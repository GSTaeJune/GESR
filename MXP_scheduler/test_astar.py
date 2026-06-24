# MXP_scheduler/test_astar.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import oracle as o
import astar as a


def test_optimize_exact_all_resident_first_touch_floor_proven():
    # Everything fits -> optimum has no reloads/spills; energy = first-touch A/W floor
    # (2 A x 2048 + 4 W x 2048 = 12288 bits), NOT zero. h0 == this floor, so A* proves it at
    # the root (nodes=0) via the bound.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)   # T=4
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    res = a.optimize_exact(w, hw)
    assert res["energy"] == 12288 * coef
    assert res["proven_optimal"] is True
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["feasible"] is True and r["energy"] == 12288 * coef


def test_astar_search_expands_and_matches_oracle():
    # A config where the warm-start incumbent is NOT already optimal, so A* must actually
    # EXPAND nodes (not just bound-break at the root). Exercises the search machinery itself.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 4]], act_bits=2)            # MT=1,KT=2,NT=2, T=4
    hw = s.HW(bank_size=38, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 38912: real pressure
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])
    assert exact["nodes_expanded"] > 0           # genuine search, not a root bound-break


@pytest.mark.parametrize("M,K,N,wb,act,bank_size,word_bits", [
    (64, 64, 32, [[2, 2], [2, 2]], 2, 2, 1024),    # cap 65536, T=4, forced pressure
    (32, 64, 32, [[2, 2]], 2, 40, 1024),           # cap 40960, T=2
    (64, 64, 32, [[2, 4], [2, 2]], 2, 2, 1024),    # mixed-precision W sizes, T=4
])
def test_astar_matches_oracle(M, K, N, wb, act, bank_size, word_bits):
    w = s.Work(M=M, K=K, N=N, wbits=wb, act_bits=act)
    hw = s.HW(bank_size=bank_size, banks=32, dram_bw=10 ** 12, word_bits=word_bits)
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])


def test_astar_deterministic():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)
    r1 = a.optimize_exact(w, hw)
    r2 = a.optimize_exact(w, hw)
    assert r1["order"] == r2["order"]
    assert r1["evictions"] == r2["evictions"]
    assert r1["energy"] == r2["energy"]


def test_astar_stall0_hard():
    # Finite BW so big mid-stream transfers cannot be hidden -> stall0 prunes them.
    # The returned schedule, scored by eval_sched, MUST be stall0_feasible.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)   # finite BW
    res = a.optimize_exact(w, hw)
    if res["feasible"]:                               # a feasible schedule was found
        r = es.eval_sched(w, hw, res["order"], res["evictions"])
        assert r["stall0_feasible"] is True
        assert r["feasible"] is True


def test_astar_matches_oracle_finite_bw():
    # Finite BW so stall=0 BINDS: A* (hard stall=0) must match the stall0-constrained oracle,
    # not just the unconstrained one. Closes the finite-BW validation gap.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)   # T=4 mixed
    hw = s.HW(bank_size=2, banks=32, dram_bw=256, word_bits=1024)      # cap 65536, finite BW
    exact = a.optimize_exact(w, hw)
    if exact["feasible"]:
        ref = o.dp_optimal(w, hw, stall0=True)
        assert exact["proven_optimal"] is True
        assert exact["energy"] == pytest.approx(ref["energy"])
    else:
        with pytest.raises(ValueError):              # oracle must also find nothing stall0-feasible
            o.dp_optimal(w, hw, stall0=True)


def test_astar_prime_tile_count_matches_oracle():
    # KT=5 (PRIME): the divisor-only closed-form gen_mappings cannot express 5-way ragged
    # blocking, but the joint optimizer can. K=160 -> KT=5; MT=NT=1 -> T=5 (<= ORACLE_MAX_T).
    # Capacity pressure forces real reload choices; A* must equal the exhaustive oracle.
    w = s.Work(M=32, K=160, N=32, wbits=[[2, 4, 8, 2, 6]], act_bits=2)  # MT=1,KT=5,NT=1,T=5
    hw = s.HW(bank_size=48, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 49152 < footprint
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])


def _w_load_counts(w, hw, order, evictions):
    """Replay a schedule and count charged W-tile loads per tile (W loads are always charged).
    Uses apply_cube for state advance (single source of truth) -- no cost logic duplicated."""
    import collections
    st = es.SchedState(frozenset(), frozenset(), -1.0)
    counts = collections.Counter()
    for i, c in enumerate(order):
        wt = es.w_tile(c)
        if wt not in st.resident:                    # not resident before this step -> a load
            counts[wt] += 1
        st = es.apply_cube(st, c, frozenset(evictions[i]), w, hw)[0]
    return counts


def test_astar_precision_adaptive_residency():
    # Mixed precision: W(0,1) is 8-bit (8192 bits, expensive), W(0,0) is 2-bit (2048, cheap).
    # Under capacity pressure with N-reuse, the EXACT optimum amortizes the expensive fetch:
    # it loads the expensive W no more often than the cheap W. Ground truth via the oracle.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 8]], act_bits=2)            # MT=1,KT=2,NT=2,T=4
    hw = s.HW(bank_size=48, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 49152 < footprint
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])             # exact, incl. mixed precision
    counts = _w_load_counts(w, hw, exact["order"], exact["evictions"])
    # the expensive 8-bit W is loaded no more often than the cheap 2-bit W (amortization)
    assert counts[("W", 0, 1)] <= counts[("W", 0, 0)]


def test_astar_empty_region_diagnostic():
    # Tiny BW + multi-cube -> NO stall=0-feasible schedule. Must NOT return a bare inf:
    # feasible=False, a positive min_steady_stall, and a human reason (invariant #11).
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=1)   # big cap (no reload), tiny BW -> all stall
    res = a.optimize_exact(w, hw)
    assert res["feasible"] is False
    assert res["proven_optimal"] is False
    assert res["min_steady_stall"] is not None and res["min_steady_stall"] > 0
    assert "stall=0" in res["reason"]


def test_astar_never_worse_than_structural_floor():
    import warmstart as ws
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=10 ** 12)   # all-resident achievable
    inc = ws.structural_incumbent(w, hw)
    res = a.optimize_exact(w, hw)
    assert inc is not None
    assert res["energy"] <= inc[0] + 1e-9        # D6 floor: M1 never regresses below structural


def test_astar_reports_gap_when_budget_zero():
    # cap = 36*32*32 = 36864 = exactly one cube working set -> the warm-start incumbent must
    # reload A/W between cubes, so its energy > h0(start). With node_budget=0 the search stops
    # before expanding any node, so optimality is NOT proven and an honest gap is reported.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=36, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap_bits = 36864
    res = a.optimize_exact(w, hw, node_budget=0)
    assert res["feasible"] is True                 # warm-start schedule exists
    assert res["proven_optimal"] is False          # budget hit -> not proven
    assert res["nodes_expanded"] == 0
    assert res["gap"] > 0.0
    assert res["lower_bound"] <= res["energy"] + 1e-9


def test_astar_selftest_runs():
    a.selftest()   # prints "astar selftest: OK"; raises on any golden mismatch
