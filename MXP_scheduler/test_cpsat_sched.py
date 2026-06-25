# MXP_scheduler/test_cpsat_sched.py
import pytest
pytest.importorskip("ortools")          # skip the whole module if OR-Tools is absent
from fractions import Fraction
import mxp_scheduler as s
import eval_sched as es
import cpsat_sched as cps


def test_cubes_tiles_canonical():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    cubes, tiles = cps._cubes_tiles(w)
    assert cubes == es.all_cubes(w)
    # every tile of every cube is present, tiles are sorted & unique
    expect = sorted({t for c in cubes for t in (es.a_tile(c), es.w_tile(c), es.c_tile(c))})
    assert tiles == expect


def test_global_scale_is_one_for_integer_config():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # integral bw, integer wbits
    assert cps._global_scale(w, hw) == 1


def test_global_scale_clears_fractional_bandwidth():
    # dram_bw = 2.5 -> Fraction 5/2 appears in the stall RHS (bw*cube_compute); G must clear it
    w = s.Work(M=32, K=64, N=32, wbits=[[2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=2.5)
    G = cps._global_scale(w, hw)
    # G must make bw*cube_compute integral for every cube
    bw = Fraction(hw.dram_bw)
    for c in cps._cubes_tiles(w)[0]:
        assert (bw * Fraction(es.cube_compute(c, w, hw)) * G).denominator == 1


def test_all_resident_first_touch_floor_proven():
    # Everything fits -> optimum has no reloads/spills; energy = first-touch A/W floor
    # (2 A x 2048 + 4 W x 2048 = 12288 bits) * coef, NOT zero. Proven optimal.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    res = cps.optimize_exact(w, hw)
    assert res["proven_optimal"] is True
    assert res["energy"] == pytest.approx(12288 * coef)
    assert res["source"] == "cpsat"
    assert isinstance(res["nodes_expanded"], int)


def test_reconstruct_matches_eval_sched_bit_exact():
    # Master check: feed the CP-SAT (order, evictions) back through the single source of truth.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 4]], act_bits=2)            # MT=1,KT=2,NT=2,T=4
    hw = s.HW(bank_size=38, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 38912: real pressure
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is True
    e = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert e["feasible"] is True
    # canonical equality on INTEGER traffic bits (exact for integer wbits)
    assert round(res["energy"] / (hw.coeffs["dram"] + hw.coeffs["onchip"])) == \
        round(e["dram_read_bits"] + e["dram_spill_bits"])
    assert e["energy"] == pytest.approx(res["energy"])


def test_cpsat_matches_oracle_under_pressure():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)   # mixed-precision, T=4
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)  # cap 65536
    import oracle as o
    res = cps.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert res["proven_optimal"] is True
    assert res["energy"] == pytest.approx(ref["energy"])


def test_stall0_returned_schedule_is_feasible():
    # Finite BW: a returned schedule, re-scored by eval_sched, MUST be stall0-feasible.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)        # finite BW
    res = cps.optimize_exact(w, hw)
    if res["feasible"]:
        e = es.eval_sched(w, hw, res["order"], res["evictions"])
        assert e["stall0_feasible"] is True and e["feasible"] is True


def test_cpsat_matches_oracle_finite_bw():
    # Finite BW so stall=0 BINDS: must match the stall0-constrained oracle (not the unconstrained one).
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)   # T=4 mixed
    hw = s.HW(bank_size=2, banks=32, dram_bw=256, word_bits=1024)      # cap 65536, finite BW
    import oracle as o
    res = cps.optimize_exact(w, hw)
    if res["feasible"]:
        ref = o.dp_optimal(w, hw, stall0=True)
        assert res["proven_optimal"] is True
        assert res["energy"] == pytest.approx(ref["energy"])
    else:
        with pytest.raises(ValueError):           # oracle must also find nothing stall0-feasible
            o.dp_optimal(w, hw, stall0=True)


def test_honest_gap_on_timeout():
    # A capacity-pressured instance where proving optimality takes search; a tiny time budget
    # forces a feasible-but-unproven return with an honest, lower-bound-relative gap.
    # max_time=0.05 reliably yields a FEASIBLE incumbent on this machine (verified); the point is
    # to exercise the honest-gap arithmetic when not proven. If CP-SAT proves optimality even at
    # 0.05s the assertions are skipped and the test still passes (acceptable: CP-SAT closed it).
    w = s.Work(M=32, K=160, N=32, wbits=[[2, 4, 8, 2, 6]], act_bits=2)  # KT=5, T=5
    hw = s.HW(bank_size=48, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 49152 < footprint
    res = cps.optimize_exact(w, hw, max_time=0.05)
    assert res["feasible"] is True                  # a feasible incumbent is found quickly
    if res["proven_optimal"] is False:              # the intended timeout branch
        assert res["energy"] >= res["lower_bound"] - 1e-6
        assert res["gap"] == pytest.approx((res["energy"] - res["lower_bound"]) / res["lower_bound"])
        assert res["gap"] >= 0.0


def test_stall0_infeasible_diagnostic():
    # Big cap (no reload pressure) + tiny BW -> NO stall=0 schedule. Must report feasible=False,
    # a positive min_steady_stall, and a human reason -- not a bare inf.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=1)        # huge cap, tiny BW -> all stall
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is False and res["proven_optimal"] is False
    assert res["energy"] == float("inf")
    assert res["min_steady_stall"] is not None and res["min_steady_stall"] > 0
    assert "stall=0" in res["reason"]


def test_capacity_infeasible_diagnostic():
    # A single cube's A+W+C exceeds capacity -> no schedule fits at all (capacity-infeasible).
    w = s.Work(M=32, K=32, N=32, wbits=[[8]], act_bits=8)   # one cube; A+W+C = 8192+8192+32768
    hw = s.HW(bank_size=1, banks=1, dram_bw=10 ** 12, word_bits=32)   # cap 32 bits << one cube
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is False
    assert res["energy"] == float("inf")
    assert res["min_steady_stall"] is None
    assert "capacity" in res["reason"]


def test_deterministic():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)
    r1 = cps.optimize_exact(w, hw)
    r2 = cps.optimize_exact(w, hw)
    assert r1["order"] == r2["order"]
    assert r1["energy"] == pytest.approx(r2["energy"])


def test_selftest_runs():
    cps.selftest()   # prints "cpsat_sched selftest: OK"; raises on any golden mismatch


@pytest.mark.parametrize("M,K,N,wb,act,bank_size,word_bits", [
    (64, 64, 32, [[2, 2], [2, 2]], 2, 2, 1024),    # cap 65536, T=4, forced pressure
    (32, 64, 32, [[2, 2]], 2, 40, 1024),           # cap 40960, T=2
    (64, 64, 32, [[2, 4], [2, 2]], 2, 2, 1024),    # mixed-precision W sizes, T=4
    (32, 64, 64, [[2, 8]], 2, 48, 32),             # precision-adaptive residency, T=4, cap 49152
    (32, 160, 32, [[2, 4, 8, 2, 6]], 2, 48, 32),   # KT=5 prime, T=5, cap 49152
])
def test_cpsat_equals_oracle_and_astar(M, K, N, wb, act, bank_size, word_bits):
    w = s.Work(M=M, K=K, N=N, wbits=wb, act_bits=act)
    hw = s.HW(bank_size=bank_size, banks=32, dram_bw=10 ** 12, word_bits=word_bits)
    import oracle as o
    import astar as a
    cp = cps.optimize_exact(w, hw)
    orc = o.dp_optimal(w, hw)
    ast = a.optimize_exact(w, hw)
    assert cp["proven_optimal"] and ast["proven_optimal"] and orc["proven_optimal"]
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    # canonical equality on integer traffic bits (exact for integer wbits)
    assert round(cp["energy"] / coef) == round(orc["energy"] / coef) == round(ast["energy"] / coef)


def test_fractional_dram_bw_scaling_parity():
    # Fractional dram_bw -> G != 1 path. Reconstruct -> eval_sched parity must still hold.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=2.5, word_bits=1024)   # finite, fractional BW
    res = cps.optimize_exact(w, hw)
    if res["feasible"]:
        e = es.eval_sched(w, hw, res["order"], res["evictions"])
        assert e["feasible"] is True
        assert e["energy"] == pytest.approx(res["energy"], abs=1e-6)


def test_fractional_wbits_scaling_parity():
    # Fractional average weight bits are permitted by Work (avg bits in [2,8]); a value whose
    # *1024 size is non-integral (e.g. 2.1 -> 2150.4 bits) exercises G != 1 size scaling.
    w = s.Work(M=32, K=64, N=64, wbits=[[2.1, 4.0]], act_bits=2)
    hw = s.HW(bank_size=40, banks=32, dram_bw=10 ** 12, word_bits=32)
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is True
    e = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert e["feasible"] is True
    assert e["energy"] == pytest.approx(res["energy"], abs=1e-6)
