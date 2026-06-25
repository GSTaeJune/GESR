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
