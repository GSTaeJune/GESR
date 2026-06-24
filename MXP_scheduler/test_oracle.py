# MXP_scheduler/test_oracle.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import oracle as o


def test_oracle_all_resident_optimum_is_first_touch_floor():
    # Everything fits -> the optimal schedule has zero reloads/spills. Its energy is the
    # mandatory first-touch A/W load floor (NOT zero): 2 A x 2048 + 4 W x 2048 = 12288 bits.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)   # MT=2,KT=2,NT=1, T=4
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)             # big cap, infinite BW
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    res = o.dp_optimal(w, hw)
    assert res["energy"] == 12288 * coef
    assert res["proven_optimal"] is True
    # the returned schedule re-scores to the same energy via the single source of truth
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["energy"] == 12288 * coef and r["feasible"] is True


def test_oracle_matches_eval_sched_on_returned_schedule():
    w = s.Work(M=32, K=64, N=32, wbits=[[2, 2]], act_bits=2)          # MT=1,KT=2,NT=1, T=2
    # cap = 40960 holds the full footprint exactly (2A+2W+1C = 4096+4096+32768) -> optimum
    # has zero variable energy; the returned schedule re-scores identically via apply_cube.
    hw = s.HW(bank_size=40, banks=1, dram_bw=10 ** 12, word_bits=1024)  # cap_bits = 40960
    res = o.dp_optimal(w, hw)
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["energy"] == res["energy"]
    assert res["proven_optimal"] is True


def test_oracle_stall0_constrained_no_better_than_unconstrained():
    # stall=0 is a HARD CONSTRAINT: the stall0-constrained optimum can only be >= the
    # unconstrained optimum (smaller feasible set). Finite BW so the constraint can bind.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)  # T=4, mixed W
    hw = s.HW(bank_size=2, banks=32, dram_bw=64, word_bits=1024)      # cap 65536, finite BW
    free = o.dp_optimal(w, hw, stall0=False)
    try:
        constrained = o.dp_optimal(w, hw, stall0=True)
        assert constrained["energy"] >= free["energy"] - 1e-9
        # the stall0 schedule must actually BE stall0-feasible when re-scored
        r = es.eval_sched(w, hw, constrained["order"], constrained["evictions"])
        assert r["stall0_feasible"] is True
    except ValueError:
        pass   # acceptable: no stall=0-feasible schedule exists at this BW (constraint empty)


def test_oracle_optimum_no_greater_than_any_structural():
    import warmstart as ws
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)  # T=4
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)  # cap 65536, infinite BW
    res = o.dp_optimal(w, hw)
    inc = ws.structural_incumbent(w, hw)
    assert inc is not None
    assert res["energy"] <= inc[0] + 1e-9     # global optimum <= any structural schedule
