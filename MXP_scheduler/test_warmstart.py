# MXP_scheduler/test_warmstart.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import warmstart as ws


def test_mapping_to_schedule_is_a_valid_permutation():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    m = s.Mapping(perm=("K", "M", "N"), m_in=1, k_in=1, n_in=1)
    order, evictions = ws.mapping_to_schedule(m, w, hw)
    assert sorted(order) == sorted(es.all_cubes(w))      # each cube exactly once
    assert len(evictions) == len(order)


def test_mapping_to_schedule_scores_feasibly_when_capacity_allows():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=10 ** 12)   # big cap + infinite BW
    m = s.Mapping(perm=("K", "M", "N"), m_in=1, k_in=1, n_in=1)
    order, evictions = ws.mapping_to_schedule(m, w, hw)
    r = es.eval_sched(w, hw, order, evictions)
    assert r["capacity_feasible"] is True
    assert r["stall0_feasible"] is True                     # infinite BW hides everything


def test_mapping_to_schedule_respects_tight_capacity():
    # Tight cap forces evictions; the produced schedule must still never exceed capacity.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    # cap holds ~1.5 cube working sets -> eviction required, but feasible
    hw = s.HW(bank_size=2, banks=32, dram_bw=32, word_bits=1024)  # cap_bits = 2*32*1024 = 65536
    m = s.Mapping(perm=("K", "M", "N"), m_in=1, k_in=1, n_in=1)
    order, evictions = ws.mapping_to_schedule(m, w, hw)
    r = es.eval_sched(w, hw, order, evictions)
    assert r["capacity_feasible"] is True                   # warm-start never overflows


def test_structural_incumbent_returns_best_stall0_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # infinite BW -> many stall0-feasible
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    inc = ws.structural_incumbent(w, hw)
    assert inc is not None
    energy, order, evictions = inc
    r = es.eval_sched(w, hw, order, evictions)
    assert r["stall0_feasible"] is True
    assert r["energy"] == energy
    # all-resident is achievable here -> no reloads/spills, energy = first-touch A/W floor
    # (4 A x 8192 + 4 W x 8192 = 65536 bits), NOT zero.
    assert energy == 65536 * coef


def test_structural_incumbent_none_when_nothing_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    # cap too small for even one cube working set AND bw tiny -> no stall0-feasible structural sched
    hw = s.HW(bank_size=1, banks=1, dram_bw=1, word_bits=32)   # cap_bits = 32
    assert ws.structural_incumbent(w, hw) is None


def test_min_structural_steady_stall_reports_positive_when_bw_tight():
    # Capacity fits, but finite BW makes every structural schedule stall mid-stream.
    # The diagnostic must report a POSITIVE min steady_stall (not None, not 0).
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=4)             # big cap, tiny BW -> stalls
    assert ws.structural_incumbent(w, hw) is None              # nothing is stall0-feasible
    diag = ws.min_structural_steady_stall(w, hw)
    assert diag is not None and diag > 0                       # explains how far from stall=0


def test_min_structural_steady_stall_zero_when_bw_huge():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=10 ** 12)      # infinite BW -> stall0 reachable
    assert ws.min_structural_steady_stall(w, hw) == 0.0
