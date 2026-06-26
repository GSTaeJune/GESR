# MXP_scheduler/test_band_sched.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import band_sched as b


def test_band_schedule_is_full_permutation_and_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 4], [2, 2]], act_bits=2)   # MT=KT=NT=2, T=8
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)             # roomy
    knobs = [(w.NT, w.KT)] * w.MT                                     # B=NT, d=KT per region
    order, evictions = b.band_schedule(w, hw, knobs)
    assert sorted(order) == sorted(es.all_cubes(w))                   # full permutation
    assert len(evictions) == len(order)
    r = es.eval_sched(w, hw, order, evictions)
    assert r["feasible"] is True                                      # feasible by construction


def test_footprint_matches_resident_peak():
    # footprint(B,d) must equal the max per-step resident bits eval_sched sees for that region.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 8]], act_bits=2)          # MT=1,KT=2,NT=2
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    B, d = 2, 2
    order, evictions = b.band_schedule(w, hw, [(B, d)])
    # replay, tracking peak resident bits
    st = es.SchedState(frozenset(), frozenset(), -1.0)
    peak = 0
    for i, c in enumerate(order):
        st, *_ = es.apply_cube(st, c, evictions[i], w, hw)
        peak = max(peak, sum(es.tile_size(t, w) for t in st.resident))
    assert peak == b.footprint_bits(w, hw, 0, B, d)
