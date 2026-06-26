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


def test_optimize_band_all_resident_floor():
    # Roomy cap -> band keeps everything resident -> energy == first-touch A/W floor, gap == 0.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    res = b.optimize_band(w, hw)
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    floor = (2 * 2048 + 4 * 2048) * coef                              # 2 A + 4 W tiles, 2048 bits each
    assert res["feasible"] is True and res["source"] == "band"
    assert res["proven_optimal"] is False
    assert res["energy"] == pytest.approx(floor)
    assert res["lower_bound"] == pytest.approx(floor)
    assert res["gap"] == pytest.approx(0.0, abs=1e-9)


def test_optimize_band_parity_and_not_below_optimum():
    # band energy == eval_sched(its order); and band cannot beat the free optimum (oracle).
    import oracle as o
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)  # cap 65536, pressure
    res = b.optimize_band(w, hw)
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["energy"] == pytest.approx(res["energy"])                # module never self-computes cost
    opt = o.dp_optimal(w, hw)["energy"]
    assert res["energy"] >= opt - 1e-6                                # restriction cannot beat optimum


def test_capacity_infeasible_reported():
    w = s.Work(M=32, K=32, N=32, wbits=[[8]], act_bits=8)             # one cube A+W+C=49152
    hw = s.HW(bank_size=1, banks=1, dram_bw=10 ** 12, word_bits=32)   # cap 32 << one cube
    res = b.optimize_band(w, hw)
    assert res["feasible"] is False and res["energy"] == float("inf")
    assert res["gap"] == float("inf") and "capacity" in res["reason"]


def test_stall_infeasible_reported():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=1)                    # roomy cap, tiny BW
    res = b.optimize_band(w, hw)
    assert res["feasible"] is False and res["energy"] == float("inf")
    assert "stall" in res["reason"] and res["min_steady_stall"] is not None


@pytest.mark.parametrize("M,K,N,wb,act,bank_size,word_bits", [
    (64, 64, 32, [[2, 2], [2, 2]], 2, 1024, 32),
    (32, 64, 64, [[2, 8]], 2, 1024, 32),
    (64, 64, 32, [[2, 4], [2, 2]], 2, 2, 1024),
])
def test_band_never_below_optimum(M, K, N, wb, act, bank_size, word_bits):
    import oracle as o
    w = s.Work(M=M, K=K, N=N, wbits=wb, act_bits=act)
    hw = s.HW(bank_size=bank_size, banks=32, dram_bw=10 ** 12, word_bits=word_bits)
    res = b.optimize_band(w, hw)
    if res["feasible"]:
        assert res["energy"] >= o.dp_optimal(w, hw)["energy"] - 1e-6


def test_band_selftest_runs():
    b.selftest()
