# MXP_scheduler/test_mxp_scheduler.py
import pytest
import mxp_scheduler as s


def test_tile_counts_and_total_w_bits():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    assert (w.MT, w.KT, w.NT) == (2, 2, 2)
    # total_w_bits = TILE*TILE * sum(all wbits) = 1024 * 32 = 32768
    assert w.total_w_bits == 32768


def test_divisors():
    assert s.divisors(1) == [1]
    assert s.divisors(4) == [1, 2, 4]
    assert s.divisors(32) == [1, 2, 4, 8, 16, 32]


def test_work_validates_wbits_shape():
    with pytest.raises(ValueError):
        s.Work(M=64, K=64, N=64, wbits=[[8, 8]], act_bits=8)  # wrong rows (MT=2)
    with pytest.raises(ValueError):
        s.Work(M=64, K=64, N=64, wbits=[[8], [8]], act_bits=8)  # wrong cols (KT=2)
    with pytest.raises(ValueError):
        s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=3)  # act not in {2,4,8}


def test_work_rejects_non_tile_multiple_and_out_of_range_wbits():
    with pytest.raises(ValueError):
        s.Work(M=40, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)   # 40 not multiple of 32
    with pytest.raises(ValueError):
        s.Work(M=0, K=64, N=64, wbits=[], act_bits=8)                  # zero dim
    with pytest.raises(ValueError):
        s.Work(M=64, K=64, N=64, wbits=[[9, 8], [8, 8]], act_bits=8)   # 9 out of [2,8]
    with pytest.raises(ValueError):
        s.Work(M=64, K=64, N=64, wbits=[[0, 8], [8, 8]], act_bits=8)   # 0 out of [2,8]


def test_work_accepts_fractional_avg_wbits():
    # per-tile AVERAGE bits can be fractional and in-range (mixed weights in a tile)
    w = s.Work(M=64, K=64, N=64, wbits=[[2.5, 7.5], [8, 2]], act_bits=8)
    assert (w.MT, w.KT) == (2, 2)


def test_hw_capacity_bits():
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    assert hw.cap_bits == 1024 * 32 * 32  # bank_size*banks*word_bits = 1048576


def test_gen_mappings_count():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    ms = s.gen_mappings(w)
    # 6 perms x divisors(2)^3 = 6 x 2^3 = 48
    assert len(ms) == 48
    # every blocking factor divides its tile count
    for m in ms:
        assert w.MT % m.m_in == 0 and w.KT % m.k_in == 0 and w.NT % m.n_in == 0
        assert set(m.perm) == {"M", "K", "N"}


def test_footprint_and_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    # all-resident mapping: m_in=k_in=n_in=2
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    # foot(A)=k_in*n_in*TILE^2*act = 2*2*1024*8 = 32768
    # foot(W)=m_in*k_in*TILE^2*max_wbits = 2*2*1024*8 = 32768
    # foot(C)=m_in*n_in*TILE^2*32 = 2*2*1024*32 = 131072
    assert s.footprint_bits(m, w) == 32768 + 32768 + 131072  # 196608
    big = s.HW(bank_size=1024, banks=32, dram_bw=64)   # cap=1048576
    tiny = s.HW(bank_size=1, banks=2, dram_bw=64)       # cap=64
    assert s.feasible(m, w, big) is True
    assert s.feasible(m, w, tiny) is False


def test_out_in_factors_roundtrip():
    w = s.Work(M=128, K=64, N=96, wbits=[[8] * 2 for _ in range(4)], act_bits=8)
    # tile counts: MT=4, KT=2, NT=3
    m = s.Mapping(perm=("M", "K", "N"), m_in=2, k_in=1, n_in=3)
    out, inn = s._out_in(m, w)
    assert inn == {"M": 2, "K": 1, "N": 3}
    assert out == {"M": 2, "K": 2, "N": 1}   # MT//m_in, KT//k_in, NT//n_in
    # outer*inner reconstructs the tile count along every dim
    assert out["M"] * inn["M"] == w.MT
    assert out["K"] * inn["K"] == w.KT
    assert out["N"] * inn["N"] == w.NT


def test_dram_bits_all_resident():
    # G1: 64^3, all-resident (K_out=1 -> no spill), uniform wbits=8, act=8
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    d = s.dram_bits(m, w)
    assert d["A"] == 64 * 64 * 8 * 1        # K*N*act * M_out(=1) = 32768
    assert d["W"] == 32768 * 1              # total_w_bits * N_out(=1)
    assert d["Cw"] == 64 * 64 * 32 * 1      # M*N*32 * K_out(=1) = 131072
    assert d["Cr"] == 0                     # K_out-1 = 0 (first touch zero-init)
    assert d["total"] == 32768 + 32768 + 131072 + 0  # 196608


def test_dram_bits_kspill():
    # G2: real psum spill needs K_out=2 (k_in=1) AND a non-trivial loop inner to K so the
    # C tile is re-touched. perm=("K","M","N") with n_in=1 -> N (inner to K) has out=2 -> spill.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=1)
    d = s.dram_bits(m, w)
    assert d["Cw"] == 64 * 64 * 32 * 2     # K_out=2 -> 262144
    assert d["Cr"] == 64 * 64 * 32 * 1     # (K_out-1)=1 -> 131072


def test_onchip_mac_rmw_and_compute_work():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    # MAC = M*K*N
    assert s.mac_ops(w) == 64 * 64 * 64                   # 262144
    # RMW = T * 1024 * disp; T=8, disp(act=8)=1
    assert s.rmw_ops(w) == 8 * 1024 * 1                   # 8192
    # compute_work = 32 * NT * sum(all wbits) = 32 * 2 * 32 = 2048
    assert s.compute_work(w) == 2048
    # onchip (G1 all-resident): A_rd=T*1024*act=8*1024*8=65536 ; W_rd=NT*total_w=2*32768=65536 ;
    # refill = DRAM(A)+DRAM(W)+DRAM(Cr) = 32768+32768+0 = 65536 ; total=196608
    assert s.onchip_bits(m, w) == 65536 + 65536 + 65536   # 196608


def test_rmw_disp_low_precision():
    w2 = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=2)
    assert s.rmw_ops(w2) == 8 * 1024 * 4                   # disp(act=2)=4
    w4 = s.Work(M=64, K=64, N=64, wbits=[[4, 4], [4, 4]], act_bits=4)
    assert s.rmw_ops(w4) == 8 * 1024 * 2                   # disp(act=4)=2 (middle dispatch value)


def test_onchip_kspill_refill():
    # onchip with K-spill (Cr>0): refill must include the psum reload (live code path).
    # perm=("K","M","N") n_in=1 -> N inner to K is non-trivial -> genuine spill.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=1)   # K_out=2, N inner -> Cr>0
    # a_rd=65536 ; w_rd=65536 ; refill=A(32768)+W(32768)+Cr(131072)=196608
    assert s.onchip_bits(m, w) == 65536 + 65536 + 196608   # 327680


def test_mixed_wbits_compute_and_footprint():
    # non-uniform per-tile avg bits: compute_work sums EVERY tile; footprint uses global max
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [8, 2]], act_bits=8)
    # compute_work = TILE * NT * Σwbits = 32 * 2 * (2+8+8+2=20) = 1280
    assert s.compute_work(w) == 1280
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    # foot W term now uses the RESIDENT-WINDOW sum (spec §6.1): all 4 tiles resident -> 2+8+8+2=20
    assert s.footprint_bits(m, w) == 32768 + 20480 + 131072  # 184320


def test_out_in_rejects_non_divisor_mapping():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)   # MT=KT=NT=2
    bad = s.Mapping(perm=("M", "K", "N"), m_in=3, k_in=3, n_in=3)       # 3 does not divide 2
    with pytest.raises(ValueError):
        s._out_in(bad, w)
    # the corruption it used to cause (negative Cr / total<0) is now blocked at the source
    with pytest.raises(ValueError):
        s.dram_bits(bad, w)
    with pytest.raises(ValueError):
        s.footprint_bits(bad, w)
    with pytest.raises(ValueError):
        s.feasible(bad, w, s.HW(1024, 32, 64))


def test_footprint_resident_window_below_global_max():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=1, n_in=2)  # window = one K-column (2 tiles)
    # worst K-column resident window = tiles [2,8] -> sum 10 -> foot_w = 10*1024 = 10240
    # (the old global-max would have used m_in*k_in*8 = 16 -> 16384, an over-count)
    foot_a = 1 * 2 * 1024 * 8     # k_in*n_in*TILE^2*act = 16384
    foot_c = 2 * 2 * 1024 * 32    # m_in*n_in*TILE^2*FP32 = 131072
    assert s.footprint_bits(m, w) == foot_a + 10240 + foot_c   # 157696


def test_energy_breakdown_g1():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)   # default coeffs 200/6/1/5
    e = s.energy_breakdown(m, w, hw)
    assert e["dram"] == 196608 * 200      # 39321600
    assert e["onchip"] == 196608 * 6      # 1179648
    assert e["mac"] == 262144 * 1         # 262144
    assert e["rmw"] == 8192 * 5           # 40960
    assert e["total"] == 39321600 + 1179648 + 262144 + 40960   # 40804352


def test_stall_fill_g3():
    # G3: 2-bit weights -> tiny compute (256/block) but FP32 C psum dominates the DRAM stall.
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m = s.Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    fill, steady, drain = s.stall_fill(m, w, hw)
    # fill = (a0+w0)/bw = (16384+8192)/32 = 768
    # boundary 0->1: A reload(16384)+C write(65536); demand=81920; steady = 81920/32 - 256 = 2304
    # trailing drain = c_blk(65536)/32 = 2048
    assert fill == 768.0
    assert steady == 2304.0
    assert drain == 2048.0
    # actual = compute_work + fill + steady + drain = 512 + 768 + 2304 + 2048 = 5632
    assert s.actual_cycle(m, w, hw) == 5632.0


def test_stall_zero_when_bw_huge():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("M", "K", "N"), m_in=2, k_in=2, n_in=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # effectively infinite BW
    fill, steady, drain = s.stall_fill(m, w, hw)
    assert steady < 1.0
    assert drain < 1.0
    assert fill < 1.0


def test_evaluate_keys():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    r = s.evaluate(m, w, hw)
    assert r["feasible"] is True
    assert r["energy"] == 40804352.0
    assert r["actual_cycle"] == s.actual_cycle(m, w, hw)
    assert r["mapping"] is m


def test_optimize_picks_min_energy_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    ranked = s.optimize(w, hw)
    assert all(r["feasible"] for r in ranked)
    assert ranked == sorted(ranked, key=lambda r: (r["energy"], r["actual_cycle"]))
    # Order-dependent energy: the min-energy mapping hits the full-reuse DRAM floor (each of
    # A, W, C moved through DRAM exactly once, no psum spill). NOTE this is NOT uniquely the
    # all-resident mapping any more -- a K-tiled mapping with K innermost is output-stationary
    # so it spills nothing and ties the floor at a smaller footprint.
    best = ranked[0]
    floor = 64 * 64 * 8 + 32768 + 64 * 64 * 32   # A + W + C, each once = 196608
    assert best["dram"]["total"] == floor
    assert best["dram"]["Cr"] == 0               # no psum spill in the optimum


def test_optimize_cycle_constraint_filters():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    tight = min(s.actual_cycle(m, w, hw)
                for m in s.gen_mappings(w) if s.feasible(m, w, hw))
    ranked = s.optimize(w, hw, max_cycle=tight)
    assert len(ranked) >= 1
    assert all(r["actual_cycle"] <= tight + 1e-9 for r in ranked)


def test_lpt_headroom_runs_and_reports_both():
    w = s.Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    best = s.optimize(w, hw)[0]["mapping"]
    h = s.lpt_headroom(best, w, hw)
    assert set(h) == {"natural_stall", "lpt_stall", "headroom"}
    assert h["headroom"] == h["natural_stall"] - h["lpt_stall"]
    assert h["natural_stall"] >= 0 and h["lpt_stall"] >= 0


# --- cross-model review hardening (Tasks 5-8) ---

def test_compute_work_fractional_no_truncation():
    # fractional avg wbits must NOT be int()-truncated (would desync from per-block compute)
    w = s.Work(M=32, K=32, N=32, wbits=[[2.1]], act_bits=8)
    cw = s.compute_work(w)
    assert cw == pytest.approx(32 * 1 * 2.1)   # 67.2
    assert cw > 67                             # exactly 67 if int()-truncated -> would fail


def test_dram_bits_fractional_no_truncation():
    # W traffic is fractional when avg wbits is fractional; must stay exact
    w = s.Work(M=32, K=32, N=32, wbits=[[2.1]], act_bits=8)
    m = s.Mapping(perm=("M", "K", "N"), m_in=1, k_in=1, n_in=1)
    d = s.dram_bits(m, w)
    assert d["W"] == pytest.approx(1024 * 2.1)  # 2150.4
    assert d["W"] > 2150                         # 2150 if int()-truncated


def test_optimize_tiebreak_by_cycle():
    # energy is order-dependent in general, but ties still exist (e.g. all-resident mappings
    # have identical DRAM traffic across perms); ties must break by ascending actual_cycle,
    # not by gen_mappings() insertion order (a hidden preference)
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    ranked = s.optimize(w, hw)
    assert ranked == sorted(ranked, key=lambda r: (r["energy"], r["actual_cycle"]))
    energies = [r["energy"] for r in ranked]
    assert len(energies) != len(set(energies))   # ties really are present


def test_hw_rejects_nonpositive_fields():
    with pytest.raises(ValueError):
        s.HW(bank_size=1024, banks=32, dram_bw=0)      # div-by-zero in stall
    with pytest.raises(ValueError):
        s.HW(bank_size=1024, banks=32, dram_bw=-8)     # negative -> impossible cycles
    with pytest.raises(ValueError):
        s.HW(bank_size=0, banks=32, dram_bw=64)        # zero capacity


def test_mapping_rejects_bad_perm():
    with pytest.raises(ValueError):
        s.Mapping(perm=("M", "K", "K"), m_in=1, k_in=1, n_in=1)   # not a permutation
    with pytest.raises(ValueError):
        s.Mapping(perm=("M", "K"), m_in=1, k_in=1, n_in=1)        # wrong length


def test_optimize_feasibility_filter_drops_infeasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=100, banks=32, dram_bw=64)   # cap=100*32*32=102400: some fit, some don't
    ranked = s.optimize(w, hw)
    assert 0 < len(ranked) < len(s.gen_mappings(w))   # filter keeps some AND drops some
    assert all(r["feasible"] for r in ranked)


def test_evaluate_infeasible_still_costs():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    tiny = s.HW(bank_size=1, banks=2, dram_bw=64)     # cap=64 bits, nothing fits
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    r = s.evaluate(m, w, tiny)
    assert r["feasible"] is False
    assert r["energy"] > 0 and r["actual_cycle"] > 0  # still a finite cost


def test_energy_breakdown_kspill_costs_more():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    resident = s.energy_breakdown(s.Mapping(("N", "K", "M"), 2, 2, 2), w, hw)
    kspill = s.energy_breakdown(s.Mapping(("K", "M", "N"), 2, 1, 1), w, hw)  # K_out=2, N inner -> spill
    assert kspill["dram"] > resident["dram"]
    assert kspill["total"] > resident["total"]


def test_lpt_headroom_nontrivial_stall():
    # multi-block mapping + low bw -> real (non-zero) stalls, exercising the actual walk
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [8, 2]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=8)
    m = s.Mapping(perm=("M", "K", "N"), m_in=1, k_in=1, n_in=1)   # out=(2,2,2) -> 8 blocks
    h = s.lpt_headroom(m, w, hw)
    assert h["natural_stall"] > 0
    assert h["headroom"] == h["natural_stall"] - h["lpt_stall"]


def test_optimize_cycle_filter_reduces_set():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    full = s.optimize(w, hw)
    cycles = sorted(r["actual_cycle"] for r in full)
    assert cycles[0] < cycles[-1]                       # there is a spread to filter on
    cutoff = (cycles[0] + cycles[-1]) / 2
    mid = s.optimize(w, hw, max_cycle=cutoff)
    assert len(mid) < len(full)
    assert all(r["actual_cycle"] <= cutoff + 1e-9 for r in mid)


# --- order-dependent DRAM/energy model (FINDING A: dram now matches the stall reload basis) ---

def test_dram_A_reload_depends_on_M_position():
    # A=[K,N] reused over M: putting M innermost keeps A resident -> fewer A loads than M outer
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m_inner = s.Mapping(perm=("K", "N", "M"), m_in=1, k_in=1, n_in=2)   # M innermost
    m_outer = s.Mapping(perm=("M", "K", "N"), m_in=1, k_in=1, n_in=2)   # M outermost
    assert s.dram_bits(m_inner, w)["A"] < s.dram_bits(m_outer, w)["A"]
    assert s.dram_bits(m_inner, w)["A"] == 64 * 64 * 8          # A streamed exactly once


def test_dram_C_no_spill_when_K_innermost():
    # K tiled (k_in=1 -> K_out=2). K innermost = output-stationary -> no psum spill.
    # A genuine spill needs K outer AND a non-trivial loop inside K (here N, n_in=1 -> out=2).
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    k_inner = s.Mapping(perm=("M", "N", "K"), m_in=2, k_in=1, n_in=2)   # K innermost
    k_outer = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=1)   # K outer, N inner -> spill
    di, do = s.dram_bits(k_inner, w), s.dram_bits(k_outer, w)
    assert di["Cw"] == 64 * 64 * 32 and di["Cr"] == 0          # one write, no reload
    assert do["Cw"] == 64 * 64 * 32 * 2 and do["Cr"] == 64 * 64 * 32   # spill per K_out


def test_energy_is_perm_dependent():
    # K innermost (output-stationary, no spill) vs a perm that genuinely spills psum
    # -> different energy. Proves energy now depends on loop order, not just blocking.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    e_kin = s.energy_breakdown(s.Mapping(("M", "N", "K"), 2, 1, 2), w, hw)["total"]   # no spill
    e_spill = s.energy_breakdown(s.Mapping(("K", "M", "N"), 2, 1, 1), w, hw)["total"]  # spill
    assert e_kin < e_spill


def test_dram_and_stall_share_reload_basis():
    # FINDING A regression lock: the A/W DRAM volume must equal the input fetch the stall
    # model walks (same _blocks, same adjacency). Independently re-count here and compare.
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [8, 2]], act_bits=8)
    for m in s.gen_mappings(w):
        a_vol = w_vol = 0.0
        prev = None
        for idx, _c, a_blk, w_blk, _cb in s._blocks(m, w):
            if prev is None or (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
                a_vol += a_blk
            if prev is None or (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
                w_vol += w_blk
            prev = idx
        d = s.dram_bits(m, w)
        assert d["A"] == pytest.approx(a_vol)
        assert d["W"] == pytest.approx(w_vol)


# --- C-in-stall: cycle model now charges C psum DRAM bandwidth too ---

def test_stall_includes_c_drain():
    # single all-resident block: no mid-stream stall, only the unavoidable trailing C drain
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("M", "K", "N"), m_in=2, k_in=2, n_in=2)   # single block
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    fill, steady, drain = s.stall_fill(m, w, hw)
    c_blk = 2 * 2 * 1024 * 32          # 131072
    assert steady == 0.0               # one block -> no boundary -> no mid-stream stall
    assert drain == c_blk / 32         # trailing C drain only (4096.0)


def test_c_spill_raises_cycle():
    # a psum-spilling mapping pays C DRAM bandwidth in cycles too, beating an output-stationary one
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    out_stationary = s.actual_cycle(s.Mapping(("M", "N", "K"), 2, 1, 2), w, hw)  # K inner, no spill
    spilling = s.actual_cycle(s.Mapping(("K", "M", "N"), 2, 1, 1), w, hw)        # spills psum
    assert spilling > out_stationary


def test_stall_c_traffic_matches_dram():
    # the C reads/writes the stall walk charges must equal dram_bits Cr/Cw (one consistent basis)
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [8, 2]], act_bits=8)
    for m in s.gen_mappings(w):
        blocks = list(s._blocks(m, w))
        seen = set()
        reads = writes = 0.0
        prev = None
        for idx, _c, _a, _w, c_blk in blocks:
            mn = (idx["M"], idx["N"])
            if prev is not None and mn != prev:
                writes += c_blk                 # prev tile left -> spill write
                if mn in seen:
                    reads += c_blk              # re-enter spilled tile -> reload read
            seen.add(mn)
            prev = mn
        writes += blocks[-1][4]                 # trailing drain write of the last tile
        d = s.dram_bits(m, w)
        assert writes == pytest.approx(d["Cw"])
        assert reads == pytest.approx(d["Cr"])


def test_no_double_buffer_exposes_fetch():
    # spec §8: if there's no room to prefetch the next block's inputs alongside the resident
    # footprint, fetches cannot overlap compute and are fully exposed -> larger stall.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=1)   # spills, 4 outer blocks
    db = s.HW(bank_size=1024, banks=32, dram_bw=32)               # cap huge -> double-buffers
    nodb = s.HW(bank_size=100, banks=32, dram_bw=32)              # cap=102400: fits 1x, no prefetch room
    assert s.feasible(m, w, nodb)                                # still feasible (footprint=90112 <= 102400)
    _f1, steady_db, _d1 = s.stall_fill(m, w, db)
    _f2, steady_no, _d2 = s.stall_fill(m, w, nodb)
    assert steady_no > steady_db                                 # no overlap -> strictly more mid-stream stall


def test_selftest_passes():
    s.selftest()   # raises AssertionError on any golden mismatch; returns None on success


def test_report_contains_mapping_and_energy():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    ranked = s.optimize(w, hw)
    txt = s.report(ranked[:3], w, hw)
    assert "perm" in txt and "energy" in txt and "actual_cycle" in txt


def test_cli_selftest_exit0():
    import subprocess, sys, pathlib
    here = pathlib.Path(s.__file__).parent
    r = subprocess.run([sys.executable, "mxp_scheduler.py", "--selftest"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- DRAM/on-chip frequency ratio: converts DRAM transfer time into on-chip cycles ---

def test_freq_ratio_default_is_one():
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)
    assert hw.freq_ratio == 1.0
    assert hw.eff_bw == 64.0          # eff_bw == dram_bw when same clock


def test_freq_ratio_scales_dram_time():
    # freq_ratio = f_chip/f_dram. A slower DRAM (chip 2x faster) doubles every DRAM transfer
    # time in on-chip cycles -> fill doubles, stall grows. compute_work and energy are unchanged.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=1)   # spills -> nonzero stall
    base = s.HW(bank_size=1024, banks=32, dram_bw=32)                  # freq_ratio default 1.0
    slow = s.HW(bank_size=1024, banks=32, dram_bw=32, freq_ratio=2.0)  # DRAM 2x slower vs chip
    fb, steady_b, drain_b = s.stall_fill(m, w, base)
    fs, steady_s, drain_s = s.stall_fill(m, w, slow)
    sb, ss = steady_b + drain_b, steady_s + drain_s
    assert fs == pytest.approx(2 * fb)               # fill is pure DRAM transfer -> exactly 2x
    assert ss > sb                                   # more un-hidden DRAM time -> more stall
    # compute term is freq-independent: actual_cycle minus the DRAM (fill+stall) part is equal
    assert s.actual_cycle(m, w, slow) - (ss + fs) == pytest.approx(s.actual_cycle(m, w, base) - (sb + fb))
    assert s.energy_breakdown(m, w, base) == s.energy_breakdown(m, w, slow)  # energy freq-independent


def test_hw_rejects_nonpositive_freq_ratio():
    with pytest.raises(ValueError):
        s.HW(bank_size=1024, banks=32, dram_bw=64, freq_ratio=0)
    with pytest.raises(ValueError):
        s.HW(bank_size=1024, banks=32, dram_bw=64, freq_ratio=-1.0)


# --- annotated twin equivalence (Task 10) ---

def test_crosscheck_passes():
    s.crosscheck()   # raises AssertionError if the two files disagree on any swept mapping


def test_annotated_imports_and_matches_one_case():
    import importlib, pathlib, sys
    sys.path.insert(0, str(pathlib.Path(s.__file__).parent))
    ann = importlib.import_module("mxp_scheduler_annotated")
    w_args = dict(M=64, K=64, N=64, wbits=[[2, 8], [8, 2]], act_bits=8)
    hw_args = dict(bank_size=1024, banks=32, dram_bw=32, freq_ratio=2.0)
    a_w, b_w = s.Work(**w_args), ann.Work(**w_args)
    a_hw, b_hw = s.HW(**hw_args), ann.HW(**hw_args)
    m_a = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=1)
    m_b = ann.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=1)
    ea, eb = s.evaluate(m_a, a_w, a_hw), ann.evaluate(m_b, b_w, b_hw)
    assert ea["energy"] == eb["energy"]
    assert ea["actual_cycle"] == eb["actual_cycle"]
    assert ea["dram"] == eb["dram"]


def test_annotated_explain_cli_exit0():
    import subprocess, sys, pathlib
    here = pathlib.Path(s.__file__).parent
    r = subprocess.run([sys.executable, "mxp_scheduler_annotated.py", "--explain",
                        "--M", "64", "--K", "64", "--N", "64", "--dram-bw", "32"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "compute_work" in r.stdout and "stall" in r.stdout


# --- Pareto front + constraint ON/OFF tradeoff (Task 11) ---

def test_pareto_front_is_nondominated():
    w = s.Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    ranked = s.optimize(w, hw)
    front = s.pareto_front(ranked)
    # no point dominates a front point in BOTH energy and cycle
    for p in front:
        for q in ranked:
            assert not (q["energy"] < p["energy"] and q["actual_cycle"] < p["actual_cycle"])
    assert front == sorted(front, key=lambda r: r["energy"])


def test_tradeoff_on_off():
    w = s.Work(M=128, K=128, N=128, wbits=[[2, 8, 2, 8]] * 4, act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    t = s.tradeoff(w, hw)
    assert "off" in t and "on" in t            # off = global min energy; on = min energy at min cycle
    assert t["off"]["energy"] <= t["on"]["energy"] + 1e-9   # constraint can only raise energy
    assert t["on"]["actual_cycle"] <= t["off"]["actual_cycle"] + 1e-9


def test_tradeoff_raises_on_no_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    tiny = s.HW(bank_size=1, banks=2, dram_bw=64)   # nothing fits
    with pytest.raises(ValueError):
        s.tradeoff(w, tiny)


def test_hw_rejects_bad_coeffs():
    with pytest.raises(ValueError):
        s.HW(bank_size=1024, banks=32, dram_bw=64, coeffs={"dram": -1.0, "onchip": 6.0, "mac": 1.0, "rmw": 5.0})
    with pytest.raises(ValueError):
        s.HW(bank_size=1024, banks=32, dram_bw=64, coeffs={"dram": 200.0, "onchip": 6.0, "mac": 1.0})  # missing rmw


def test_hw_rejects_unknown_coeff_key():
    # a typo'd key (e.g. "darm") must raise, not silently leave the real coeff at its default
    bad = dict(s.DEFAULT_COEFFS)
    bad["darm"] = 150.0
    with pytest.raises(ValueError, match="unknown key"):
        s.HW(bank_size=1024, banks=32, dram_bw=64, coeffs=bad)


def test_cli_friendly_error_on_bad_dims():
    # bad input must exit via argparse error (rc=2, message on stderr), not a raw traceback
    import subprocess, sys, pathlib
    here = pathlib.Path(s.__file__).parent
    r = subprocess.run([sys.executable, "mxp_scheduler.py", "--M", "100", "--K", "64", "--N", "64"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 2
    assert "multiple of TILE" in r.stderr
    assert "Traceback" not in r.stderr


def test_evaluate_shared_walk_matches_public_functions():
    # evaluate() now computes _blocks/dram_bits once and shares them; results must equal the
    # standalone public functions (the no-precompute path) exactly
    w = s.Work(M=64, K=64, N=64, wbits=[[2.5, 7.5], [8, 2]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32, freq_ratio=2.0)
    for m in s.gen_mappings(w):
        r = s.evaluate(m, w, hw)
        assert r["feasible"] == s.feasible(m, w, hw)
        assert r["dram"] == s.dram_bits(m, w)
        assert r["energy_breakdown"] == s.energy_breakdown(m, w, hw)
        fill, steady, drain = s.stall_fill(m, w, hw)
        assert (r["fill"], r["steady_stall"], r["drain"]) == (fill, steady, drain)
        assert r["stall"] == steady + drain


def test_evaluate_exposes_steady_stall_and_drain():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 2], [2, 2]], act_bits=8)
    m = s.Mapping(perm=("N", "M", "K"), m_in=2, k_in=2, n_in=1)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    r = s.evaluate(m, w, hw)
    assert r["steady_stall"] == 2304.0
    assert r["drain"] == 2048.0
    assert r["fill"] == 768.0
    # back-compat: r["stall"] stays the combined steady+drain
    assert r["stall"] == 2304.0 + 2048.0
    assert r["actual_cycle"] == float(s.compute_work(w)) + 768.0 + 2304.0 + 2048.0
