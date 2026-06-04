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
    # G2: force K_out=2 (k_in=1) -> psum spill doubles C write, adds C read
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=2)
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
    # onchip with K-spill (Cr>0): refill must include the psum reload (live code path)
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=2)   # K_out=2 -> Cr>0
    # a_rd=65536 ; w_rd=65536 ; refill=A(32768)+W(32768)+Cr(131072)=196608
    assert s.onchip_bits(m, w) == 65536 + 65536 + 196608   # 327680


def test_mixed_wbits_compute_and_footprint():
    # non-uniform per-tile avg bits: compute_work sums EVERY tile; footprint uses global max
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [8, 2]], act_bits=8)
    # compute_work = TILE * NT * Σwbits = 32 * 2 * (2+8+8+2=20) = 1280
    assert s.compute_work(w) == 1280
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)
    # foot W term uses max_wbits=8 (global max), so same as uniform-8: 196608
    assert s.footprint_bits(m, w) == 32768 + 32768 + 131072  # 196608


def test_out_in_rejects_non_divisor_mapping():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)   # MT=KT=NT=2
    bad = s.Mapping(perm=("M", "K", "N"), m_in=3, k_in=3, n_in=3)       # 3 does not divide 2
    with pytest.raises(ValueError):
        s._out_in(bad, w)
    # the corruption it used to cause (negative Cr / total<0) is now blocked at the source
    with pytest.raises(ValueError):
        s.dram_bits(bad, w)


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
