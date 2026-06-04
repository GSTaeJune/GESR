# MXP_scheduler/test_mxp_scheduler.py
import math
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
        s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=3)  # act not in {2,4,8}


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
