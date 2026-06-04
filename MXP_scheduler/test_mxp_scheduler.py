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
