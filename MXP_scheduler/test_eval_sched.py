# MXP_scheduler/test_eval_sched.py
import pytest
import mxp_scheduler as s
import eval_sched as es


def test_all_cubes_canonical_order():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [8, 2]], act_bits=4)  # MT=2,KT=2,NT=1
    cubes = es.all_cubes(w)
    assert len(cubes) == 2 * 2 * 1
    # canonical order = product(range(MT), range(KT), range(NT))
    assert cubes == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]


def test_tile_identities():
    c = (1, 0, 2)  # (mt, kt, nt)
    assert es.a_tile(c) == ("A", 0, 2)   # A indexed (kt, nt)
    assert es.w_tile(c) == ("W", 1, 0)   # W indexed (mt, kt)
    assert es.c_tile(c) == ("C", 1, 2)   # C indexed (mt, nt)


def test_tile_sizes():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [4, 6]], act_bits=4)
    assert es.tile_size(("A", 0, 0), w) == s.TILE * s.TILE * 4          # act_bits
    assert es.tile_size(("W", 0, 1), w) == 8 * s.TILE * s.TILE          # wbits[0][1]
    assert es.tile_size(("W", 1, 0), w) == 4 * s.TILE * s.TILE          # wbits[1][0]
    assert es.tile_size(("C", 0, 0), w) == s.TILE * s.TILE * s.FP32_BITS


def test_cube_compute_sums_to_compute_work():
    # per-cube compute summed over all cubes must equal compute_work(w, cpb)
    w = s.Work(M=64, K=96, N=64, wbits=[[2, 8, 4], [6, 2, 8]], act_bits=8)  # MT=2,KT=3,NT=2
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32, cycles_per_bit=1.5)
    total = sum(es.cube_compute(c, w, hw) for c in es.all_cubes(w))
    assert total == s.compute_work(w, hw.cycles_per_bit)


def test_cube_compute_value():
    w = s.Work(M=64, K=64, N=64, wbits=[[2, 8], [4, 6]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32, cycles_per_bit=2.0)
    # cube (mt=0,kt=1,nt=0): cycles_per_bit * TILE * wbits[0][1] = 2.0 * 32 * 8 = 512
    assert es.cube_compute((0, 1, 0), w, hw) == 2.0 * 32 * 8
