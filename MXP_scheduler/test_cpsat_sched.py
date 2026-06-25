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
