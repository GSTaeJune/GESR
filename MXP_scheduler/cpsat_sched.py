# MXP_scheduler/cpsat_sched.py
"""cpsat_sched -- CP-SAT joint (cube-order + tile-eviction) EXACT optimizer for the MXP scheduler.

Offline tool (depends on ortools); NEVER imported by the stdlib runtime. Step-indexed CP-SAT
model with COSMA-style C/P/S/R residency actions, demand-driven loads, exact-rational integer
scaling, M1-bit-exact cost (reproduces eval_sched.apply_cube), stall=0 hard constraint, honest
gap on timeout. Returns a dict shaped like astar.optimize_exact.

Spec: docs/superpowers/specs/2026-06-25-mxp-scheduler-cpsat-design.md
"""
import math
from fractions import Fraction
from ortools.sat.python import cp_model
import mxp_scheduler as s
import eval_sched as es

INT64_MARGIN = 2 ** 62


def _cubes_tiles(w):
    """Canonical cube list and the sorted set of distinct A/W/C tiles they touch."""
    cubes = es.all_cubes(w)
    tiles = sorted({t for c in cubes
                    for t in (es.a_tile(c), es.w_tile(c), es.c_tile(c))})
    return cubes, tiles


def _global_scale(w, hw):
    """Single positive-integer scale G that integerizes EVERY rational coefficient in the model:
    tile sizes, cap_bits, the stall LHS (size*freq_ratio) and the stall RHS (dram_bw*cube_compute).
    Exact via fractions.Fraction (NO round()). G == 1 for integer-wbits / integral-BW configs."""
    cubes, tiles = _cubes_tiles(w)
    fr = Fraction(hw.freq_ratio).limit_denominator(10 ** 12)
    bw = Fraction(hw.dram_bw).limit_denominator(10 ** 12)
    rats = []
    for t in tiles:
        sz = Fraction(es.tile_size(t, w)).limit_denominator(10 ** 12)
        rats.append(sz)            # capacity + objective
        rats.append(sz * fr)       # stall LHS
    rats.append(Fraction(hw.cap_bits).limit_denominator(10 ** 12))
    for c in cubes:
        rats.append(bw * Fraction(es.cube_compute(c, w, hw)).limit_denominator(10 ** 12))  # stall RHS
    g = 1
    for r in rats:
        g = g * r.denominator // math.gcd(g, r.denominator)
    return g
