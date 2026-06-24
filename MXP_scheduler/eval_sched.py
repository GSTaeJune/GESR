# MXP_scheduler/eval_sched.py
"""eval_sched -- order+eviction schedule evaluator for the MXP scheduler (M1).

Single source of truth for per-step cost. The closed-form model in mxp_scheduler.py
is unchanged; this module addresses the GEMM as a stream of cube-ops (one SA pass each)
with three resident tile types and explicit eviction. stdlib only.

Spec: docs/superpowers/specs/2026-06-23-mxp-scheduler-precision-adaptive-design.md
"""
import itertools
from dataclasses import dataclass
from mxp_scheduler import TILE, FP32_BITS, compute_work


def all_cubes(w):
    """Canonical fixed cube order = product(MT, KT, NT). Index into this list is the
    stable cube id used for bitmask / signature / determinism."""
    return [(mt, kt, nt)
            for mt in range(w.MT) for kt in range(w.KT) for nt in range(w.NT)]


def a_tile(c):
    """Activation tile feeding cube c = (mt,kt,nt). A is indexed (K,N), reused over M."""
    return ("A", c[1], c[2])


def w_tile(c):
    """Weight tile feeding cube c. W is indexed (M,K), reused over N. Size varies with
    the tile's average weight bits -- this is where precision-adaptive residency lives."""
    return ("W", c[0], c[1])


def c_tile(c):
    """Output psum tile of cube c. C is indexed (M,N), reduced over K. FP32, fixed size."""
    return ("C", c[0], c[2])


def tile_size(tile, w):
    kind = tile[0]
    if kind == "A":
        return TILE * TILE * w.act_bits
    if kind == "W":
        mt, kt = tile[1], tile[2]
        return w.wbits[mt][kt] * TILE * TILE
    if kind == "C":
        return TILE * TILE * FP32_BITS
    raise ValueError(f"unknown tile kind {kind!r}")


def cube_compute(c, w, hw):
    """On-chip cycles to compute one cube. Sum over all cubes == compute_work(w, cpb).
    Includes cycles_per_bit so the stall=0 hide budget scales consistently (M1 fixes the
    M0 'hide budget unscaled' note)."""
    mt, kt, _nt = c
    return hw.cycles_per_bit * TILE * w.wbits[mt][kt]
