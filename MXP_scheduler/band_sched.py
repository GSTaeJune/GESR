# MXP_scheduler/band_sched.py
"""band_sched -- band-serpentine (B,d)-per-region heuristic scheduler for large T.

Offline, stdlib only. Two per-region knobs: B = open C tiles (band width), d = K-depth
(k-chunk size). Order = band-serpentine over (m,n), k swept inside each band (k-outer/n-inner
so W is amortized across the band's B columns). Eviction = exact set-difference of the resident
set across steps. Cost is scored by eval_sched (single source of truth) -- this module computes
no cost itself. Honest lower bound = first-touch A/W floor. Returns an astar.optimize_exact-shaped
dict. NOT imported by the stdlib runtime path that avoids heavy deps -- but it IS pure stdlib.

Spec: docs/superpowers/specs/2026-06-26-mxp-scheduler-band-bd-design.md
"""
import math
import mxp_scheduler as s
import eval_sched as es

TILE = s.TILE
FP32_BITS = s.FP32_BITS


def _groups(NT, B, reverse):
    g = [list(range(c, min(c + B, NT))) for c in range(0, NT, B)]
    return list(reversed(g)) if reverse else g


def _chunks(KT, d):
    return [list(range(c, min(c + d, KT))) for c in range(0, KT, d)]


def _emit_region(order, evictions, w, m, B, d, last_chunk_tiles, last_band_c):
    """Append region m's band-serpentine cubes + per-step evictions. Eviction = set-difference
    of the resident set across steps, realized as: at the first cube of each chunk drop the prior
    chunk's W/A; at the first cube of each band (also chunk 0) additionally drop the prior band's
    (completed) C tiles. Carried state (last_chunk_tiles, last_band_c) makes region boundaries a
    correct union too. Returns updated (last_chunk_tiles, last_band_c)."""
    for group in _groups(w.NT, B, reverse=(m % 2 == 1)):
        band_c = {es.c_tile((m, group[0], n)) for n in group}      # ("C",m,n) for n in group
        for ci, chunk in enumerate(_chunks(w.KT, d)):
            ev = set(last_chunk_tiles)                             # prior chunk's W/A
            if ci == 0:
                ev |= last_band_c                                  # band boundary: prior band's C
            this_chunk = set()
            first = True
            for k in chunk:                                        # k-outer
                for n in group:                                    # n-inner -> W(m,k) amortized
                    c = (m, k, n)
                    order.append(c)
                    evictions.append(frozenset(ev) if first else frozenset())
                    first = False
                    this_chunk.add(es.w_tile(c))
                    this_chunk.add(es.a_tile(c))
            last_chunk_tiles = this_chunk
        last_band_c = band_c
    return last_chunk_tiles, last_band_c


def band_schedule(w, hw, knobs):
    """knobs: list of (B,d) per region m (len == w.MT). Returns (order, evictions): order is a full
    permutation of all_cubes; evictions[i] applied by eval_sched before loading cube i."""
    if len(knobs) != w.MT:
        raise ValueError("knobs must have one (B,d) per region (w.MT=%d)" % w.MT)
    order, evictions = [], []
    lct, lbc = set(), set()
    for m in range(w.MT):
        B, d = knobs[m]
        if not (1 <= B <= w.NT and 1 <= d <= w.KT):
            raise ValueError("region %d: B in [1,NT], d in [1,KT]; got B=%s d=%s" % (m, B, d))
        lct, lbc = _emit_region(order, evictions, w, m, B, d, lct, lbc)
    return order, evictions


def footprint_bits(w, hw, m, B, d):
    """Resident peak for region m with (B,d): C(B) + W(max k-chunk) + A(d*B). Sizes via eval_sched."""
    c_region = B * TILE * TILE * FP32_BITS
    a_region = d * B * TILE * TILE * w.act_bits
    w_region = 0
    for chunk in _chunks(w.KT, d):
        cw = sum(es.tile_size(("W", m, k), w) for k in chunk)
        if cw > w_region:
            w_region = cw
    return c_region + w_region + a_region
