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


def _emit_region(order, evictions, w, hw, m, B, d, resident):
    """Append region m's band-serpentine cubes + per-step evictions, mutating the carried
    `resident` set (a set of tile tuples currently on-chip). Eviction = lazy, capacity-aware
    set-difference: tiles stay resident across chunk/band/region boundaries (so reused A/W is
    never gratuitously reloaded -> an all-resident schedule hits the first-touch floor), and a
    tile is dropped ONLY when the next cube's load would otherwise exceed cap. When a drop is
    forced, the largest tiles outside the current working set (band C + this cube's needs) are
    evicted first -- evicting a complete/zero C is free, a partial C spills, A/W is free; pricing
    is left entirely to eval_sched. The per-region (B,d) footprint bound (footprint_bits) caps the
    forced working set, so footprint_bits(B,d) <= cap guarantees a feasible build. `resident` is
    passed by reference and carried across regions for global reuse."""
    cap = hw.cap_bits
    for group in _groups(w.NT, B, reverse=(m % 2 == 1)):
        band_c = {es.c_tile((m, group[0], n)) for n in group}      # ("C",m,n) for n in group
        for chunk in _chunks(w.KT, d):
            for k in chunk:                                        # k-outer
                for n in group:                                    # n-inner -> W(m,k) amortized
                    c = (m, k, n)
                    need = {es.a_tile(c), es.w_tile(c), es.c_tile(c)}
                    to_load = need - resident
                    protect = set(band_c) | need                  # current working set: never evict
                    cand = sorted((t for t in resident if t not in protect),
                                  key=lambda t: -es.tile_size(t, w))   # largest-first
                    ev = set()
                    ci = 0
                    while (sum(es.tile_size(t, w) for t in ((resident - ev) | to_load)) > cap
                           and ci < len(cand)):
                        ev.add(cand[ci])
                        ci += 1
                    resident.difference_update(ev)
                    resident.update(to_load)
                    order.append(c)
                    evictions.append(frozenset(ev))


def band_schedule(w, hw, knobs):
    """knobs: list of (B,d) per region m (len == w.MT). Returns (order, evictions): order is a full
    permutation of all_cubes; evictions[i] applied by eval_sched before loading cube i. Residency
    is carried across regions so A/W shared between m-rows is reused (lazy capacity-aware drop)."""
    if len(knobs) != w.MT:
        raise ValueError("knobs must have one (B,d) per region (w.MT=%d)" % w.MT)
    order, evictions = [], []
    resident = set()
    for m in range(w.MT):
        B, d = knobs[m]
        if not (1 <= B <= w.NT and 1 <= d <= w.KT):
            raise ValueError("region %d: B in [1,NT], d in [1,KT]; got B=%s d=%s" % (m, B, d))
        _emit_region(order, evictions, w, hw, m, B, d, resident)
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


def _score_region(w, hw, m, B, d):
    """Partial apply_cube fold over region m alone (fresh resident). Returns
    (bits, capacity_ok, steady_stall): bits = read+spill, steady_stall = sum of mid-stream
    un-hidden transfer. Region-isolated (its first cube treated as fill) -- used only to SELECT
    (B,d); the whole-schedule eval_sched pass is authoritative."""
    order, evictions = [], []
    _emit_region(order, evictions, w, hw, m, B, d, set())
    st = es.SchedState(frozenset(), frozenset(), -1.0)
    bits = 0.0
    steady = 0.0
    for i, c in enumerate(order):
        st, read, spill, unhidden, cap_ok = es.apply_cube(st, c, evictions[i], w, hw)
        if not cap_ok:
            return float("inf"), False, float("inf")
        if i > 0:
            steady += unhidden
        bits += read + spill
    return bits, True, steady


def _candidates(n):
    """(B,d) candidate values in [1,n]. Full range for small n; for large n a logged subset
    (powers of two + n). Ragged tails are allowed, so divisibility is NOT required."""
    if n <= 16:
        return list(range(1, n + 1))
    cand = {1, n}
    p = 1
    while p <= n:
        cand.add(p)
        p *= 2
    return sorted(cand)


def first_touch_floor(w, hw):
    """Valid lower bound on the objective: every distinct A and W tile is loaded at least once."""
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    cubes = es.all_cubes(w)
    a_tiles = {es.a_tile(c) for c in cubes}
    w_tiles = {es.w_tile(c) for c in cubes}
    bits = (sum(es.tile_size(t, w) for t in a_tiles)
            + sum(es.tile_size(t, w) for t in w_tiles))
    return bits * coef


def optimize_band(w, hw, skiplog=None):
    """Per-region (B,d) search + band-serpentine schedule, scored by eval_sched. Returns the
    astar.optimize_exact-shaped dict. Heuristic: proven_optimal=False; gap vs first-touch floor.
    skiplog: optional list; appended with a note when the (B,d) candidate set is capped (no silent
    truncation)."""
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    Bcand, dcand = _candidates(w.NT), _candidates(w.KT)
    if skiplog is not None and (len(Bcand) < w.NT or len(dcand) < w.KT):
        skiplog.append("(B,d) candidates capped: B in %s of 1..%d, d in %s of 1..%d"
                       % (Bcand, w.NT, dcand, w.KT))
    knobs = []
    n_eval = 0
    bad = None                                          # (m, reason, min_stall)
    for m in range(w.MT):
        best = None                                     # (bits, B, d)
        for B in Bcand:
            for d in dcand:
                if footprint_bits(w, hw, m, B, d) > hw.cap_bits:
                    continue
                n_eval += 1
                bits, cap_ok, steady = _score_region(w, hw, m, B, d)
                if not cap_ok or steady > 0:            # capacity already filtered; enforce stall=0
                    continue
                if best is None or bits < best[0]:
                    best = (bits, B, d)
        if best is None and bad is None:                # region infeasible -> diagnose once
            if footprint_bits(w, hw, m, 1, 1) > hw.cap_bits:
                bad = (m, "capacity", None)
            else:
                _, _, steady = _score_region(w, hw, m, 1, 1)
                bad = (m, "stall", steady)
        knobs.append((best[1], best[2]) if best else (1, 1))

    lb = first_touch_floor(w, hw)
    if bad is not None:
        m, reason, stall = bad
        if reason == "capacity":
            msg = ("region %d: no capacity-feasible (B,d); even one cube's A+W+C exceeds cap "
                   "(%d bits). Increase banks/bank_size/word_bits." % (m, hw.cap_bits))
        else:
            msg = ("region %d: capacity-fits at (B=1,d=1) but no stall=0 schedule "
                   "(min steady stall ~%.1f). Raise dram_bw (eff_bw=%g)." % (m, stall, hw.eff_bw))
        return {"energy": float("inf"), "order": None, "evictions": None, "feasible": False,
                "proven_optimal": False, "lower_bound": lb, "gap": float("inf"),
                "nodes_expanded": n_eval, "source": "band", "min_steady_stall": stall, "reason": msg}

    order, evictions = band_schedule(w, hw, knobs)
    r = es.eval_sched(w, hw, order, evictions)
    energy = r["energy"]
    gap = (energy - lb) / lb if (r["feasible"] and lb > 0) else float("inf")
    return {"energy": energy if r["feasible"] else float("inf"),
            "order": order if r["feasible"] else None,
            "evictions": evictions if r["feasible"] else None,
            "feasible": bool(r["feasible"]), "proven_optimal": False,
            "lower_bound": lb, "gap": gap if r["feasible"] else float("inf"),
            "nodes_expanded": n_eval, "source": "band",
            "min_steady_stall": None if r["feasible"] else r["steady_stall"],
            "reason": None if r["feasible"] else "constructed schedule infeasible under eval_sched"}
