# MXP_scheduler/warmstart.py
"""warmstart -- structural Mapping -> concrete (order, evictions) schedule.

Provides the A* incumbent (spec §D6 'M1 >= M0' floor). The eviction policy is Belady
(furthest-next-use among non-needed resident tiles): feasible and a valid upper bound,
explicitly NOT claimed optimal (rev5 retracted that; A* searches for better). stdlib only.
"""
import itertools
import mxp_scheduler as s
import eval_sched as es


def _loop_order(m, w):
    """Cube visitation order induced by the loop nest (perm outermost-first, with the
    resident inner block expanded into its cubes). Mirrors mxp_scheduler._blocks' walk but
    emits individual cubes. Returns a list of cube tuples (a permutation of all_cubes)."""
    out, inn = s._out_in(m, w)
    order = []
    outer_ranges = [range(out[d]) for d in m.perm]
    for combo in itertools.product(*outer_ranges):
        oidx = dict(zip(m.perm, combo))
        # within this outer block, iterate the inner cubes in the same perm order
        inner_ranges = [range(inn[d]) for d in m.perm]
        for icombo in itertools.product(*inner_ranges):
            iidx = dict(zip(m.perm, icombo))
            mt = oidx["M"] * inn["M"] + iidx["M"]
            kt = oidx["K"] * inn["K"] + iidx["K"]
            nt = oidx["N"] * inn["N"] + iidx["N"]
            order.append((mt, kt, nt))
    return order


def _next_use(order):
    """For each position i, map each tile to the next position > i that needs it
    (or +inf). Precomputed once and consumed backwards for Belady victim choice."""
    n = len(order)
    nxt = [dict() for _ in range(n + 1)]   # nxt[i][tile] = next index >= i needing tile
    # walk backwards
    future = {}
    for i in range(n - 1, -1, -1):
        c = order[i]
        nxt[i] = dict(future)              # snapshot BEFORE adding i's own tiles as 'next'
        for t in (es.a_tile(c), es.w_tile(c), es.c_tile(c)):
            future[t] = i
        # store i's tiles' next-use as i itself for position i (needed now)
        for t in (es.a_tile(c), es.w_tile(c), es.c_tile(c)):
            nxt[i][t] = i
    return nxt


def mapping_to_schedule(m, w, hw):
    """Return (order, evictions). order = loop-nest cube order; evictions[i] = the tiles
    dropped immediately before loading cube order[i], chosen by Belady (furthest next use)
    among resident tiles NOT needed by cube i, evicting just enough to fit capacity."""
    order = _loop_order(m, w)
    nxt = _next_use(order)
    resident = set()
    evictions = []
    INF = len(order) + 1
    for i, c in enumerate(order):
        needed = {es.a_tile(c), es.w_tile(c), es.c_tile(c)}
        # prospective load: tiles not yet resident
        to_load = [t for t in needed if t not in resident]
        ev = set()
        # bits if we load everything with no eviction
        prospective = set(resident) | set(to_load)
        bits = sum(es.tile_size(t, w) for t in prospective)
        if bits > hw.cap_bits:
            # evict non-needed resident tiles, furthest-next-use first, until it fits
            evictable = [t for t in resident if t not in needed]
            evictable.sort(key=lambda t: nxt[i].get(t, INF), reverse=True)  # furthest first
            for t in evictable:
                if bits <= hw.cap_bits:
                    break
                ev.add(t)
                bits -= es.tile_size(t, w)
        evictions.append(frozenset(ev))
        # advance resident exactly as apply_cube will
        resident -= ev
        resident |= needed
    return order, evictions


def structural_incumbent(w, hw):
    """Best stall=0-feasible structural schedule, scored by eval_sched. Returns
    (energy, order, evictions) or None if no structural mapping is stall=0-feasible.
    This is the A* incumbent and the spec §D6 floor (report = min(A*, this))."""
    best = None
    for m in s.gen_mappings(w):
        order, evictions = mapping_to_schedule(m, w, hw)
        r = es.eval_sched(w, hw, order, evictions)
        if not r["feasible"]:                      # capacity AND stall0
            continue
        if best is None or r["energy"] < best[0]:
            best = (r["energy"], order, evictions)
    return best


def min_structural_steady_stall(w, hw):
    """Diagnostic for the empty-feasible-region case (invariant #11 / spec §8). Returns the
    smallest steady_stall achievable by any CAPACITY-feasible structural schedule, or None if
    no structural schedule even fits in capacity. Used to explain WHY stall=0 is infeasible
    (e.g. 'closest schedule still stalls X cycles; raise DRAM bw or shrink the resident
    window'). Never used as a silent fallback -- only to report."""
    best = None
    for m in s.gen_mappings(w):
        order, evictions = mapping_to_schedule(m, w, hw)
        r = es.eval_sched(w, hw, order, evictions)
        if not r["capacity_feasible"]:
            continue
        if best is None or r["steady_stall"] < best:
            best = r["steady_stall"]
    return best
