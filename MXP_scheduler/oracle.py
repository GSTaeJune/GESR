# MXP_scheduler/oracle.py
"""oracle -- DP-exact joint (order+eviction) optimizer for SMALL problems.

Deliberately INDEPENDENT of astar.py: enumerates every cube ordering and, for each fixed
order, solves optimal eviction with a (step, resident_set) DP. min over orders = provable
global optimum. Used only to cross-validate A* on small T. stdlib only.

Hard guard: raises if T or the per-order state space is too large to be exhaustive, so a
test never silently runs a non-exhaustive 'oracle'.

Performance note: per fixed order the DP enumerates es.eviction_choices (up to 2**|evictable|)
per layer, so even within the guard a T=6 case with many low-precision (small) tiles can take
minutes. The committed tests use T<=5 and stay sub-second; reach for larger T sparingly.
"""
import itertools
import math
import eval_sched as es

ORACLE_MAX_T = 6            # T! orderings; 6! = 720 is the practical ceiling
ORACLE_MAX_TILES = 14       # 2**tiles resident-subset DP per step


def _all_tiles(w):
    tiles = set()
    for c in es.all_cubes(w):
        tiles.add(es.a_tile(c)); tiles.add(es.w_tile(c)); tiles.add(es.c_tile(c))
    return sorted(tiles)


def _fixed_order_optimal(w, hw, order, stall0=False):
    """Optimal eviction cost (variable DRAM energy) for a FIXED cube order, via DP keyed by
    (resident frozenset). Enumerates all eviction subsets per step via the SHARED
    es.eviction_choices (invariant #10) -> exact for the fixed order. Returns
    (energy, evictions_list) or (inf, None) if no feasible completion.

    For a fixed order, last_compute at step i is determined (cube_compute(order[i-1])), so the
    state stays (resident) -- no search variable added. When stall0=True, transitions whose
    mid-stream un-hidden transfer time > 0 are pruned, enforcing the hard stall=0 constraint."""
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    # DP layer: resident frozenset -> (best_cost, evictions_path). Start: nothing resident.
    layer = {frozenset(): (0.0, [])}
    for i, c in enumerate(order):
        last_compute = -1.0 if i == 0 else es.cube_compute(order[i - 1], w, hw)
        done = frozenset(order[:i])             # path-independent for a fixed order
        nxt = {}
        for resident, (cost, path) in layer.items():
            st = es.SchedState(done=done, resident=resident, last_compute=last_compute)
            for ev in es.eviction_choices(st, c, w, hw):
                ns, read, spill, unhidden, cap_ok = es.apply_cube(st, c, ev, w, hw)
                if not cap_ok:
                    continue                    # eviction_choices guarantees capacity; defensive
                if stall0 and i > 0 and unhidden > 0:
                    continue                    # hard stall=0 prune (mid-stream)
                cand = cost + (read + spill) * coef
                key = ns.resident
                if key not in nxt or cand < nxt[key][0]:
                    nxt[key] = (cand, path + [ev])
        if not nxt:
            return math.inf, None
        layer = nxt
    best_cost, best_path = min(layer.values(), key=lambda v: v[0])
    return best_cost, best_path


def dp_optimal(w, hw, stall0=False):
    """Global optimum over (order, eviction) for small problems. Returns
    {energy, order, evictions, proven_optimal: True}. With stall0=True, optimizes subject to
    the hard stall=0 constraint (comparable to optimize_exact at finite BW). Raises if too
    large to be exhaustive (so a test never silently runs a non-exhaustive 'oracle')."""
    cubes = es.all_cubes(w)
    T = len(cubes)
    tiles = _all_tiles(w)
    if T > ORACLE_MAX_T:
        raise ValueError(f"oracle: T={T} exceeds ORACLE_MAX_T={ORACLE_MAX_T} (not exhaustive)")
    if len(tiles) > ORACLE_MAX_TILES:
        raise ValueError(f"oracle: {len(tiles)} tiles exceeds ORACLE_MAX_TILES={ORACLE_MAX_TILES}")
    best = (math.inf, None, None)
    for order in itertools.permutations(cubes):
        cost, evictions = _fixed_order_optimal(w, hw, list(order), stall0=stall0)
        if cost < best[0]:
            best = (cost, list(order), evictions)
    if best[1] is None:
        suffix = " with stall=0" if stall0 else ""
        raise ValueError(f"oracle: no feasible (order, eviction){suffix} within capacity")
    return {"energy": best[0], "order": best[1], "evictions": best[2], "proven_optimal": True}
