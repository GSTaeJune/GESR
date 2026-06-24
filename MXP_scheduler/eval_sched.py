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


@dataclass(frozen=True)
class SchedState:
    """Markov state for the joint search. done/resident identify the state; last_compute
    is included because stall=0 feasibility of the NEXT cube depends on it (spec §1.3).
    last_compute = -1.0 means 'no cube executed yet' (the next load is fill, not a stall)."""
    done: frozenset       # frozenset of cube tuples already scheduled
    resident: frozenset   # frozenset of tile tuples currently on-chip
    last_compute: float


def c_counter(done, mt, nt):
    """Accumulation contributions made to C tile (mt,nt) so far = # scheduled cubes
    (mt, *, nt). Derived from `done` (not stored). When this is > 0 a non-resident C tile
    must have been spilled, so loading it is a charged DRAM reload."""
    return sum(1 for (m, _k, n) in done if m == mt and n == nt)


def apply_cube(state, c, evict, w, hw):
    """Execute one cube. Returns (new_state, read_bits, spill_bits, unhidden_time, capacity_ok).

    Steps (single source of truth for cost):
      1. Apply `evict` (a frozenset of resident tiles to drop). Evicting a partial C
         (0 < counter < KT) charges a spill write. Evicting a complete C (counter == KT)
         or a zero-init C (counter == 0) is free (the one final write is a schedule-invariant
         constant accounted elsewhere). Evicting A/W is free (read-only inputs).
      2. Load c's three tiles if not resident. A/W loads are charged reads. A C load is a
         charged read ONLY if its counter > 0 (reload of a spilled partial); counter == 0 is
         a free zero-init that still occupies space.
      3. capacity_ok = resident_bits (after load) <= cap_bits.
      4. unhidden_time = transfer_time if first cube (last_compute < 0; this is `fill`),
         else max(0, transfer_time - last_compute) (the steady-stall contribution; must be
         0 for a stall=0-feasible mid-stream step). transfer_time = (read+spill)/eff_bw.
      5. Advance: done |= {c}; last_compute = cube_compute(c).
    """
    mt, kt, nt = c
    resident = set(state.resident)

    # 1. evictions
    spill = 0.0
    for t in evict:
        if t not in resident:
            raise ValueError(f"cannot evict non-resident tile {t}")
        resident.discard(t)
        if t[0] == "C":
            k = c_counter(state.done, t[1], t[2])
            if 0 < k < w.KT:
                spill += tile_size(t, w)          # partial -> spill write (variable)
            # k == w.KT (complete -> invariant final write) or k == 0 (zero data): free

    # 2. load missing tiles for c
    read = 0.0
    for t in (a_tile(c), w_tile(c), c_tile(c)):
        if t not in resident:
            if t[0] == "C":
                if c_counter(state.done, mt, nt) > 0:
                    read += tile_size(t, w)       # reload of spilled partial
                # else zero-init: free, but still occupies space
            else:
                read += tile_size(t, w)
            resident.add(t)

    # 3. capacity
    resident_bits = sum(tile_size(t, w) for t in resident)
    capacity_ok = resident_bits <= hw.cap_bits

    # 4. stall / fill
    transfer_time = (read + spill) / hw.eff_bw
    if state.last_compute < 0:
        unhidden = transfer_time                  # first cube -> fill (allowed)
    else:
        unhidden = max(0.0, transfer_time - state.last_compute)

    # 5. advance
    new_state = SchedState(done=state.done | {c},
                           resident=frozenset(resident),
                           last_compute=cube_compute(c, w, hw))
    return new_state, read, spill, unhidden, capacity_ok


def eviction_choices(state, c, w, hw):
    """Yield EVERY capacity-feasible eviction set for running cube c from `state` -- the full
    exact move set (invariant #10), shared by the oracle and A*. Each yielded frozenset E is a
    subset of evictable (= resident minus c's needed tiles) such that after evicting E and
    loading c's missing tiles the resident footprint fits cap_bits.

    When the cube already fits (deficit <= 0) EVERY subset qualifies (voluntary eviction is a
    legal move -- shedding a not-yet-needed tile early can prevent a forced C-spill later).
    When it does not fit, only subsets freeing >= deficit qualify, INCLUDING non-minimal ones
    (a larger free eviction now can avoid a charged spill later). Cardinality-minimal pruning
    would be unsound (see invariant #10). Exponential in |evictable| by necessity (NP-hard
    variable-size offline caching); A* caps the work via node budget + gap report."""
    needset = {a_tile(c), w_tile(c), c_tile(c)}
    cur = set(state.resident)
    after_load = cur | needset                                   # tiles occupying space post-load
    deficit = sum(tile_size(t, w) for t in after_load) - hw.cap_bits
    evictable = [t for t in cur if t not in needset]
    sizes = [tile_size(t, w) for t in evictable]
    n = len(evictable)
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            freed = sum(sizes[i] for i in combo)
            if freed >= deficit:                                 # deficit may be <= 0 -> all qualify
                yield frozenset(evictable[i] for i in combo)
