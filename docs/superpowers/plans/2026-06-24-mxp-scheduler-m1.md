# MXP_scheduler M1 — exact joint (order + eviction) scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the order+eviction joint exact optimizer for the MXP scheduler — a single-source-of-truth forward-pass evaluator, an A*/branch-and-bound joint optimizer (exact, or honest gap report), and an independent DP-exact oracle for cross-validation — so that all mapping levers (precision-adaptive residency, ragged blocking, C-window, ordering, eviction) are optimized jointly under a hard stall=0 constraint.

**Architecture:** Four new stdlib-only modules layered on M0's closed-form model. `eval_sched.py` holds the tile/cube primitives and the `apply_cube` transition function that is the *single source of truth* for cost (both the evaluator and the optimizer fold over it, so the two-model-drift bug class is impossible). `warmstart.py` turns a structural `Mapping` into a concrete schedule (the A* incumbent / D6 floor). `oracle.py` is a deliberately-independent exhaustive optimizer (all orderings × per-order eviction DP) for small problems. `astar.py` is the best-first joint optimizer plus the public `optimize_exact` entry point. The closed-form model and its `mxp_scheduler_annotated.py` twin are **unchanged** — these new modules are single-source (no twin mirror).

**Tech Stack:** Python 3.13, stdlib only (`heapq`, `itertools`, `dataclasses`), pytest. Run everything from `MXP_scheduler/`.

**Spec:** `docs/superpowers/specs/2026-06-23-mxp-scheduler-precision-adaptive-design.md` (rev5). Section refs below (§N) point there.

**Scope note (read before starting):** This is a large, algorithmically subtle plan (7 tasks). Tasks 1–5 (**Phase A**) build the evaluator + warm-start + oracle — each produces working, independently-testable software (you can *score* any schedule and *optimally* schedule tiny GEMMs without A* at all). Tasks 6–7 (**Phase B**) build the A* optimizer that scales Phase A's exactness to larger problems. There is a natural review/commit checkpoint between Phase A and Phase B. **Per the M0 precedent, this plan should be reviewed to convergence (agent review of the algorithm/heuristic correctness) before execution.**

---

## Design invariants (every task must preserve these)

1. **Single source of truth.** All per-step cost (DRAM bits, stall) flows through `eval_sched.apply_cube`. The evaluator folds it over a full `(order, evictions)`; A* and the oracle call it per expansion. No second cost formula anywhere. (§6, kills two-model drift.)
2. **Final C writes are a schedule-invariant constant**, *not* part of the search objective `g`. Every C tile `(mt,nt)` is written to DRAM exactly once when complete; total `= MT·NT·TILE·TILE·FP32_BITS` regardless of schedule. Modeling it as a constant (excluded from `g` and `h`) is an *exact* reformulation of spec §7's "final-C-write floor in h" (a constant offset changes neither the argmin nor A* optimality), and it lets the A* state stay minimal (no `written` set). This is the only deviation from the spec's literal §7 h; it is documented in `astar.py` and is an equivalence, not an approximation.
3. **A C-tile load is a charged DRAM read iff its accumulation counter > 0** (it was spilled as a partial and must come back from DRAM). A first-touch C (counter == 0) is zero-init on-chip — free. The counter is *derived* from the set of scheduled cubes, not stored separately.
4. **stall=0 is a hard feasibility constraint** (§D4): for every mid-stream transition (every cube after the first), the step's DRAM traffic must be fully hidden by the previous cube's compute (`transfer_time <= prev_compute`). The first cube's traffic is `fill` and the trailing resident-C flush is `drain` — both allowed (not stall violations).
5. **Hide-budget is consistently `cycles_per_bit`-scaled.** `cube_compute` includes `hw.cycles_per_bit`, and the stall check compares `transfer_time` (also in on-chip cycles via `eff_bw`) against it. This fixes the M0 conservative-approximation note (M0 left the per-block hide budget unscaled; M1 scales both sides).
6. **A* state is Markov:** `(done, resident, last_compute)`. `last_compute` is in the state because stall=0 feasibility of the *next* move depends on it (§1.3); omitting it loses optimality or admits false-feasible states.
7. **Determinism (§D9):** the A* open-set is keyed by `(f, g, order_signature, seq)` where `order_signature` is the tuple of cube indices scheduled so far and `seq` is a monotonic insertion counter. The `seq` is the **final** tiebreaker so heapq never compares the unorderable `SchedState` payload; `(f, g, order_signature)` decides priority (lexicographically smallest prefix first), and `seq` is deterministic because expansion order is deterministic. Same input → identical output, every run. **(Without `seq`, two same-prefix-different-eviction entries can tie on `(f, g, sig)` and heapq crashes comparing `SchedState` — a confirmed bug; `seq` is mandatory, not optional.)**
8. **Exact or honest gap (§D2):** A* runs until the frontier is exhausted (proven optimal) **and** the node budget was never hit, otherwise it returns the incumbent + a lower bound (min open `f`) with `gap = (incumbent − lower_bound)/lower_bound`. `proven_optimal = (not budget_hit) and (frontier empty or incumbent <= min_open_f)`. Hitting the node budget **forces** `proven_optimal = False` regardless of frontier state. Never silently truncate.
9. **D6 floor (§D6):** A* is seeded with the best stall=0-feasible structural schedule (from `warmstart`) as its incumbent, so the reported result is `min(A*, structural)` — M1 can never regress below M0's structural answer.
10. **Exact eviction move set (§D1/§3).** Eviction is a full search dimension. At each cube the candidate eviction sets are **every** subset of evictable tiles (= resident minus the cube's needed tiles) that leaves the post-load footprint within capacity — this includes *non-minimal* sets and *voluntary* eviction when the cube already fits. Cardinality-minimal pruning is **unsound** (a free A/W eviction now can avoid a charged C-spill later, and shedding a not-yet-needed tile early can prevent a forced C-spill when it later becomes needed). This move set is generated by ONE shared helper `eval_sched.eviction_choices` used by *both* the oracle and A*, so their move spaces are identical by construction; only the *search* (exhaustive DP vs best-first) differs, keeping A*==oracle a real cross-check. It is exponential in `|evictable|` — that is the inherent cost of exact variable-size eviction (NP-hard offline caching); A* bounds the work with the node budget and reports a gap for large problems (§7/§12).
11. **No silent infeasibility (§8, `feedback_optimize_everything_no_silent_drop`).** When no stall=0-feasible schedule exists, the optimizer does **not** return a bare `inf`/`None`. It returns `feasible=False` plus a diagnostic: the minimum achievable `steady_stall` across capacity-feasible structural schedules and a human reason; `astar.main()` prints an explicit error, never `energy = inf`.

> **Heuristic note (admissible, not consistent).** The A* heuristic `h` (invariant: remaining needed-and-non-resident A/W tiles × coef, C reload/spill lower-bounded to 0) is a true lower bound on remaining `g` (admissible → A* returns the optimum) but is **not monotone/consistent**: a free eviction of a still-needed A/W tile raises `h` while costing 0, so `h(state) > step_cost + h(child)` can occur. Optimality therefore **relies on the exact-state `best_g_to_state` re-expansion** (invariant in Task 7) which re-opens a state when a strictly cheaper `g` to it is found. This is correct for admissible-but-inconsistent heuristics; do not "optimize away" the re-expansion.

---

## File Structure

- `MXP_scheduler/eval_sched.py` — **new.** Tile/cube primitives (`all_cubes`, `a_tile`/`w_tile`/`c_tile`, `tile_size`, `cube_compute`), `SchedState`, `apply_cube` (the single transition), `eviction_choices` (the shared exact move-set generator — invariant #10), `eval_sched` (fold). Imports `TILE`, `FP32_BITS`, `compute_work` from `mxp_scheduler`.
- `MXP_scheduler/warmstart.py` — **new.** `mapping_to_schedule(m, w, hw)` (loop-nest order + Belady-by-next-use eviction), `structural_incumbent(w, hw)` (best stall=0-feasible structural schedule), `min_structural_steady_stall(w, hw)` (diagnostic for invariant #11). Bridges `mxp_scheduler` ↔ `eval_sched`.
- `MXP_scheduler/oracle.py` — **new.** `dp_optimal(w, hw, stall0=False)` — exhaustive (all orderings × per-order eviction DP via `eviction_choices`), independent of A*, for small-T validation. The `stall0` flag enforces the hard stall=0 constraint per order (using the order-determined `last_compute`) so A*==oracle can be checked in the finite-BW regime too.
- `MXP_scheduler/astar.py` — **new.** `astar(...)`, `optimize_exact(w, hw, ...)` public API, `selftest()`, `main()`.
- `MXP_scheduler/test_eval_sched.py` — **new.** Primitives, `apply_cube`, `eval_sched`, closed-form parity.
- `MXP_scheduler/test_warmstart.py` — **new.** schedule validity + structural incumbent.
- `MXP_scheduler/test_oracle.py` — **new.** oracle optimality on hand-checkable cases.
- `MXP_scheduler/test_astar.py` — **new.** A* == oracle, determinism, stall=0 hard, D6 floor, gap reporting.

**No twin.** `mxp_scheduler_annotated.py` is the closed-form twin only. The M1 modules are single-source. Do **not** mirror them. `python mxp_scheduler.py --crosscheck` must still print `crosscheck: OK` after every task (it will — M1 touches none of the closed-form functions).

---

# Phase A — evaluator, warm-start, oracle (Tasks 1–5)

## Task 1: Tile/cube primitives

**Why:** Everything downstream addresses three tile *types* (A/W/C) with distinct index spaces and sizes, and a per-cube compute. Pin these down once, in one module, with tests.

**Files:**
- Create: `MXP_scheduler/eval_sched.py`
- Test: `MXP_scheduler/test_eval_sched.py`

- [ ] **Step 1: Write the failing tests**

Create `MXP_scheduler/test_eval_sched.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test_eval_sched.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval_sched'`.

- [ ] **Step 3: Implement the primitives**

Create `MXP_scheduler/eval_sched.py`:

```python
# MXP_scheduler/eval_sched.py
"""eval_sched — order+eviction schedule evaluator for the MXP scheduler (M1).

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
    the tile's average weight bits — this is where precision-adaptive residency lives."""
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_eval_sched.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/eval_sched.py MXP_scheduler/test_eval_sched.py
git commit -m "feat(scheduler): M1 tile/cube primitives (eval_sched foundation)"
```

---

## Task 2: `apply_cube` — the single transition function

**Why:** This is invariant #1. Every cost the evaluator and optimizer report comes from here. It executes one cube: apply the given evictions (charging C spills), load missing tiles (charging reads, with the C zero-init-vs-reload rule), check capacity, compute the un-hidden transfer time (fill or steady-stall contribution), and advance the state.

**Files:**
- Modify: `MXP_scheduler/eval_sched.py` (add `import itertools`, `SchedState`, `c_counter`, `apply_cube`, `eviction_choices`)
- Test: `MXP_scheduler/test_eval_sched.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_eval_sched.py`:

```python
def _state0():
    return es.SchedState(done=frozenset(), resident=frozenset(), last_compute=-1.0)


def test_apply_cube_first_cube_loads_a_w_and_zero_inits_c():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)  # MT=2,KT=2,NT=1
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    c = (0, 0, 0)
    ns, read, spill, unhidden, cap_ok = es.apply_cube(_state0(), c, frozenset(), w, hw)
    a_sz = s.TILE * s.TILE * 2     # 2048
    w_sz = 2 * s.TILE * s.TILE     # 2048
    assert read == a_sz + w_sz     # C is zero-init (counter 0) -> free
    assert spill == 0.0
    assert cap_ok is True
    assert ns.last_compute == es.cube_compute(c, w, hw)
    assert (es.a_tile(c) in ns.resident and es.w_tile(c) in ns.resident
            and es.c_tile(c) in ns.resident)        # C occupies space even though load was free
    # first cube: unhidden == fill == (read+spill)/eff_bw
    assert unhidden == (read + spill) / hw.eff_bw


def test_apply_cube_evicting_partial_c_charges_spill():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    # state: C(0,0) resident with counter 1 (cube (0,0,0) already done), about to run (1,0,0)
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 0, 0), ("W", 0, 0), ("C", 0, 0)}),
                       last_compute=es.cube_compute((0, 0, 0), w, hw))
    c = (1, 0, 0)  # needs A(0,0)[resident], W(1,0)[new], C(1,0)[new zero-init]
    ns, read, spill, unhidden, cap_ok = es.apply_cube(st, c, frozenset({("C", 0, 0)}), w, hw)
    c_sz = s.TILE * s.TILE * s.FP32_BITS   # 32768
    assert spill == c_sz                   # C(0,0) counter 1, 0<1<KT=2 -> partial spill
    assert read == 2 * s.TILE * s.TILE     # only W(1,0); A resident, C(1,0) zero-init free
    assert ("C", 0, 0) not in ns.resident


def test_apply_cube_reload_of_spilled_c_is_charged():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    # C(0,0) counter 1 but NOT resident (was spilled). Re-running a cube into it reloads.
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 1, 0), ("W", 0, 1)}),
                       last_compute=10.0 ** 9)   # huge -> nothing stalls
    c = (0, 1, 0)  # needs A(1,0)[resident], W(0,1)[resident], C(0,0)[counter 1 -> reload]
    ns, read, spill, unhidden, cap_ok = es.apply_cube(st, c, frozenset(), w, hw)
    c_sz = s.TILE * s.TILE * s.FP32_BITS
    assert read == c_sz            # only the C reload
    assert spill == 0.0
    assert unhidden == 0.0         # transfer fully hidden by the huge prev compute


def test_apply_cube_capacity_violation_flagged():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    # cap smaller than a single cube's working set (A+W+C = 2048+2048+32768 = 36864)
    hw = s.HW(bank_size=1, banks=1, dram_bw=32, word_bits=32)   # cap_bits = 32
    ns, read, spill, unhidden, cap_ok = es.apply_cube(_state0(), (0, 0, 0), frozenset(), w, hw)
    assert cap_ok is False


def test_eviction_choices_includes_empty_when_fits():
    # cube fits with no eviction -> empty set is among the choices (and, since deficit<=0,
    # so is every subset of evictable -> voluntary eviction is offered).
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)   # big cap
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 0, 0), ("W", 0, 0), ("C", 0, 0)}),
                       last_compute=1.0)
    choices = set(es.eviction_choices(st, (1, 0, 0), w, hw))
    assert frozenset() in choices
    # evictable = resident - needed(cube (1,0,0)=A(0,0),W(1,0),C(1,0)) = {W(0,0), C(0,0)}
    # all 4 subsets are capacity-feasible (big cap)
    assert frozenset({("W", 0, 0)}) in choices
    assert frozenset({("C", 0, 0)}) in choices
    assert len(choices) == 4


def test_eviction_choices_excludes_insufficient_subsets_when_tight():
    # cap holds exactly one cube working set (36864). Running a 2nd cube that needs a fresh
    # C tile forces evicting enough; subsets that DON'T free enough must NOT be offered.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=36, banks=32, dram_bw=32, word_bits=32)   # cap_bits = 36*32*32 = 36864
    st = es.SchedState(done=frozenset({(0, 0, 0)}),
                       resident=frozenset({("A", 0, 0), ("W", 0, 0), ("C", 0, 0)}),
                       last_compute=1.0)
    # cube (1,0,0) needs A(0,0)[resident], W(1,0)[+2048], C(1,0)[+32768]. After load w/o evict:
    # 2048+2048+32768 + 2048 + 32768 = 71680 > 36864 -> deficit = 34816.
    # evictable = {W(0,0)=2048, C(0,0)=32768}. Only subsets freeing >= 34816 qualify:
    #   {W(0,0),C(0,0)} frees 34816 (==deficit) -> OK; {C(0,0)} frees 32768 < deficit -> NO;
    #   {W(0,0)} frees 2048 -> NO; {} -> NO.
    choices = set(es.eviction_choices(st, (1, 0, 0), w, hw))
    assert choices == {frozenset({("W", 0, 0), ("C", 0, 0)})}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test_eval_sched.py -k "apply_cube or eviction_choices" -v`
Expected: FAIL — `AttributeError: module 'eval_sched' has no attribute 'SchedState'`.

- [ ] **Step 3: Implement `SchedState`, `c_counter`, `apply_cube`, `eviction_choices`**

Append to `eval_sched.py`:

```python
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
```

> Add `import itertools` at the top of `eval_sched.py` (next to the existing `from mxp_scheduler import ...`). `eviction_choices` is the single move-set generator; the oracle and A* MUST both consume it (do not re-enumerate eviction sets anywhere else — that would reintroduce the two-model drift invariant #1 forbids).

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_eval_sched.py -k "apply_cube or eviction_choices" -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/eval_sched.py MXP_scheduler/test_eval_sched.py
git commit -m "feat(scheduler): M1 apply_cube transition + eviction_choices (single source of truth)"
```

---

## Task 3: `eval_sched` — fold the transition over a full schedule

**Why:** Given a `(order, evictions)` pair, produce the complete cost picture: feasibility (capacity + stall=0), variable DRAM energy (the optimization objective), the DRAM-bit breakdown (read/spill/final, for parity checks), and `actual_cycle` (fill + steady + drain + compute, matching M0's formula). This is what the optimizer's incumbents and the oracle's outputs are scored by.

**Files:**
- Modify: `MXP_scheduler/eval_sched.py` (add `eval_sched`)
- Test: `MXP_scheduler/test_eval_sched.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_eval_sched.py`:

```python
def test_eval_sched_rejects_non_permutation():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    cubes = es.all_cubes(w)
    with pytest.raises(ValueError):
        es.eval_sched(w, hw, cubes[:-1], [frozenset()] * (len(cubes) - 1))  # missing a cube
    with pytest.raises(ValueError):
        es.eval_sched(w, hw, cubes, [frozenset()] * (len(cubes) - 1))       # evictions length mismatch


def test_eval_sched_all_resident_no_eviction():
    # All tiles fit -> zero reloads/spills; read = first-touch A+W; final = all C written once.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)   # MT=KT=NT=2, T=8
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)                    # cap_bits = 1048576
    cubes = es.all_cubes(w)
    r = es.eval_sched(w, hw, cubes, [frozenset()] * len(cubes))
    assert r["capacity_feasible"] is True
    assert r["dram_spill_bits"] == 0.0
    # 4 A tiles x 8192 + 4 W tiles x 8192 = 65536 (C zero-init free)
    assert r["dram_read_bits"] == 65536
    assert r["dram_final_bits"] == 4 * (s.TILE * s.TILE * s.FP32_BITS)  # 131072
    # energy = (read + spill) * coef. No reloads/spills, but the mandatory first-touch A/W loads
    # are REAL DRAM energy (final C writes are the excluded invariant constant) -> NOT zero.
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    assert r["energy"] == 65536 * coef


def test_eval_sched_total_dram_matches_closed_form_all_resident():
    # Parity (spec §8): for a fully-resident schedule, read+spill+final == closed-form total.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    cubes = es.all_cubes(w)
    r = es.eval_sched(w, hw, cubes, [frozenset()] * len(cubes))
    total_dram = r["dram_read_bits"] + r["dram_spill_bits"] + r["dram_final_bits"]
    m = s.Mapping(perm=("N", "K", "M"), m_in=2, k_in=2, n_in=2)   # single all-resident block
    assert total_dram == s.dram_bits(m, w)["total"]               # both 196608


def test_eval_sched_hand_built_spill_accounting():
    # Force one partial-C eviction + its reload via explicit evictions; check exact bits.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)  # MT=2,KT=2,NT=1, T=4
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)                   # cap large: everything fits
    order = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    evictions = [frozenset(), frozenset({("C", 0, 0)}), frozenset(), frozenset()]
    r = es.eval_sched(w, hw, order, evictions)
    c_sz = s.TILE * s.TILE * s.FP32_BITS    # 32768
    a_sz = w_sz = s.TILE * s.TILE * 2       # 2048
    # reads: every A loaded once (2), every W once (4), one C reload (the spilled C(0,0))
    assert r["dram_read_bits"] == 2 * a_sz + 4 * w_sz + c_sz        # 4096 + 8192 + 32768 = 45056
    assert r["dram_spill_bits"] == c_sz                            # exactly one partial spill
    assert r["dram_final_bits"] == 2 * c_sz                        # 2 C tiles, one final write each
    # variable energy uses the unified (dram + onchip) per-bit coefficient
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    assert r["energy"] == (45056 + c_sz) * coef                    # (read + spill) * coef


def test_eval_sched_stall0_flag_and_actual_cycle():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # infinite BW -> all transfers hidden
    cubes = es.all_cubes(w)
    r = es.eval_sched(w, hw, cubes, [frozenset()] * len(cubes))
    assert r["steady_stall"] < 1.0
    assert r["stall0_feasible"] is True
    # actual_cycle = compute_work(cpb) + sa_fill + fill + steady + drain + sa_drain
    assert r["actual_cycle"] == pytest.approx(
        float(s.compute_work(w, hw.cycles_per_bit)) + hw.sa_fill_cycles
        + r["fill"] + r["steady_stall"] + r["drain"] + hw.sa_drain_cycles)


def test_cycles_per_bit_scales_stall_hide_budget():
    # Invariant #5: cube_compute (the per-step hide budget) scales with cycles_per_bit, so a
    # fetch that stalls at cpb=1 becomes hidden at cpb=8. (M0's note: the hide budget was
    # unscaled; M1 scales BOTH sides consistently.) Big cap -> no reload, fetch is first-touch.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    cubes = es.all_cubes(w)
    ev = [frozenset()] * len(cubes)
    slow = s.HW(bank_size=4096, banks=32, dram_bw=32, cycles_per_bit=1.0)   # cap holds everything
    fast = s.HW(bank_size=4096, banks=32, dram_bw=32, cycles_per_bit=8.0)
    r1 = es.eval_sched(w, slow, cubes, ev)
    r8 = es.eval_sched(w, fast, cubes, ev)
    assert r1["steady_stall"] > 0          # at cpb=1 the per-cube A+W fetch is NOT fully hidden
    assert r8["steady_stall"] == 0         # 8x compute hides every fetch (consistent scaling)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test_eval_sched.py -k eval_sched -v`
Expected: FAIL — `AttributeError: module 'eval_sched' has no attribute 'eval_sched'`.

- [ ] **Step 3: Implement `eval_sched`**

Append to `eval_sched.py`:

```python
def eval_sched(w, hw, order, evictions):
    """Fold apply_cube over a full (order, evictions). Returns a dict:
      capacity_feasible, stall0_feasible, feasible (both), reason (None or str),
      dram_read_bits, dram_spill_bits, dram_final_bits,
      fill, steady_stall, drain, actual_cycle, energy (VARIABLE dram energy = objective).

    `energy` = (read + spill) * (coeffs.dram + coeffs.onchip) -- the DRAM read+write energy the
    optimizer minimizes (matches spec §3/§6 'energy = (load + spill bits)*(dram+onchip)'). `read`
    includes the MANDATORY first-touch A/W loads (a schedule-invariant floor) PLUS variable
    reloads; `spill` is variable C-partial spills. EXCLUDED (schedule-invariant constants that do
    not change the argmin -- invariant #2): final C writes, SA-facing on-chip reads, mac, rmw.
    So an all-resident schedule has energy = first-touch-loads * coef (NOT zero -- the loads are
    real DRAM energy; only reloads/spills are zero). The spec §7 heuristic counts exactly these
    first-touch loads, so h0(start) == the all-resident optimum (tight). NOT directly comparable
    in magnitude to mxp_scheduler.energy_breakdown()['total']."""
    cubes = all_cubes(w)
    if sorted(order) != sorted(cubes):
        raise ValueError("order must be a permutation of all cubes (each exactly once)")
    if len(evictions) != len(order):
        raise ValueError(f"evictions must have one set per cube ({len(order)}), got {len(evictions)}")

    state = SchedState(done=frozenset(), resident=frozenset(), last_compute=-1.0)
    read_total = spill_total = 0.0
    fill = 0.0
    steady = 0.0
    for i, c in enumerate(order):
        new_state, read, spill, unhidden, cap_ok = apply_cube(state, c, frozenset(evictions[i]), w, hw)
        if not cap_ok:
            return {"capacity_feasible": False, "stall0_feasible": False, "feasible": False,
                    "reason": f"capacity exceeded at step {i} (cube {c})",
                    "dram_read_bits": read_total, "dram_spill_bits": spill_total,
                    "dram_final_bits": w.MT * w.NT * TILE * TILE * FP32_BITS,
                    "fill": fill, "steady_stall": steady, "drain": 0.0,
                    "actual_cycle": float("inf"), "energy": float("inf")}
        if i == 0:
            fill = unhidden
        else:
            steady += unhidden
        read_total += read
        spill_total += spill
        state = new_state

    final_bits = w.MT * w.NT * TILE * TILE * FP32_BITS          # invariant: each C written once
    resident_c_bits = sum(tile_size(t, w) for t in state.resident if t[0] == "C")
    drain = resident_c_bits / hw.eff_bw                        # trailing flush, no compute to hide
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    stall0 = (steady == 0.0)
    return {
        "capacity_feasible": True,
        "stall0_feasible": stall0,
        "feasible": stall0,                                    # capacity already true here
        "reason": None if stall0 else "steady_stall > 0 (stall=0 violated)",
        "dram_read_bits": read_total,
        "dram_spill_bits": spill_total,
        "dram_final_bits": final_bits,
        "fill": fill,
        "steady_stall": steady,
        "drain": drain,
        "actual_cycle": (float(compute_work(w, hw.cycles_per_bit))
                         + hw.sa_fill_cycles + fill + steady + drain + hw.sa_drain_cycles),
        "energy": (read_total + spill_total) * coef,
    }
```

> `compute_work` is already imported at the top of `eval_sched.py` (Task 1 header: `from mxp_scheduler import TILE, FP32_BITS, compute_work`). Call it directly — no inline `__import__`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_eval_sched.py -v`
Expected: PASS (all Task 1–3 tests).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/eval_sched.py MXP_scheduler/test_eval_sched.py
git commit -m "feat(scheduler): M1 eval_sched fold + closed-form DRAM parity"
```

---

## Task 4: Verify the §8 parity contract explicitly

**Why:** Spec §8 requires "평가기 == 기존 closed-form" for the structurally-natural, fully-resident regime, and an inequality (evaluator ≤ closed-form upper bound) elsewhere. Task 3 already proved the all-resident equality for one case; this task adds a parameterized parity sweep so a future change to either accounting trips immediately.

**Files:**
- Test: `MXP_scheduler/test_eval_sched.py`

- [ ] **Step 1: Write the parity sweep test**

Append to `test_eval_sched.py`:

```python
import itertools


@pytest.mark.parametrize("M,K,N,wb,act", [
    (64, 64, 64, [[8, 8], [8, 8]], 8),
    (64, 64, 64, [[2, 8], [4, 6]], 8),
    (96, 64, 32, [[2, 4], [8, 2], [6, 6]], 4),   # MT=3,KT=2,NT=1
])
def test_all_resident_parity_sweep(M, K, N, wb, act):
    # When the full footprint fits, the fully-resident schedule's total DRAM bits
    # (read + spill + final) equals the closed-form single-block dram total.
    w = s.Work(M=M, K=K, N=N, wbits=wb, act_bits=act)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=32)   # big cap so everything is resident
    cubes = es.all_cubes(w)
    r = es.eval_sched(w, hw, cubes, [frozenset()] * len(cubes))
    assert r["capacity_feasible"] is True
    total_dram = r["dram_read_bits"] + r["dram_spill_bits"] + r["dram_final_bits"]
    # closed-form single-block mapping: every dim fully resident (m_in=MT,k_in=KT,n_in=NT)
    m = s.Mapping(perm=("M", "K", "N"), m_in=w.MT, k_in=w.KT, n_in=w.NT)
    assert s.feasible(m, w, hw)                       # confirm closed-form agrees it fits
    assert total_dram == s.dram_bits(m, w)["total"]


def test_inner_blocked_evaluator_le_closed_form_upper_bound():
    # Spec §1.4/§8: the closed-form C-spill model is an all-or-nothing UPPER bound for
    # inner-blocked mappings. A real schedule that follows the SAME loop order (via
    # warmstart.mapping_to_schedule) must have total DRAM bits <= the closed-form total.
    import warmstart as ws
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)   # MT=KT=NT=2
    # K-outer, N inner -> genuine psum spill in closed-form (selftest G2 style)
    m = s.Mapping(perm=("K", "M", "N"), m_in=2, k_in=1, n_in=1)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=10 ** 12)             # big cap (no extra eviction)
    order, evictions = ws.mapping_to_schedule(m, w, hw)
    r = es.eval_sched(w, hw, order, evictions)
    total_dram = r["dram_read_bits"] + r["dram_spill_bits"] + r["dram_final_bits"]
    assert total_dram <= s.dram_bits(m, w)["total"] + 1e-9            # evaluator <= closed-form bound
```

- [ ] **Step 2: Run to verify pass (or surface a real accounting bug)**

Run: `python -m pytest test_eval_sched.py -k "parity or inner_blocked" -v`
Expected: PASS for all three parity parametrizations AND the inner-blocked upper-bound test. If the parity equality fails, the evaluator's read/spill/final accounting diverges from closed-form for the all-resident regime — STOP and reconcile before proceeding (this is the §8 contract). If the inner-blocked inequality fails (evaluator total > closed-form), the evaluator is over-counting spill/reload relative to the closed-form upper bound — also STOP.

- [ ] **Step 3: Commit**

```bash
git add MXP_scheduler/test_eval_sched.py
git commit -m "test(scheduler): M1 closed-form parity sweep for eval_sched"
```

---

## Task 5: `warmstart` — structural mapping → schedule (A* incumbent / D6 floor)

**Why:** A* needs a feasible incumbent to seed its bound and to guarantee `M1 >= M0` (spec §D6). `mapping_to_schedule` turns any structural `Mapping` into a concrete `(order, evictions)` using the loop-nest traversal order and a Belady-by-next-use eviction policy (which is feasible and a valid upper bound — *not* claimed optimal; that claim was retracted in rev5, which is exactly why A* exists). `structural_incumbent` scores every structural mapping with `eval_sched` and returns the cheapest one that is stall=0-feasible.

**Files:**
- Create: `MXP_scheduler/warmstart.py`
- Test: `MXP_scheduler/test_warmstart.py`

- [ ] **Step 1: Write the failing tests**

Create `MXP_scheduler/test_warmstart.py`:

```python
# MXP_scheduler/test_warmstart.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import warmstart as ws


def test_mapping_to_schedule_is_a_valid_permutation():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=32)
    m = s.Mapping(perm=("K", "M", "N"), m_in=1, k_in=1, n_in=1)
    order, evictions = ws.mapping_to_schedule(m, w, hw)
    assert sorted(order) == sorted(es.all_cubes(w))      # each cube exactly once
    assert len(evictions) == len(order)


def test_mapping_to_schedule_scores_feasibly_when_capacity_allows():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=10 ** 12)   # big cap + infinite BW
    m = s.Mapping(perm=("K", "M", "N"), m_in=1, k_in=1, n_in=1)
    order, evictions = ws.mapping_to_schedule(m, w, hw)
    r = es.eval_sched(w, hw, order, evictions)
    assert r["capacity_feasible"] is True
    assert r["stall0_feasible"] is True                     # infinite BW hides everything


def test_mapping_to_schedule_respects_tight_capacity():
    # Tight cap forces evictions; the produced schedule must still never exceed capacity.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    # cap holds ~1.5 cube working sets -> eviction required, but feasible
    hw = s.HW(bank_size=2, banks=32, dram_bw=32, word_bits=1024)  # cap_bits = 2*32*1024 = 65536
    m = s.Mapping(perm=("K", "M", "N"), m_in=1, k_in=1, n_in=1)
    order, evictions = ws.mapping_to_schedule(m, w, hw)
    r = es.eval_sched(w, hw, order, evictions)
    assert r["capacity_feasible"] is True                   # warm-start never overflows


def test_structural_incumbent_returns_best_stall0_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)   # infinite BW -> many stall0-feasible
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    inc = ws.structural_incumbent(w, hw)
    assert inc is not None
    energy, order, evictions = inc
    r = es.eval_sched(w, hw, order, evictions)
    assert r["stall0_feasible"] is True
    assert r["energy"] == energy
    # all-resident is achievable here -> no reloads/spills, energy = first-touch A/W floor
    # (4 A x 8192 + 4 W x 8192 = 65536 bits), NOT zero.
    assert energy == 65536 * coef


def test_structural_incumbent_none_when_nothing_feasible():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    # cap too small for even one cube working set AND bw tiny -> no stall0-feasible structural sched
    hw = s.HW(bank_size=1, banks=1, dram_bw=1, word_bits=32)   # cap_bits = 32
    assert ws.structural_incumbent(w, hw) is None


def test_min_structural_steady_stall_reports_positive_when_bw_tight():
    # Capacity fits, but finite BW makes every structural schedule stall mid-stream.
    # The diagnostic must report a POSITIVE min steady_stall (not None, not 0).
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=4)             # big cap, tiny BW -> stalls
    assert ws.structural_incumbent(w, hw) is None              # nothing is stall0-feasible
    diag = ws.min_structural_steady_stall(w, hw)
    assert diag is not None and diag > 0                       # explains how far from stall=0


def test_min_structural_steady_stall_zero_when_bw_huge():
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=10 ** 12)      # infinite BW -> stall0 reachable
    assert ws.min_structural_steady_stall(w, hw) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test_warmstart.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'warmstart'`.

- [ ] **Step 3: Implement `warmstart.py`**

Create `MXP_scheduler/warmstart.py`:

```python
# MXP_scheduler/warmstart.py
"""warmstart — structural Mapping -> concrete (order, evictions) schedule.

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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_warmstart.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/warmstart.py MXP_scheduler/test_warmstart.py
git commit -m "feat(scheduler): M1 warm-start + steady-stall diagnostic (D6 floor)"
```

> **Phase A checkpoint.** The evaluator + warm-start are complete and independently useful. Recommended: request a code review of `eval_sched.py` + `warmstart.py` (accounting correctness, the C zero-init/reload rule, the final-write-as-constant equivalence) before building the optimizer.

---

# Phase B — exact optimizer + oracle (Tasks 6–7)

## Task 6: `oracle` — independent DP-exact optimizer (small T)

**Why:** A* must be validated against a *different* exact algorithm or a bug in A* validates itself. The oracle enumerates **all** cube orderings and, for each fixed order, finds the optimal eviction via a `(step, resident_set)` DP — a genuinely different method from best-first search. The minimum over all orders is the provable global optimum. Tractable only for small T (≤ ~6 cubes, tiny capacity), which is exactly the validation regime.

**Files:**
- Create: `MXP_scheduler/oracle.py`
- Test: `MXP_scheduler/test_oracle.py`

- [ ] **Step 1: Write the failing tests**

Create `MXP_scheduler/test_oracle.py`:

```python
# MXP_scheduler/test_oracle.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import oracle as o


def test_oracle_all_resident_optimum_is_first_touch_floor():
    # Everything fits -> the optimal schedule has zero reloads/spills. Its energy is the
    # mandatory first-touch A/W load floor (NOT zero): 2 A x 2048 + 4 W x 2048 = 12288 bits.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)   # MT=2,KT=2,NT=1, T=4
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)             # big cap, infinite BW
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    res = o.dp_optimal(w, hw)
    assert res["energy"] == 12288 * coef
    assert res["proven_optimal"] is True
    # the returned schedule re-scores to the same energy via the single source of truth
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["energy"] == 12288 * coef and r["feasible"] is True


def test_oracle_matches_eval_sched_on_returned_schedule():
    w = s.Work(M=32, K=64, N=32, wbits=[[2, 2]], act_bits=2)          # MT=1,KT=2,NT=1, T=2
    # cap = 40960 holds the full footprint exactly (2A+2W+1C = 4096+4096+32768) -> optimum
    # has zero variable energy; the returned schedule re-scores identically via apply_cube.
    hw = s.HW(bank_size=40, banks=1, dram_bw=10 ** 12, word_bits=1024)  # cap_bits = 40960
    res = o.dp_optimal(w, hw)
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["energy"] == res["energy"]
    assert res["proven_optimal"] is True


def test_oracle_stall0_constrained_no_better_than_unconstrained():
    # stall=0 is a HARD CONSTRAINT: the stall0-constrained optimum can only be >= the
    # unconstrained optimum (smaller feasible set). Finite BW so the constraint can bind.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)  # T=4, mixed W
    hw = s.HW(bank_size=2, banks=32, dram_bw=64, word_bits=1024)      # cap 65536, finite BW
    free = o.dp_optimal(w, hw, stall0=False)
    try:
        constrained = o.dp_optimal(w, hw, stall0=True)
        assert constrained["energy"] >= free["energy"] - 1e-9
        # the stall0 schedule must actually BE stall0-feasible when re-scored
        r = es.eval_sched(w, hw, constrained["order"], constrained["evictions"])
        assert r["stall0_feasible"] is True
    except ValueError:
        pass   # acceptable: no stall=0-feasible schedule exists at this BW (constraint empty)


def test_oracle_optimum_no_greater_than_any_structural():
    import warmstart as ws
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)  # T=4
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)  # cap 65536, infinite BW
    res = o.dp_optimal(w, hw)
    inc = ws.structural_incumbent(w, hw)
    assert inc is not None
    assert res["energy"] <= inc[0] + 1e-9     # global optimum <= any structural schedule
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'oracle'`.

- [ ] **Step 3: Implement `oracle.py`**

Create `MXP_scheduler/oracle.py`:

```python
# MXP_scheduler/oracle.py
"""oracle — DP-exact joint (order+eviction) optimizer for SMALL problems.

Deliberately INDEPENDENT of astar.py: enumerates every cube ordering and, for each fixed
order, solves optimal eviction with a (step, resident_set) DP. min over orders = provable
global optimum. Used only to cross-validate A* on small T. stdlib only.

Hard guard: raises if T or the per-order state space is too large to be exhaustive, so a
test never silently runs a non-exhaustive 'oracle'.
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
```

> **Independence & shared move set.** The oracle shares `eviction_choices` (move generation) with A* — that is deliberate (invariant #10: identical move spaces). What stays independent is the *search*: exhaustive permutations + forward DP here, vs heuristic best-first in A*. So `A*==oracle` validates the search/pruning/heuristic, not the move set. A bug in `apply_cube` itself would not be caught by `A*==oracle` (shared) but IS caught by Task 4's closed-form parity (independent accounting). The `stall0` flag lets A*==oracle be checked at finite BW (where stall=0 binds), closing the gap the reviewers flagged; `stall0=False` remains for pure energy/eviction validation at huge BW.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_oracle.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/oracle.py MXP_scheduler/test_oracle.py
git commit -m "feat(scheduler): M1 independent DP-exact oracle (shared move set + stall0 mode)"
```

---

## Task 7: `astar` — joint optimizer, public API, validation, CLI

**Why:** The deliverable. Best-first joint (order + eviction) search with an admissible heuristic, hard stall=0 pruning, deterministic tie-break, D6 warm-start floor, and exact-or-gap termination. Validated against the oracle on small T.

**Files:**
- Create: `MXP_scheduler/astar.py`
- Test: `MXP_scheduler/test_astar.py`

- [ ] **Step 1: Write the failing tests**

Create `MXP_scheduler/test_astar.py`:

```python
# MXP_scheduler/test_astar.py
import pytest
import mxp_scheduler as s
import eval_sched as es
import oracle as o
import astar as a


def test_optimize_exact_all_resident_first_touch_floor_proven():
    # Everything fits -> optimum has no reloads/spills; energy = first-touch A/W floor
    # (2 A x 2048 + 4 W x 2048 = 12288 bits), NOT zero. h0 == this floor, so A* proves it at
    # the root (nodes=0) via the bound.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)   # T=4
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    res = a.optimize_exact(w, hw)
    assert res["energy"] == 12288 * coef
    assert res["proven_optimal"] is True
    r = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert r["feasible"] is True and r["energy"] == 12288 * coef


def test_astar_search_expands_and_matches_oracle():
    # A config where the warm-start incumbent is NOT already optimal, so A* must actually
    # EXPAND nodes (not just bound-break at the root). Exercises the search machinery itself.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 4]], act_bits=2)            # MT=1,KT=2,NT=2, T=4
    hw = s.HW(bank_size=38, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 38912: real pressure
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])
    assert exact["nodes_expanded"] > 0           # genuine search, not a root bound-break


@pytest.mark.parametrize("M,K,N,wb,act,bank_size,word_bits", [
    (64, 64, 32, [[2, 2], [2, 2]], 2, 2, 1024),    # cap 65536, T=4, forced pressure
    (32, 64, 32, [[2, 2]], 2, 40, 1024),           # cap 40960, T=2
    (64, 64, 32, [[2, 4], [2, 2]], 2, 2, 1024),    # mixed-precision W sizes, T=4
])
def test_astar_matches_oracle(M, K, N, wb, act, bank_size, word_bits):
    w = s.Work(M=M, K=K, N=N, wbits=wb, act_bits=act)
    hw = s.HW(bank_size=bank_size, banks=32, dram_bw=10 ** 12, word_bits=word_bits)
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])


def test_astar_deterministic():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)
    r1 = a.optimize_exact(w, hw)
    r2 = a.optimize_exact(w, hw)
    assert r1["order"] == r2["order"]
    assert r1["evictions"] == r2["evictions"]
    assert r1["energy"] == r2["energy"]


def test_astar_stall0_hard():
    # Finite BW so big mid-stream transfers cannot be hidden -> stall0 prunes them.
    # The returned schedule, scored by eval_sched, MUST be stall0_feasible.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)   # finite BW
    res = a.optimize_exact(w, hw)
    if res["feasible"]:                               # a feasible schedule was found
        r = es.eval_sched(w, hw, res["order"], res["evictions"])
        assert r["stall0_feasible"] is True
        assert r["feasible"] is True


def test_astar_matches_oracle_finite_bw():
    # Finite BW so stall=0 BINDS: A* (hard stall=0) must match the stall0-constrained oracle,
    # not just the unconstrained one. Closes the finite-BW validation gap.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)   # T=4 mixed
    hw = s.HW(bank_size=2, banks=32, dram_bw=256, word_bits=1024)      # cap 65536, finite BW
    exact = a.optimize_exact(w, hw)
    if exact["feasible"]:
        ref = o.dp_optimal(w, hw, stall0=True)
        assert exact["proven_optimal"] is True
        assert exact["energy"] == pytest.approx(ref["energy"])
    else:
        with pytest.raises(ValueError):              # oracle must also find nothing stall0-feasible
            o.dp_optimal(w, hw, stall0=True)


def test_astar_prime_tile_count_matches_oracle():
    # KT=5 (PRIME): the divisor-only closed-form gen_mappings cannot express 5-way ragged
    # blocking, but the joint optimizer can. K=160 -> KT=5; MT=NT=1 -> T=5 (<= ORACLE_MAX_T).
    # Capacity pressure forces real reload choices; A* must equal the exhaustive oracle.
    w = s.Work(M=32, K=160, N=32, wbits=[[2, 4, 8, 2, 6]], act_bits=2)  # MT=1,KT=5,NT=1,T=5
    hw = s.HW(bank_size=48, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 49152 < footprint
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])


def _w_load_counts(w, hw, order, evictions):
    """Replay a schedule and count charged W-tile loads per tile (W loads are always charged).
    Uses apply_cube for state advance (single source of truth) -- no cost logic duplicated."""
    import collections
    st = es.SchedState(frozenset(), frozenset(), -1.0)
    counts = collections.Counter()
    for i, c in enumerate(order):
        wt = es.w_tile(c)
        if wt not in st.resident:                    # not resident before this step -> a load
            counts[wt] += 1
        st = es.apply_cube(st, c, frozenset(evictions[i]), w, hw)[0]
    return counts


def test_astar_precision_adaptive_residency():
    # Mixed precision: W(0,1) is 8-bit (8192 bits, expensive), W(0,0) is 2-bit (2048, cheap).
    # Under capacity pressure with N-reuse, the EXACT optimum amortizes the expensive fetch:
    # it loads the expensive W no more often than the cheap W. Ground truth via the oracle.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 8]], act_bits=2)            # MT=1,KT=2,NT=2,T=4
    hw = s.HW(bank_size=48, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 49152 < footprint
    exact = a.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert exact["proven_optimal"] is True
    assert exact["energy"] == pytest.approx(ref["energy"])             # exact, incl. mixed precision
    counts = _w_load_counts(w, hw, exact["order"], exact["evictions"])
    # the expensive 8-bit W is loaded no more often than the cheap 2-bit W (amortization)
    assert counts[("W", 0, 1)] <= counts[("W", 0, 0)]


def test_astar_empty_region_diagnostic():
    # Tiny BW + multi-cube -> NO stall=0-feasible schedule. Must NOT return a bare inf:
    # feasible=False, a positive min_steady_stall, and a human reason (invariant #11).
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=1)   # big cap (no reload), tiny BW -> all stall
    res = a.optimize_exact(w, hw)
    assert res["feasible"] is False
    assert res["proven_optimal"] is False
    assert res["min_steady_stall"] is not None and res["min_steady_stall"] > 0
    assert "stall=0" in res["reason"]


def test_astar_never_worse_than_structural_floor():
    import warmstart as ws
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=10 ** 12)   # all-resident achievable
    inc = ws.structural_incumbent(w, hw)
    res = a.optimize_exact(w, hw)
    assert inc is not None
    assert res["energy"] <= inc[0] + 1e-9        # D6 floor: M1 never regresses below structural


def test_astar_reports_gap_when_budget_zero():
    # cap = 36*32*32 = 36864 = exactly one cube working set -> the warm-start incumbent must
    # reload A/W between cubes, so its energy > h0(start). With node_budget=0 the search stops
    # before expanding any node, so optimality is NOT proven and an honest gap is reported.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=36, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap_bits = 36864
    res = a.optimize_exact(w, hw, node_budget=0)
    assert res["feasible"] is True                 # warm-start schedule exists
    assert res["proven_optimal"] is False          # budget hit -> not proven
    assert res["nodes_expanded"] == 0
    assert res["gap"] > 0.0
    assert res["lower_bound"] <= res["energy"] + 1e-9


def test_astar_selftest_runs():
    a.selftest()   # prints "astar selftest: OK"; raises on any golden mismatch
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test_astar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'astar'`.

- [ ] **Step 3: Implement `astar.py`**

Create `MXP_scheduler/astar.py`:

```python
# MXP_scheduler/astar.py
"""astar — exact joint (order + eviction) GEMM-mapping optimizer for the MXP scheduler (M1).

Best-first search over SchedState = (done, resident, last_compute). g = cumulative VARIABLE
DRAM energy (read+spill)*(dram+onchip). h = admissible lower bound = remaining needed-and-
non-resident A/W tiles * coef (C reloads/spills lower-bound to 0; final C writes are a
schedule-invariant constant, excluded from g and h alike -- an exact reformulation of spec
§7's final-write floor, not an approximation). stall=0 is a hard prune. Determinism via
(f, g, order-signature) keys. Exact when the frontier empties; otherwise incumbent + min open f
= honest gap. Seeded with the best structural schedule (spec §D6 floor). stdlib only.
"""
import heapq
import itertools
import eval_sched as es
import warmstart as ws

DEFAULT_NODE_BUDGET = 200_000


def _heuristic(state, cubes, w, coef):
    """Admissible lower bound on remaining g: every remaining cube's A and W tile, if not
    resident, needs >= 1 load; C reload/spill lower-bounds to 0 (final C writes are the
    excluded invariant constant). ADMISSIBLE but NOT consistent (see the plan's heuristic
    note) -- optimality relies on the exact-state best_g_to_state re-expansion below."""
    needed_a, needed_w = set(), set()
    for c in cubes:
        if c not in state.done:
            needed_a.add(es.a_tile(c))
            needed_w.add(es.w_tile(c))
    bits = 0.0
    for t in needed_a | needed_w:
        if t not in state.resident:
            bits += es.tile_size(t, w)
    return bits * coef


def astar(w, hw, node_budget=DEFAULT_NODE_BUDGET, warm=None):
    """Joint (order+eviction) search. Returns a dict:
    {energy, order, evictions, feasible, proven_optimal, lower_bound, gap, nodes_expanded,
     source, min_steady_stall, reason}.

    Move set = es.eviction_choices (the full exact set, shared with the oracle; invariant #10).
    Determinism = (f, g, sig, seq) heap key with a monotonic seq so SchedState is never compared
    (invariant #7). Termination = exact when the frontier empties without hitting the node
    budget; hitting the budget forces proven_optimal=False with an honest gap (invariant #8).
    Infeasible (no stall=0 schedule) -> feasible=False + diagnostic (invariant #11)."""
    cubes = es.all_cubes(w)
    T = len(cubes)
    cube_id = {c: i for i, c in enumerate(cubes)}
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    seq = itertools.count()                          # monotonic tie-breaker (never compares state)

    start = es.SchedState(done=frozenset(), resident=frozenset(), last_compute=-1.0)

    # incumbent (spec §D6 floor): best structural schedule, if any
    if warm is None:
        warm = ws.structural_incumbent(w, hw)
    inc_energy = warm[0] if warm is not None else float("inf")
    inc_order = list(warm[1]) if warm is not None else None
    inc_evict = list(warm[2]) if warm is not None else None
    inc_source = "structural" if warm is not None else None

    h0 = _heuristic(start, cubes, w, coef)
    open_heap = [(h0, 0.0, (), next(seq), start, [], [])]
    best_g_to_state = {start: 0.0}
    nodes = 0
    min_open_f = h0
    budget_hit = False

    while open_heap:
        f, g, sig, _s, state, order_list, evict_list = heapq.heappop(open_heap)
        min_open_f = f                               # popped in nondecreasing f order
        if g > best_g_to_state.get(state, float("inf")):
            continue                                 # stale (a cheaper path to state was found)
        if f >= inc_energy:                          # cannot beat the incumbent -> done
            break                                    # (heap is f-ordered: nothing better remains)
        if len(state.done) == T:                     # goal
            if g < inc_energy:
                inc_energy, inc_order, inc_evict, inc_source = g, order_list, evict_list, "astar"
            continue
        if nodes >= node_budget:                     # budget exhausted BEFORE expanding -> stop
            budget_hit = True
            break
        nodes += 1

        for c in cubes:                              # deterministic: canonical cube order
            if c in state.done:
                continue
            for ev in es.eviction_choices(state, c, w, hw):   # full exact move set
                ns, read, spill, unhidden, cap_ok = es.apply_cube(state, c, ev, w, hw)
                if not cap_ok:
                    continue                         # eviction_choices guarantees this; defensive
                if state.last_compute >= 0 and unhidden > 0:
                    continue                         # stall=0 HARD prune (mid-stream)
                ng = g + (read + spill) * coef
                if ng >= best_g_to_state.get(ns, float("inf")):
                    continue
                nf = ng + _heuristic(ns, cubes, w, coef)
                if nf >= inc_energy:                 # bound: cannot beat incumbent
                    continue
                best_g_to_state[ns] = ng
                nsig = sig + (cube_id[c],)
                heapq.heappush(open_heap,
                               (nf, ng, nsig, next(seq), ns, order_list + [c], evict_list + [ev]))

    feasible = inc_order is not None
    if budget_hit:
        proven = False
    else:
        proven = (not open_heap) or (inc_energy <= min_open_f + 1e-9)
    lower_bound = inc_energy if (proven and feasible) else min_open_f
    gap = (0.0 if (proven and feasible)
           else (inc_energy - lower_bound) / lower_bound if (lower_bound > 0 and feasible) else 0.0)

    # invariant #11: never return a bare inf -- diagnose the empty feasible region
    min_ss, reason = None, None
    if not feasible:
        min_ss = ws.min_structural_steady_stall(w, hw)
        if min_ss is None:
            reason = ("no capacity-feasible schedule exists (footprint exceeds on-chip capacity "
                      "for every mapping); increase banks / bank_size / word_bits.")
        else:
            reason = (f"no stall=0-feasible schedule; closest structural schedule still stalls "
                      f"{min_ss:.1f} cycles mid-stream. Raise dram_bw (eff_bw={hw.eff_bw:g}) or "
                      f"shrink the resident window so each fetch hides under the prior cube's compute.")

    return {"energy": inc_energy, "order": inc_order, "evictions": inc_evict,
            "feasible": feasible, "proven_optimal": (feasible and proven),
            "lower_bound": lower_bound, "gap": gap, "nodes_expanded": nodes,
            "source": inc_source, "min_steady_stall": min_ss, "reason": reason}


def optimize_exact(w, hw, node_budget=DEFAULT_NODE_BUDGET):
    """Public entry point. Joint (order+eviction) exact optimizer with D6 structural floor.
    Returns the astar() result dict: energy, order, evictions, feasible, proven_optimal,
    lower_bound, gap, nodes_expanded, source, min_steady_stall, reason. When feasible is False,
    energy is inf, order/evictions are None, and reason/min_steady_stall carry the diagnostic."""
    return astar(w, hw, node_budget=node_budget)


def selftest():
    import mxp_scheduler as s
    import oracle as o
    # G1: all-resident -> no reloads/spills; energy = first-touch A/W floor (12288 bits * coef),
    # NOT zero. Proven at the root.
    w1 = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw1 = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef1 = hw1.coeffs["dram"] + hw1.coeffs["onchip"]
    r1 = optimize_exact(w1, hw1)
    assert r1["energy"] == 12288 * coef1 and r1["proven_optimal"], r1
    # G2: A* == oracle under capacity pressure (mixed-precision W sizes)
    w2 = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw2 = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)   # cap 65536
    r2 = optimize_exact(w2, hw2)
    ref2 = o.dp_optimal(w2, hw2)
    assert r2["proven_optimal"] and abs(r2["energy"] - ref2["energy"]) < 1e-6, (r2, ref2)
    # G3: determinism
    assert optimize_exact(w2, hw2)["order"] == r2["order"]
    print("astar selftest: OK")


def main(argv=None):
    import argparse
    import mxp_scheduler as s
    p = argparse.ArgumentParser(description="MXP_scheduler M1 - exact joint (order+eviction) optimizer")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0)
    p.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    a = p.parse_args(argv)
    if a.selftest:
        selftest(); return 0
    if not (a.M and a.K and a.N):
        p.error("provide --M --K --N (or --selftest)")
    MT, KT = a.M // s.TILE, a.K // s.TILE
    w = s.Work(M=a.M, K=a.K, N=a.N, wbits=[[a.act] * KT for _ in range(MT)], act_bits=a.act)
    hw = s.HW(bank_size=a.bank_size, banks=a.banks, dram_bw=a.dram_bw)
    res = optimize_exact(w, hw, node_budget=a.node_budget)
    if not res["feasible"]:
        print(f"NO FEASIBLE SCHEDULE: {res['reason']}")
        return 1                                      # CLI error, never a silent inf (invariant #11)
    status = "PROVEN OPTIMAL" if res["proven_optimal"] else f"gap={res['gap']*100:.1f}% (lb={res['lower_bound']:.0f})"
    print(f"energy(variable DRAM) = {res['energy']:.0f}   {status}   nodes={res['nodes_expanded']}   source={res['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_astar.py -v`
Expected: PASS (all). If `test_astar_matches_oracle` fails, A* and the independent oracle disagree on the exact optimum — STOP and reconcile (this is the core correctness gate).

- [ ] **Step 5: Run the selftest binary**

Run: `python astar.py --selftest`
Expected: `astar selftest: OK`

Run: `python astar.py --M 64 --K 64 --N 32 --act 2 --bank-size 1024 --banks 32 --dram-bw 1000000000000`
Expected: `energy(variable DRAM) = 2531328   PROVEN OPTIMAL   nodes=0   source=structural` (all-resident -> the first-touch A/W load floor = 12288 bits * 206 coef; no reloads/spills; proven at the root).

Run: `python astar.py --M 64 --K 64 --N 64 --act 8 --bank-size 4096 --banks 32 --dram-bw 1`
Expected: `NO FEASIBLE SCHEDULE: no stall=0-feasible schedule; ...` and exit code 1 (invariant #11 diagnostic; never a silent inf).

- [ ] **Step 6: Confirm the closed-form twin is untouched**

Run: `python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck`
Expected: `selftest: OK` then `crosscheck: OK` (M1 added new modules only; the closed-form model and its twin are unchanged).

- [ ] **Step 7: Commit**

```bash
git add MXP_scheduler/astar.py MXP_scheduler/test_astar.py
git commit -m "feat(scheduler): M1 A* joint (order+eviction) optimizer + oracle validation"
```

---

## Final verification (run after all tasks)

- [ ] Full M1 + regression suites:

Run: `python -m pytest test_eval_sched.py test_warmstart.py test_oracle.py test_astar.py test_mxp_scheduler.py test_hwconfig.py -q`
Expected: all PASS.

- [ ] Embedded checks (new + existing):

Run: `python astar.py --selftest && python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck`
Expected: `astar selftest: OK`, `selftest: OK`, `crosscheck: OK`.

- [ ] Update the memory pointer:

Edit `~/.claude/projects/.../memory/project_scheduler_m0_landed_m1_next.md` (or write a new `project_scheduler_m1_landed.md`) to record M1 landed (modules `eval_sched`/`warmstart`/`oracle`/`astar`, A*==oracle validated, exact-or-gap), and update `MEMORY.md`'s index line. Note that A* proves optimality only for small T and reports an honest gap for large T (spec §7/§12).

---

## Self-review notes (author)

- **Prerequisite — M0 landed.** This plan consumes M0 (`HW.cycles_per_bit`/`sa_fill_cycles`/`sa_drain_cycles`, the `fill`/`steady_stall`/`drain` split, `dram_presets.json`, `compute_work(w, cycles_per_bit)`). Verify `python mxp_scheduler.py --selftest` passes before starting. Note: spec §11 lists the §6 evaluator under M0, but it was *deferred* — only the stall-split + cycle params + DRAM coeffs landed in M0. This plan builds the evaluator (Tasks 2–3) alongside the optimizer, which is the right coupling.
- **Spec coverage:** §5 architecture → eval_sched (Tasks 1–3) + astar (Task 7) + DP oracle (Task 6) + closed-form warm-start (Task 5). §6 evaluator → Tasks 2–3 (single-source `apply_cube`, each-cube-once invariant, on-chip refill in the (dram+onchip) coef). §7 A* → Task 7 (state incl. last_compute, full eviction move set, admissible h, stall=0 prune, determinism, warm-start, exact-or-gap). §8 validation → Task 4 (closed-form parity + inner-blocked ≤ upper-bound), Task 6/7 (DP==A* at huge BW *and* finite BW via `stall0` oracle), Task 7 (determinism, stall=0 binding, D6 floor, gap, prime KT=5, precision-adaptive residency, empty-region diagnostic), Task 3 (cycles_per_bit hide-budget scaling). §D1–D9 → invariants list + Task 7. §1.4 oracle (closed-form is upper-bound sanity, DP is the real oracle) → Task 6.
- **Review fixes applied (rev2, after 3-agent review):**
  - *(Critical)* Minimal-eviction pruning was unsound (dropped cost-optimal super/voluntary evictions) → replaced with the **full exact move set** `eval_sched.eviction_choices`, shared by the oracle and A* (invariant #10). The reviewers' partial-C-spill-avoidance counterexample is now reachable in the search.
  - *(Critical)* Heap `(f, g, sig)` ties would compare the unorderable `SchedState` and crash → added a monotonic `seq` tiebreaker as the final heap-key element (invariant #7).
  - *(Critical)* The gap-budget test contradicted the termination logic → `budget_hit` now **forces** `proven_optimal=False` (invariant #8), and `test_astar_reports_gap_when_budget_zero` uses `node_budget=0` on a config whose warm-start needs reloads (so the bound-prune doesn't pre-empt the budget path).
  - *(Important)* Empty feasible region → `feasible=False` + `min_steady_stall` + human `reason`, `main()` prints a CLI error (invariant #11; `test_astar_empty_region_diagnostic`). No silent `inf`.
  - *(Important)* Finite-BW optimality was untested → `oracle.dp_optimal(stall0=True)` + `test_astar_matches_oracle_finite_bw`.
  - *(Important)* Precision-adaptive residency (the rev5 motivation) → `test_astar_precision_adaptive_residency` (oracle-exact + expensive-W loaded ≤ cheap-W). Prime ragged blocking → `test_astar_prime_tile_count_matches_oracle` (KT=5). cycles_per_bit hide-budget → `test_cycles_per_bit_scales_stall_hide_budget`. Inner-blocked ≤ closed-form → `test_inner_blocked_evaluator_le_closed_form_upper_bound`.
- **Deliberate equivalence (not an approximation):** final C writes modeled as a schedule-invariant constant excluded from g/h (invariant #2). Exact reformulation of spec §7's "final-write floor in h" (constant offset preserves argmin and A* optimality); the only literal deviation from §7's h, surfaced in `astar.py` + invariants.
- **Heuristic admissible-not-consistent:** `h` is admissible (→ optimum) but not monotone; optimality relies on the exact-state `best_g_to_state` re-expansion (heuristic note in invariants). Do not remove the re-expansion.
- **Independence of the oracle:** Task 6 enumerates orderings × per-order eviction DP — a different *search* from best-first A* (the move set `eviction_choices` is intentionally shared; the search is not), so A*==oracle is a real cross-check. `apply_cube` bugs are caught instead by Task 4's independent closed-form parity.
- **Honesty on scale (§7/§12):** A* proves optimality only when the frontier empties without a budget hit; otherwise honest gap. The full eviction move set is exponential in `|evictable|` (NP-hard variable-size caching) — small T closes (validated vs oracle), large T reports a gap. No silent truncation.
- **No twin drift:** M1 modules are single-source; the closed-form `--crosscheck` is asserted unchanged after every task.
- **Type/name consistency:** `SchedState(done, resident, last_compute)`; `apply_cube(state, c, evict, w, hw) -> (new_state, read, spill, unhidden, cap_ok)`; `eviction_choices(state, c, w, hw) -> iterator[frozenset]`; `eval_sched(...) -> dict` keys `dram_read_bits/dram_spill_bits/dram_final_bits/energy/feasible/stall0_feasible/capacity_feasible/fill/steady_stall/drain/actual_cycle`; `mapping_to_schedule -> (order, evictions)`; `structural_incumbent -> (energy, order, evictions)|None`; `min_structural_steady_stall -> float|None`; `dp_optimal(w, hw, stall0=False) -> {energy, order, evictions, proven_optimal}`; `optimize_exact -> {energy, order, evictions, feasible, proven_optimal, lower_bound, gap, nodes_expanded, source, min_steady_stall, reason}` — used identically across all tasks.
- **Known residual risk to verify in review:** the hand-derived golden in `test_eval_sched_hand_built_spill_accounting` (read=45056, spill=32768, final=65536) is reasoned (and independently re-derived by a reviewer), not executed; Task 3 Step 2→4 is the TDD gate. If it fails on a correct implementation, re-derive the trace before "fixing" the implementation. The capacity-pressure configs in the new tests (cap=49152 for KT=5/precision cases) assume the full footprint exceeds cap — if a config turns out to fit (trivial optimum), tighten `bank_size` so pressure actually occurs.
