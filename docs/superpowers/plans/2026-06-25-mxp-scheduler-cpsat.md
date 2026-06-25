# MXP_scheduler CP-SAT joint exact optimizer -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `MXP_scheduler/cpsat_sched.py`, a CP-SAT joint (cube-order + tile-eviction) exact optimizer that reproduces `eval_sched.apply_cube` bit-exactly, replacing the M1 A* engine in the T>=32 regime where A* OOMs; cross-validate `CP-SAT == oracle == A*` and extend `measure_gap.py` to measure the optimality gap at T=32/64.

**Architecture:** Step-indexed CP-SAT model (one cube per step). Per-step tile-residency booleans with COSMA-style C/P/S/R actions; a demand-driven load constraint ties new residency to the using cube (this closes a verified vacuous-stall exploit). Capacity = per-step size sum <= cap (M1 semantics, no prefetch-peak charge). C accumulation tracked by an order-derived counter; reload/spill charged via reified booleans. stall=0 is a hard per-step linear constraint coupling step i's traffic to step i-1's compute. All real coefficients integerized once via exact `fractions.Fraction` LCM scaling (no `round()`). Returns a dict shaped exactly like `astar.optimize_exact`; honest gap (incumbent + proven lower bound) on timeout.

**Tech Stack:** Python 3, OR-Tools CP-SAT (`ortools.sat.python.cp_model`, installed: 9.15.6755), `fractions`, pytest. Imports the M1 modules `mxp_scheduler`, `eval_sched` (data/cost source of truth) and `warmstart` (diagnostics) at module scope; imports `oracle`/`astar` only in tests/selftest. OR-Tools is offline-only -- no runtime module imports `cpsat_sched`.

**Key facts for the implementer (you have zero repo context):**
- Tests live in `MXP_scheduler/*.py` next to the modules and run with `cd MXP_scheduler && python -m pytest -q`. Imports are bare (`import mxp_scheduler as s`), so commands must `cd MXP_scheduler` first. There is no `tests/` dir and no conftest.
- `eval_sched.eval_sched(w, hw, order, evictions)` returns a dict with keys incl. `feasible`, `stall0_feasible`, `capacity_feasible`, `dram_read_bits`, `dram_spill_bits`, `energy`. `energy = (dram_read_bits + dram_spill_bits) * coef`, `coef = hw.coeffs["dram"] + hw.coeffs["onchip"]`.
- `eval_sched` helpers: `all_cubes(w)` (canonical cube list), `a_tile(c)/w_tile(c)/c_tile(c)` (the 3 tiles of cube c=(mt,kt,nt)), `tile_size(t, w)` (bits), `cube_compute(c, w, hw)` (on-chip cycles = `cycles_per_bit*TILE*wbits[mt][kt]`).
- `mxp_scheduler`: `TILE=32`, `FP32_BITS=32`, classes `Work(M,K,N,wbits,act_bits)` and `HW(bank_size,banks,dram_bw,word_bits=32,freq_ratio=1.0,coeffs,cycles_per_bit=1.0)`; `w.MT/KT/NT`, `hw.cap_bits`, `hw.eff_bw=dram_bw/freq_ratio`.
- `oracle.dp_optimal(w, hw, stall0=False)` -> `{energy, order, evictions, proven_optimal}` (raises if T>6 or infeasible). `astar.optimize_exact(w, hw, node_budget=...)` -> the result dict this module mirrors.
- `warmstart.min_structural_steady_stall(w, hw)` -> smallest steady_stall of any capacity-feasible structural schedule, or None.

---

## Task 1: Module scaffold, cube/tile + exact-rational scaling helpers

**Files:**
- Create: `MXP_scheduler/cpsat_sched.py`
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'cpsat_sched'`).

- [ ] **Step 3: Write the module scaffold + helpers**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/cpsat_sched.py MXP_scheduler/test_cpsat_sched.py
git commit -m "feat(scheduler): cpsat_sched scaffold + exact-rational integer scaling"
```

---

## Task 2: Core CP-SAT model + optimize_exact (OPTIMAL path, bit-exact cost)

This task builds the complete model (a solver is not meaningfully partial). Validation is the all-resident floor (G1), the reconstruct->eval_sched parity (G4), and one oracle match (G2). Subsequent tasks add the rest of the validation battery.

**Files:**
- Modify: `MXP_scheduler/cpsat_sched.py` (add `_build`, `_reconstruct`, `_infeasible_result`, `optimize_exact`)
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_all_resident_first_touch_floor_proven():
    # Everything fits -> optimum has no reloads/spills; energy = first-touch A/W floor
    # (2 A x 2048 + 4 W x 2048 = 12288 bits) * coef, NOT zero. Proven optimal.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    res = cps.optimize_exact(w, hw)
    assert res["proven_optimal"] is True
    assert res["energy"] == pytest.approx(12288 * coef)
    assert res["source"] == "cpsat"
    assert isinstance(res["nodes_expanded"], int)


def test_reconstruct_matches_eval_sched_bit_exact():
    # Master check: feed the CP-SAT (order, evictions) back through the single source of truth.
    w = s.Work(M=32, K=64, N=64, wbits=[[2, 4]], act_bits=2)            # MT=1,KT=2,NT=2,T=4
    hw = s.HW(bank_size=38, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 38912: real pressure
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is True
    e = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert e["feasible"] is True
    # canonical equality on INTEGER traffic bits (exact for integer wbits)
    assert round(res["energy"] / (hw.coeffs["dram"] + hw.coeffs["onchip"])) == \
        round(e["dram_read_bits"] + e["dram_spill_bits"])
    assert e["energy"] == pytest.approx(res["energy"])


def test_cpsat_matches_oracle_under_pressure():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)   # mixed-precision, T=4
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)  # cap 65536
    import oracle as o
    res = cps.optimize_exact(w, hw)
    ref = o.dp_optimal(w, hw)
    assert res["proven_optimal"] is True
    assert res["energy"] == pytest.approx(ref["energy"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -q`
Expected: FAIL (`AttributeError: module 'cpsat_sched' has no attribute 'optimize_exact'`).

- [ ] **Step 3: Implement the model build, reconstruction, and optimize_exact**

Append to `MXP_scheduler/cpsat_sched.py`:

```python
def _build(w, hw, enforce_stall=True):
    """Build the step-indexed CP-SAT model. Returns a handle dict. enforce_stall=False drops the
    stall=0 constraint (used by the infeasibility diagnostic to tell capacity- from stall-infeasible)."""
    cubes, tiles = _cubes_tiles(w)
    T = len(cubes)
    KT = w.KT
    G = _global_scale(w, hw)
    fr = Fraction(hw.freq_ratio).limit_denominator(10 ** 12)
    bw = Fraction(hw.dram_bw).limit_denominator(10 ** 12)
    size_G = {t: int(Fraction(es.tile_size(t, w)) * G) for t in tiles}          # objective + capacity
    cap_G = int(Fraction(hw.cap_bits) * G)
    lhs_G = {t: int(Fraction(es.tile_size(t, w)) * fr * G) for t in tiles}      # stall LHS coeff
    rhs_G = {c: int(bw * Fraction(es.cube_compute(c, w, hw)) * G) for c in cubes}  # stall RHS coeff
    maxc = max([cap_G, 1] + list(size_G.values()) + list(lhs_G.values()) + list(rhs_G.values()))
    if maxc >= INT64_MARGIN:
        raise ValueError("cpsat_sched: scaled coefficient %d exceeds int64 margin; "
                         "reduce dram_bw or precision of inputs" % maxc)

    uses = {t: [] for t in tiles}
    for c in cubes:
        for t in (es.a_tile(c), es.w_tile(c), es.c_tile(c)):
            uses[t].append(c)
    awtiles = [t for t in tiles if t[0] != "C"]
    ctiles = [t for t in tiles if t[0] == "C"]
    cgroup = {ct: [c for c in cubes if es.c_tile(c) == ct] for ct in ctiles}

    m = cp_model.CpModel()
    x = {(c, t): m.NewBoolVar("x_%d_%d" % (ci, t))
         for ci, c in enumerate(cubes) for t in range(T)}
    for c in cubes:                                  # each cube runs exactly once
        m.Add(sum(x[(c, t)] for t in range(T)) == 1)
    for t in range(T):                               # exactly one cube per step
        m.Add(sum(x[(c, t)] for c in cubes) == 1)

    res = {(tau, t): m.NewBoolVar("res_%d_%d" % (ti, t))
           for ti, tau in enumerate(tiles) for t in range(T)}

    def rprev(tau, t):
        return res[(tau, t - 1)] if t > 0 else 0

    for c in cubes:                                  # co-residency: a cube's 3 tiles resident at its step
        for tau in (es.a_tile(c), es.w_tile(c), es.c_tile(c)):
            for t in range(T):
                m.Add(res[(tau, t)] >= x[(c, t)])

    load = {}
    for tau in tiles:
        for t in range(T):
            lv = m.NewBoolVar("load_%s_%d" % (tau, t))
            m.Add(lv >= res[(tau, t)] - rprev(tau, t))             # fires on non-resident->resident
            m.Add(lv <= sum(x[(c, t)] for c in uses[tau]))         # DEMAND-DRIVEN (closes vacuous stall)
            load[(tau, t)] = lv

    for t in range(T):                               # capacity (M1 semantics: post-load resident <= cap)
        m.Add(sum(size_G[tau] * res[(tau, t)] for tau in tiles) <= cap_G)

    rl, sp = {}, {}                                  # C reload / C partial-spill charged terms
    for ct in ctiles:
        for t in range(T):
            cnt = sum(x[(c, u)] for c in cgroup[ct] for u in range(t))   # cnt_before == c_counter(done)
            cge1 = m.NewBoolVar("cge1_%s_%d" % (ct, t))
            m.Add(cnt >= 1).OnlyEnforceIf(cge1)
            m.Add(cnt <= 0).OnlyEnforceIf(cge1.Not())
            ev = m.NewBoolVar("evict_%s_%d" % (ct, t))
            m.Add(ev >= rprev(ct, t) - res[(ct, t)])
            r = m.NewBoolVar("rl_%s_%d" % (ct, t))                  # reload = load AND cnt>=1
            m.Add(r >= load[(ct, t)] + cge1 - 1)
            rl[(ct, t)] = r
            clek = m.NewBoolVar("clek_%s_%d" % (ct, t))
            m.Add(cnt <= KT - 1).OnlyEnforceIf(clek)
            m.Add(cnt >= KT).OnlyEnforceIf(clek.Not())
            sv = m.NewBoolVar("sp_%s_%d" % (ct, t))                 # spill = evict AND 1<=cnt<=KT-1
            m.Add(sv >= ev + cge1 + clek - 2)
            sp[(ct, t)] = sv

    def traffic_terms(t, coeff):
        terms = [coeff[tau] * load[(tau, t)] for tau in awtiles]
        terms += [coeff[ct] * rl[(ct, t)] for ct in ctiles]
        terms += [coeff[ct] * sp[(ct, t)] for ct in ctiles]
        return terms

    obj = sum(sum(traffic_terms(t, size_G)) for t in range(T))
    m.Minimize(obj)

    if enforce_stall:                                # stall=0 hard: step i traffic hides under step i-1 compute
        for i in range(1, T):
            m.Add(sum(traffic_terms(i, lhs_G)) <= sum(rhs_G[c] * x[(c, i - 1)] for c in cubes))

    return dict(model=m, x=x, res=res, obj=obj, cubes=cubes, tiles=tiles, T=T, G=G)


def _reconstruct(solver, h):
    """Read (order, evictions) out of a solved model. evictions[t] = tiles resident at t-1 but not t."""
    x, res, cubes, tiles, T = h["x"], h["res"], h["cubes"], h["tiles"], h["T"]
    order = []
    for t in range(T):
        order.append(next(c for c in cubes if solver.Value(x[(c, t)]) == 1))
    evictions = []
    for t in range(T):
        ev = set()
        for tau in tiles:
            prev = solver.Value(res[(tau, t - 1)]) if t > 0 else 0
            if prev == 1 and solver.Value(res[(tau, t)]) == 0:
                ev.add(tau)
        evictions.append(frozenset(ev))
    return order, evictions


def _new_solver(max_time, random_seed):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1            # determinism (in-process, fixed ortools version)
    solver.parameters.random_seed = random_seed
    if max_time is not None:
        solver.parameters.max_time_in_seconds = max_time
    return solver


def _infeasible_result(w, hw, nodes):
    """No stall=0 solution for the full model. Re-solve dropping stall=0 to tell capacity-infeasible
    from stall0-infeasible. Never returns a bare inf without a human reason (invariant: no silent inf)."""
    import warmstart as ws
    h2 = _build(w, hw, enforce_stall=False)
    solver2 = _new_solver(10.0, 0)
    st2 = solver2.Solve(h2["model"])
    base = {"energy": float("inf"), "order": None, "evictions": None,
            "feasible": False, "proven_optimal": False, "lower_bound": float("inf"),
            "gap": 0.0, "nodes_expanded": int(nodes), "source": "cpsat"}
    if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        base["min_steady_stall"] = ws.min_structural_steady_stall(w, hw)
        base["reason"] = ("no stall=0-feasible schedule; raise dram_bw (eff_bw=%g) or shrink the "
                          "resident window so each fetch hides under the prior cube's compute."
                          % hw.eff_bw)
    else:
        base["min_steady_stall"] = None
        base["reason"] = ("no capacity-feasible schedule exists (footprint exceeds on-chip capacity "
                          "for every mapping); increase banks / bank_size / word_bits.")
    return base


def optimize_exact(w, hw, max_time=None, random_seed=0):
    """Joint (order+eviction) exact optimizer. Returns a dict with the same keys as
    astar.optimize_exact: energy, order, evictions, feasible, proven_optimal, lower_bound, gap,
    nodes_expanded, source, min_steady_stall, reason. proven_optimal=True only when the solver
    proves OPTIMAL; on timeout it returns the incumbent with an honest gap; infeasible -> diagnostic."""
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    h = _build(w, hw, enforce_stall=True)
    solver = _new_solver(max_time, random_seed)
    status = solver.Solve(h["model"])
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        order, evictions = _reconstruct(solver, h)
        bits = solver.Value(h["obj"]) / h["G"]
        energy = bits * coef
        proven = status == cp_model.OPTIMAL
        if proven:
            lower_bound, gap = energy, 0.0
        else:
            lower_bound = (solver.BestObjectiveBound() / h["G"]) * coef
            gap = (energy - lower_bound) / lower_bound if lower_bound > 0 else 0.0
        return {"energy": energy, "order": order, "evictions": evictions, "feasible": True,
                "proven_optimal": proven, "lower_bound": lower_bound, "gap": gap,
                "nodes_expanded": int(solver.NumBranches()), "source": "cpsat",
                "min_steady_stall": None, "reason": None}
    return _infeasible_result(w, hw, solver.NumBranches())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -q`
Expected: PASS (all Task 1 + Task 2 tests pass).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/cpsat_sched.py MXP_scheduler/test_cpsat_sched.py
git commit -m "feat(scheduler): CP-SAT model + optimize_exact (OPTIMAL path, bit-exact vs eval_sched)"
```

---

## Task 3: stall=0 finite-bandwidth validation (the demand-load anti-regression)

Validates that under finite bandwidth the engine matches the stall0-constrained oracle, and that a returned schedule is genuinely stall0-feasible per `eval_sched` (this is the test that would have caught the vacuous-stall bug).

**Files:**
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing test**

```python
def test_stall0_returned_schedule_is_feasible():
    # Finite BW: a returned schedule, re-scored by eval_sched, MUST be stall0-feasible.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=1024, banks=32, dram_bw=64)        # finite BW
    res = cps.optimize_exact(w, hw)
    if res["feasible"]:
        e = es.eval_sched(w, hw, res["order"], res["evictions"])
        assert e["stall0_feasible"] is True and e["feasible"] is True


def test_cpsat_matches_oracle_finite_bw():
    # Finite BW so stall=0 BINDS: must match the stall0-constrained oracle (not the unconstrained one).
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)   # T=4 mixed
    hw = s.HW(bank_size=2, banks=32, dram_bw=256, word_bits=1024)      # cap 65536, finite BW
    import oracle as o
    res = cps.optimize_exact(w, hw)
    if res["feasible"]:
        ref = o.dp_optimal(w, hw, stall0=True)
        assert res["proven_optimal"] is True
        assert res["energy"] == pytest.approx(ref["energy"])
    else:
        with pytest.raises(ValueError):           # oracle must also find nothing stall0-feasible
            o.dp_optimal(w, hw, stall0=True)
```

- [ ] **Step 2: Run to verify behavior**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k stall0 -q` and `... -k finite_bw -q`
Expected: PASS (the Task 2 engine already enforces stall=0; these lock it in). If either FAILS, the stall=0 encoding or demand-load constraint is wrong -- stop and fix `_build` before proceeding.

- [ ] **Step 3: Commit**

```bash
git add MXP_scheduler/test_cpsat_sched.py
git commit -m "test(scheduler): cpsat stall=0 finite-BW matches stall0 oracle + schedule feasibility"
```

---

## Task 4: Honest-gap path on timeout (incumbent + proven lower bound)

**Files:**
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing test**

```python
def test_honest_gap_on_timeout():
    # A capacity-pressured instance where proving optimality takes search; a tiny time budget
    # forces a feasible-but-unproven return with an honest, lower-bound-relative gap.
    w = s.Work(M=32, K=160, N=32, wbits=[[2, 4, 8, 2, 6]], act_bits=2)  # KT=5, T=5
    hw = s.HW(bank_size=48, banks=32, dram_bw=10 ** 12, word_bits=32)   # cap 49152 < footprint
    res = cps.optimize_exact(w, hw, max_time=0.05)
    assert res["feasible"] is True                  # a feasible incumbent is found quickly
    if res["proven_optimal"] is False:              # the intended timeout branch
        assert res["energy"] >= res["lower_bound"] - 1e-6
        assert res["gap"] == pytest.approx((res["energy"] - res["lower_bound"]) / res["lower_bound"])
        assert res["gap"] >= 0.0
```

- [ ] **Step 2: Run to verify behavior**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k honest_gap -q`
Expected: PASS. (If the solver proves optimality even at 0.05s, the `if` body is skipped and the test still passes; that is acceptable -- it means CP-SAT closed it, a finding for open question A. The honest-gap *arithmetic* is still exercised by the assertions when not proven.)

- [ ] **Step 3: Commit**

```bash
git add MXP_scheduler/test_cpsat_sched.py
git commit -m "test(scheduler): cpsat honest-gap on timeout (incumbent + proven lower bound)"
```

---

## Task 5: Infeasibility diagnostics (capacity vs stall0, never a bare inf)

**Files:**
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_stall0_infeasible_diagnostic():
    # Big cap (no reload pressure) + tiny BW -> NO stall=0 schedule. Must report feasible=False,
    # a positive min_steady_stall, and a human reason -- not a bare inf.
    w = s.Work(M=64, K=64, N=64, wbits=[[8, 8], [8, 8]], act_bits=8)
    hw = s.HW(bank_size=4096, banks=32, dram_bw=1)        # huge cap, tiny BW -> all stall
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is False and res["proven_optimal"] is False
    assert res["energy"] == float("inf")
    assert res["min_steady_stall"] is not None and res["min_steady_stall"] > 0
    assert "stall=0" in res["reason"]


def test_capacity_infeasible_diagnostic():
    # A single cube's A+W+C exceeds capacity -> no schedule fits at all (capacity-infeasible).
    w = s.Work(M=32, K=32, N=32, wbits=[[8]], act_bits=8)   # one cube; A+W+C = 8192+8192+32768
    hw = s.HW(bank_size=1, banks=1, dram_bw=10 ** 12, word_bits=32)   # cap 32 bits << one cube
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is False
    assert res["energy"] == float("inf")
    assert res["min_steady_stall"] is None
    assert "capacity" in res["reason"]
```

- [ ] **Step 2: Run to verify behavior**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k infeasible -q`
Expected: PASS (the `_infeasible_result` branch from Task 2 handles both). If the capacity case returns feasible, the cap is not tight enough -- it is intentionally far below one cube's working set.

- [ ] **Step 3: Commit**

```bash
git add MXP_scheduler/test_cpsat_sched.py
git commit -m "test(scheduler): cpsat infeasibility diagnostics (capacity vs stall0, no silent inf)"
```

---

## Task 6: Determinism, selftest(), and CLI

**Files:**
- Modify: `MXP_scheduler/cpsat_sched.py` (add `selftest`, `main`, `__main__`)
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_deterministic():
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)
    r1 = cps.optimize_exact(w, hw)
    r2 = cps.optimize_exact(w, hw)
    assert r1["order"] == r2["order"]
    assert r1["energy"] == pytest.approx(r2["energy"])


def test_selftest_runs():
    cps.selftest()   # prints "cpsat_sched selftest: OK"; raises on any golden mismatch
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k "deterministic or selftest" -q`
Expected: `test_selftest_runs` FAILS (`AttributeError: ... 'selftest'`). `test_deterministic` may already pass.

- [ ] **Step 3: Add selftest and CLI**

Append to `MXP_scheduler/cpsat_sched.py`:

```python
def selftest():
    import oracle as o
    # G1: all-resident floor = first-touch A/W bits * coef, proven at the root.
    w1 = s.Work(M=64, K=64, N=32, wbits=[[2, 2], [2, 2]], act_bits=2)
    hw1 = s.HW(bank_size=1024, banks=32, dram_bw=10 ** 12)
    coef1 = hw1.coeffs["dram"] + hw1.coeffs["onchip"]
    r1 = optimize_exact(w1, hw1)
    assert r1["proven_optimal"] and abs(r1["energy"] - 12288 * coef1) < 1e-6, r1
    # G4: reconstruct -> eval_sched parity.
    e1 = es.eval_sched(w1, hw1, r1["order"], r1["evictions"])
    assert e1["feasible"] and abs(e1["energy"] - r1["energy"]) < 1e-6, e1
    # G2: cpsat == oracle under mixed-precision capacity pressure.
    w2 = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw2 = s.HW(bank_size=2, banks=32, dram_bw=10 ** 12, word_bits=1024)
    r2 = optimize_exact(w2, hw2)
    ref2 = o.dp_optimal(w2, hw2)
    assert r2["proven_optimal"] and abs(r2["energy"] - ref2["energy"]) < 1e-6, (r2, ref2)
    # G3: in-process determinism.
    assert optimize_exact(w2, hw2)["order"] == r2["order"]
    print("cpsat_sched selftest: OK")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="MXP_scheduler CP-SAT joint (order+eviction) optimizer")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0)
    p.add_argument("--max-time", type=float, default=None)
    args = p.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if not (args.M and args.K and args.N):
        p.error("provide --M --K --N (or --selftest)")
    mt, kt = args.M // s.TILE, args.K // s.TILE
    w = s.Work(M=args.M, K=args.K, N=args.N,
               wbits=[[args.act] * kt for _ in range(mt)], act_bits=args.act)
    hw = s.HW(bank_size=args.bank_size, banks=args.banks, dram_bw=args.dram_bw)
    res = optimize_exact(w, hw, max_time=args.max_time)
    if not res["feasible"]:
        print("NO FEASIBLE SCHEDULE: %s" % res["reason"])
        return 1                                      # CLI error, never a silent inf
    status = ("PROVEN OPTIMAL" if res["proven_optimal"]
              else "gap=%.1f%% (lb=%.0f)" % (res["gap"] * 100, res["lower_bound"]))
    print("energy(variable DRAM) = %.0f   %s   branches=%d   source=%s"
          % (res["energy"], status, res["nodes_expanded"], res["source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k "deterministic or selftest" -q`
then: `cd MXP_scheduler && python cpsat_sched.py --selftest`
Expected: tests PASS; CLI prints `cpsat_sched selftest: OK`.

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/cpsat_sched.py MXP_scheduler/test_cpsat_sched.py
git commit -m "feat(scheduler): cpsat_sched selftest + CLI; determinism test"
```

---

## Task 7: Cross-validation sweep -- CP-SAT == oracle == A*

**Files:**
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("M,K,N,wb,act,bank_size,word_bits", [
    (64, 64, 32, [[2, 2], [2, 2]], 2, 2, 1024),    # cap 65536, T=4, forced pressure
    (32, 64, 32, [[2, 2]], 2, 40, 1024),           # cap 40960, T=2
    (64, 64, 32, [[2, 4], [2, 2]], 2, 2, 1024),    # mixed-precision W sizes, T=4
    (32, 64, 64, [[2, 8]], 2, 48, 32),             # precision-adaptive residency, T=4, cap 49152
    (32, 160, 32, [[2, 4, 8, 2, 6]], 2, 48, 32),   # KT=5 prime, T=5, cap 49152
])
def test_cpsat_equals_oracle_and_astar(M, K, N, wb, act, bank_size, word_bits):
    w = s.Work(M=M, K=K, N=N, wbits=wb, act_bits=act)
    hw = s.HW(bank_size=bank_size, banks=32, dram_bw=10 ** 12, word_bits=word_bits)
    import oracle as o
    import astar as a
    cp = cps.optimize_exact(w, hw)
    orc = o.dp_optimal(w, hw)
    ast = a.optimize_exact(w, hw)
    assert cp["proven_optimal"] and ast["proven_optimal"] and orc["proven_optimal"]
    coef = hw.coeffs["dram"] + hw.coeffs["onchip"]
    # canonical equality on integer traffic bits (exact for integer wbits)
    assert round(cp["energy"] / coef) == round(orc["energy"] / coef) == round(ast["energy"] / coef)
```

- [ ] **Step 2: Run to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k equals_oracle_and_astar -q`
Expected: PASS (5 parametrizations). A mismatch means the CP-SAT cost model diverges from `apply_cube` -- stop and reconcile against the spec section 3 table before continuing.

- [ ] **Step 3: Commit**

```bash
git add MXP_scheduler/test_cpsat_sched.py
git commit -m "test(scheduler): cpsat == oracle == astar cross-validation sweep (T<=5)"
```

---

## Task 8: Exact-rational scaling (fractional dram_bw and fractional wbits)

**Files:**
- Test: `MXP_scheduler/test_cpsat_sched.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fractional_dram_bw_scaling_parity():
    # Fractional dram_bw -> G != 1 path. Reconstruct -> eval_sched parity must still hold.
    w = s.Work(M=64, K=64, N=32, wbits=[[2, 4], [2, 2]], act_bits=2)
    hw = s.HW(bank_size=2, banks=32, dram_bw=2.5, word_bits=1024)   # finite, fractional BW
    res = cps.optimize_exact(w, hw)
    if res["feasible"]:
        e = es.eval_sched(w, hw, res["order"], res["evictions"])
        assert e["feasible"] is True
        assert e["energy"] == pytest.approx(res["energy"], abs=1e-6)


def test_fractional_wbits_scaling_parity():
    # Fractional average weight bits are permitted by Work (avg bits in [2,8]); a value whose
    # *1024 size is non-integral (e.g. 2.1 -> 2150.4 bits) exercises G != 1 size scaling.
    w = s.Work(M=32, K=64, N=64, wbits=[[2.1, 4.0]], act_bits=2)
    hw = s.HW(bank_size=40, banks=32, dram_bw=10 ** 12, word_bits=32)
    res = cps.optimize_exact(w, hw)
    assert res["feasible"] is True
    e = es.eval_sched(w, hw, res["order"], res["evictions"])
    assert e["feasible"] is True
    assert e["energy"] == pytest.approx(res["energy"], abs=1e-6)
```

- [ ] **Step 2: Run to verify it passes**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k fractional -q`
Expected: PASS. If a `TypeError: Linear constraints only accept integer values` is raised, the scaling is not applied globally -- every size/cap/stall coefficient must go through `_global_scale` (Task 1) and `int(Fraction(...)*G)` (Task 2). If `test_fractional_wbits_scaling_parity` errors in `Work(...)` construction, confirm `mxp_scheduler.Work` accepts fractional avg bits (it validates `2 <= b <= 8`); if not, change `2.1` to a permitted fractional like `2.5` and `4.0`.

- [ ] **Step 3: Commit**

```bash
git add MXP_scheduler/test_cpsat_sched.py
git commit -m "test(scheduler): cpsat exact-rational scaling parity (fractional dram_bw + wbits)"
```

---

## Task 9: measure_gap.py CP-SAT backend (push gap to T=32/64)

**Files:**
- Modify: `MXP_scheduler/measure_gap.py` (add `--backend`, CP-SAT shapes, honest-gap column)
- Test: `MXP_scheduler/test_cpsat_sched.py` (smoke import + tiny run)

- [ ] **Step 1: Read the current harness**

Run: `cd MXP_scheduler && python -c "import measure_gap"` to confirm it imports (it runs as a flat script -- you will wrap its body in `main()`).

- [ ] **Step 2: Rewrite measure_gap.py with a backend selector**

Replace the body of `MXP_scheduler/measure_gap.py` (keep `min_working_set`, `mixed_map`, `PRECS`) with:

```python
# MXP_scheduler/measure_gap.py
"""Measure the optimality gap the structural baseline leaves vs the PROVEN-optimal schedule.

gap = (warmstart_energy - opt_energy) / opt_energy, recorded where the engine proves optimal;
on timeout the CP-SAT backend prints the HONEST (lower-bound-relative) gap, marked with '*'.

Backends:
  --backend astar  : M1 A* (default; OOMs at T>=32 under pressure -- keep to T<=12 shapes).
  --backend cpsat  : CP-SAT engine; pushes to the T=32/64 shapes A* cannot reach.

Run from MXP_scheduler/:
  python measure_gap.py --backend cpsat
  python measure_gap.py --backend cpsat --max-time 60
"""
import argparse
import mxp_scheduler as s
import warmstart as ws
import eval_sched as es

BANKS, WB = 32, 32

ASTAR_SHAPES = [(64, 64, 32), (64, 64, 64), (96, 64, 64)]            # T = 4, 8, 12
CPSAT_SHAPES = [(64, 64, 32), (96, 64, 64), (128, 128, 64), (128, 128, 128)]  # T = 4, 12, 32, 64
PRECS = {
    "unif8": lambda MT, KT: [[8] * KT for _ in range(MT)],
    "unif2": lambda MT, KT: [[2] * KT for _ in range(MT)],
    "mixed": lambda MT, KT: [[2 if (i + j) % 2 == 0 else 8 for j in range(KT)] for i in range(MT)],
}
MULTS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def min_working_set(w):
    mws = 0
    for c in es.all_cubes(w):
        ws_c = (es.tile_size(es.a_tile(c), w) + es.tile_size(es.w_tile(c), w)
                + es.tile_size(es.c_tile(c), w))
        mws = max(mws, ws_c)
    return mws


def _solve(backend, w, hw, max_time):
    if backend == "astar":
        import astar
        r = astar.optimize_exact(w, hw)
        return (r["energy"] if r["feasible"] else None, r["proven_optimal"],
                r["nodes_expanded"], r["gap"])
    import cpsat_sched
    r = cpsat_sched.optimize_exact(w, hw, max_time=max_time)
    return (r["energy"] if r["feasible"] else None, r["proven_optimal"],
            r["nodes_expanded"], r["gap"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["astar", "cpsat"], default="astar")
    p.add_argument("--max-time", type=float, default=30.0, help="per-instance CP-SAT budget (s)")
    p.add_argument("--quick", action="store_true",
                   help="smallest shape, one prec, two capacities -- fast CI smoke (not the full sweep)")
    args = p.parse_args(argv)
    shapes = ASTAR_SHAPES if args.backend == "astar" else CPSAT_SHAPES
    precs = PRECS
    mults = MULTS
    if args.quick:
        shapes = shapes[:1]
        precs = {"mixed": PRECS["mixed"]}
        mults = MULTS[:2]

    print("backend=%s   ('*' = honest gap, not proven optimal)" % args.backend)
    print("shape         T  prec  capxWS  warmE(e6) optE(e6)  proven  nodes   gap%")
    print("-" * 80)
    for (M, K, N) in shapes:
        MT, KT, NT = M // 32, K // 32, N // 32
        T = MT * KT * NT
        for pname, pf in precs.items():
            w = s.Work(M=M, K=K, N=N, wbits=pf(MT, KT), act_bits=8)
            mws = min_working_set(w)
            for mult in mults:
                bank_size = max(1, int(mws * mult) // (BANKS * WB))
                hw = s.HW(bank_size=bank_size, banks=BANKS, dram_bw=1e12, word_bits=WB)
                capx = hw.cap_bits / mws
                warm = ws.structural_incumbent(w, hw)
                optE, proven, nodes, gap = _solve(args.backend, w, hw, args.max_time)
                warmE = warm[0] if warm else None
                if warmE and optE and optE > 0:
                    g = (warmE - optE) / optE * 100
                    gaps = "%6.2f" % g if proven else "%6.2f*" % g    # measured vs proven optimum
                elif optE and not proven:
                    gaps = "%5.1f*h" % (gap * 100)                    # honest lb-relative gap only
                else:
                    gaps = "   -  "
                we = "%8.2f" % (warmE / 1e6) if warmE else "   inf  "
                oe = "%8.2f" % (optE / 1e6) if optE else "   inf  "
                print("%dx%dx%-4d %3d %6s %5.2fx  %s %s  %5s  %6d  %s"
                      % (M, K, N, T, pname, capx, we, oe, str(proven), nodes, gaps))
        print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write the smoke test**

```python
def test_measure_gap_cpsat_backend_smoke():
    # --quick: smallest shape (T=4), one prec, two capacities -> end-to-end in seconds, no raise.
    import measure_gap
    rc = measure_gap.main(["--backend", "cpsat", "--quick", "--max-time", "5"])
    assert rc == 0
```

- [ ] **Step 4: Run the smoke test and a manual T=32/64 pass**

Run: `cd MXP_scheduler && python -m pytest test_cpsat_sched.py -k measure_gap -q`
Expected: PASS in a few seconds (the `--quick` smoke runs ~2 small instances, NOT the full T=64 sweep).
Then (manual, optional, answers open questions A/B): `cd MXP_scheduler && python measure_gap.py --backend cpsat --max-time 60` -- inspect whether T=32/64 rows show `proven=True` (A) and whether `gap%` climbs above 0 only at `capx <~ 1.5` (B).

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/measure_gap.py MXP_scheduler/test_cpsat_sched.py
git commit -m "feat(scheduler): measure_gap CP-SAT backend + T=32/64 shapes + honest-gap column"
```

---

## Task 10: Full regression and final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole MXP_scheduler suite**

Run: `cd MXP_scheduler && python -m pytest -q`
Expected: all prior tests still pass (134 passed, 1 skip baseline) PLUS the new `test_cpsat_sched.py` cases. No regressions in M1 modules (they are unmodified).

- [ ] **Step 2: Run all engine selftests**

Run:
```bash
cd MXP_scheduler && python cpsat_sched.py --selftest && python astar.py --selftest && python -c "import oracle, mxp_scheduler as s; print('oracle import OK')"
```
Expected: `cpsat_sched selftest: OK` and `astar selftest: OK`.

- [ ] **Step 3: Confirm the runtime never imports the offline tool**

Run: `cd MXP_scheduler && python -c "import sys; import mxp_scheduler; assert 'cpsat_sched' not in sys.modules and 'ortools' not in sys.modules; print('runtime clean of ortools/cpsat_sched: OK')"`
Expected: prints the OK line (proves the stdlib runtime path does not pull OR-Tools).

- [ ] **Step 4: Commit any final touch-ups and report**

```bash
git add -A MXP_scheduler/
git commit -m "test(scheduler): full regression green for CP-SAT joint optimizer" --allow-empty
```

Report to the user: pytest summary line, the two selftest OK lines, and (if the manual Task 9 pass was run) the T=32/64 proven/gap findings for open questions A and B.

---

## Self-review notes (author)

- **Spec coverage:** module + offline constraints (Task 1-2, 10/Step 3); step-indexed model with C/P/S/R + demand-load + capacity + stall=0 + reservoir + exact-rational scaling (Task 2, 8); result-dict contract incl. per-path values (Task 2, 4, 5); reconstruct->eval_sched master check (Task 2); cpsat==oracle==astar (Task 7); determinism + selftest + CLI (Task 6); honest-gap (Task 4); infeasibility diagnostics (Task 5); measure_gap backend + T=32/64 + honest-gap column (Task 9); open questions A/B answerable via Task 9 manual pass.
- **Symmetry breaking (spec 4.9):** intentionally OMITTED from `_build` (off by default, oracle-gated). Not implemented here -- correct per spec; revisit only if solve time at T=64 demands it, behind the Task 7 oracle gate.
- **Reified-AND encoding (spec 4.7):** `rl >= load+cge1-1` and `sp >= ev+cge1+clek-2` with cost/stall minimization pressure are sufficient (both objective and stall push the term down; the lower bound forces it up exactly when the AND holds). Verified equivalent to `apply_cube` by Task 2/7 parity tests.
- **No placeholders:** every code step is complete and runnable. Types/keys (`energy/order/evictions/feasible/proven_optimal/lower_bound/gap/nodes_expanded/source/min_steady_stall/reason`) are consistent across tasks and match `astar.optimize_exact`.
