# MXP_scheduler CP-SAT joint (order+eviction) exact optimizer -- design

Date: 2026-06-25
Status: DRAFT rev2 (post multi-agent review; 3 reviewers, CRITICAL/IMPORTANT findings folded in)
Supersedes engine choice in: 2026-06-23 precision-adaptive design (rev5) for the small/medium-T exact engine; M1 A* (`astar.py`) is retained for cross-validation, not removed.

## 1. Motivation and decision context

The M1 joint exact optimizer (`astar.py`) is correct (joint order+eviction, stall=0 hard prune, exact C-spill accounting, `A* == oracle` verified) but materializes the eviction-subset search as child nodes, so under capacity pressure at T >= 32 the open heap OOMs (~14.8 GB). The 2026-06-25 six-agent evaluation confirmed (6/6) that the engine for the small/medium-T exact regime should be **CP-SAT** (OR-Tools): lazy clause generation avoids state materialization, so it either closes the T=64 cases A* could not, or degrades to an **honest gap** (incumbent + proven lower bound) instead of OOM. Decomposition (order then eviction) is rejected: stall=0 couples the two and the optimal order is often a non-loop permutation (repo oracle: mixed +10.5%, uniform serpentine +25%). The problem must be solved **jointly**.

### 1.1 Prior-art grounding (deep-research, 2026-06-25, 21 primary sources, 3-vote verified)

- **COSMA** (Li/Gupta/Malik, arXiv 2311.18246) is the closest match: one ILP jointly optimizing operator schedule + scratchpad allocation + tensor replacement (eviction), encoded as **four per-tensor-per-timestep action binaries Creation/Preservation/Spilling/Retrieval (C/P/S/R)** with state-transition constraints. We BORROW the C/P/S/R residency-action skeleton. COSMA does NOT transfer on: operator (not tile/cube) granularity, objective EXCLUDES compulsory first-touch transfers (we INCLUDE the A/W floor, per M1), and it models none of our three hard features.
- **CoSA** (ISCA'21) proves a one-shot exact MIP over tiling+order is fast, and independently corroborates our M1 finding that **loop-order enumeration != optimal**. Its static log-linearized capacity model (no time, no eviction, no reservoir) does NOT transfer.
- **RCPSP/max + lazy clause generation** (Schutt/Feydy/Stuckey/Wallace, arXiv 1009.0347), the algorithmic ancestor of OR-Tools CP-SAT, is an EXACT solver for cumulative-resource scheduling with **generalized (min/max time-lag) precedences** -- the right family for both K-accumulation ordering and the stall=0 "fetch within previous compute window" (a max time-lag). The one piece it does not hand us -- durations that depend on tile sizes -- is benign here because wbits is DATA, not a decision: per-step compute and traffic are linear in the assignment (see 4.6).
- **Writeback-aware caching** (Beckmann/Gibbons/Haeupler/McGuffey, APOCS'20) proves offline psum-style eviction is NP-complete and Max-SNP/APX-hard even at unit cost, and that Belady is only (omega+1)-competitive (tight). This justifies **honest-gap** reporting and kills any Belady/min-cost-flow C-spill decomposition. The named optimal online policy, **Writeback-Aware Landlord**, is the correct large-T heuristic baseline (deferred to the next phase).
- **Buffets** (ASPLOS'19): strict double-buffering charges the in-flight prefetch peak at a literal 2x, psum is a distinct long-lived in-place-updated type, and a unified soft-partitioned SRAM serving A/W/C is silicon-proven. We retain the **M1 capacity semantics (post-load resident <= cap, no separate prefetch-peak charge)** for this build -- a deliberate decision (see 6) so the CP-SAT objective stays bit-exact with `apply_cube` and the `CP-SAT == oracle == A*` cross-check holds.

## 2. Scope

In scope (this build):
- New offline module `MXP_scheduler/cpsat_sched.py` (depends on `ortools`), returning a result dict shaped exactly like `astar.optimize_exact`.
- `selftest()` and a new `test_cpsat_sched.py` (skipped if `ortools` is absent).
- Cross-validation `CP-SAT == oracle == A*` on small T (<=6), plus a reconstruct->`eval_sched` bit-exact parity check, plus the timeout/honest-gap, infeasible-diagnostic, determinism, and exact-rational-scaling tests enumerated in 5.5.
- A CP-SAT backend in `measure_gap.py` to push gap measurement to the T=32/64 cases A* OOMs on (edit plan in 5.6).

Out of scope (next phase): large-T (10^3-10^6) structural heuristic (Writeback-Aware Landlord) + honest lower bound (FOO-L / Lagrangian). The C/P/S/R model here is the exact reference those will be measured against.

Immutable (do not edit): RTL, `mxp_scheduler.py`, `mxp_scheduler_annotated.py`, `hwconfig.py`, and the M1 four modules `eval_sched.py`, `warmstart.py`, `oracle.py`, `astar.py`. `cpsat_sched.py` is a pure consumer of `eval_sched` data (TILE/FP32_BITS, `Work`, `HW`, tile helpers, `cube_compute`) -- it imports them at module scope, never mutates them, and imports `oracle`/`astar` only inside test/selftest code (so a runtime that lacks ortools never pulls cpsat_sched, and cpsat_sched never makes the M1 search modules a module-scope dependency). OR-Tools is an offline-only dependency; no runtime module imports `cpsat_sched`. All user-facing print is ASCII.

## 3. Cost model = single source of truth

The objective and per-step traffic must reproduce `eval_sched.apply_cube` BIT-EXACT (`eval_sched.py:75-134`). Definitions (unchanged):
- Cubes: canonical `all_cubes(w)` = product(MT, KT, NT); cube id = index. T = MT*KT*NT.
- Tiles of cube c=(mt,kt,nt): `a_tile=("A",kt,nt)`, `w_tile=("W",mt,kt)`, `c_tile=("C",mt,nt)`.
- Sizes (bits): A = TILE*TILE*act_bits (fixed); W = wbits[mt][kt]*TILE*TILE (variable, and **may be fractional** -- `Work` permits average bits in [2,8], `mxp_scheduler.py:94-96`); C = TILE*TILE*FP32_BITS (fixed).
- `cube_compute(c) = cycles_per_bit * TILE * wbits[mt][kt]` (on-chip cycles).
- `coef = coeffs["dram"] + coeffs["onchip"]` -- a positive real, **generally non-integer** (default DRAM/on-chip pJ/bit presets). This is why the canonical equality is on integer traffic bits, not on the float energy (5.3).
- `eff_bw = dram_bw / freq_ratio` (`mxp_scheduler.py:68`).

Objective (energy units): `coef * sum_i traffic_bits[i]`, where `traffic_bits[i]` is the charged read+spill at step i, in BITS:

| event at step i | charged bits |
|---|---|
| A or W load (tile becomes resident) | size(tile) -- INCLUDES the first-touch floor (every A/W's first load) AND reloads, per M1 (`apply_cube` 109-117 charges unconditionally for A/W) |
| C load with cnt_before > 0 | size(C) (reload of a spilled partial; `apply_cube` 111-114) |
| C load with cnt_before == 0 | 0 (zero-init; occupies space, no DRAM read) |
| C evict with 0 < cnt_before < KT | size(C) (partial-psum spill; `apply_cube` 101-104) |
| C evict with cnt_before in {0, KT} | 0 (zero data / complete -- final write is a schedule-invariant constant, excluded) |
| A or W evict | 0 (read-only inputs) |

`cnt_before[C(mt,nt), i]` = number of cubes (mt, *, nt) scheduled at steps u < i = exactly `c_counter(state.done, mt, nt)` (`apply_cube` uses `state.done` = cubes strictly before the current cube; reviewer-confirmed these are the SAME value at step i, including the case where the current cube's own C is in the eviction set).

`eval_sched.energy = (read_total + spill_total) * coef` (`eval_sched.py:223`) and `sum_i traffic_bits[i] = read_total + spill_total` over ALL steps including step 0 (step-0 traffic IS in energy; only the stall=0 CONSTRAINT exempts step 0). Excluded schedule-invariant constants: final C write (`dram_final_bits`), SA on-chip reads, mac, rmw, and the trailing `drain` term (which is in `actual_cycle` only, never in `energy`). An all-resident schedule therefore has energy = first-touch A/W floor * coef, NOT zero.

## 4. CP-SAT model

Step-indexed (T steps, exactly one cube per step). Tiles: the set of distinct A/W/C tiles touched by any cube. `uses(tau)` = set of cubes whose a/w/c tile equals tau.

### 4.1 Variables
- `x[c,t] in {0,1}` -- cube c runs at step t.
- `res[tau,t] in {0,1}` -- tile tau is resident (occupies space) during step t. Convention res[tau,-1] = 0.
- Derived (reified), the C/P/S/R skeleton:
  - `load[tau,t] = res[tau,t] AND NOT res[tau,t-1]` (Creation if first-touch / zero-init C; Retrieval otherwise).
  - `evict[tau,t] = res[tau,t-1] AND NOT res[tau,t]` (Spilling for partial C; free otherwise).
  - Preservation (`res[tau,t] AND res[tau,t-1]`) is implicit and free.

### 4.2 Assignment (permutation)
- `sum_t x[c,t] = 1` for all c; `sum_c x[c,t] = 1` for all t.

### 4.3 Co-residency (lower bound on residency)
- For each cube c and each of its three tiles tau: `res[tau,t] >= x[c,t]`. A cube's A/W/C must be resident at its step.

### 4.4 Demand-driven load (CRITICAL -- closes the vacuous-stall exploit)
- For every tile tau and every step t: `load[tau,t] <= sum_{c in uses(tau)} x[c,t]`.
- Rationale (reviewer-reproduced bug): without this, `res[tau,t]` is free, so the solver makes every tile resident at the fill step t=0 (stall-exempt, 4.6), placing all loads in step 0 and making the stall=0 constraint vacuous -- CP declares schedules feasible that `apply_cube` rejects (verified counterexample: M=32,K=64,N=64,wbits=[[2,4]],act_bits=8 at loose capacity cap >= 2x single-cube working set and finite bandwidth e.g. dram_bw=24 -> CP "feasible" but the reconstructed schedule has eval_sched steady_stall > 0 i.e. infeasible; with 4.4 added, CP-SAT returns INFEASIBLE matching eval_sched, and CP == oracle bit-exact where feasible). Since exactly one cube runs per step, the RHS is 1 iff the step-t cube uses tau, so a tile may only become newly resident at a step whose cube uses it -- exactly `apply_cube`'s demand-driven load (it only loads cube_i's own tiles). This does NOT restrict voluntary/non-minimal eviction (4.7) or later reload (a tile may be re-loaded at any later using step). No "evict only if capacity forces it" constraint may be added -- that would prune optimal schedules (`eviction_choices` yields the FULL subset move set, including voluntary; `eval_sched.py:137-160`).

### 4.5 Capacity (M1 semantics, no prefetch-peak charge)
- `sum_tau res[tau,t] * size_S(tau) <= cap_S` for all t, where `size_S`/`cap_S` are the integer-scaled sizes/capacity (4.8).

### 4.6 stall = 0 (hard constraint)
For i >= 1 only (i = 0 is the fill cube, exempt -- matches `apply_cube`'s `last_compute < 0` branch, `eval_sched.py:125-127`; note i = 1 is NOT exempt, its hide budget is `compute[0]`):
- Semantics: `(read_i + spill_i) / eff_bw <= cube_compute(cube_{i-1})`, i.e. step i's transfer hides under step i-1's compute. Equivalent to `apply_cube`'s `max(0, transfer_time - last_compute) == 0`.
- `traffic_bits[i]` here is the SAME read+spill quantity as the objective (4.7) -- it INCLUDES the C spill term, not reads only.
- `compute[i-1] = sum_c x[c,i-1] * cube_compute(c)` (linear; reconstructs `state.last_compute`).
- Encoded exactly via integer scaling (4.8): multiply both sides by the global scale so `traffic_bits[i] * num(eff_bw_inv_scaled) <= sum_c x[c,i-1] * coeff_c` becomes an all-integer linear inequality. NO float `round()` -- use exact rationals (4.8), or the stall feasible region diverges from `apply_cube`'s exact float comparison (reviewer-flagged: a blanket "S=1" truncation breaks G2/5.3 when eff_bw or wbits is non-integral).

### 4.7 C accumulation counter (reservoir) and traffic assembly
- `cnt_before[C,t] = sum_{c in (mt,*,nt)} sum_{u<t} x[c,u]` (linear integer expression; domain hinted 0..KT for solver efficiency).
- C charging conditionals (reified booleans channeled to `cnt_before`):
  - C-reload term `rl[C,t]` active iff `load[C,t] AND (cnt_before[C,t] >= 1)`.
  - C-spill term `sp[C,t]` active iff `evict[C,t] AND (cnt_before[C,t] >= 1) AND (cnt_before[C,t] <= KT-1)`.
  - (`load` and `evict` are mutually exclusive for one tile in one step, so a C is never both reloaded and spilled at the same step -- no double count.)
- `traffic_bits[i] = sum_{tau in A,W} size_S(tau)*load[tau,i] + sum_C size_S(C)*rl[C,i] + sum_C size_S(C)*sp[C,i]`.

### 4.8 Exact-rational integer scaling (global)
CP-SAT requires integer coefficients. Define ONE positive integer scale `S` = LCM of the denominators (lowest terms, via `fractions.Fraction`) of: all tile sizes (only W can be fractional), `cap_bits`, and the 4.6 stall factors (`freq_ratio`, `dram_bw`, `cycles_per_bit`). Apply S to ALL sizes, capacity, and stall coefficients (argmin-invariant since S > 0). For every committed cross-validation/measure_gap config (`wbits in {2,4,8}`, integral `dram_bw`, `freq_ratio = cycles_per_bit = 1`) S = 1; the LCM path is exercised by the fractional-wbits and fractional-dram_bw unit tests (5.5). At model-build time assert every coefficient fits int64 with margin (max coefficient < 2^62) -- the stall RHS carries `S*dram_bw*...` (~1e14 at dram_bw=1e12, well within range, but assert to be safe).

### 4.9 Symmetry breaking -- OFF by default, oracle-gated (do NOT risk exactness)
Default: NO symmetry-breaking constraint (rely on CP-SAT's internal symmetry handling). Distinct cubes here are rarely truly interchangeable: two cubes are cost-equivalent only with IDENTICAL tile identities (same A, same W, same C) AND identical accumulation role -- equal tile *sizes* is NOT sufficient, and cubes sharing a C-accumulation group `(mt,*,nt)` are order-sensitive via `cnt_before` (4.7) and must never be lex-ordered. Any optional symmetry-breaking flag must be gated behind a passing `cpsat == oracle` test on a mixed-precision capacity-pressure family and is never enabled on the proven-optimal path unless that gate is green. This protects the "no silent approximation" invariant (a too-aggressive symmetry cut would report a non-optimum as proven-optimal).

### 4.10 Objective
`minimize sum_i traffic_bits[i]` (integer, in scaled bits). Energy is derived post-solve (5.2), not minimized as a float.

## 5. Result contract and validation

### 5.1 Result dict (must match `astar.optimize_exact` keys EXACTLY, `astar.py:127-137`)
Keys: `energy, order, evictions, feasible, proven_optimal, lower_bound, gap, nodes_expanded, source, min_steady_stall, reason`.
- `nodes_expanded` is EXACTLY this key, typed `int` (populate from `int(solver.NumBranches())`). Never substitute `status`/`branches`; `measure_gap.py:68` hard-formats `res['nodes_expanded']:6d`. A CP-SAT status string, if wanted, goes in an ADDITIONAL key no consumer reads.
- `source` = "cpsat".
- Per-path field values (mirror astar, including its `gap=0.0` on the infeasible branch, `astar.py:112-113`):

| path | energy | order/evictions | feasible | proven_optimal | lower_bound | gap | reason | min_steady_stall |
|---|---|---|---|---|---|---|---|---|
| OPTIMAL | optimum | solution | True | True | energy | 0.0 | None | None |
| FEASIBLE+timeout | incumbent | incumbent | True | False | bound (5.2) | (energy-lb)/lb | None | None |
| INFEASIBLE | inf | None | False | False | inf | 0.0 | diagnostic (5.4) | value or None |

### 5.2 Energy and lower_bound derivation (float, from integer bits)
- `total_bits` = sum of charged sizes from the solution's load/spill events in RAW (un-scaled) bits.
- `energy = float(total_bits) * coef` -- computed ONCE from the bit total, matching `eval_sched.py:223`'s `(read_total+spill_total)*coef` form (so 5.3 is bit-for-bit, not float-summation-order-dependent).
- `lower_bound` (energy units, to match astar) = `(solver.BestObjectiveBound() / S) * coef` (un-scale the scaled-bit bound to bits, then * coef). NOT `/coef`.

### 5.3 Reconstruct -> eval_sched bit-exact (master check)
From any CP-SAT solution reconstruct (order, evictions) and run `eval_sched.eval_sched(w, hw, order, evictions)`. The canonical equality is on INTEGER traffic bits: `total_bits == round(eval.dram_read_bits + eval.dram_spill_bits)` EXACTLY when wbits is integral (and `abs(...) < 1e-6` for the fractional-wbits case). Then `energy` matches by construction. Any mismatch is a model bug. Also assert `eval.feasible` agrees with CP-SAT feasibility (this is what would have caught the 4.4 vacuous-stall bug).

### 5.4 Determinism and feasibility diagnostics
`num_search_workers=1` + fixed `random_seed`; build the model in `sorted(...)` cube/tile order; determinism is guaranteed only IN-PROCESS for a fixed OR-Tools version (do NOT claim cross-version determinism). For proven-optimal cases solve to OPTIMAL (no time limit) so the returned order is canonical; if ties are possible add an explicit lexicographic tie-break on the order. INFEASIBLE -> re-solve a relaxed model that drops ONLY the 4.6 stall constraint (keep 4.4 demand-load -- it is residency correctness, not stall): if relaxed is feasible, report stall0-infeasible with the M1-style `min_steady_stall` diagnostic (`warmstart.min_structural_steady_stall`); else capacity-infeasible. Never return a bare inf (CLI exit 1).

### 5.5 Required tests (`test_cpsat_sched.py`, skipped if ortools absent) + `selftest()`
- selftest G1: all-resident floor = first-touch A/W bits * coef, proven at root.
- selftest G2: `cpsat == oracle` under mixed-precision capacity pressure.
- selftest G3: in-process determinism (same order on repeated solve).
- selftest G4: reconstruct->eval_sched parity (5.3).
- Sweep: `cpsat == oracle == astar` (energy AND proven_optimal) over small T (<=6), mixed/uniform precision x capacity multipliers; equality asserted on integer bit totals (or `<1e-6`), never bare float `==`.
- Honest-gap path: a known-hard instance with a tiny `max_time_in_seconds` -> assert `proven_optimal == False`, `energy >= lower_bound`, `gap == (energy-lower_bound)/lower_bound`.
- Infeasible paths: one capacity-infeasible and one stall0-infeasible instance -> assert `feasible == False`, `energy == inf`, and the stall0 case yields a non-None `min_steady_stall`.
- Exact-rational scaling: one fractional-wbits (e.g. 2.5) and one fractional-`dram_bw` config -> S != 1 path, reconstruct->eval_sched parity holds (`<1e-6`).

### 5.6 measure_gap.py backend edit plan
- Add `import cpsat_sched`; add a `--backend {astar,cpsat}` selector (introduce a small `main()`/argparse; current file is a flat script).
- Extend `SHAPES` for the CP-SAT backend to reach the OOM regime: T=32 = `(128,128,64)` (MT=KT=4, NT=2); T=64 = `(128,128,128)` (MT=KT=NT=4). Keep the existing T<=12 shapes for the astar cross-check.
- At T >= 32 the reference column is **CP-SAT** (astar OOMs); `warmstart.structural_incumbent` stays as the cheap structural baseline (no search). Do not call `astar.optimize_exact` at T >= 32.
- Columns: carry CP-SAT `proven_optimal` (False on timeout) and print the HONEST gap even when not proven (mark proven vs honest, e.g. a trailing `*` on honest-gap rows), so the table answers open question (A)/(B). `nodes_expanded` prints CP-SAT branches.

## 6. Deliberate modeling decisions (with rationale)

- **Capacity = M1 (post-load resident <= cap; no prefetch-peak 2x).** Keeps the CP-SAT objective bit-exact with `apply_cube` so `CP-SAT == oracle == A*` holds. Physically valid if the HW has Buffets-style decoupled fill/read (recovers the 2x). Charging the double-buffer peak is a separate future decision that would also change eval_sched/oracle/astar/warmstart and force re-cross-validation.
- **First-touch A/W floor INCLUDED in the objective** (opposite of COSMA, per M1 energy semantics).
- **C/P/S/R residency-action skeleton ADOPTED** (COSMA-verified template) over a bare res+derived encoding, for a cleaner state machine and easier oracle cross-checks. Demand-driven load (4.4) is the added constraint that makes the skeleton match `apply_cube`.
- **Step-indexed boolean** (not interval/AddCumulative-AddReservoir) because C/P/S/R is inherently per-step and the per-step capacity sum + assignment-determined stall=0 are linear; this is the most faithful reproduction of `apply_cube`.
- **Exact-rational scaling, symmetry-breaking off by default**: both protect exactness/the no-silent-approximation invariant over micro-optimizations.

## 7. Open questions this build answers

- (A) Does CP-SAT actually CLOSE the T=32/64 pressure cases A* OOMs on (proven_optimal), i.e., a real upgrade over A*?
- (B) Does the gap-(1) opportunity (structural baseline vs optimal), which `measure_gap` saw as 15-44% only at cap <~ 1.5x single-cube working set, persist at larger T -- or does the structural baseline stay optimal (gap 0) so the scheduler is unnecessary at the default loose HW?

## 8. Risks

- CP-SAT may not close T=64 within budget -> acceptable: report honest gap (graceful, not OOM). That IS a finding for (A).
- Reified C-counter conditionals + per-step residency enlarge the model -> verify model size/solve on T=32 before T=64; domain-hint `cnt_before` 0..KT.
- Integer scaling: float `round()` would silently diverge the stall feasible region -> use exact `Fraction` LCM (4.8); unit-test fractional wbits and dram_bw.
- int64 coefficient domain: stall RHS carries `S*dram_bw*...` -> assert max coefficient < 2^62 at build (4.8).
- Symmetry-breaking soundness: a too-aggressive cut would report a non-optimum as proven-optimal -> OFF by default, oracle-gated (4.9).
- Demand-load omission (the reproduced bug) -> 4.4 is mandatory and 5.3 asserts feasibility-agreement to catch any regression.
