# MXP_scheduler band-serpentine (B,d)-per-region scheduler -- design

Date: 2026-06-26
Status: DRAFT rev2 (post 2-reviewer round; CRITICAL/IMPORTANT findings folded in)
Builds on: the cube-stream cost model in `eval_sched.py` (single source of truth, unchanged) and the exact engines `oracle.py`/`astar.py`/`cpsat_sched.py` (kept as small-T cross-checks). Problem statement: `docs/cpsat-scheduler-problem-statement.md`.

## 1. Motivation

The exact engines do not scale past T ~ 12-32; real workloads are T = 10^3-10^6. We need a **tractable heuristic that scales to any T, plus an honest lower bound** to certify its gap. This spec collapses the schedule's degrees of freedom to **two per-region knobs** and produces a feasible schedule by construction, scored by the unchanged cost model.

Design intent (user): partition on-chip SRAM into separate A / W / C regions sized by two knobs chosen **per region**:
- **B** = band width = number of C (output accumulator) tiles kept open simultaneously.
- **d** = K-depth = number of k-steps of W/A pulled in / kept resident at once.

Order is a **band-serpentine** sweep of the (m,n) output plane with k swept inside each band. Eviction is **implied by (B,d) and band/chunk advance**, not a free search.

## 2. W tile sizing (correctness item) -- RESOLVED, no cost-model change

Confirmed semantics (user, 2026-06-26):
- A W tile = `32 blocks x 32 elements = 1024 elements`; each block uniform 2/4/8 bits; different blocks may differ, so per-tile average bits/element lies on a fine lattice in [2,8].
- The cost model needs **only each tile's total resident bits**, not the 32 per-block values.
- W tile size = `avg_bits[m][k] * 1024` (= `32 * sum(block_bits)`). **Any block/word padding is pre-applied by the caller**: `Work.wbits[m][k]` already encodes the true (padded) per-tile total / 1024.
- Activation A stays **layer-uniform** `act_bits in {2,4,8}`.

Consequence: `eval_sched.tile_size("W") = wbits[mt][kt]*TILE*TILE` is **already correct** (per-tile, fractional-capable; `mxp_scheduler.py:80,94-96`, `eval_sched.py:44`). No "3-value / uniform-W" assumption exists to fix (the only uniform/3-value assumption is on A=`act_bits`, kept). `cpsat_sched._global_scale` already handles fractional `wbits`. **No change to the cost model's W sizing.** The band module sources every tile size from `eval_sched.tile_size` (never re-derives).
NOTE: `Work.__post_init__` rejects any `wbits` entry outside [2,8] (`mxp_scheduler.py:95`). The padded per-tile average the caller supplies MUST still land in [2,8]; if real padding could push it above 8, the caller normalizes or we document the cap. (No A or C change.)

## 3. Scope

In scope:
- New offline module `MXP_scheduler/band_sched.py` (stdlib only; no ortools): the (B,d)-per-region band-serpentine constructor + per-region search + honest lower bound.
- New `MXP_scheduler/demo_qwen.py` (or a `band_sched` CLI subcommand): take a real model's Q/K/V projection GEMM dimensions (Qwen2.5 presets), assign an arbitrary per-tile avg-bit map, run `optimize_band`, and emit the schedule (order summary + per-region (B,d) + energy + honest gap), writing the full cube order to a file. **This is the end-to-end deliverable.**
- A `measure_gap.py` backend `--backend band` (scales to large T; no exact reference -> reports the honest first-touch-floor gap).
- `test_band_sched.py`: feasibility-by-construction, reconstruct->`eval_sched` parity, restriction-gap vs exact on small T, C-spill-fallback and single-cube-infeasible paths, uniform/Belady special case, and a Qwen-shape smoke.

Out of scope: changing the cost model or stall=0 semantics; per-block (sub-tile) load/evict; A mixed precision; cross-region (cross-m-row) reuse (v2); tightening the CP-SAT bound (separate task).

Keep unchanged: RTL; `eval_sched.py`; the exact engines `oracle.py`/`astar.py`/`cpsat_sched.py`; `mxp_scheduler.py`/`hwconfig.py`. The old uniform-size / Belady path (`warmstart.structural_incumbent`) is **kept as a verification baseline** (Belady-optimal on the uniform-size special case; a reference the band schedule is compared to but is NOT required to beat -- see Section 10).

## 4. Region, band, and order geometry

- **Region = one m-row**: region `m` covers all `(m,n)`, `n in [0,NT)`, all `k`. There are `MT` regions, processed `m = 0..MT-1`. Regions are **independent** in v1 (no cross-row reuse modeled), which is what makes per-region (B,d) selection valid (Section 8). Honest limitation: A(k,n) is reused over m in principle, but v1 reloads A per region; the restriction-gap test (Section 10) quantifies the cost. (Cross-row A reuse = v2.)
- Within region `m` with `(B,d)`:
  - `n` swept in **groups of B**: group `g` covers `n in [g*B, min((g+1)*B, NT))` (ragged tail allowed; B need not divide NT). `ceil(NT/B)` groups. Each n-group is a **band**.
  - `k` swept in **chunks of d**: chunk `c` covers `k in [c*d, min((c+1)*d, KT))` (ragged tail allowed). `ceil(KT/d)` chunks.
  - **Cube emission order within a band** (k-outer, n-inner -> W amortized across the band's B columns):
    ```
    for g in n-groups (serpentine: reverse the group order on odd m-rows):   # band
      for c in k-chunks:
        for k in chunk c:
          for n in group g:
            emit cube (m, k, n)
    ```
    This makes W(m,k) loaded **once per (band, k)** and reused by all B cubes of that k in the band -> larger B amortizes W (fewer bands = fewer W reloads). C(m,n) for n in the band stays resident across the whole k-sweep -> completes with one final write, **no C spill** (nominal case).
  - **Serpentine** = reverse the n-group order on odd m-rows. v1 does NOT model cross-row reuse, so serpentine does not change cost here; it is kept as the order label (matches design intent) and as a v2 hook. (Within a region, n-group order also does not change cost: each band is independent.)
- The flattened cube sequence is the `order` that `eval_sched` consumes; the per-step `evictions` come from the resident-set state machine (Section 5.1).

## 5. Footprint and feasibility (per region, true variable W sizes)

For region `m` with `(B,d)`, the **forced working set** -- the minimum that must be co-resident to run the band, i.e. the selection-filter lower bound (NOT the resident peak; see 5.1) -- is:
- **C region** = `B * (TILE*TILE*FP32_BITS)` (B open accumulators, fixed FP32).
- **W region** = `max over k-chunks c of  sum_{k in c} tile_size(("W",m,k))` (the d-deep W window; max over chunks because chunks differ in bits -- 8-bit-heavy rows cost more). Matches `mxp_scheduler.footprint_bits`'s `max(w_blk)` pattern.
- **A region** = `d * B * (TILE*TILE*act_bits)` (d k's x B n's of activation).
- `footprint(m,B,d) = C_region + W_region + A_region`. Tile sizes from `eval_sched.tile_size`.

Feasibility filter for `(B,d)`: `footprint(m,B,d) <= cap` (sufficient -- the lazy eviction in 5.1 can always reduce resident to this forced working set) AND the constructed schedule is **stall=0-feasible** (decided by `eval_sched`, Section 6).

### 5.1 Eviction policy: LAZY capacity-aware (implemented; reviewed SOUND)

The order (band-serpentine, Section 4) is `(B,d)`-determined; eviction is **lazy and capacity-aware** (an improvement over a rigid deterministic-partition eviction, which strands shared A tiles at boundaries and never reaches the all-resident floor even with roomy cap). Per cube `(m,k,n)`:

```
need     = { A(k,n), W(m,k), C(m,n) }
protect  = { C(m,n') : n' in band g } UNION need      # never evict the band's C or this cube's tiles
to_load  = need - resident
# evict largest tiles OUTSIDE protect, only while the next load would exceed cap:
ev = {}; cand = sorted(resident - protect, by size desc)
while size((resident - ev) UNION to_load) > cap and cand left: ev += next(cand)
resident = (resident - ev) UNION to_load            # eval_sched applies ev before loading
```
- Tiles **stay resident across chunk/band/region boundaries** (one global `resident` carried across regions), so reused A/W (A is shared over m; W over n within a row) is not gratuitously reloaded -> a roomy-cap schedule reaches the **first-touch floor** (gap 0). This captures the cross-region A reuse the v1 spec had deferred; it is strictly better and remains capacity-safe.
- **Capacity safety (proven + 159 builds, 0 violations)**: when the while-loop exhausts candidates, resident reduces to `protect UNION to_load` = `B C-tiles + {A(k,n),W(m,k)}`, whose size `= B*C + W(m,k) + A(k,n) <= B*C + max-chunk-W + d*B*A = footprint(m,B,d) <= cap`. So `footprint(B,d) <= cap` is a **sufficient** feasibility filter (Section 5). The actual per-step peak may be LARGER than `footprint` (lazy retention fills spare capacity up to `cap`) -- `footprint` is the forced-working-set lower bound, not the peak.
- Pricing is entirely `eval_sched`'s: evicting a complete/zero C is free, a partial C spills, A/W is free; loads charge per `apply_cube`. The flattened `(order, evictions)` is run once through `eval_sched.eval_sched` for the authoritative energy/feasibility.
- Eviction victim choice is largest-first (not Belady/furthest-future). Because each band completes its full k-sweep before advancing, prior-chunk A/W dropped under pressure are not reused within the band, so largest-first showed no measurable extra reloads vs Belady on this access pattern (reviewer note); a Belady victim rule is a possible v2 refinement.
- **Reviewed (2026-06-26): deviation from the rigid-partition draft ruled SOUND** -- capacity-safe (proof above), `band_energy >= oracle.dp_optimal` in 378 cross-checks (never beats the free optimum), stall=0 enforced. Tradeoff: `footprint_bits` is a selection bound, not the resident peak (test `test_footprint_equals_peak_when_no_eviction` guards only the no-eviction case).

## 6. stall = 0 (unchanged semantics)

Deferred to `eval_sched` via demand-driven, cube-by-cube loads. Each cube's small load must hide under the previous cube's compute -- `eval_sched`'s per-cube stall=0 check (`eval_sched.py:124-128,55`). We do NOT re-implement stall: feasibility = `eval_sched(...).stall0_feasible`. Stall-risk points are the first cube of each **chunk** AND each **band** (their loads hide under the prior cube's compute, which at a band/chunk boundary is the prior band/chunk's last cube, NOT the fill exemption -- only the very first cube of the whole schedule is exempt, `eval_sched.py:125`).

## 7. Region with no feasible (B,d) (user-confirmed policy)

If no `(B,d)` with `B>=1, d>=1` gives `footprint <= cap` AND stall=0 for a region:
1. **Report honestly** (never silent): region index, smallest infeasible footprint, binding reason (capacity vs stall).
2. **C-spill fallback** (well-defined pattern): pick the largest feasible `B' < B_min` for the C+W+A budget by **splitting the band's k-sweep**: process the band's C tiles over k in two (or more) passes, evicting each C tile **while partial** (`0 < counter < KT`) at the end of a pass and reloading it for the next pass. This produces exactly the `0 < counter < KT` eviction + `counter > 0` reload pattern that `eval_sched` charges as spill/reload (`eval_sched.py:101-105,111-113`). Emit that order+evictions; `eval_sched` prices the spill. If the resulting schedule then fails stall=0 (the reload cannot hide), it is **stall-infeasible** -> report (do not silently keep it).
3. If even single-cube residency `(B=1,d=1)` exceeds `cap` (one cube's A+W+C > cap), it is genuinely capacity-infeasible -> report and stop for that region (no schedule), mirroring the exact engines' diagnostic. The whole-schedule result is then `feasible=False`, `energy=inf`, `gap` reported as `inf` / N/A (not a number).

## 8. Per-region search and honest lower bound

- **Per-region scorer**: a region's `(B,d)` candidates are scored by a **partial `apply_cube` fold** over that region's cube sub-sequence starting from an empty resident set (regions independent, so no carry assumed for selection). This reuses `eval_sched.apply_cube` (the single source of truth) -- the module computes NO cost itself. Returns the region's (read+spill) bits.
- **Enumeration**: B in `1..NT`, d in `1..KT` (ragged tails allowed, so no divisor requirement). For very large NT/KT, restrict candidates to a logged subset (e.g. `{1,2,4,...,2^k} ∪ {NT}` for B, similarly d) -- this is a heuristic prune, NOT a correctness requirement; **LOG exactly what was skipped** (no silent truncation). Pick the min-traffic feasible `(B,d)` per region.
- **Whole-schedule scoring**: concatenate all regions' (order, evictions) and run `eval_sched.eval_sched` once for the authoritative total energy/feasibility. (Per-region scores are only for choosing (B,d); the single pass captures any incidental cross-region reuse and is the reported number.)
- **Honest lower bound** (certificate): `LB = first-touch floor = (sum over all distinct A tiles of tile_size) + (sum over all distinct W tiles of tile_size)`, times `coef`. Every input is loaded at least once, so this is a valid, unavoidable lower bound on `eval_sched.energy` (which = `(read+spill)*coef` and `read` always includes first-touch A/W; `eval_sched.py:223,172-176`). `gap = (band_energy - LB) / LB >= 0`. (Stronger bounds -- read-only A/W min-cost-flow, Lagrangian -- are future work. Do NOT claim this beats CP-SAT's bound; on capacity-pressured cases it may be loose, reported honestly.)

## 9. Result contract

`optimize_band(w, hw)` returns the `astar.optimize_exact`-shaped dict (exact key set, `astar.py:127-130`):
`{energy, order, evictions, feasible, proven_optimal, lower_bound, gap, nodes_expanded, source, min_steady_stall, reason}` with `source="band"`, `proven_optimal=False` always (heuristic; `gap` is the honest first-touch-floor gap, not a proof), `nodes_expanded` = number of (B,d) candidates evaluated (note: this means a different thing than astar/cpsat search effort -- documented), `energy`/`feasible` from the final `eval_sched` pass, and `reason`/`min_steady_stall` carrying the Section-7 diagnostic. On any reported-infeasible region: `feasible=False, energy=inf, lower_bound=LB, gap=inf, order/evictions=None`.

## 10. Validation (`test_band_sched.py`)

- **Feasibility by construction**: every returned schedule, re-scored by `eval_sched`, has `feasible == True` (capacity + stall0), except reported-infeasible regions.
- **Reconstruct -> eval_sched parity**: the module's reported `energy` equals `eval_sched(order,evictions).energy` exactly (the module never computes energy itself).
- **Restriction gap vs exact (small T <= ~8)**: `band_energy >= oracle.dp_optimal.energy` (the (B,d) restriction cannot beat the free optimum); RECORD `(band_energy - opt)/opt` to quantify the two-knob cost. Compare band_energy vs `warmstart.structural_incumbent` and **report the ratio (informational; do NOT assert band <= warmstart** -- the band forbids M-inner mappings, so it can legitimately lose on A-reuse-dominated shapes; reviewer C2).
- **C-spill fallback path**: a region with cap below the no-spill floor -> assert a feasible spilling schedule is returned and its `dram_spill_bits > 0` (spill actually surfaced, cross-checked via `eval_sched`).
- **Single-cube-infeasible path**: cap below one cube's A+W+C -> assert `feasible=False`, `energy=inf`, honest `reason`, no silent inf.
- **Uniform special case**: all tiles equal size -> band eviction reduces to Belady on the band order; cross-check cost matches `warmstart` Belady on that case (eviction-logic sanity).
- **Large-T honest gap**: `measure_gap.py --backend band` runs at T >= 10^3 and prints the honest first-touch-floor gap.
- **Qwen demo smoke**: `demo_qwen.py` on a small Qwen preset runs end-to-end, returns a feasible schedule with a finite honest gap, writes the order file.

## 11. measure_gap.py wiring (`--backend band`)

Concrete edits (else an implementer must guess):
- `measure_gap.py:55` add `"band"` to `--backend choices`.
- Add `BAND_SHAPES` (large T, beyond the exact range), e.g. `[(256,256,256), (512,512,256), (512,512,512)]` (T = 512, 2048, 4096) plus one Qwen-scale row; `shapes = BAND_SHAPES` when `backend == "band"`.
- `_solve` (`measure_gap.py:41-50`) add a `band` branch: `import band_sched; r = band_sched.optimize_band(w, hw); return (r["energy"] if r["feasible"] else None, r["proven_optimal"], r["nodes_expanded"], r["gap"])`.
- **No exact reference at large T**: for the band backend, do NOT compute a "vs optimum" gap. `warmstart.structural_incumbent` is the only baseline; it is an **O(#mappings x T) fold** (it materializes a length-T order per mapping and folds `eval_sched` over it -- NOT closed-form), fine at BAND_SHAPES (T<=4096) but slow at Qwen scale (T up to ~10^6). Compute it as `warmE` for the structural-vs-band ratio column **only when feasible within a time/size budget**; if too slow on a given shape (e.g. T above a threshold), **skip it and print `warm = -` (LOG the skip)**. The headline is always the **honest first-touch-floor gap** (`r["gap"]`, printed as `lb%.0f`); the proven column is always `False` for band.
- Columns for band rows: `shape | T | prec | capxWS | warmE(e6) | bandE(e6) | proven(False) | nodes(=#(B,d)) | gap%(= struct-rel) lb<honest>`.

## 12. Qwen Q/K/V demo (`demo_qwen.py`) -- the end-to-end deliverable

- **Presets**: a small dict of Qwen2.5 hidden dims, e.g. `{"qwen2.5-0.5b": H=896, "qwen2.5-7b": H=3584, "qwen2.5-72b": H=8192}` (Q/K/V projection GEMM: activation `[S,H] x` weight `[H, H_out]`). For Q projection `H_out = H`; for GQA K/V, `H_out = n_kv_heads*head_dim` (a smaller N) -- expose both via a `--proj {q,k,v}` flag with the kv dim from the preset. Map to `Work`: `M = S` (sequence length, `--seq`), `K = H`, `N = H_out`, all rounded to multiples of 32.
- **Arbitrary per-tile avg bits**: a pluggable assigner producing the `MT x KT` `wbits` map. Provide a couple: `uniform(b)`, `random_2_4_8(seed)` (each tile's avg drawn from a per-block mix, e.g. mean of 32 draws from {2,4,8}), `checker(lo,hi)`. Default = `random_2_4_8` so tiles genuinely mix (lattice sizes). `act_bits` from `--act` (uniform).
- **Run**: build `HW` from a config/flags (cap, banks, dram_bw), call `band_sched.optimize_band(w, hw)`.
- **Output**: print a summary -- shape, T, per-region chosen `(B,d)`, total energy, honest gap, feasible/infeasible; write the **full cube order** (and per-step evictions) to `work/<label>/band_order.txt` (or .json). For very large T, also print head/tail of the order so the result is inspectable without opening the file.
- This is the "pull Qwen dims -> assign arbitrary per-tile avg bits -> extract the order" deliverable.

## 13. Risks

- Two-knob (B,d) restriction may leave a large gap vs the free optimum on some shapes -> measured by the small-T restriction-gap test; deliberate tractability trade, honest gap reports it.
- v1 m-row regions do not reuse A across rows -> over-counts A loads vs the free optimum; noted, restriction-gap test quantifies it, v2 hook via serpentine.
- First-touch-floor LB may be loose on capacity-pressured large-T -> honest gap may be wide; stronger bounds are future work (stated, not hidden).
- (B,d) enumeration over very large NT/KT -> capped candidate set with explicit logging (no silent truncation).
- Padded `wbits` must stay in [2,8] or `Work` rejects it (Section 2 note).
