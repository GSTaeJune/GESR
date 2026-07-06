# timeloop-sweep — Timeloop-scheduled SRAM partition sweep (design)

Date: 2026-07-06. Status: direction approved by user (this session); details in flux
until the smoke test lands.

## 0. Why this exists (correction of buffer_sweep v1)

buffer_sweep v1 FIXED the inter-tile schedule to O-stationary. User clarification:
"dataflow 확정" meant the INTRA-tile dataflow only (the SA's bit-serial row=K /
col=N / cycle=M — fixed by RTL). The inter-tile schedule (loop order + tiling) is
a SEARCH SPACE, exactly what mappers like Timeloop explore. Decision: do not
hand-roll that search — use Timeloop.

## 1. Division of labor

```
driver (Windows Python)                 Timeloop (WSL, conda env `timeloop`)
  sweep: cap -> (W,A,O) partition  -->  arch.yaml (32x32 spatial mesh, 3 dedicated
  (ping-pong = capacity/2)              buffers w/ dataspace bypass, DRAM)
  workload: model GEMM list       -->  problem.yaml per GEMM
                                        timeloop-mapper: searches (m,k,n) tiling
                                        + loop permutation  <- the search space
  parse mapping (tiles + perm)   <---  map.txt / stats
  score with OUR evaluator       ----> final cycles & energy (bit-serial x8,
  (buffer_sweep cost model,             ping-pong prefetch/stall, RMW, LPDDR5
   extended to arbitrary perms)         coefficients)
```

- Timeloop's internal cost model is used ONLY to rank mappings during search.
  Final numbers come from our evaluator (it knows bit-serial timing, the ping-pong
  drain semantics, and the project's energy coefficients).
- The 32-multiple tile granularity is NOT a sweep grid: it falls out of the arch
  (32x32 spatial mesh) — hardware condition, as the user noted.
- Sweep axis = partition of the cap into (W_cap, A_cap, O_cap) (the only remaining
  arch DOF). Best (m,k,n) and loop order are Timeloop OUTPUTS per partition.

## 2. Why Timeloop (and not CoSA / newer mappers) — user asked, resolved

Our mapspace is small: one on-chip level (3 dedicated buffers) + DRAM, 3 loops
(M,K,N), tiles quantized by the 32x32 spatial level. Timeloop-mapper covers it
exhaustively (or linear/random-pruned) in seconds-to-minutes = optimal under its
model. CoSA / GAMMA / DOSA / Mind-Mappings etc. exist to AVOID exhaustive search
in huge multi-level mapspaces; they return approximate solutions (proxy
objectives), CoSA additionally needs Gurobi, and they still evaluate on Timeloop
infrastructure. Re-visit trigger: multi-level SRAM hierarchy or fusion scheduling
(mapspace explosion) -> then CoSA/DOSA on top of the same Timeloop base.

## 3. Components

1. `buffer_sweep/timeloop/` — arch/problem/mapper YAML templates + `run_smoke.sh`
   + NOTES.md (smoke test deliverable; exact dialect learned from the working
   ~/tl-verify-v4 run in WSL).
2. Driver `buffer_sweep/timeloop_sweep.py` (new, standalone like v1):
   - partition enumeration per cap (compositions of usable bits across W/A/O,
     step = cap/16 or similar; ping-pong halves the usable capacity)
   - YAML generation per (partition, GEMM), invocation via
     `wsl bash -lc 'source ~/miniconda3/bin/activate timeloop && timeloop-mapper ...'`
   - mapping parser: per-level tile sizes -> (m,k,n); temporal permutation
   - result cache keyed by (partition, GEMM) so re-runs are incremental
3. Evaluator extension in `buffer_sweep/buffer_sweep.py`: `walk_gemm` generalized
   to an arbitrary tile-loop permutation, with partial-psum spill/reload charging
   (32b write + read per eviction of an incomplete O tile; zero-init free,
   final write once) and the O-drain window generalized to "until the shadow O
   is next needed". The grouped analytic stays O-stationary-only (v1); Timeloop
   mappings arrive one per sweep point, so the slow reference walk is fast enough
   to be the scorer. Selftest: walk-vs-grouped equality retained for the
   O-stationary perm; spill accounting hand-checked cases for one non-O-stationary
   perm.
4. Outputs (same style as v1, `buffer_sweep/results/`): CSV per model with
   partition, Timeloop-chosen (m,k,n, perm), our energy/cycles, Timeloop's own
   energy/cycles (sanity columns); plots: best energy/cycles vs partition per cap;
   cross-check column vs v1's O-stationary best (v1 stays as the baseline).

## 4. Open items (resolve during implementation)

- Exact YAML dialect / constraint syntax: from the smoke test NOTES.md.
- Bit-serial timing inside Timeloop's search ranking: acceptable to leave
  unscaled (uniform compute scaling rarely flips traffic-dominated rankings);
  our evaluator applies the real x WBITS timing. If rankings look wrong at the
  cross-check, derate the arch DRAM bandwidth by 1/WBITS instead.
- Partition step granularity (runtime vs resolution) — start coarse (16 steps),
  refine near the winner if needed.
- BMM shapes (S x d x S per head): mapper cost amortizes over count x layers.

## 5. Non-goals

Replacing v1 (it stays as the fast baseline + cross-check), CoSA integration,
multi-level hierarchies, fusion, sparsity, mixed precision.
