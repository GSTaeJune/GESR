# buffer_sweep — SRAM (m,k,n) buffer-partition sweep (design)

Date: 2026-07-03. Status: approved by user (this session).

## 0. Context / direction change

The joint order+eviction scheduler line (M1 astar/oracle, cpsat, band) is being
retired. New approach: **fix the SRAM organization and dataflow up front**, then map
GEMMs onto it. This program finds the best buffer partition for that fixed dataflow.

## 1. What it is

A **single standalone Python file** `buffer_sweep/buffer_sweep.py` (new top-level
directory, deliberately separate from `MXP_scheduler/`; imports nothing from it —
the user will read and hand-edit this file). stdlib + matplotlib (plots are a
required output, not optional).

## 2. SRAM organization (fixed)

Three dedicated spaces, each **ping-pong duplicated (x2)**:

| space | holds        | size (bits)      |
|-------|--------------|------------------|
| W     | m x k weights| m * k * 8        |
| A     | k x n acts   | k * n * 8        |
| O     | m x n psums  | m * n * 32       |

Budget: `2 * (8*m*k + 8*k*n + 32*m*n) <= cap_bits`.
Caps swept: **64, 128, 256 KB** (physical total incl. ping-pong). m, k, n are all
multiples of 32 (SA is 32x32: row=K, col=N, cycle=M). Precision fixed 8/8/32.

## 3. Dataflow (fixed): O-stationary

For each output tile (i,j) (i over M, j over N), sweep kk innermost, accumulating
into the O buffer; write the finished FP32 tile to DRAM **once** (no psum
spill/reload). Loop order: `for i: for j: for kk`.

Ping-pong semantics: while computing step t, prefetch step t+1's W/A tiles into the
shadow copies; a finished O tile drains from the shadow O while the next tile
accumulates. A tile is fetched **iff its tile id differs from the one currently
held** (so e.g. Kt==1 keeps W(i) resident across the whole j sweep — the traffic
and stall models must both honor this reuse rule).

## 4. Cost model

Dims padded up to multiples of 32 per GEMM (only S=197 -> 224 is actually ragged);
traffic/compute use padded dims (models HW zero-padding; slightly pessimistic).

- **DRAM traffic**: derived from the same tile-id-change walk as the stall model
  (single source of truth). For the generic case (Kt>1) it equals the closed form
  `W = ceil(N/n)*M*K*8`, `A = ceil(M/m)*K*N*8`, `O_write = M*N*32`, but reuse
  corner cases (Kt==1, Nt==1) are handled by the walk, not by special-cased formulas.
- **Compute cycles** (order-invariant): per 32^3 cube = `cycles_per_bit * 32 * 8`
  (bit-serial, 8b weights); total = #cubes * that.
- **Stall** (ping-pong prefetch): per step transition,
  `stall = max(0, fetch_bits/eff_bw - compute_cycles(current step))`.
  fill = step 0's W+A fetch; each output tile's O write is charged on its final kk
  step (hidden by the next tile's compute via O ping-pong); the last O tile is drain.
  `total_cycles = compute + fill + steady_stall + drain`.
- **Grouped evaluation**: steps are NOT enumerated one by one (too slow in Python).
  Tile sizes per axis take at most 2 values (full, residual), so transitions are
  grouped into O(1) classes with multiplicities and summed. A `--selftest` compares
  the grouped model against a naive step-by-step walk on small shapes (exact match
  required), plus hand-checked traffic asserts.
- **Energy** (M0-style structure): `dram_bits*c_dram + onchip_bits*c_onchip +
  mac_ops*c_mac + rmw_ops*c_rmw`. onchip = DRAM refill writes + SA-facing A/W reads
  per cube. mac/rmw are config-invariant but included for absolute totals. EDP
  reported too.
- **Coeffs** live in an editable CONFIG block at the top of the file: DRAM = 9.0
  pJ/b (LPDDR5 full-system, docs/dram-energy/), onchip default 6.0 pJ/b with a
  per-cap comment (refresh from CACTI later), DRAM BW/freqs -> eff_bw, cycles_per_bit.

## 5. Workloads (embedded table)

Per-layer GEMM list x L layers, summed per model. Attention BMMs included
(count = #Q heads). Q/K/V are separate GEMMs (matches Qwen GQA).

| model | dims | per-layer GEMMs (M x K x N, count) |
|---|---|---|
| deit_tiny  | D=192, H=3, d=64, MLP=768, L=12, S=197->224 | Q/K/V 224x192x192 x3; out 224x192x192; QK^T 224x64x224 x3; attnV 224x224x64 x3; fc1 224x192x768; fc2 224x768x192 |
| deit_small | D=384, H=6, d=64, MLP=1536, L=12, S=197->224 | same structure with D=384 |
| qwen2.5-0.5b | D=896, Qh=14, KVh=2, d=64, I=4864, L=24, S=128 | Q 128x896x896; K/V 128x896x128 x2; out 128x896x896; QK^T 128x64x128 x14; attnV 128x128x64 x14; gate/up 128x896x4864 x2; down 128x4864x896 |
| qwen2.5-1.5b | D=1536, Qh=12, KVh=2, d=128, I=8960, L=28, S=128 | same structure with 1.5b dims |

Seq: DeiT 197 (fixed), Qwen prefill 128 (seq sweep is future work).

## 6. Sweep driver + outputs

Loop: model x cap x k in K_LIST (default powers of two 32..1024) x all feasible
(m,n) 32-multiples. **One (m,k,n) per model** (buffer partition is a hardware
decision): cost = sum over all its GEMMs.

Outputs to `work/buffer_sweep/`:
- `<model>.csv`: cap_kb, k, m, n, dram_bits, energy_pj, cycles, steady_stall, edp
- plots (matplotlib, per model x cap): `<model>_cap<KB>_energy.png` and
  `<model>_cap<KB>_cycles.png` — (m,n) heatmaps, one subplot per k, best cell marked
- summary plot per metric: best energy / best cycles vs cap, per model
- console: per model x cap, top-10 by energy (cycles column alongside) + best-by-cycles
  row. ASCII-only output.

CLI: `python buffer_sweep.py [--model NAME|all] [--caps 64,128,256]
[--klist 32,64,...] [--out DIR] [--selftest]`. No args = all models, all caps.

## 7. Non-goals (v1)

Mixed precision (8/8/32 fixed), seq-length sweep, psum-spill dataflows, per-GEMM
buffer reconfiguration, DRAM latency modeling beyond bandwidth, MXP_scheduler
integration.
