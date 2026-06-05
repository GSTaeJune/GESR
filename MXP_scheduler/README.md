# MXP_scheduler

GEMM mapping cost-model + optimizer for the MXP bit-serial mixed-precision systolic array.
Given a GEMM shape and HW parameters, it enumerates loop-order × tile-blocking mappings,
costs each one (energy + cycles), and ranks the feasible ones so you can pick a schedule.

Spec: [`../docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md`](../docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md)

## Two files, identical results

| File | Role |
|---|---|
| `mxp_scheduler.py` | Standard (clean) build. stdlib only. This is the one you run. |
| `mxp_scheduler_annotated.py` | Line-by-line 주석 상세본 + a `--explain` step-by-step trace. Same model. |

The two files are kept bit-for-bit equivalent in their numeric output. `mxp_scheduler.py --crosscheck`
imports the annotated twin and asserts that `evaluate()`, `dram_bits`, `energy_breakdown`,
`lpt_headroom`, `pareto_front`, and `tradeoff` all agree across a sweep of cases (every
perm/blocking, spill + no-spill, `freq_ratio != 1`, fractional wbits, non-default coeffs).
If you edit one file's logic, run `--crosscheck` — it fails on any drift.

## CLI

```
usage: mxp_scheduler.py [-h] [--selftest] [--crosscheck] [--M M] [--K K] [--N N]
                        [--bank-size BANK_SIZE] [--banks BANKS] [--dram-bw DRAM_BW]
                        [--freq-ratio FREQ_RATIO] [--act ACT] [--bits-file BITS_FILE]
                        [--max-cycle MAX_CYCLE] [--coeffs COEFFS]
```

| Flag | Meaning | Default |
|---|---|---|
| `--M --K --N` | GEMM dims. Each must be a positive multiple of 32. | (required for a run) |
| `--bank-size` | Words per SRAM bank. | 1024 |
| `--banks` | Number of banks. | 32 |
| `--dram-bw` | DRAM bus throughput, bits per DRAM cycle. | 64 |
| `--freq-ratio` | On-chip cycles per DRAM cycle (`f_chip / f_dram`); 1.0 = same clock. | 1.0 |
| `--act` | Layer-uniform activation precision (2/4/8). | 8 |
| `--bits-file` | JSON MT×KT map of avg weight bits per 32×32 tile. Default: all tiles = `--act`. | none |
| `--max-cycle` | Drop mappings whose `actual_cycle` exceeds this before ranking. | none |
| `--coeffs` | JSON override of the energy coefficients. | DEFAULT_COEFFS |
| `--selftest` | Run embedded golden-value asserts, print `selftest: OK`, exit. | |
| `--crosscheck` | Assert the annotated twin agrees, print `crosscheck: OK`, exit. | |

The annotated file takes the **same flags** except it has `--explain` (and not `--selftest` /
`--crosscheck`). `--explain` prints a step-by-step intermediate-value trace for the optimal
mapping (or the first mapping if none is feasible).

### Examples

```bash
# Project workload: uniform 8-bit, 128^3, current HW
python mxp_scheduler.py --M 128 --K 128 --N 128 --bank-size 1024 --banks 32 --dram-bw 64 --act 8

# Mixed-precision via a per-tile weight-bits map (MT x KT)
printf '[[2,8],[8,2]]' > /tmp/wb.json
python mxp_scheduler.py --M 64 --K 64 --N 64 --dram-bw 32 --bits-file /tmp/wb.json

# Different chip/DRAM clock ratio (chip 2x faster than DRAM) + tighter cycle budget
python mxp_scheduler.py --M 128 --K 128 --N 128 --dram-bw 64 --freq-ratio 2.0 --max-cycle 30000

# Override energy coefficients
printf '{"dram":150,"onchip":8,"mac":1,"rmw":4}' > /tmp/co.json
python mxp_scheduler.py --M 128 --K 128 --N 128 --coeffs /tmp/co.json

# Step-by-step explain trace (annotated twin)
python mxp_scheduler_annotated.py --explain --M 64 --K 64 --N 64 --dram-bw 32 --bits-file /tmp/wb.json

# Verification
python mxp_scheduler.py --selftest
python mxp_scheduler.py --crosscheck
```

## Model

What the cost model actually computes:

**Search space.** For each candidate mapping the optimizer enumerates `(loop-order perm) ×
(tile blocking per dim)`. There are ≤6 perms of `(M, K, N)` (`perm[0]` = outermost), and for
each dim the resident inner-tile count is any divisor of that dim's tile count (`MT=M/32`,
`KT=K/32`, `NT=N/32`). The 32×32 physical systolic array is a **fixed** spatial mapping — it is
not part of the search.

**Energy = event counts × coefficients.** Counts are gathered first, then weighted in a
separate step (`energy_breakdown`). `DEFAULT_COEFFS = {dram:200, onchip:6, mac:1, rmw:5}`
(override with `--coeffs`). The four buckets:
- `dram` — DRAM transfer bits (`dram_bits().total`), **order-dependent** (see below).
- `onchip` — on-chip buffer accesses: SA-facing A/W reads plus the on-chip write of DRAM refills.
- `mac` — `M*K*N` multiply-accumulates.
- `rmw` — read-modify-write dispatches: one per col fire, scaled by the activation mode
  (`A8→1, A4→2, A2→4` dispatches per col).

Energy is **frequency-independent** (it counts events, not time).

**DRAM traffic is order-dependent** (`dram_bits`), walked on the same outer-block sequence as
the stall model so the two never disagree:
- **A** (activation `[K,N]`, reused over M) reloads whenever the `(K,N)` outer index changes
  (first block always loads). Putting M inner to both K and N keeps A resident.
- **W** (weight `[M,K]`, reused over N) reloads whenever the `(M,K)` outer index changes.
- **C** (output `[M,N]`, reduced over K) is **output-stationary** — one final write, no spill —
  *unless* K is tiled (`K_out > 1`) **and** some loop inner to K is non-trivial so other `(M,N)`
  tiles interleave the K visits. Only then does the psum get evicted/reloaded
  (`K_out` writes, `K_out − 1` reads).

**Cycles** = `compute_work + fill + stall`:
- `compute_work` is the order-independent ideal: `32 · NT · Σ wbits[mt][kt]` — bit-serial, so
  it scales with the *sum* of per-tile weight bits.
- `fill` is the first block's unhidable input-fetch startup.
- `stall` is a sequence-aware **shared-DRAM-bandwidth** model: one DRAM channel of bandwidth
  `eff_bw` is shared by A/W input fetch **and** C psum spill-writes / reload-reads. Each block's
  compute window hides the transfers around it; leftover transfer time is stall (plus a trailing
  C drain for the last tile).
- `freq_ratio = f_chip / f_dram` converts DRAM transfer time into on-chip cycles via
  `eff_bw = dram_bw / freq_ratio`. With `freq_ratio = 1` it is back-compatible (`eff_bw == dram_bw`).
  Energy is unaffected by `freq_ratio`.

**Constraints / shapes.** M, K, N must be positive multiples of 32 (the caller zero-pads to
get there). `wbits` is an MT×KT map of *average* weight bits per 32×32 tile, each in `[2, 8]`
(may be **fractional**; the model never int()-truncates a footprint or a compute count).

**Ranking.** `optimize` keeps only feasible mappings (resident A/W/C footprint ≤ bank capacity),
optionally filters by `--max-cycle`, and sorts by `(energy, actual_cycle)`. Because energy
depends only on the blocking (the outer factors), all perms of a given blocking tie on energy
and the cycle tiebreak picks the genuinely faster schedule. `pareto_front` returns the
non-dominated `(energy, cycle)` set; `tradeoff` exposes the min-energy mapping (OFF) vs. the
cheapest-energy mapping among the fastest schedules (ON).

## `wbits.json` format

A JSON array of shape **MT×KT** (`MT = M/32` rows, `KT = K/32` columns) giving the **average
weight bits** of each 32×32 tile, each entry in `[2, 8]` (fractional allowed):

```json
[[2, 8],
 [8, 2]]
```

This example is for M=K=64 (MT=KT=2): top-left and bottom-right tiles are 2-bit, the other two
are 8-bit. If `--bits-file` is omitted every tile defaults to `--act`.

## Example output

`python mxp_scheduler.py --M 128 --K 128 --N 128 --bank-size 1024 --banks 32 --dram-bw 64 --act 8`:

```
# GEMM (128x128x128)  cap_bits=1048576  dram_bw=64.0
perm       m_in k_in n_in          energy    actual_cycle       stall
MKN           1    4    4       167575552         20992.0      2048.0
MKN           4    4    1       167575552         20992.0      2048.0
MNK           1    4    4       167575552         20992.0      2048.0
MNK           4    4    1       167575552         20992.0      2048.0
KMN           1    4    4       167575552         20992.0      2048.0
KMN           4    4    1       167575552         20992.0      2048.0
KNM           1    4    4       167575552         20992.0      2048.0
KNM           4    4    1       167575552         20992.0      2048.0
NMK           1    4    4       167575552         20992.0      2048.0
NMK           4    4    1       167575552         20992.0      2048.0
```

For this fully-resident uniform workload every mapping costs the same (all tiles fit, no
reload), so the table is a flat tie — exactly what you expect when the whole problem fits
on-chip. Mixed-precision and/or bandwidth-constrained shapes spread the rows out.

### `--explain` trace

`python mxp_scheduler_annotated.py --explain --M 64 --K 64 --N 64 --dram-bw 32 --bits-file /tmp/wb.json`:

```
=== MXP_scheduler explain (annotated) ===
workload : M=64 K=64 N=64  (MT=2 KT=2 NT=2)  act_bits=8
hardware : cap_bits=1048576  dram_bw=32.0  freq_ratio=1.0  eff_bw=32.0
mapping  : perm=MKN  inner(M,K,N)=(1,2,2)  outer(M,K,N)=(2,1,1)
feasible : True   (footprint_bits=114688 <= cap_bits=1048576)
blocks   : 2 outer block(s) in perm order
compute_work : 1280  (= TILE*NT*Σwbits ; Σ per-block compute = 1280)
dram_bits    : A=32768.0  W=20480.0  Cw=131072  Cr=0  total=184320.0
onchip_bits  : 159744.0
mac_ops      : 262144   rmw_ops : 8192
energy       : dram=36864000.0  onchip=958464.0  mac=262144.0  rmw=40960.0  total=38125568.0
stall / fill : stall=3776.0  fill=1344.0   (DRAM bw shared by A/W fetch + C spill/reload)
actual_cycle : 6400.0  (= compute_work 1280 + fill 1344.0 + stall 3776.0)
lpt_headroom : natural_stall=3776.0  lpt_stall=3776.0  headroom=0.0
```

## Tests

```bash
python -m pytest test_mxp_scheduler.py -q   # unit suite (51 cases)
python mxp_scheduler.py --selftest          # embedded golden-value asserts -> "selftest: OK"
python mxp_scheduler.py --crosscheck        # standard == annotated twin -> "crosscheck: OK"
```
