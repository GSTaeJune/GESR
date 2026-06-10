# MXP_scheduler

A cost model + optimizer that answers one question for the MXP bit-serial mixed-precision
systolic array: **"Given this GEMM and this chip, which tiling + loop order should I run?"**

You hand it a matrix-multiply shape and your hardware parameters. It lists every sensible way
to schedule the work, estimates the **energy** and **cycles** of each, throws out the ones that
don't fit in SRAM, and ranks the rest. You read off the winner.

It's plain Python (stdlib only — no solver library). Spec:
[`../docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md`](../docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md)

## TL;DR

```bash
# "What's the best schedule for a 128x128x128 GEMM on my current chip?"
python mxp_scheduler.py --M 128 --K 128 --N 128 --bank-size 1024 --banks 32 --dram-bw 64 --act 8
```

The top row of the printed table is the recommended schedule. That's the whole tool.

---

## The mental model (read this first)

Everything below is easier once these five ideas are in your head.

### 1. The matrices: A, W, C

We compute `C = W · A` (output `C[M,N]` = weight `W[M,K]` times activation `A[K,N]`).
Three tensors, three different reuse patterns — this is what makes loop order matter:

| Tensor | Shape | Reused over | Role |
|---|---|---|---|
| **A** activation | `[K, N]` | M | read |
| **W** weight | `[M, K]` | N | read (bit-serial) |
| **C** output | `[M, N]` | — (reduced over K) | written, accumulated |

### 2. block → tile

The array is physically **32×32**, so everything is counted in 32-sized chunks:

- A **block** = **32 elements** that share one quantization scale, quantized to **2, 4, or 8 bits**.
- A **tile** = a **32×32** chunk of a matrix = **32 blocks** (32×32 = 1024 elements).
- So a matrix of shape `M×K` is `MT×KT` tiles, where `MT = M/32`, `KT = K/32`.

Because a tile holds 32 independently-quantized blocks, its **average weight bits can be
fractional** — e.g. 16 blocks at 2-bit + 16 at 8-bit → average **5.0**; 24 at 2-bit + 8 at
8-bit → **3.5**. The model carries these averages as real numbers and **never rounds them off.**

### 3. A "mapping" = one schedule candidate

A mapping is **(loop order) + (how much of each dimension you stage in SRAM at once)**.
Example: `perm=MKN, inner=(1,4,4)` means "loop M outermost, then K, then N; keep 1 M-tile,
4 K-tiles, 4 N-tiles resident." The optimizer's whole job is to score every such candidate.

### 4. The 32×32 array is fixed

The spatial mapping onto the physical PEs is **not** searched — it's your hardware. The tool
only searches *around* it: the loop order and the blocking.

### 5. Why loop order changes the bill

SRAM is small, so tiles stream in and out from DRAM. **A tile that's still resident when you
need it again is free; one that got evicted has to be re-fetched.** Loop order decides what
stays resident and what gets kicked out — so it directly drives DRAM traffic (energy) and
stalls (cycles). That's the entire reason this tool exists.

---

## Two files, identical results

| File | Role |
|---|---|
| `mxp_scheduler.py` | **The one you run.** Clean build, stdlib only. |
| `mxp_scheduler_annotated.py` | **The one you read.** Same logic, heavy 한국어 주석 + a `--explain` trace. |

The two are kept **bit-for-bit identical in their numbers**. `python mxp_scheduler.py --crosscheck`
imports the annotated twin and asserts `evaluate / dram_bits / energy_breakdown / lpt_headroom /
pareto_front / tradeoff` all agree across a sweep of cases (every perm/blocking, spill +
no-spill, `freq_ratio != 1`, fractional wbits, non-default coeffs). Edit one file's logic and
`--crosscheck` fails on any drift — so you can trust the annotated file as ground truth while
reading.

---

## CLI

```
usage: mxp_scheduler.py [-h] [--selftest] [--crosscheck] [--M M] [--K K] [--N N]
                        [--bank-size BANK_SIZE] [--banks BANKS] [--dram-bw DRAM_BW]
                        [--freq-ratio FREQ_RATIO] [--act ACT] [--bits-file BITS_FILE]
                        [--max-cycle MAX_CYCLE] [--coeffs COEFFS]
```

**What to multiply**
| Flag | Meaning | Default |
|---|---|---|
| `--M --K --N` | GEMM dims. Each a positive multiple of 32 (caller zero-pads). | required for a run |
| `--act` | Activation precision, 2 / 4 / 8. | 8 |
| `--bits-file` | JSON `MT×KT` map of avg weight bits per tile (mixed precision). | all tiles = `--act` |

**Which chip**
| Flag | Meaning | Default |
|---|---|---|
| `--bank-size` | Words per SRAM bank. | 1024 |
| `--banks` | Number of banks. | 32 |
| `--dram-bw` | DRAM throughput, bits per DRAM cycle. | 64 |
| `--freq-ratio` | On-chip cycles per DRAM cycle (`f_chip / f_dram`); 1.0 = same clock. | 1.0 |
| `--coeffs` | JSON override of the energy coefficients. | DEFAULT_COEFFS |

**Filtering / verification**
| Flag | Meaning |
|---|---|
| `--max-cycle` | Drop mappings slower than this before ranking. |
| `--selftest` | Run embedded golden-value asserts, print `selftest: OK`, exit. |
| `--crosscheck` | Assert the annotated twin agrees, print `crosscheck: OK`, exit. |

The annotated file takes the **same flags** but swaps `--selftest`/`--crosscheck` for
**`--explain`**, which prints a step-by-step trace of the optimal mapping (or the first mapping
if none is feasible).

### Examples

```bash
# Project workload: uniform 8-bit, 128^3, current HW
python mxp_scheduler.py --M 128 --K 128 --N 128 --bank-size 1024 --banks 32 --dram-bw 64 --act 8

# Mixed precision: a per-tile weight-bits map (MT x KT)
printf '[[2,8],[8,2]]' > /tmp/wb.json
python mxp_scheduler.py --M 64 --K 64 --N 64 --dram-bw 32 --bits-file /tmp/wb.json

# Chip 2x faster than DRAM + a cycle budget
python mxp_scheduler.py --M 128 --K 128 --N 128 --dram-bw 64 --freq-ratio 2.0 --max-cycle 30000

# Plug in your own measured energy coefficients
printf '{"dram":150,"onchip":8,"mac":1,"rmw":4}' > /tmp/co.json
python mxp_scheduler.py --M 128 --K 128 --N 128 --coeffs /tmp/co.json

# See every intermediate value for the winning schedule
python mxp_scheduler_annotated.py --explain --M 64 --K 64 --N 64 --dram-bw 32 --bits-file /tmp/wb.json

# Verify nothing drifted
python mxp_scheduler.py --selftest
python mxp_scheduler.py --crosscheck
```

## `--config`: 물리 스펙으로 HW 파라미터 자동 도출

`hw_config.json` 에 칩을 물리적으로 기술하면 CACTI(SRAM)와 `dram_presets.json`(DRAM
datasheet 값)으로 `dram_bw / freq_ratio / coeffs.dram / coeffs.onchip` 을 자동 도출한다.
명시 CLI 플래그는 항상 config 를 이긴다. CACTI 설치: `docs/cacti-setup.md`.

```bash
cp hw_config.example.json hw_config.json   # 편집: SRAM 스펙 / DRAM 표준명 / 칩 클럭
python mxp_scheduler.py --config hw_config.json --M 128 --K 128 --N 128 --act 8
```

- `dram` 은 `dram_presets.json` 의 키와 정확히 일치해야 한다 (불일치 시 사용 가능 목록을
  에러로 보여줌). 프리셋은 JSON 이라 자유롭게 추가 가능 — `pj_per_bit` 는 출처(`source`)와
  함께 적을 것.
- `coeffs` 는 부분 오버라이드: 적은 키만 자동 도출값/기본값 위에 덮인다. mac/rmw 계수는
  자동 도출 범위 밖(로직 에너지)이라 기본값 유지 — 매핑-상수항이라 랭킹에는 영향 없음.
  실측값이 생기면 여기로 주입.
- 프리셋 pj_per_bit 의 기준 주의: LPDDR 계열은 device-internal, DDR 계열은 off-chip I/O
  포함 기준이라 절대값을 계열 간 직접 비교하지 말 것 (한 칩 안에서의 매핑 랭킹에는 무관).
  근거는 각 항목 `source` 참조.

---

## How the cost model works

### What it searches

For each candidate it enumerates **(loop order) × (blocking per dimension)**:

- **Loop order**: a permutation of `(M, K, N)` — at most 6. `perm[0]` is the outermost loop.
- **Blocking**: for each dimension, how many inner tiles stay resident — any **divisor** of that
  dimension's tile count (`MT`, `KT`, `NT`). Divisors only, so tiles partition cleanly.

The space stays small (thousands of candidates even for the largest transformer GEMMs), which is
why a plain exhaustive search finds the **exact** global optimum — no solver needed.

### Energy = how many events happen × what each costs

Counts are gathered first, then weighted in a **separate** step (`energy_breakdown`), so no
preference is baked in — the costs come out of events that actually occur.
`DEFAULT_COEFFS = {dram: 200, onchip: 6, mac: 1, rmw: 5}` (override with `--coeffs`). Four buckets:

| Bucket | Counts | Depends on the mapping? |
|---|---|---|
| `dram` | DRAM transfer bits (`dram_bits().total`) | **Yes — order-dependent** (see below) |
| `onchip` | on-chip buffer accesses: SA-facing A/W reads + the on-chip write of each DRAM refill | Yes (tracks DRAM refills) |
| `mac` | `M·K·N` scalar multiply-accumulates | **No — constant** |
| `rmw` | read-modify-write dispatches: one per column fire, scaled by activation mode (A8→1, A4→2, A2→4) | No |

Two things worth knowing:

- **Energy is frequency-independent** — it counts events, not time, so `--freq-ratio` never
  touches it.
- **`mac` is one scalar multiply-accumulate** (one weight element × one activation element,
  accumulated) — *not* a per-tile matmul. There are exactly `M·K·N` of them for the whole GEMM,
  regardless of precision or schedule. Because it's the same for every mapping, **it never
  changes the ranking** — it's there so the absolute energy total is physically complete. The
  precision effect lives in cycles (`compute_work`) and in `rmw` dispatches, not here. The only
  buckets that separate one mapping from another are `dram` and the refill part of `onchip`.

### DRAM traffic is order-dependent (the heart of it)

Picture SRAM as a small desk. Tiles get carried in from DRAM (the warehouse) as you need them.
**If the tile you need next is still on the desk, it's free. If it got pushed off to make room,
you pay to fetch it again.** Loop order decides what stays on the desk. `dram_bits` walks the
exact same block sequence the stall model walks, so energy and cycles can never disagree about
how much traffic there was.

- **A** (`[K,N]`, reused over M): stays resident while only M moves; **re-fetched whenever the
  `(K,N)` index changes** (the first block always loads). → Put M *inside* both K and N and A
  rarely leaves the desk.
- **W** (`[M,K]`, reused over N): symmetric — re-fetched whenever `(M,K)` changes.
- **C** (`[M,N]`, accumulated over K): this one is written, not read, and it builds up across K.
  Two cases:
  - **Output-stationary (no spill):** if you finish *all* of K for a given `(M,N)` tile before
    moving on, the partial sum just sits on the desk accumulating, then gets written **once** at
    the end. This happens when **K is the innermost loop, or K isn't tiled** (`K_out == 1`).
  - **Spill (extra traffic):** if you split K into chunks (`K_out > 1`) **and** visit other
    `(M,N)` tiles between the chunks, each half-finished C tile gets pushed off the desk and must
    be **read back** to add the next K-chunk. Cost: `K_out` writes + `K_out − 1` reads (the first
    touch starts from zero, so one read is saved).

Both conditions are required. If K is outer but the inner M and N aren't split, there's only one
`(M,N)` tile — nothing to interleave, nothing gets evicted, no spill. In code:

```python
k_pos = m.perm.index("K")
interleaved = any(out[d] > 1 for d in m.perm[k_pos + 1:])   # is anything split inside K?
if out["K"] > 1 and interleaved:
    cw, cr = full_c * out["K"], full_c * (out["K"] - 1)      # spill
else:
    cw, cr = full_c, 0                                       # output-stationary
```

A concrete feel for it:
- `perm = MNK` (K innermost): fix an `(M,N)` tile, sweep all of K, write once. **No spill.**
- `perm = KMN` with M, N split: do K-chunk 0 across every `(M,N)`, then K-chunk 1 across every
  `(M,N)`… each C tile wakes up once per chunk, but you touched other tiles in between, so you
  re-read it every time. **Spill.**

### Cycles = compute_work + fill + stall

- **`compute_work`** — the order-independent ideal: `32 · NT · Σ wbits`. Bit-serial, so it scales
  with the *sum* of per-tile weight bits (an all-2-bit layer runs ~4× faster than all-8-bit).
- **`fill`** — the first block's input fetch, which has no prior compute to hide behind.
- **`stall`** — a sequence-aware, **shared-bandwidth** model. One DRAM channel of bandwidth
  `eff_bw` carries A/W fetches **and** C spill-writes / reload-reads. Each block's compute window
  hides the transfers around it; whatever doesn't fit shows up as stall (plus a trailing C drain
  for the last tile).
  - **Double buffering is conditional:** compute can only hide a fetch if SRAM has room to
    prefetch the *next* tiles while computing the current ones — specifically
    `cap ≥ footprint + a second copy of the streamed A+W windows`. If it doesn't fit, the mapping
    is still feasible, but its fetches are **fully exposed** (no overlap). This is why a schedule
    that fits the single-buffer footprint can still stall heavily.
- **`freq_ratio = f_chip / f_dram`** turns DRAM transfer time into on-chip cycles via
  `eff_bw = dram_bw / freq_ratio`. At `1.0` it's a no-op (`eff_bw == dram_bw`). Energy is
  unaffected.

### Constraints

M, K, N must be positive multiples of 32 (the caller zero-pads). `wbits` is the `MT×KT` map of
*average* weight bits per tile, each in `[2, 8]` and possibly fractional — the model never
`int()`-truncates a footprint or a compute count.

### Ranking

`optimize` keeps only feasible mappings (resident A/W/C footprint ≤ bank capacity), optionally
filters by `--max-cycle`, and sorts by `(energy, actual_cycle)`. Energy depends on **both** loop
order and blocking (DRAM is order-dependent), so different perms generally cost differently;
where two mappings still tie on energy, the cycle count breaks the tie toward the genuinely
faster schedule. `pareto_front` returns the non-dominated `(energy, cycle)` set; `tradeoff`
exposes the min-energy mapping (OFF) vs. the cheapest-energy mapping among the fastest schedules
(ON).

---

## Code map — where each idea lives in the code

`mxp_scheduler.py` is ~30 small functions, most under 15 lines. These are the ones that carry
the weight; the rest are one-line count helpers (`mac_ops`, `rmw_ops`, `compute_work`, …) and
CLI plumbing. Line numbers are into `mxp_scheduler.py`.

**The inputs (3 dataclasses)**
| Function | What it is |
|---|---|
| `HW` (`:19`) | The chip: `bank_size`, `banks`, `dram_bw`, `freq_ratio`, `coeffs`. Props `cap_bits` (total SRAM) and `eff_bw` (`dram_bw / freq_ratio`). Validates everything positive; rejects unknown coeff keys (typo guard). |
| `Work` (`:67`) | The GEMM: `M, K, N` + the `wbits` map + `act_bits`. Enforces 32-multiples and the `[2,8]` wbits range. Props `MT/KT/NT`. |
| `Mapping` (`:107`) | One schedule candidate: `perm` (loop order) + `m_in, k_in, n_in` (blocking). Checks `perm` is a real permutation of M/K/N. |

**Search → cost → rank (the pipeline)**
| Function | Role | Implements (README §) |
|---|---|---|
| `gen_mappings(w)` (`:135`) | Yields **every** candidate: `≤6 perms × divisor blockings`. The enumerator. | *What it searches* |
| `footprint_bits` / `feasible` (`:145`/`:159`) | Resident A/W/C bits (W = resident-window max), then "fits in `cap_bits`?". The capacity gate. | *Constraints* |
| **`dram_bits(m,w)`** (`:163`) | **The heart.** Order-dependent DRAM traffic `{A, W, Cw, Cr, total}` — walks the outer-block sequence, reloading A/W on index change and spilling C only when K is split *and* interleaved. | *DRAM is order-dependent* |
| `energy_breakdown(m,w,hw)` (`:239`) | Turns event counts into energy: `counts × coeffs → {dram, onchip, mac, rmw, total}`. | *Energy = events × cost* |
| `_blocks` + `_stall_of_order` (`:252`/`:286`) | The cycle engine: `_blocks` walks outer blocks in loop order; `_stall_of_order` hides each block's transfers behind the neighbouring compute and sums the leftover as stall. `_double_buffered` (`:272`) gates whether hiding is even allowed. | *Cycles = compute + fill + stall* |
| `evaluate(m,w,hw)` (`:347`) | Bundles one mapping's full metric dict (footprint, dram, energy, cycle, stall). Walks `_blocks` **once** and shares it across all metrics. One call = one table row. | — |
| **`optimize(w,hw,max_cycle)`** (`:370`) | **The ranking.** Keep feasible → filter `max_cycle` → sort by `(energy, actual_cycle)`. Returns the ordered list. | *Ranking* |
| `tradeoff` / `pareto_front` (`:459`/`:445`) | `pareto_front` = the non-dominated `(energy, cycle)` curve; `tradeoff` picks its two ends: OFF (min energy) and ON (fastest, then cheapest). | *Ranking* |

**If you read only five**, read them in pipeline order: `gen_mappings` → `dram_bits` →
`energy_breakdown` → `_stall_of_order` → `optimize`. That's the whole tool; everything else
feeds one of these. The matching `mxp_scheduler_annotated.py` has the same five with line-by-line
한국어 주석, and `--explain` prints their intermediate values for one mapping.

---

## `wbits.json` format

A JSON array of shape **MT×KT** (`MT = M/32` rows, `KT = K/32` columns): the **average weight
bits** of each 32×32 tile, each entry in `[2, 8]`. Fractional is allowed and expected — a tile is
32 independently-quantized blocks, so its average is rarely a round number.

```json
[[2, 8],
 [8, 2]]
```

This is M=K=64 (MT=KT=2): the two diagonal tiles are all-2-bit, the other two all-8-bit. Omit
`--bits-file` and every tile defaults to `--act`.

---

## Example output

`python mxp_scheduler.py --M 128 --K 128 --N 128 --bank-size 1024 --banks 32 --dram-bw 64 --act 8`:

```
# GEMM (128x128x128)  cap_bits=1048576  dram_bw=64.0  freq_ratio=1.0  eff_bw=64.0
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

Here the whole problem fits on-chip, so nothing ever reloads and every mapping ties — exactly
what you'd expect. Mixed precision and/or tighter bandwidth spread the rows apart.

### `--explain` trace

`python mxp_scheduler_annotated.py --explain --M 64 --K 64 --N 64 --dram-bw 32 --bits-file /tmp/wb.json`:

```
=== MXP_scheduler explain (annotated) ===
workload : M=64 K=64 N=64  (MT=2 KT=2 NT=2)  act_bits=8
hardware : cap_bits=1048576  dram_bw=32.0  freq_ratio=1.0  eff_bw=32.0
mapping  : perm=MKN  inner(M,K,N)=(1,2,2)  outer(M,K,N)=(2,1,1)
feasible : True   (footprint_bits=108544 <= cap_bits=1048576)
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

---

## Tests

```bash
python -m pytest test_mxp_scheduler.py -q   # unit suite (58 cases)
python -m pytest test_hwconfig.py -q        # hwconfig suite (23 cases; CACTI 없으면 1 skip)
python mxp_scheduler.py --selftest          # embedded golden-value asserts -> "selftest: OK"
python mxp_scheduler.py --crosscheck        # standard == annotated twin -> "crosscheck: OK"
```
