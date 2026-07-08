# RMW BF16 accumulator — design

**Date**: 2026-07-08
**Status**: design approved (brainstorming), pending spec review → writing-plans
**Scope**: change the RMW inter-tile accumulator from FP32 to BF16, golden-first (bit-exact preserved), then RTL.

---

## 1. Motivation

The RMW datapath uses SRAM as an **inter-tile psum accumulator**: SRAM holds a running
partial sum per output element across K-tiles. Each RMW step reads the prior value from
SRAM, adds this K-tile's dequantized GEMM contribution, and writes it back. Today that
running sum is **FP32** (32-bit words).

Narrowing the accumulator to **BF16** (16-bit words) buys two things that the sibling
`buffer_sweep` line already identified as dominant:

- **SRAM psum storage halves** (32→16 bit/word). `buffer_sweep` found the O (psum) buffer
  is the binding capacity constraint (best partitions are always O-heavy 1:1:6); halving
  the psum word doubles effective O-buffer capacity.
- **DRAM O-write energy halves.** O write = `M·N·32b` is a fixed, dominant DRAM energy
  term; at 16 bit it is exactly halved.

**Why BF16, not FP16**: this is an *accumulator*. BF16 keeps FP32's exponent range (bias
127, 8-bit exponent) so it cannot overflow for wide dot products / deep accumulation; FP16
(5-bit exponent, max 65504) can. The cost of BF16 is a coarse 8-bit significand — accepted
here (see §7).

**Non-goal**: the intra-tile INT reduction (32 K spatially reduced across SA rows, combined
by the `Accumulator`/`Accumulator_Col` adder tree) stays exact INT and is **unchanged**.
"Change RMW to BF16" means precisely: change the inter-tile accumulator that lives in SRAM.

---

## 2. Decisions (locked during brainstorming)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Golden is parameterized**: `mxint_gemm_golden(accum_dtype ∈ {fp32(default), bf16})`. fp32 path stays byte-identical. | Preserves the FP32 9-mode bit-exact regression; lets both be validated. |
| D2 | **BF16 via `ml_dtypes.bfloat16`**, added as an **optional extra** + lazy-imported in the bf16 branch. | Correct IEEE bf16 RNE for free; keeps the numpy-only default path unaffected. |
| D3 | **RTL is hard-replaced** to bf16 (16-bit). The fp32 RTL *datapath* is preserved via a verified git tag; fp32 *unit* RTL + unit TBs stay at HEAD (see §7). | User choice; simplest RTL. |
| D4 | **No blocking accuracy gate.** bf16 accuracy (vs true FP32) is accepted; the fp32 golden path is the fallback if bf16 proves too lossy. bf16 `sw_fp32` SNR is printed, with a **loud warning** on catastrophic (negative-dB) SNR. | fp32 fallback makes bf16-badness non-catastrophic; loud warning keeps the fallback decision actionable. |
| D5 | **Loop-order scope**: the bf16 golden accumulates K-blocks in order `0..n-1`, valid **only** for the current K-outermost sequential feeding (the sole order the RTL/TB drive). Enforced by a guard, not just a comment. | bf16 is non-associative; a silent loop-order change would make a wrong result look green. |
| D6 | **Phase-1 golden is PROVISIONAL** until the Phase-2a HardFloat↔ml_dtypes cross-check passes. | The golden's bit-exact match to HardFloat is only *proven* once bf16 RTL primitives exist. |

---

## 3. Bit-exactness model

The golden is a **software model of the HW arithmetic**, not a truth reference (the truth is
`C_fp32`). The compare gate is literally bit-exact: `hw_sw n_nonzero_diff == 0`
(`compare.py`). It works today because the golden mirrors the HW reduction order and width.
For BF16 the golden must mirror the BF16 datapath exactly.

### 3.1 Datapath the golden must mirror

Per output element, per K-block `blk = 0,1,…,n-1` (K-outermost order), the HW does:

1. `Accumulator`/`Accumulator_Col` produce the **exact int32 block sum** (intra-tile INT).
2. `int_to_bf16`: `INToRecFN`(int32 → bf16, RNE) → recoded-domain **exponent add** by the
   9-bit combined scale (exact power-of-2 shift) → `FNFromRecFN` (rounds if the value lands
   in bf16 subnormal range).
3. `bf16_adder`: `AddRecFN`(bf16 + bf16, RNE, tininess-after-rounding) with the prior SRAM
   value.

### 3.2 Golden formula (corrected — single exponent shift)

```python
from ml_dtypes import bfloat16          # lazy, inside the bf16 branch only
C = np.zeros((M, N), dtype=bfloat16)     # accumulator is bf16 (visible in the type)
for blk in range(n_blocks):              # order 0..n-1 (D5, guarded)
    block_int = int_A[:, sl].astype(np.int64) @ int_B[sl, :].astype(np.int64)  # exact
    # Combined integer exponent, per (m, n). e_a=scale_A (uint8 E8M0), e_b=scale_B (uint8).
    # impl_a_blk = per-(row) implicit exponent for this block (scalar-broadcast, or the
    # mixed-prec array column impl_a_mat[:, blk]); impl_b is the scalar activation implicit.
    #   E = (e_a - 0) + e_b - 254 - impl_a_blk - impl_b     (cast to int64 BEFORE arithmetic)
    E = (e_a[:, blk:blk+1].astype(np.int64) + e_b[blk:blk+1, :].astype(np.int64)
         - 254 - impl_a_blk - impl_b)
    r    = block_int.astype(bfloat16)                              # int->bf16 RNE  (= INToRecFN)
    fp_a = np.ldexp(r.astype(np.float64), E).astype(bfloat16)      # exact 2^E shift + 1 round
    C    = C + fp_a                                                # native bf16 add (= AddRecFN)
    assert C.dtype == bfloat16                                     # guard: no float32 promotion
return C.astype(np.float32)               # exact upcast; compare stays fp32
```

### 3.3 Why this is bit-exact to HardFloat (review-verified)

- **Double-rounding of the add is safe.** `ml_dtypes` bf16 add computes the fp32 (or wider)
  sum then rounds RNE; `round_bf16(round_wide(exact)) == round_bf16(exact)` because the
  intermediate significand (24-bit fp32 / 52-bit fp64) satisfies `p1 ≥ 2·p2+2` (18 for
  bf16). Reviewer A confirmed **0 divergences across 10M+ adversarial cases** (random,
  boundary straddles, near-overflow, subnormal-result, ties). Equals HardFloat `AddRecFN`.
- **Scales are exact powers of two (E8M0).** `const.IMPLICIT_SCALE_EXP`, `quant.py`,
  `gemm._e8m0_to_fp32` all confirm this. So dequant is a pure exponent shift; the **only**
  rounding points are int→bf16 (step 2 significand) and each bf16 add (step 3).
- **The scale must be a single exponent shift, not a collapsed bf16 scalar.** The combined
  exponent ranges `[-266, +254]`; a collapsed `bf16(2^E)` underflows to 0 / overflows to inf
  even when `block_int·2^E` is representable (Reviewer A measured 26% cell divergence in the
  small-magnitude regime). `np.ldexp` on the rounded int, computed in float64 (no premature
  underflow) then rounded once, matches HW in **all** regimes including bf16 subnormals.
- **`block_int` is small.** Max `|block_int|` (A8×B8) = `32·127·127 = 516128 < 2^20 < 2^24`,
  so int→fp32 is exact and `int32.astype(bfloat16)` is RNE (Reviewer A: 0 non-RNE), matching
  `INToRecFN`.
- **Tininess mode does not affect delivered values.** `detectTininess` only sets the IEEE
  underflow *flag*, never the stored result; the compare gate reads values only. So a
  tininess-convention difference between HardFloat and ml_dtypes cannot cause a divergence
  (Reviewer A).

`np.ldexp`/float64 and the `assert` are the concrete fixes for the two bugs review found:
the scale-collapse underflow and the numpy `bf16 * float32 → float32` promotion trap.

---

## 4. Phasing

| Phase | Deliverable | Gate |
|---|---|---|
| **1. Golden BF16** | `accum_dtype` param + bf16 path; `ref --accum`; optional dep | Phase-1 test suite (§5.2) + fp32 9-mode sweep still bit-exact. **RTL untouched.** Golden marked PROVISIONAL. |
| **2a. HardFloat + primitives** | Re-vendor bf16 HardFloat; `int_to_bf16.v`, `bf16_adder.v` + unit TBs | Unit TBs cross-check HardFloat bf16 == ml_dtypes bf16 on near-exhaustive/large vectors (§6.1). Clears D6 (golden confirmed). |
| **2b. RTL widths + integration** | RMW/SRAM/top/TB 32→16; hwio bf16 reader; sweep wiring | Integration sweep (bf16 golden) returns to **bit-exact PASS**. |

Splitting 2a/2b front-loads the one risk that could force a golden change (HardFloat's actual
rounding), before the width-propagation churn.

---

## 5. Phase 1 — golden (detailed)

All changes in `MXP_Tools/`. No RTL, no sweep-script changes.

### 5.1 Code

- **`mxp_tools/gemm.py`**: add `accum_dtype="fp32"` param. `"fp32"` → current code,
  byte-identical. `"bf16"` → §3.2. Lazy `from ml_dtypes import bfloat16` inside the branch;
  clear error if the extra is absent. Both scalar-prec and mixed-prec (array) paths round to
  bf16 at identical points. Add the D5 loop-order guard (accumulation is `0..n-1`; assert /
  parameterize so a non-sequential order fails loud).
- **`mxp_tools/cli.py`**: `ref` gains `--accum {fp32,bf16}` (default `fp32`). Default →
  filename/behavior identical to today. `bf16` → `C_sw_{pa}_{pb}_bf16.npz` (coexists with the
  fp32 ref), `accum` recorded in the npz. `compare.py` **unchanged** (both are fp32
  containers; bf16→fp32 upcast is exact).
- **`pyproject.toml`**: `[project.optional-dependencies] bf16 = ["ml_dtypes"]` (pinned).
  Update the "numpy-only" wording in `gemm.py`.

### 5.2 Phase-1 gate (`tests/test_gemm_bf16.py` + regression)

| # | Test | What it proves (non-tautological) |
|---|---|---|
| 1 | **Independent bf16-RNE reference** | Pure-python bit-level fp32→bf16 RNE (23→7 mantissa round-half-to-even, subnormal, overflow→inf), asserted == `ml_dtypes` over large random + directed sets (ties, subnormal, max/min, overflow). **Anchors ml_dtypes to the IEEE spec** — this is what makes Phase 1 a gate, not `ml_dtypes==ml_dtypes`. |
| 2 | **Rounding topology** | A constructed case where per-K-tile rounding ≠ round-once; golden must match **per-add** (proves the golden rounds every add, not fp32-accumulate-then-cast). |
| 3 | **Subnormal dequant** | Small-magnitude blocks (E8M0 well below 127) forcing bf16-subnormal partial products; exercises the ldexp path the 9-mode random-normal inputs never reach. |
| 4 | uniform == mixed | scalar `prec_A` == array of same value (mirrors existing fp32 invariant), bf16 path. |
| 5 | bf16-subset | every `C_sw` element satisfies `x == fp32(bf16(x))`. |
| 6 | **fp32 frozen-output regression** | fixed-seed fp32 `C` stored and asserted bit-equal — pins "byte-identical" against the refactor (stronger than the loose 5% test). |
| 7 | reference-semantics | small hand-derived examples (values computed from the spec, **not** ml_dtypes). |
| 8 | (informational) accuracy | bf16 `sw_fp32` SNR on representative shapes, printed; **loud warning** on negative-dB SNR. Not a gate. |
| — | (human) | `run_integration_sweep.sh` 9-mode fp32 still bit-exact. |

---

## 6. Phase 2 — RTL (detailed)

### 6.1 Phase 2a — HardFloat + primitives

- **Re-vendor bf16 HardFloat** (e8/s8) via `docs/hardfloat-setup.md` (sbt/Chisel). The bundle
  is currently FP32-only (`INToRecFN_i32_e8_s24`, `AddRecFN` e8/s24, `RecFN/FN` wrappers with
  baked-in widths); e8/s8 is a valid config but requires **regeneration**, not a Verilog
  parameter edit.
- **`int_to_bf16.v`**: `int_to_fp32.v` with narrower significand. Because bf16 and fp32
  **share expWidth=8**, the recoded exponent field is 9 bits in both and the scale-exponent-
  add logic is width-invariant (the same 9-bit combined scale is reused). Concrete edits:
  `REC_W 33→17`, `sig_field [22:0]→[6:0]`, primitives `_e8_s24 → _e8_s8`, and re-derive the
  `in_int==0` zero-passthrough for bf16.
- **`bf16_adder.v`**: `fp32_adder.v` with `AddRecFN` e8/s8, RNE, `detectTininess=0`.
- **Unit TBs** `int_to_bf16_tb`, `bf16_adder_tb` cross-check **HardFloat bf16 == ml_dtypes
  bf16**:
  - `int_to_bf16`: **near-exhaustive** over the realizable domain (`|block_int| ≲ 2^20` ×
    the finite scale set), incl. int RNE ties at the 8-bit boundary, INT32_MIN, mantissa
    carry-out across a power of two, scale extremes exercising `new_exp` truncation/wrap,
    zero-passthrough-with-nonzero-scale.
  - `bf16_adder`: **large randomized** (1e5–1e6) + directed edges: tie-to-even, ±0 from
    cancellation, subnormal±subnormal, carry→exponent increment/overflow→inf, inf+(−inf)→NaN,
    and a **defined NaN-equality policy** (HardFloat vs ml_dtypes NaN payloads may differ).

Passing 2a clears D6 (the golden is now a proven model of the HW).

### 6.2 Phase 2b — width propagation + integration

Datapath narrows 32→16 for the psum/SRAM side; `in_GEMM` (INT32 into RMW) stays 32.

Concrete sites review flagged (change list must cover all):
- **`RMW.v`**: `in_SRAM`/`out_RMW`/`sram_dly` → 16-bit; swap `int_to_fp32`→`int_to_bf16`,
  `fp32_adder`→`bf16_adder`.
- **`sram_1rw*`**: `DATA_WIDTH 32→16` (halves the SRAM; leaf + banked already parameterized).
- **`gemm_sram_top.v`**: `Q[c*32+:32]`, `rmw_out_RMW[c*32+:32]`, `sram_D_w[c*32+:32]` →
  `*16`; zero-mux `32'h0 → 16'h0` (`:75,80,81`). `rmw_in_GEMM[c*32+:32]` **stays** 32.
- **`tb/gemm_sram_top_tb.v`**: `sram_Q`/`sram_WMASK`/`rmw_out_RMW` wire widths → `*16`; WMASK
  write/init `32'hFFFFFFFF → 16'hFFFF`; `dump_banks` `reg[31:0]→[15:0]`, `[bi*32+:32]→[bi*16+:16]`,
  `%08x→%04x`. Lane/FIFO carry INT32 psum + 9-bit scale + addr — **unchanged** (A2/A4 lane
  packing is INT-domain, orthogonal to bf16).
- **`hwio.py`**: new `read_writememh_bf16` (read 16-bit words, upcast `fp32 =
  uint32(bf16_bits) << 16`, lossless). **Gap sentinel must change**: a written bf16 `0x0000`
  upcasts to `+0.0` and collides with the `uint32(0)` unwritten-slot sentinel — use a separate
  written-mask (as `gather_banks` already tracks), do not reuse the NaN-gap trick.
  `interleaved_row_major_32bank` mapping is word-width-agnostic — unchanged.
- **Sweep**: wire `ref --accum bf16`; integration sweep = bf16-RTL vs bf16-golden → bit-exact.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| HardFloat bf16 rounding ≠ ml_dtypes (the bit-exact link) | Phase-2a cross-check TBs with near-exhaustive/large vectors + NaN policy (§6.1). This is the load-bearing proof; golden is PROVISIONAL until it passes (D6). |
| bf16 accuracy too lossy (8-bit significand, deep-K swamping for qwen ~28 blocks) | Accepted (D4). fp32 golden path is the fallback; `sw_fp32` SNR printed + loud negative-dB warning. |
| fp32 RTL regression lost (hard-replace) | git tag taken pre-swap **and verified to build green**; documented checkout-and-run recipe; **fp32 unit RTL (`int_to_fp32.v`, `fp32_adder.v`) + unit TBs kept at HEAD** (the bf16 top just doesn't instantiate them) so unit regressions run without a checkout — only the width-coupled integration sweep forks. |
| Silent loop-order violation (bf16 non-associative) | D5 guard: order is `0..n-1`, asserted/parameterized to fail loud. |
| numpy `bf16 * float32 → float32` silently widens the accumulator | Force bf16 through the chain; `assert C.dtype == bfloat16` per block (§3.2). |
| Phase-1 gate tautological (ml_dtypes vs itself) | Independent bf16-RNE reference + rounding-topology test (§5.2 #1,#2); golden PROVISIONAL. |

---

## 8. Scope limits / non-goals

- Only the current **K-outermost sequential** loop order is supported (D5). Other loop orders
  (the `buffer_sweep`/`timeloop` search space) would require the golden to replicate that
  order — out of scope.
- **FP16 is not pursued** (overflow risk for an accumulator; §1). The dtype is swappable in
  the golden if ever revisited.
- The **RTL is not parameterized** for fp32/bf16 coexistence (D3); only the golden is.

---

## 9. Review provenance

Design reviewed pre-spec by three independent opus reviewers (numerical correctness /
RTL faithfulness+completeness / verification+phasing+risk), each verifying against the actual
code. Key outcomes folded in above:

- Confirmed: double-rounding safety (10M+ empirical cases), "1 block = 1 RMW in order 0..n-1"
  for this RTL/workload, expWidth=8 shared trick, A2/A4 orthogonality, HardFloat bf16
  feasibility.
- Fixed: scale-collapse subnormal bug → single exponent shift (§3.2); bf16×float32 promotion
  → dtype guard; tautological Phase-1 gate → independent reference + topology test + PROVISIONAL
  (§5.2); Phase-2 split 2a/2b + near-exhaustive edge vectors (§6); optional-dep + lazy import;
  loop-order guard; fp32 frozen regression; loud SNR warning; git-tag verify + fp32 unit files
  at HEAD; concrete Phase-2b sites enumerated.
