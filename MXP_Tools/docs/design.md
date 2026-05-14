# mxp_tools — design notes

## Goal

HW (FPGA, FP32 output) bring-up/verification helper. Generate test inputs,
produce a SW golden reference, parse HW output, do a 3-way FP32 comparison,
and visualize where things diverge.

## Data flow

```
1. gen     seed              → FP32 A (M×K), FP32 B (K×N), C_fp32 = A @ B
2. quant   FP32 A, B         → (int_A, scale_A), (int_B, scale_B)   [block=32, E8M0]
3. emit    quantized A, B    → $readmemh files (HW input SRAM image)

   ── HW runs on FPGA ──

4. read    HW $writememh     → flat FP32 array per bank
   gather  + bank layout     → C_hw (M, N)
5. ref     (int, scale)      → SW golden via mxint_gemm_golden → C_sw
6. truth   (A, B)            → C_fp32 (already from step 1)

7. diff    (C_hw, C_sw, C_fp32) → stats (max, RMSE, SNR) + diff matrices
8. viz     heatmap 2×3 grid + 9-mode summary table
```

Key property: HW input emit (step 3) and SW golden compute (step 5) use the
**same** `(int, scale)` from step 2. Anything `C_hw` and `C_sw` disagree on
is HW logic; `C_sw` vs `C_fp32` is pure quantization error.

## Module layout

- `const.py` — INT2/4/8, PREC_NAMES, IMPLICIT_SCALE_EXP, MAX_INT, BLOCK_SIZE=32
- `quant.py` — `quantize_block_mx`, `quantize_matrix_mx`. Bit-exact with TransformSerial.py.
- `gemm.py` — `mxint_gemm_golden`. Per-block int dot → FP32 scale → accumulate.
- `hwio.py` — `$writememh` read/write, bank gather, bit-serial/parallel emitters.
- `compare.py` — `diff_3way`, `print_stats`.
- `viz.py` — `heatmap_3way`, `summary_dashboard`.
- `cli.py` / `__main__.py` — `python -m mxp_tools {gen, emit, ref, compare, viz}`.

## Numerical contract

- All ops bit-exact-aligned with OCP MX Spec §6.3 and `MXP/TransformSerial.py`.
- Strict dtype: `quantize_block_mx` rejects non-`float32`. `mxint_gemm_golden`
  rejects non-`int8` mantissas / non-`uint8` scales.
- K must be a multiple of 32 (no auto-pad inside `quant` / `gemm`); the CLI
  pads at the boundary if M/K/N aren't multiples of 32.
- Implicit scale: `2^-(IMPLICIT_SCALE_EXP[a] + IMPLICIT_SCALE_EXP[b])` per block.
  For MXINT8×MXINT8 = `2^-12`. Power-of-two so exact in FP32.

## HW output contract

- One `$writememh` file per SRAM bank. Each payload line is one 32-bit hex
  word, treated as IEEE-754 FP32 bit pattern. `@<addr>` headers and `//`
  comments are skipped.
- Bank → matrix-element mapping is **plug-in**: user supplies a callable
  `(bank_idx, word_offset, M, N) -> (m, n) | None`. `None` = unused slot.
- Unmapped C[m, n] slots default to NaN; `gather_banks` raises if any
  element of C is still NaN at the end. (= layout config wrong, fail loud.)

## What's NOT here

- No torch dependency. Pure numpy + matplotlib (matplotlib only for viz).
- No tests yet; design is small enough that the smoke script + visual diff
  catches regressions. Add pytest later if the toolkit grows.
- No CUDA / acceleration. M=K=N=128 in CLI defaults; smoke script runs in
  well under a second.
