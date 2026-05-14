# mxp_tools

MXP FPGA bring-up / verification helper. Companion to the MXP HW repo.

## What it does

```
FP32 A,B  ─►  quantize MXINT{2,4,8}  ─►  $readmemh files (HW input)
                       │
                       └────────────►  SW golden GEMM (FP32 accumulator)
                                         │
HW (FPGA)  ─►  $writememh FP32 dump  ────┴──►  3-way diff (HW / SW / FP32 truth)
                                                  └─►  heatmap + 9-mode summary
```

Same MX quantization algorithm as `MXP/TransformSerial.py`, lifted from `MXP_Soft/mxint/`
into a self-contained numpy package with `$writememh` I/O and visualization.

## Install

```bash
pip install numpy matplotlib    # matplotlib only needed for viz
```

(No package install needed — run from this folder.)

## Quick start (no real HW)

```bash
cd MXP_Tools
python -m mxp_tools gen     --out work/ -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit    --out work/                          # all 9 modes
python -m mxp_tools ref     --out work/                          # SW golden
python examples/01_smoke.py work/                                # fake HW + compare + viz
```

Output in `work/`:
- `ab_fp32.npz` — A, B, C_fp32 (truth)
- `hw_input/a_input_BS_*.hex`, `b_input_*.hex`, `a_scale_*.hex`, `b_scale_*.hex`, `quant_*.npz`
- `sw_ref/C_sw_mxint{A}_mxint{B}.npz` — SW golden for each of 9 modes
- (`01_smoke.py` adds) `hw_out/`, `compare_*.npz`, `heatmap_*.png`, `summary.png`

## With real HW

```bash
python -m mxp_tools emit --out work/ --prec 8
# … run FPGA on work/hw_input/*_mxint8.hex, dump result to work/hw_out/bank0.mem
python -m mxp_tools ref --out work/ --prec-a 8 --prec-b 8
python -m mxp_tools compare \
    --ref work/sw_ref/C_sw_mxint8_mxint8.npz \
    --hw-banks work/hw_out/bank0.mem \
    --layout single \
    --save work/compare_8x8.npz
python -m mxp_tools viz --npz work/compare_8x8.npz --save work/heatmap_8x8.png
```

## Bank layout

The HW SRAM is 128 KB × 16 banks; how C is laid out across banks is still
TBD. `hwio.gather_banks(bank_paths, M, N, mapping)` accepts any callable
`(bank_idx, word_offset, M, N) -> (m, n) | None`. Two defaults are provided:

- `default_single_bank_row_major` — all of C in one bank, row-major flat
- `default_banks_split_rows(n_banks)` — M rows evenly split across banks

When the actual layout is decided, drop in a new mapping function — that's
the only thing that changes.

## See also

- `docs/design.md` — short design notes
- `examples/01_smoke.py` — runnable end-to-end without real HW
