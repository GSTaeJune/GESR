# Berkeley HardFloat — vendored copy

- **Source**: https://github.com/ucb-bar/berkeley-hardfloat
- **Commit**: `072fe42b1cfc2ead6ca56e725adf964ec06f9a4c`
- **License**: BSD-3-Clause (see `LICENSE`)
- **Date vendored**: 2026-05-14
- **Generated from**: Chisel 3.5.6 (Scala 2.13.10) → FIRRTL → Verilog.
  Driver: `hardfloat/src/main/scala/EmitVerilog.scala` (kept in vendoring source tree, not in this directory).
  See `docs/hardfloat-setup.md` in the project root for regeneration steps.

## Module names + ports (verified from emitted Verilog)

| Logical role | Emitted module | Ports |
|---|---|---|
| Signed INT32 → recoded FP32 | `INToRecFN_i32_e8_s24` | `input io_signedIn`, `input [31:0] io_in`, `input [2:0] io_roundingMode`, `output [32:0] io_out`, `output [4:0] io_exceptionFlags` |
| Recoded FP32 add | `AddRecFN` | `input io_subOp`, `input [32:0] io_a`, `input [32:0] io_b`, `input [2:0] io_roundingMode`, `input io_detectTininess`, `output [32:0] io_out`, `output [4:0] io_exceptionFlags` |
| IEEE FP32 → recoded FP32 | `RecFNFromFN_wrapper` | `input [31:0] in`, `output [32:0] out` |
| Recoded FP32 → IEEE FP32 | `FNFromRecFN_wrapper` | `input [32:0] in`, `output [31:0] out` |

**Note**: `INToRecFN_i32_e8_s24` has NO `io_detectTininess` port (Chisel FIRRTL optimization removed it as unused for our config). Only `AddRecFN` has `io_detectTininess`. Drive it to `1'b0` (default IEEE tininess detection).

The wrapper modules (`RecFNFromFN_wrapper`, `FNFromRecFN_wrapper`) use bare `in` / `out` (no `io_` prefix) because they were defined as `RawModule` with bare `IO(...)` declarations rather than an `io` bundle.

Internal helper modules (also in `HardFloatBundle.v`, do not edit / do not instantiate from user code):
- `RoundAnyRawFNToRecFN_ie6_is32_oe8_os24`
- `RoundAnyRawFNToRecFN_ie8_is26_oe8_os24`
- `RoundRawFNToRecFN_e8_s24`
- `AddRawFN`
- `HardFloatBundle` — emit-time container only, not used.

## Key API differences vs old pure-Verilog HardFloat

Older Verilog HardFloat (`HardFloat-1`, Hauser ~2010) used:
- `control` port + `HardFloat_consts.vi` / `HardFloat_specialize.vi` include files.
- Module names without param suffix (`iNToRecFN`, `addRecFN`).

The Chisel 3.5.6 version emitted here has:
- **No `control` port** (replaced by `detectTininess` direct bit input).
- **No include files** — all constants inlined.
- Param-suffixed names (`INToRecFN_i32_e8_s24`).

User RTL must drive `detectTininess = 1'b0` (default tininess detection after rounding, standard IEEE-754 behavior).

## roundingMode encoding (3-bit input)

| Value | Mode |
|---|---|
| 0 | round to nearest, ties to even (RNE) |
| 1 | round toward zero (truncation) |
| 2 | round toward −∞ |
| 3 | round toward +∞ |
| 4 | round to nearest, ties to max magnitude |
| 6 | round to odd |

For `gemm_sram` RMW: always drive `roundingMode = 3'd0` (RNE, IEEE-754 default; matches `np.float32 + np.float32`).

## Local modification

The vendored `HardFloatBundle.v` has **one** local edit: a `` `timescale 1ns/1ps `` directive prepended at line 1. This is required because Vivado XSim mandates "if any module has timescale, all must" and our TBs/RTL use `1ns/1ps`. The edit does not affect synthesis (timescale is sim-only).

Other than that one line, the file is verbatim Chisel emit output.

## Do not edit (otherwise)

To re-vendor a different commit, follow `docs/hardfloat-setup.md`. Remember to re-apply the timescale prepend after regeneration.
