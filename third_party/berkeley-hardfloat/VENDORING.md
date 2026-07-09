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

## bf16 (e8/s8) subset - `HardFloatBundle_bf16.v`

A separate bundle vendored for the RMW bf16 datapath (Phase 2a).

- **Date vendored**: 2026-07-09
- **Upstream**: same commit as the fp32 bundle (`072fe42b...`), BSD-3-Clause.
- **Generated from**: Chisel 3.5.6 via `sbt`, run under **Java 17**. Java 17 needs the
  reflection unlock flags for the Chisel 3.5.6 / FIRRTL toolchain, passed via
  `JAVA_TOOL_OPTIONS=--add-opens=java.base/java.lang=ALL-UNNAMED` (add the sibling
  `--add-opens=java.base/java.util=ALL-UNNAMED` if sbt/FIRRTL still trips on reflection).
  See `docs/hardfloat-setup.md` for the full regeneration recipe.
- **Local modification**: same as the fp32 bundle - a `` `timescale 1ns/1ps `` directive is
  prepended at line 1 (XSim "all-or-none timescale" rule). A short comment header naming the
  modules follows it. Everything below the header is verbatim Chisel emit output.
- **No cross-bundle module-name collision** with `HardFloatBundle.v` (checked: the two files
  share no `module` name), so both bundles may be compiled into the same `work` library.

### Module names + ports (verified from emitted Verilog)

| Logical role | Emitted module | Ports |
|---|---|---|
| Signed INT32 -> recoded bf16 | `INToRecFN_i32_e8_s8` | `input io_signedIn`, `input [31:0] io_in`, `input [2:0] io_roundingMode`, `input io_detectTininess`, `output [16:0] io_out`, `output [4:0] io_exceptionFlags` |
| Recoded bf16 -> IEEE bf16 | `FNFromRecFN_bf16_wrapper` | `input [16:0] in`, `output [15:0] out` |

Recoded bf16 is 17-bit (1 sign + 9 exp + 7 sig); IEEE bf16 is 16-bit (1 sign + 8 exp + 7 sig).

**Note**: unlike the fp32 `INToRecFN_i32_e8_s24` (whose `io_detectTininess` was optimized away),
the bf16 `INToRecFN_i32_e8_s8` **does** expose `io_detectTininess`. Drive it to `1'b0` (default
IEEE tininess detection), and `io_roundingMode` to `3'd0` (RNE) for the RMW datapath.

Internal helper module (in `HardFloatBundle_bf16.v`, do not instantiate from user code):
- `RoundAnyRawFNToRecFN_ie6_is32_oe8_os8`

### AddRecFN(8,8) limitation - why there is no bf16 adder

HardFloat's `AddRecFN` / `AddRawFN` adder core does **not** support `sigWidth = 8` (bf16):
generating the adder with `expWidth = 8, sigWidth = 8` fails to elaborate. This bundle therefore
provides **no bf16 adder** - only the INT->bf16 (`INToRecFN_i32_e8_s8`) and recoded-bf16->IEEE-bf16
(`FNFromRecFN_bf16_wrapper`) converters.

bf16 addition in the RMW path is done in the **fp32 domain** (a later task): widen bf16 -> fp32
(bf16's 8-bit exponent and 7-bit mantissa are a strict subset of fp32, so the widening is exact),
add with the fp32 `AddRecFN` from `HardFloatBundle.v`, then narrow fp32 -> bf16.

## Do not edit (otherwise)

To re-vendor a different commit, follow `docs/hardfloat-setup.md`. Remember to re-apply the timescale prepend after regeneration (both `HardFloatBundle.v` and `HardFloatBundle_bf16.v`).
