# RMW Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `RMW` (Read-Modify-Write) module that dequantizes a GEMM (INT32, 9-bit scale) result to IEEE-754 FP32 and adds it to a prior FP32 partial sum read from SRAM, plus the supporting test infrastructure.

**Architecture:** Berkeley HardFloat (open-source, vendor-neutral) backend, decomposed into two sub-modules wrapped by `RMW`:
- `int_to_fp32.v` — INT32 + 9-bit scale → IEEE-754 FP32. Internally `iNToRecFN` + recoded exp adjust + `fNFromRecFN`.
- `fp32_adder.v` — FP32 + FP32 → FP32. Internally `recFNFromFN ×2` + `addRecFN` + `fNFromRecFN`.
- `RMW.v` — thin top wrapper instantiating both, with `L_CONV`-cycle delay chain on the `in_SRAM` path for alignment.

All HardFloat modules are combinational; each sub-module inserts its own pipeline registers (`L_CONV` / `L_ADD` parameters).

**Tech Stack:** Verilog-2001 RTL, Vivado 2024.1 XSim batch (xvlog/xelab/xsim), Berkeley HardFloat pre-generated Verilog, Python (numpy) for golden vector generation in MXP_Tools.

**Spec:** `docs/superpowers/specs/2026-05-14-rmw-design.md`

---

## File Structure (created/modified)

```
gemm_sram/
├── third_party/berkeley-hardfloat/         [NEW] vendored HardFloat Verilog
│   ├── LICENSE
│   ├── VENDORING.md
│   ├── HardFloat_consts.vi
│   ├── HardFloat_specialize.vi
│   ├── HardFloat_primitives.v
│   ├── HardFloat_rawFN.v
│   ├── isSigNaNRecFN.v
│   ├── iNToRecFN.v
│   ├── recFNFromFN.v
│   ├── fNFromRecFN.v
│   ├── addRecFN.v
│   ├── addRecFNToRaw.v
│   ├── roundAnyRawFNToRecFN.v
│   └── roundRawFNToRecFN.v
├── gemm_sram.srcs/sources_1/new/
│   ├── int_to_fp32.v                       [NEW] INT32+scale → FP32 sub-module
│   ├── fp32_adder.v                        [NEW] FP32+FP32 → FP32 sub-module
│   └── RMW.v                               [MODIFY] thin top wrapper
├── gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/
│   └── Accumulator_Col.v                   [MODIFY] IMPLICIT_total subtract (local copy only)
├── tb/                                     [NEW]
│   ├── rmw_smoke_tb.v                      HardFloat round-trip sanity
│   ├── int_to_fp32_tb.v                    sub-module unit test
│   ├── fp32_adder_tb.v                     sub-module unit test
│   └── rmw_tb.v                            RMW top, vector-driven
├── sim/                                    [NEW]
│   ├── clean.sh
│   ├── run_rmw_smoke.sh
│   ├── run_int_to_fp32.sh
│   ├── run_fp32_adder.sh
│   └── run_rmw.sh
├── MXP_Tools/mxp_tools/
│   └── rmw_gen.py                          [NEW] vector generator
├── MXP_Tools/mxp_tools/cli.py              [MODIFY] register `rmw-gen` sub-command
└── CLAUDE.md                               [MODIFY] add third_party/ + sim/ + tb/ entries
```

Each file has one responsibility:
- `third_party/berkeley-hardfloat/` — read-only vendored backend; no edits to vendored source.
- `int_to_fp32.v` — only INT→FP32 with scale apply.
- `fp32_adder.v` — only FP32+FP32 → FP32.
- `RMW.v` — only the wrapper: instantiate both sub-modules, in_SRAM delay chain.
- `Accumulator_Col.v` (local copy) — only the implicit-total subtraction; no other changes.
- `tb/rmw_smoke_tb.v` — HardFloat round-trip sanity (no RMW yet). Verifies build before sub-modules.
- `tb/int_to_fp32_tb.v`, `tb/fp32_adder_tb.v` — sub-module unit tests (directed cases each).
- `tb/rmw_tb.v` — RMW top, full TB driven by `$readmemh` vectors from `rmw_gen.py`.
- `sim/*.sh` — XSim batch scripts (one per scenario).
- `rmw_gen.py` — vector generator (numpy → hex files).

---

## Task 1: Vendor Berkeley HardFloat into `third_party/`

**Files:**
- Create: `third_party/berkeley-hardfloat/{LICENSE, VENDORING.md, *.v}`
- (No tests yet — verified via elaboration in Task 2.)

**Important**: The canonical `ucb-bar/berkeley-hardfloat` is **Chisel (Scala DSL)** only. Pre-generated Verilog must be produced by compiling Chisel → Verilog via `sbt`. This requires Java JDK + sbt installed on the host machine. Detailed install steps are in `docs/hardfloat-setup.md`.

- [ ] **Step 1: Install Java JDK + sbt + clone HardFloat**

Follow `docs/hardfloat-setup.md` (sections 1, 2, 3) on the host machine. This produces:
- Working `java -version` and `sbt --version` commands
- HardFloat clone at `/tmp/hardfloat_src/` (or path of your choice)

- [ ] **Step 2: Write Chisel → Verilog emit driver**

Create `/tmp/hardfloat_src/hardfloat/src/main/scala/EmitVerilog.scala` with the content given in `docs/hardfloat-setup.md` section 4 (the `package hardfloat` / `object EmitVerilog extends App` block).

- [ ] **Step 3: Run sbt to generate Verilog**

```bash
cd /tmp/hardfloat_src
sbt "runMain hardfloat.EmitVerilog ./generated_verilog"
```

First run: ~5–10 minutes (dependency download + compile). Subsequent runs: ~30 seconds.

Expected output: `./generated_verilog/*.v` files including (at minimum):
- `INToRecFN.v`
- `RecFNFromFN.v`
- `AddRecFN.v`
- `FNFromRecFN_wrapper.v`

Plus any internal helper modules that Chisel did not inline (e.g., `RoundAnyRawFNToRecFN`, `MulAddRecFN_preMul` if used).

If sbt fails on dependency resolution / Chisel version / etc., see Troubleshooting in `docs/hardfloat-setup.md`.

- [ ] **Step 4: Create vendored directory + copy generated files**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
mkdir -p third_party/berkeley-hardfloat
cp /tmp/hardfloat_src/LICENSE                third_party/berkeley-hardfloat/LICENSE
cp /tmp/hardfloat_src/generated_verilog/*.v  third_party/berkeley-hardfloat/
```

**Note on module names**: Chisel emit may produce mangled names like `INToRecFN_1` due to its dedup pass. After copying, inspect each `.v` file's `module <name>` declaration line and remember the exact name — RTL instantiations in later Tasks (3, 4, 5) reference them by name. If names contain a hash suffix, either:
- (a) Rename the module declaration in the `.v` files to drop the suffix (manual edit OK; comment what was changed), OR
- (b) Update `int_to_fp32.v` / `fp32_adder.v` instantiations in Tasks 3, 4 to use the actual mangled names.

Option (a) is cleaner; option (b) avoids editing vendored files. Choose one consistently.

- [ ] **Step 5: Write VENDORING.md**

Get the commit hash:
```bash
COMMIT=$(git -C /tmp/hardfloat_src rev-parse HEAD)
echo "$COMMIT"
```

Create `third_party/berkeley-hardfloat/VENDORING.md`:

```markdown
# Berkeley HardFloat — vendored copy

Source: https://github.com/ucb-bar/berkeley-hardfloat
Commit: <paste $COMMIT here>
License: BSD-3-Clause (see LICENSE)
Date vendored: 2026-05-14
Generated from: Chisel (Scala) → Verilog via `sbt "runMain hardfloat.EmitVerilog ..."`.
                See docs/hardfloat-setup.md for regeneration steps.

## Modules used by gemm_sram

- `INToRecFN`     — signed INT32 → recoded FP32 (33-bit)
- `RecFNFromFN`   — IEEE-754 FP32 → recoded FP32
- `AddRecFN`      — recoded FP32 add
- `FNFromRecFN_wrapper` — recoded FP32 → IEEE-754 FP32 (function-to-module wrapper)

Helper modules that Chisel emitted inline or as separate files:
- `RoundAnyRawFNToRecFN` (used by AddRecFN, INToRecFN)
- Internal primitives (CLZ, shift logic) — usually inlined

## Do not edit

These files are vendored read-only. If a bug fix is needed, file an upstream
issue and re-vendor from the fixed commit. To re-vendor a different version,
follow `docs/hardfloat-setup.md`.
```

- [ ] **Step 6: Verify file presence**

```bash
ls third_party/berkeley-hardfloat/
```

Expected: `LICENSE`, `VENDORING.md`, plus several `.v` files (likely 4–8 depending on how Chisel splits modules).

- [ ] **Step 7: Commit (optional)**

If the project is a git repo:
```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
git add third_party/berkeley-hardfloat docs/hardfloat-setup.md
git commit -m "third_party: vendor Berkeley HardFloat (Chisel→Verilog) for FP32 backend"
```

If not a git repo, skip this step (don't `git init` without explicit user consent).

---

## Task 2: Sim scaffold + smoke-test the vendored HardFloat

**Files:**
- Create: `tb/rmw_smoke_tb.v`
- Create: `sim/clean.sh`, `sim/run_rmw_smoke.sh`

The smoke TB instantiates a `iNToRecFN → fNFromRecFN` round-trip ONLY (not the RMW module). This verifies the vendored HardFloat elaborates and produces the right answer for a trivial case before we wire it into RMW. Catches missing-dependency errors early.

- [ ] **Step 1: Write `sim/clean.sh`**

Create `sim/clean.sh`:

```bash
#!/usr/bin/env bash
# sim/clean.sh - Remove XSim artifacts.
set -e
cd "$(dirname "$0")"
rm -rf xsim.dir webtalk*.log xvlog.log xvlog.pb xelab.log xelab.pb xsim.log xsim.pb .Xil
rm -f *.vcd *.wdb
```

```bash
chmod +x sim/clean.sh
```

- [ ] **Step 2: Write `tb/rmw_smoke_tb.v`**

Create `tb/rmw_smoke_tb.v`:

```verilog
`timescale 1ns/1ps

// HardFloat include files (macros for control / rounding modes).
`include "HardFloat_consts.vi"
`include "HardFloat_specialize.vi"

module rmw_smoke_tb;

    // ── HardFloat parameters ────────────────────────────────────────
    localparam EXP_W = 8;
    localparam SIG_W = 24;             // includes hidden bit
    localparam REC_W = EXP_W + SIG_W + 1;   // 33-bit recoded FP32

    // ── Stimuli ─────────────────────────────────────────────────────
    reg signed [31:0] int_in;
    wire [REC_W-1:0]  rec;
    wire [31:0]       fp_out;

    // ── DUT: int → recFN → IEEE-754 round trip ──────────────────────
    iNToRecFN #(.intWidth(32), .expWidth(EXP_W), .sigWidth(SIG_W)) u_i2f (
        .control       (`flControl_default),
        .signedIn      (1'b1),
        .in            (int_in),
        .roundingMode  (`round_near_even),
        .out           (rec),
        .exceptionFlags()
    );

    fNFromRecFN #(.expWidth(EXP_W), .sigWidth(SIG_W)) u_f2f (
        .in  (rec),
        .out (fp_out)
    );

    integer errors;

    initial begin
        $dumpfile("rmw_smoke_tb.vcd");
        $dumpvars(0, rmw_smoke_tb);

        errors = 0;

        // Test 1: int 1 → FP32 1.0 (0x3F800000)
        int_in = 32'sd1; #1;
        if (fp_out !== 32'h3F800000) begin
            $display("FAIL: int=1, got fp=%h, expected 3F800000", fp_out);
            errors = errors + 1;
        end

        // Test 2: int 0 → FP32 +0.0 (0x00000000)
        int_in = 32'sd0; #1;
        if (fp_out !== 32'h00000000) begin
            $display("FAIL: int=0, got fp=%h, expected 00000000", fp_out);
            errors = errors + 1;
        end

        // Test 3: int -1 → FP32 -1.0 (0xBF800000)
        int_in = -32'sd1; #1;
        if (fp_out !== 32'hBF800000) begin
            $display("FAIL: int=-1, got fp=%h, expected BF800000", fp_out);
            errors = errors + 1;
        end

        if (errors == 0) $display("rmw_smoke_tb: ALL TESTS PASSED");
        else              $display("rmw_smoke_tb: %0d FAILURES", errors);

        $finish;
    end

endmodule
```

- [ ] **Step 3: Write `sim/run_rmw_smoke.sh`**

Create `sim/run_rmw_smoke.sh`:

```bash
#!/usr/bin/env bash
# sim/run_rmw_smoke.sh - Compile + run rmw_smoke_tb (HardFloat round-trip).
set -e
cd "$(dirname "$0")"
bash clean.sh

HF_DIR=../third_party/berkeley-hardfloat

xvlog -nolog -i "$HF_DIR" \
    "$HF_DIR"/HardFloat_primitives.v \
    "$HF_DIR"/HardFloat_rawFN.v \
    "$HF_DIR"/isSigNaNRecFN.v \
    "$HF_DIR"/iNToRecFN.v \
    "$HF_DIR"/recFNFromFN.v \
    "$HF_DIR"/fNFromRecFN.v \
    "$HF_DIR"/addRecFN.v \
    "$HF_DIR"/addRecFNToRaw.v \
    "$HF_DIR"/roundAnyRawFNToRecFN.v \
    "$HF_DIR"/roundRawFNToRecFN.v \
    ../tb/rmw_smoke_tb.v

xelab -nolog -debug typical rmw_smoke_tb -s rmw_smoke_tb_sim
xsim rmw_smoke_tb_sim -nolog -R
```

```bash
chmod +x sim/run_rmw_smoke.sh
```

The `-i "$HF_DIR"` flag tells xvlog where to find the `.vi` include files. The file order doesn't matter for xvlog — elaboration resolves cross-references.

- [ ] **Step 4: Run the smoke test — expect PASS**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
bash sim/run_rmw_smoke.sh
```

Expected final lines of output:
```
rmw_smoke_tb: ALL TESTS PASSED
```

**If you see elaboration errors** like `ERROR: [VRFC 10-XXXX] cannot find module 'XYZ'`, the named module is a HardFloat dependency that wasn't vendored. Locate `XYZ.v` in the original HardFloat source, copy it to `third_party/berkeley-hardfloat/`, add the file path to `sim/run_rmw_smoke.sh`'s xvlog argument list, and re-run.

**If the smoke tests FAIL** (e.g., `int=1, got fp=...`), the HardFloat modules are not behaving as IEEE-754 expects. This is unlikely if the vendored copy is from the canonical repo. Recheck step 1 of Task 1.

- [ ] **Step 5: Commit**

```bash
git add sim/clean.sh sim/run_rmw_smoke.sh tb/rmw_smoke_tb.v
git commit -m "sim: smoke-test HardFloat round-trip (iNToRecFN + fNFromRecFN)"
```

---

## Task 3: `int_to_fp32` sub-module — TDD

**Files:**
- Create: `gemm_sram.srcs/sources_1/new/int_to_fp32.v`
- Create: `tb/int_to_fp32_tb.v`
- Create: `sim/run_int_to_fp32.sh`

- [ ] **Step 1: Write failing TB `tb/int_to_fp32_tb.v`**

```verilog
`timescale 1ns/1ps
`include "HardFloat_consts.vi"
`include "HardFloat_specialize.vi"

module int_to_fp32_tb;
    localparam L_CONV     = 2;
    localparam CLK_PERIOD = 10;

    reg              clk;
    reg              rst;
    reg  [31:0]      in_int;
    reg  [8:0]       scale;
    wire [31:0]      out_fp32;

    int_to_fp32 #(.L_CONV(L_CONV)) dut (
        .clk(clk), .rst(rst),
        .in_int(in_int), .scale(scale),
        .out_fp32(out_fp32)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("int_to_fp32_tb.vcd"); $dumpvars(0, int_to_fp32_tb); end

    integer errors;
    integer i;

    task check(input [31:0] expected, input [255:0] label);
        begin
            #1;
            if (out_fp32 !== expected) begin
                $display("FAIL %0s: got %h expected %h", label, out_fp32, expected);
                errors = errors + 1;
            end else begin
                $display("PASS %0s", label);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst    = 1; in_int = 0; scale = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Case 1: int=1, scale=127 → FP32 1.0 (0x3F800000)
        @(negedge clk); in_int = 32'sd1;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h3F800000, "int=1 scale=127 -> 1.0");

        // Case 2: int=-1 → -1.0
        @(negedge clk); in_int = -32'sd1;  scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'hBF800000, "int=-1 scale=127 -> -1.0");

        // Case 3: int=2 → 2.0
        @(negedge clk); in_int = 32'sd2;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h40000000, "int=2 scale=127 -> 2.0");

        // Case 4: int=0 → 0.0
        @(negedge clk); in_int = 32'sd0;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h00000000, "int=0 -> 0.0");

        // Case 5: int=1, scale=128 → 2.0 (1 × 2^1)
        @(negedge clk); in_int = 32'sd1;   scale = 9'sd128;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h40000000, "int=1 scale=128 -> 2.0");

        if (errors == 0) $display("int_to_fp32_tb: ALL TESTS PASSED");
        else              $display("int_to_fp32_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
```

- [ ] **Step 2: Write `sim/run_int_to_fp32.sh`**

```bash
#!/usr/bin/env bash
# sim/run_int_to_fp32.sh - Compile + run int_to_fp32_tb.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF_DIR=../third_party/berkeley-hardfloat
SRC=../gemm_sram.srcs/sources_1/new

xvlog -nolog -i "$HF_DIR" \
    "$HF_DIR"/HardFloat_primitives.v \
    "$HF_DIR"/HardFloat_rawFN.v \
    "$HF_DIR"/isSigNaNRecFN.v \
    "$HF_DIR"/iNToRecFN.v \
    "$HF_DIR"/recFNFromFN.v \
    "$HF_DIR"/fNFromRecFN.v \
    "$HF_DIR"/addRecFN.v \
    "$HF_DIR"/addRecFNToRaw.v \
    "$HF_DIR"/roundAnyRawFNToRecFN.v \
    "$HF_DIR"/roundRawFNToRecFN.v \
    "$SRC"/int_to_fp32.v \
    ../tb/int_to_fp32_tb.v

xelab -nolog -debug typical int_to_fp32_tb -s int_to_fp32_tb_sim
xsim int_to_fp32_tb_sim -nolog -R
```

```bash
chmod +x sim/run_int_to_fp32.sh
```

- [ ] **Step 3: Run — expect FAIL (module does not exist)**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
bash sim/run_int_to_fp32.sh
```

Expected: xelab error like `ERROR: [VRFC 10-XXXX] cannot find module 'int_to_fp32'`. This confirms the TB compiles but the DUT is absent.

- [ ] **Step 4: Implement `gemm_sram.srcs/sources_1/new/int_to_fp32.v`**

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// int_to_fp32 — signed INT32 + 9-bit signed scale → IEEE-754 FP32.
//
// Output value = in_int × 2^(scale - 127).
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md (`int_to_fp32` section)
// Backend: Berkeley HardFloat (iNToRecFN + fNFromRecFN).
// Internal pipeline: L_CONV cycles between in and out (≥ 1).
//////////////////////////////////////////////////////////////////////////////////

`include "HardFloat_consts.vi"

module int_to_fp32 #(
    parameter L_CONV = 2
)(
    input  wire        clk,
    input  wire        rst,        // unused (HardFloat is combinational; regs power up to 0)
    input  wire [31:0] in_int,
    input  wire [8:0]  scale,
    output wire [31:0] out_fp32
);
    localparam EXP_W = 8;
    localparam SIG_W = 24;
    localparam REC_W = EXP_W + SIG_W + 1;   // 33 bits

    // 1) INT32 → recoded FP32
    wire [REC_W-1:0] recFN_int;
    iNToRecFN #(.intWidth(32), .expWidth(EXP_W), .sigWidth(SIG_W)) u_i2f (
        .control       (`flControl_default),
        .signedIn      (1'b1),
        .in            (in_int),
        .roundingMode  (`round_near_even),
        .out           (recFN_int),
        .exceptionFlags()
    );

    // 2) Exponent bias adjust on recoded FP exp field.
    wire        sign_bit  = recFN_int[REC_W-1];
    wire [8:0]  exp_field = recFN_int[REC_W-2 -: 9];
    wire [SIG_W-2:0] sig_field = recFN_int[SIG_W-2:0];

    wire signed [9:0] exp_ext   = $signed({1'b0, exp_field});
    wire signed [9:0] scale_ext = $signed(scale);
    wire signed [9:0] new_exp10 = exp_ext + scale_ext - 10'sd127;
    wire [8:0]        new_exp   = new_exp10[8:0];

    wire [REC_W-1:0] recFN_scaled = {sign_bit, new_exp, sig_field};

    // 3) Pipeline register chain (L_CONV stages on recFN_scaled)
    reg [REC_W-1:0] recFN_dly [0:L_CONV-1];
    integer di;
    always @(posedge clk) begin
        recFN_dly[0] <= recFN_scaled;
        for (di = 1; di < L_CONV; di = di + 1)
            recFN_dly[di] <= recFN_dly[di-1];
    end

    // 4) recoded → IEEE-754 (combinational at output)
    fNFromRecFN #(.expWidth(EXP_W), .sigWidth(SIG_W)) u_out (
        .in  (recFN_dly[L_CONV-1]),
        .out (out_fp32)
    );
endmodule
```

- [ ] **Step 5: Run — expect PASS**

```bash
bash sim/run_int_to_fp32.sh
```

Expected:
```
PASS int=1 scale=127 -> 1.0
PASS int=-1 scale=127 -> -1.0
PASS int=2 scale=127 -> 2.0
PASS int=0 -> 0.0
PASS int=1 scale=128 -> 2.0
int_to_fp32_tb: ALL TESTS PASSED
```

If FAIL on case 5 (scale=128 → 2.0) but case 1 passes: the exponent bias arithmetic is off by one. Check the `- 10'sd127` constant.

- [ ] **Step 6: Commit**

```bash
git add gemm_sram.srcs/sources_1/new/int_to_fp32.v \
        tb/int_to_fp32_tb.v sim/run_int_to_fp32.sh
git commit -m "int_to_fp32: HardFloat-based INT32+scale → FP32 sub-module"
```

---

## Task 4: `fp32_adder` sub-module — TDD

**Files:**
- Create: `gemm_sram.srcs/sources_1/new/fp32_adder.v`
- Create: `tb/fp32_adder_tb.v`
- Create: `sim/run_fp32_adder.sh`

- [ ] **Step 1: Write failing TB `tb/fp32_adder_tb.v`**

```verilog
`timescale 1ns/1ps
`include "HardFloat_consts.vi"
`include "HardFloat_specialize.vi"

module fp32_adder_tb;
    localparam L_ADD      = 3;
    localparam CLK_PERIOD = 10;

    reg              clk;
    reg              rst;
    reg  [31:0]      a, b;
    wire [31:0]      sum;

    fp32_adder #(.L_ADD(L_ADD)) dut (
        .clk(clk), .rst(rst),
        .a(a), .b(b), .sum(sum)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("fp32_adder_tb.vcd"); $dumpvars(0, fp32_adder_tb); end

    integer errors;
    integer i;

    task check(input [31:0] expected, input [255:0] label);
        begin
            #1;
            if (sum !== expected) begin
                $display("FAIL %0s: got %h expected %h", label, sum, expected);
                errors = errors + 1;
            end else begin
                $display("PASS %0s", label);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst    = 1; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Case 1: 1.0 + 1.0 = 2.0
        @(negedge clk); a = 32'h3F800000; b = 32'h3F800000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h40000000, "1.0 + 1.0 = 2.0");

        // Case 2: 1.0 + (-1.0) = 0.0
        @(negedge clk); a = 32'h3F800000; b = 32'hBF800000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h00000000, "1.0 + -1.0 = 0.0");

        // Case 3: 1.5 + 2.5 = 4.0
        @(negedge clk); a = 32'h3FC00000; b = 32'h40200000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h40800000, "1.5 + 2.5 = 4.0");

        // Case 4: 0 + 3.14 ≈ 3.14 (0x4048F5C3)
        @(negedge clk); a = 32'h00000000; b = 32'h4048F5C3;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h4048F5C3, "0 + 3.14 = 3.14");

        if (errors == 0) $display("fp32_adder_tb: ALL TESTS PASSED");
        else              $display("fp32_adder_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
```

- [ ] **Step 2: Write `sim/run_fp32_adder.sh`**

```bash
#!/usr/bin/env bash
# sim/run_fp32_adder.sh - Compile + run fp32_adder_tb.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF_DIR=../third_party/berkeley-hardfloat
SRC=../gemm_sram.srcs/sources_1/new

xvlog -nolog -i "$HF_DIR" \
    "$HF_DIR"/HardFloat_primitives.v \
    "$HF_DIR"/HardFloat_rawFN.v \
    "$HF_DIR"/isSigNaNRecFN.v \
    "$HF_DIR"/iNToRecFN.v \
    "$HF_DIR"/recFNFromFN.v \
    "$HF_DIR"/fNFromRecFN.v \
    "$HF_DIR"/addRecFN.v \
    "$HF_DIR"/addRecFNToRaw.v \
    "$HF_DIR"/roundAnyRawFNToRecFN.v \
    "$HF_DIR"/roundRawFNToRecFN.v \
    "$SRC"/fp32_adder.v \
    ../tb/fp32_adder_tb.v

xelab -nolog -debug typical fp32_adder_tb -s fp32_adder_tb_sim
xsim fp32_adder_tb_sim -nolog -R
```

```bash
chmod +x sim/run_fp32_adder.sh
```

- [ ] **Step 3: Run — expect FAIL (module does not exist)**

```bash
bash sim/run_fp32_adder.sh
```

Expected: xelab error `cannot find module 'fp32_adder'`.

- [ ] **Step 4: Implement `gemm_sram.srcs/sources_1/new/fp32_adder.v`**

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// fp32_adder — IEEE-754 FP32 add (RNE rounding).
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md (`fp32_adder` section)
// Backend: Berkeley HardFloat (recFNFromFN × 2 → addRecFN → fNFromRecFN).
// Internal pipeline: L_ADD cycles between (a, b) and sum.
//////////////////////////////////////////////////////////////////////////////////

`include "HardFloat_consts.vi"

module fp32_adder #(
    parameter L_ADD = 3
)(
    input  wire        clk,
    input  wire        rst,        // unused
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] sum
);
    localparam EXP_W = 8;
    localparam SIG_W = 24;
    localparam REC_W = EXP_W + SIG_W + 1;

    // IEEE-754 → recoded
    wire [REC_W-1:0] recFN_a, recFN_b;
    recFNFromFN #(.expWidth(EXP_W), .sigWidth(SIG_W)) u_in_a (.in (a), .out (recFN_a));
    recFNFromFN #(.expWidth(EXP_W), .sigWidth(SIG_W)) u_in_b (.in (b), .out (recFN_b));

    // L_ADD-stage pipeline on the recoded inputs (simple layout — all stages
    // before addRecFN. If timing closure fails, redistribute stages around
    // addRecFN. The contract is unchanged: total latency = L_ADD cycles.)
    reg [REC_W-1:0] recFN_a_dly [0:L_ADD-1];
    reg [REC_W-1:0] recFN_b_dly [0:L_ADD-1];
    integer di;
    always @(posedge clk) begin
        recFN_a_dly[0] <= recFN_a;
        recFN_b_dly[0] <= recFN_b;
        for (di = 1; di < L_ADD; di = di + 1) begin
            recFN_a_dly[di] <= recFN_a_dly[di-1];
            recFN_b_dly[di] <= recFN_b_dly[di-1];
        end
    end

    wire [REC_W-1:0] recFN_sum;
    addRecFN #(.expWidth(EXP_W), .sigWidth(SIG_W)) u_add (
        .control      (`flControl_default),
        .subOp        (1'b0),
        .a            (recFN_a_dly[L_ADD-1]),
        .b            (recFN_b_dly[L_ADD-1]),
        .roundingMode (`round_near_even),
        .out          (recFN_sum),
        .exceptionFlags()
    );

    fNFromRecFN #(.expWidth(EXP_W), .sigWidth(SIG_W)) u_out (
        .in  (recFN_sum),
        .out (sum)
    );
endmodule
```

- [ ] **Step 5: Run — expect PASS**

```bash
bash sim/run_fp32_adder.sh
```

Expected:
```
PASS 1.0 + 1.0 = 2.0
PASS 1.0 + -1.0 = 0.0
PASS 1.5 + 2.5 = 4.0
PASS 0 + 3.14 = 3.14
fp32_adder_tb: ALL TESTS PASSED
```

- [ ] **Step 6: Commit**

```bash
git add gemm_sram.srcs/sources_1/new/fp32_adder.v \
        tb/fp32_adder_tb.v sim/run_fp32_adder.sh
git commit -m "fp32_adder: HardFloat-based FP32+FP32 → FP32 sub-module"
```

---

## Task 5: `RMW.v` top wrapper — TDD

**Files:**
- Modify: `gemm_sram.srcs/sources_1/new/RMW.v`
- Create: `tb/rmw_tb.v` (minimal 1-case directed; expanded to vector-driven in Task 7)
- Create: `sim/run_rmw.sh`

- [ ] **Step 1: Write RMW.v skeleton (port + DEAD_BEEF placeholder)**

Replace the contents of `gemm_sram.srcs/sources_1/new/RMW.v` with:

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// RMW — Read-Modify-Write FP32 accumulator boundary (top wrapper).
//
// Instantiates int_to_fp32 (dequant path) + fp32_adder (accumulate path),
// with L_CONV-cycle delay chain on in_SRAM for alignment.
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md
// Total RMW latency = L_CONV + L_ADD cycles.
//////////////////////////////////////////////////////////////////////////////////

module RMW #(
    parameter L_CONV = 2,
    parameter L_ADD  = 3
)(
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] in_SRAM,
    input  wire [31:0] in_GEMM,
    input  wire [8:0]  scale,
    output wire [31:0] out_RMW
);

    // Body implemented in Step 5. For now drive a recognizably-wrong constant
    // so the TB compiles and the first run fails with a clear mismatch.
    assign out_RMW = 32'hDEAD_BEEF;

endmodule
```

- [ ] **Step 2: Write minimal TB `tb/rmw_tb.v` (one directed case)**

Create `tb/rmw_tb.v`:

```verilog
`timescale 1ns/1ps
`include "HardFloat_consts.vi"
`include "HardFloat_specialize.vi"

module rmw_tb;
    localparam L_CONV     = 2;
    localparam L_ADD      = 3;
    localparam L_TOTAL    = L_CONV + L_ADD;
    localparam CLK_PERIOD = 10;

    reg              clk;
    reg              rst;
    reg  [31:0]      in_SRAM;
    reg  [31:0]      in_GEMM;
    reg  [8:0]       scale;
    wire [31:0]      out_RMW;

    RMW #(.L_CONV(L_CONV), .L_ADD(L_ADD)) dut (
        .clk(clk), .rst(rst),
        .in_SRAM(in_SRAM), .in_GEMM(in_GEMM), .scale(scale),
        .out_RMW(out_RMW)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("rmw_tb.vcd"); $dumpvars(0, rmw_tb); end

    integer errors;
    integer i;

    initial begin
        errors  = 0;
        rst     = 1;
        in_SRAM = 0; in_GEMM = 0; scale = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Directed case 1: 1 × 2^0 + 0.0 = 1.0
        @(negedge clk);
        in_GEMM = 32'sd1;  scale = 9'sd127;  in_SRAM = 32'h0000_0000;

        for (i = 0; i < L_TOTAL; i = i + 1) @(posedge clk);
        #1;

        if (out_RMW !== 32'h3F800000) begin
            $display("FAIL case1: got %h expected 3F800000", out_RMW);
            errors = errors + 1;
        end else begin
            $display("PASS case1: dequant 1 x 2^0 + 0 = 1.0");
        end

        if (errors == 0) $display("rmw_tb: ALL TESTS PASSED");
        else              $display("rmw_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
```

- [ ] **Step 3: Write `sim/run_rmw.sh`**

Create `sim/run_rmw.sh`:

```bash
#!/usr/bin/env bash
# sim/run_rmw.sh - Compile + run rmw_tb.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF_DIR=../third_party/berkeley-hardfloat
SRC=../gemm_sram.srcs/sources_1/new

xvlog -nolog -i "$HF_DIR" \
    "$HF_DIR"/HardFloat_primitives.v \
    "$HF_DIR"/HardFloat_rawFN.v \
    "$HF_DIR"/isSigNaNRecFN.v \
    "$HF_DIR"/iNToRecFN.v \
    "$HF_DIR"/recFNFromFN.v \
    "$HF_DIR"/fNFromRecFN.v \
    "$HF_DIR"/addRecFN.v \
    "$HF_DIR"/addRecFNToRaw.v \
    "$HF_DIR"/roundAnyRawFNToRecFN.v \
    "$HF_DIR"/roundRawFNToRecFN.v \
    "$SRC"/int_to_fp32.v \
    "$SRC"/fp32_adder.v \
    "$SRC"/RMW.v \
    ../tb/rmw_tb.v

xelab -nolog -debug typical rmw_tb -s rmw_tb_sim
xsim rmw_tb_sim -nolog -R
```

```bash
chmod +x sim/run_rmw.sh
```

- [ ] **Step 4: Run — expect FAIL with `DEAD_BEEF` mismatch**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
bash sim/run_rmw.sh
```

Expected output (near end):
```
FAIL case1: got DEADBEEF expected 3F800000
rmw_tb: 1 FAILURES
```

This confirms the TB is wired correctly and the placeholder body is in place. If you see elaboration errors, check that `int_to_fp32.v` and `fp32_adder.v` were committed in Tasks 3 and 4.

- [ ] **Step 5: Implement RMW.v body — wrapper around the two sub-modules**

Replace the body of `RMW.v` (between `output wire [31:0] out_RMW);` and `endmodule`) with:

```verilog
    // Path A: in_GEMM + scale → FP32 (L_CONV cycles)
    wire [31:0] fp_a;
    int_to_fp32 #(.L_CONV(L_CONV)) u_conv (
        .clk(clk), .rst(rst),
        .in_int(in_GEMM), .scale(scale),
        .out_fp32(fp_a)
    );

    // Path B: in_SRAM delayed by L_CONV cycles to align with fp_a at fp32_adder.b
    reg [31:0] sram_dly [0:L_CONV-1];
    integer di;
    always @(posedge clk) begin
        sram_dly[0] <= in_SRAM;
        for (di = 1; di < L_CONV; di = di + 1)
            sram_dly[di] <= sram_dly[di-1];
    end
    wire [31:0] in_SRAM_aligned = sram_dly[L_CONV-1];

    // Add (L_ADD cycles)
    fp32_adder #(.L_ADD(L_ADD)) u_add (
        .clk(clk), .rst(rst),
        .a(fp_a), .b(in_SRAM_aligned),
        .sum(out_RMW)
    );
```

Also delete the placeholder `assign out_RMW = 32'hDEAD_BEEF;` line. Final `RMW.v` ends with:

```verilog
    fp32_adder #(.L_ADD(L_ADD)) u_add (
        .clk(clk), .rst(rst),
        .a(fp_a), .b(in_SRAM_aligned),
        .sum(out_RMW)
    );
endmodule
```

- [ ] **Step 6: Run TB — expect PASS**

```bash
bash sim/run_rmw.sh
```

Expected output:
```
PASS case1: dequant 1 x 2^0 + 0 = 1.0
rmw_tb: ALL TESTS PASSED
```

If FAIL:
- **`got 00000000` instead of `3F800000`**: in_SRAM_aligned is reading garbage / `sram_dly[L_CONV-1]` reset issue. Bump rst-held cycles in TB or check delay chain.
- **`got <near 1.0 but not exact>`**: latency mismatch — TB waiting L_TOTAL cycles but RMW takes L_CONV+L_ADD+1. Recheck cycle counting.

- [ ] **Step 7: Commit**

```bash
git add gemm_sram.srcs/sources_1/new/RMW.v tb/rmw_tb.v sim/run_rmw.sh
git commit -m "rmw: wrap int_to_fp32 + fp32_adder with L_CONV-cycle SRAM align"
```

---

## Task 6: MXP_Tools — `rmw_gen` vector generator

**Files:**
- Create: `MXP_Tools/mxp_tools/rmw_gen.py`
- Modify: `MXP_Tools/mxp_tools/cli.py` (register `rmw-gen` sub-command)

This generates `(in_GEMM, scale, in_SRAM, expected_out)` quadruples for the TB to read via `$readmemh`. Vectors cover dequant-only, accumulation, and back-to-back streaming.

- [ ] **Step 1: Write `rmw_gen.py`**

Create `MXP_Tools/mxp_tools/rmw_gen.py`:

```python
"""RMW test vector generator.

Generates (in_GEMM, scale, in_SRAM, expected_out) quadruples for the RMW unit
TB. Mirrors the RMW data contract:

    fp32_partial = int_value * 2^(scale - 127)             [via IEEE-754 cast]
    out_RMW      = in_SRAM + fp32_partial                  [IEEE-754 RNE add]

Output: four .hex files (one column each), one line per vector.
"""
import argparse
import os
import struct

import numpy as np


def fp32_to_hex(x):
    """np.float32 → 8-char IEEE-754 hex string (lowercase)."""
    return f"{struct.unpack('<I', struct.pack('<f', np.float32(x)))[0]:08x}"


def int32_to_hex(x):
    return f"{int(x) & 0xFFFFFFFF:08x}"


def scale9_to_hex(x):
    """signed 9-bit → 3-char hex (low 9 bits, sign-magnitude in 2's complement)."""
    return f"{int(x) & 0x1FF:03x}"


def dequant(int_val, scale):
    """RMW's dequant model — bit-exact with the RTL pipeline."""
    return np.float32(int_val) * np.float32(2.0) ** (np.int32(scale) - 127)


def add_fp32(a, b):
    """IEEE-754 round-to-nearest-even add. numpy float32 is RNE by default."""
    return np.float32(a) + np.float32(b)


def gen_vectors(n_random=64, seed=0):
    """Returns list of (in_GEMM, scale, in_SRAM, expected_out) tuples."""
    rng = np.random.default_rng(seed)
    vectors = []

    # ── Directed cases ──────────────────────────────────────────────
    # case 1: trivial 1.0
    vectors.append((1, 127, np.float32(0.0)))
    # case 2: -1.0
    vectors.append((-1, 127, np.float32(0.0)))
    # case 3: accumulation: 1 × 2^0 + 1.5 = 2.5
    vectors.append((1, 127, np.float32(1.5)))
    # case 4: zero input
    vectors.append((0, 127, np.float32(3.14)))
    # case 5: large positive int with negative scale (small fp value)
    vectors.append((1_000_000, 127 - 20, np.float32(0.0)))
    # case 6: scale = 0 (means 2^(-127), nearly zero)
    vectors.append((127, 0, np.float32(0.0)))
    # case 7: typical MXINT8 mid-range dot product
    vectors.append((12345, 127, np.float32(-100.0)))

    # ── Random cases ────────────────────────────────────────────────
    # int range: ±2^20 (well within int32). scale range: 127 ± 30
    # in_SRAM: float in [-1e6, 1e6].
    for _ in range(n_random):
        int_val = int(rng.integers(-(1 << 20), (1 << 20)))
        scale   = int(rng.integers(127 - 30, 127 + 30))
        sram_fp = np.float32(rng.uniform(-1e6, 1e6))
        vectors.append((int_val, scale, sram_fp))

    out = []
    for int_val, scale, sram_fp in vectors:
        dq       = dequant(int_val, scale)
        expected = add_fp32(sram_fp, dq)
        out.append((int_val, scale, sram_fp, expected))
    return out


def write_vectors(out_dir, vectors):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "in_GEMM.hex"), "w") as f:
        for v in vectors:
            f.write(int32_to_hex(v[0]) + "\n")
    with open(os.path.join(out_dir, "scale.hex"), "w") as f:
        for v in vectors:
            f.write(scale9_to_hex(v[1]) + "\n")
    with open(os.path.join(out_dir, "in_SRAM.hex"), "w") as f:
        for v in vectors:
            f.write(fp32_to_hex(v[2]) + "\n")
    with open(os.path.join(out_dir, "expected_out.hex"), "w") as f:
        for v in vectors:
            f.write(fp32_to_hex(v[3]) + "\n")
    with open(os.path.join(out_dir, "N.txt"), "w") as f:
        f.write(str(len(vectors)) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rmw-gen")
    p.add_argument("--out",  required=True, help="output directory for .hex files")
    p.add_argument("--n",    type=int, default=64, help="random vector count")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    vectors = gen_vectors(args.n, args.seed)
    write_vectors(args.out, vectors)
    print(f"rmw-gen: wrote {len(vectors)} vectors to {args.out}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register `rmw-gen` sub-command in `cli.py`**

Open `MXP_Tools/mxp_tools/cli.py`. Find the bottom of the `main()` function where subparsers are registered (look for `g = sp.add_parser("viz", ...)`). Add a new subparser registration right before `args = p.parse_args(argv)`:

```python
    from . import rmw_gen
    g = sp.add_parser("rmw-gen", help="generate RMW unit-test vectors")
    g.add_argument("--out",  required=True, help="output directory")
    g.add_argument("--n",    type=int, default=64)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=lambda a: rmw_gen.main([
        "--out", a.out, "--n", str(a.n), "--seed", str(a.seed)
    ]))
```

- [ ] **Step 3: Run generator and verify output**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram/MXP_Tools
python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0
```

Expected:
```
rmw-gen: wrote 71 vectors to work/rmw/
```
(7 directed + 64 random = 71 vectors.)

```bash
ls work/rmw/
```
Expected files: `in_GEMM.hex`, `scale.hex`, `in_SRAM.hex`, `expected_out.hex`, `N.txt`.

Sanity check the first directed case (line 1 of each file):
```bash
head -1 work/rmw/in_GEMM.hex       # expect "00000001"  (int 1)
head -1 work/rmw/scale.hex         # expect "07f"       (127)
head -1 work/rmw/in_SRAM.hex       # expect "00000000"  (FP32 +0)
head -1 work/rmw/expected_out.hex  # expect "3f800000"  (FP32 1.0)
```

- [ ] **Step 4: Commit**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
git add MXP_Tools/mxp_tools/rmw_gen.py MXP_Tools/mxp_tools/cli.py
git commit -m "mxp_tools: add rmw-gen sub-command for RMW unit TB vectors"
```

---

## Task 7: Expand `rmw_tb.v` to read generated vectors

**Files:**
- Modify: `tb/rmw_tb.v` (replace with vector-driven version)
- Modify: `sim/run_rmw.sh` (point at generated vectors)

- [ ] **Step 1: Replace `tb/rmw_tb.v` with vector-driven TB**

Open `tb/rmw_tb.v` and replace its **entire** contents with the following. (The previous one-case TB from Task 5 is discarded.) The new TB issues vectors back-to-back (1 per cycle), captures `out_RMW` with an output-index lag equal to `L_TOTAL = L_CONV + L_ADD`, then compares all results.

```verilog
`timescale 1ns/1ps

`include "HardFloat_consts.vi"
`include "HardFloat_specialize.vi"

module rmw_tb;

    localparam L_CONV     = 2;
    localparam L_ADD      = 3;
    localparam L_TOTAL    = L_CONV + L_ADD;
    localparam CLK_PERIOD = 10;     // 100 MHz

    // Path to the vector files. Defaults relative to xsim's cwd (sim/).
`ifndef VECTOR_DIR
  `define VECTOR_DIR "../MXP_Tools/work/rmw"
`endif

    localparam MAX_N = 1024;

    // ── Vector storage ─────────────────────────────────────────────
    reg [31:0] mem_int   [0:MAX_N-1];
    reg [8:0]  mem_scale [0:MAX_N-1];
    reg [31:0] mem_sram  [0:MAX_N-1];
    reg [31:0] mem_exp   [0:MAX_N-1];
    reg [31:0] captured_out [0:MAX_N-1];

    integer N;
    integer fd;

    initial begin
        $readmemh({`VECTOR_DIR, "/in_GEMM.hex"},      mem_int);
        $readmemh({`VECTOR_DIR, "/scale.hex"},        mem_scale);
        $readmemh({`VECTOR_DIR, "/in_SRAM.hex"},      mem_sram);
        $readmemh({`VECTOR_DIR, "/expected_out.hex"}, mem_exp);

        fd = $fopen({`VECTOR_DIR, "/N.txt"}, "r");
        if (fd == 0) begin $display("ERROR: cannot open N.txt"); $finish; end
        if ($fscanf(fd, "%d", N) != 1) begin
            $display("ERROR: cannot parse N.txt"); $finish;
        end
        $fclose(fd);
        $display("rmw_tb: loaded %0d vectors", N);
    end

    // ── DUT ────────────────────────────────────────────────────────
    reg              clk;
    reg              rst;
    reg  [31:0]      in_SRAM;
    reg  [31:0]      in_GEMM;
    reg  [8:0]       scale;
    wire [31:0]      out_RMW;

    RMW #(.L_CONV(L_CONV), .L_ADD(L_ADD)) dut (
        .clk(clk), .rst(rst),
        .in_SRAM(in_SRAM), .in_GEMM(in_GEMM), .scale(scale),
        .out_RMW(out_RMW)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("rmw_tb.vcd"); $dumpvars(0, rmw_tb); end

    // ── Stimulus + capture + check ─────────────────────────────────
    // Streams N vectors at 1/cycle, captures out_RMW with L_TOTAL lag.
    integer i;
    integer errors;
    integer out_idx;

    initial begin : main_seq
        errors  = 0;
        out_idx = -L_TOTAL;   // out_idx < 0 means "pipeline still filling, ignore output"

        rst     = 1;
        in_SRAM = 0; in_GEMM = 0; scale = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;
        @(posedge clk);

        for (i = 0; i < N + L_TOTAL; i = i + 1) begin
            @(negedge clk);
            if (i < N) begin
                in_GEMM = mem_int[i];
                scale   = mem_scale[i];
                in_SRAM = mem_sram[i];
            end else begin
                in_GEMM = 0; scale = 0; in_SRAM = 0;
            end

            @(posedge clk);
            #1;
            if (out_idx >= 0 && out_idx < N) begin
                captured_out[out_idx] = out_RMW;
            end
            out_idx = out_idx + 1;
        end

        // ─── Compare ────────────────────────────────────────────────
        for (i = 0; i < N; i = i + 1) begin
            if (captured_out[i] !== mem_exp[i]) begin
                $display("FAIL vec %0d: int=%h scale=%h sram=%h → got %h expected %h",
                         i, mem_int[i], mem_scale[i], mem_sram[i],
                         captured_out[i], mem_exp[i]);
                errors = errors + 1;
            end
        end

        if (errors == 0) $display("rmw_tb: ALL %0d TESTS PASSED", N);
        else              $display("rmw_tb: %0d / %0d FAILURES", errors, N);
        $finish;
    end

endmodule
```

The file has exactly four `initial` blocks (vector load, clock, VCD dump, `main_seq` stimulus). Verify:

```bash
grep -c "^    initial" tb/rmw_tb.v       # expect 4
grep -c "main_seq"      tb/rmw_tb.v       # expect 1
```

- [ ] **Step 2: Run — expect PASS**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
bash sim/run_rmw.sh
```

Expected output:
```
rmw_tb: loaded 71 vectors
rmw_tb: ALL 71 TESTS PASSED
```

If FAIL on specific vectors:
- **All fail with off-by-N bit pattern**: pipeline latency mismatch. Verify `L_CONV=2` and `L_ADD=3` in both the RMW module instantiation and the TB localparams (`L_TOTAL=5`).
- **Specific corner cases fail (e.g., overflow/underflow)**: HardFloat's special-encoding handling differs from numpy's. Adjust `dequant()` in `rmw_gen.py` to match (or restrict vector ranges to avoid edge cases).
- **First few vectors fail, rest pass**: capture index off by one. Check `out_idx` initialization and the `if (out_idx >= 0 && out_idx < N)` guard.

- [ ] **Step 3: Commit**

```bash
git add tb/rmw_tb.v
git commit -m "rmw: vector-driven TB reading MXP_Tools-generated cases"
```

---

## Task 8: Modify `Accumulator_Col.v` (project-local copy) for IMPLICIT_total

**Files:**
- Modify: `gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v`

This change is independent of RMW unit tests — it integrates RMW into the larger GEMM pipeline. No new TB is required at this stage; the GEMM-integration TB (out of scope for this plan) will verify behavior.

- [ ] **Step 1: Open the project-local copy**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
# Confirm we are editing the project-local copy, NOT the upstream sibling:
ls -l gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v
# Should NOT touch:  ../../MXP/MXP.srcs/sources_1/new/Accumulator_Col.v
```

- [ ] **Step 2: Locate the `comb_s0..3` block (around line 99)**

Search for `comb_s0`:
```bash
grep -n "comb_s0" gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v
```

Expected location: one assignment per `comb_s0..3`, around lines 99–102.

- [ ] **Step 3: Insert IMPLICIT_total computation + subtract**

Replace the existing `comb_s0..3` block with:

```verilog
// IMPLICIT_total = IMPLICIT[A_prec] + IMPLICIT[W_prec]  (∈ {0,4,8,12} for the 5 modes).
// A side: in_Mode_oh = {is_A2, is_A4, is_A8}.  is_A8 = in_Mode_oh[0], is_A4 = in_Mode_oh[1].
// W side: in_Wcontrol  W_INT8 = 2'b11, W_INT4 = 2'b10, W_INT2 = 2'b01, IDLE = 2'b00.
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md (Q3a, Accumulator_Col modification).
wire [3:0] impl_a = is_A8 ? 4'd6 : is_A4 ? 4'd2 : 4'd0;           // A2 / idle → 0
wire [3:0] impl_w = (in_Wcontrol == 2'b11) ? 4'd6 :
                    (in_Wcontrol == 2'b10) ? 4'd2 : 4'd0;          // INT2 / IDLE → 0
wire [4:0] implicit_total = impl_a + impl_w;                       // 5-bit safe

wire signed [scale_len:0] comb_s0 = {1'b0, in_scale_act[  7: 0]} + {1'b0, in_scale_weight}
                                  - 9'sd127 - {{4{1'b0}}, implicit_total};
wire signed [scale_len:0] comb_s1 = {1'b0, in_scale_act[ 15: 8]} + {1'b0, in_scale_weight}
                                  - 9'sd127 - {{4{1'b0}}, implicit_total};
wire signed [scale_len:0] comb_s2 = {1'b0, in_scale_act[ 23:16]} + {1'b0, in_scale_weight}
                                  - 9'sd127 - {{4{1'b0}}, implicit_total};
wire signed [scale_len:0] comb_s3 = {1'b0, in_scale_act[ 31:24]} + {1'b0, in_scale_weight}
                                  - 9'sd127 - {{4{1'b0}}, implicit_total};
```

- [ ] **Step 4: Verify elaboration of modified Accumulator_Col**

Quick sanity by elaborating with a wrapper that just instantiates `Accumulator_Col` with default params. Create `tb/accumulator_col_elab.v`:

```verilog
`timescale 1ns/1ps
module accumulator_col_elab;
    // Elaboration-only wrapper. Verifies the syntax change compiles.
    reg clk, rst;
    Accumulator_Col dut (
        .clk(clk), .rst(rst),
        .in_a0(0), .in_a1(0), .in_a2(0), .in_a3(0),
        .in_start_accumulate(0),
        .in_Wcontrol(2'b11),
        .in_Mode_oh(3'b001),
        .in_scale_weight(0),
        .in_scale_act(0),
        .out_start_accumulate(),
        .out_Wcontrol(),
        .out_accumulate(),
        .out_scale_weight(),
        .out_scale(),
        .fire()
    );
endmodule
```

Run:
```bash
cd sim
xvlog -nolog \
    ../gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator.v \
    ../gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v \
    ../tb/accumulator_col_elab.v
xelab -nolog accumulator_col_elab -s accumulator_col_elab_sim
```

Expected: no errors. `xelab` exits with status 0.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
git add gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v
git add tb/accumulator_col_elab.v
git commit -m "accumulator_col: subtract IMPLICIT_total from comb_s for RMW dequant"
```

---

## Task 9: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read current file layout section**

```bash
grep -n "## File layout" CLAUDE.md
```

- [ ] **Step 2: Update file layout block**

Find the `## File layout` section in `CLAUDE.md` and replace it with:

```markdown
## File layout

\`\`\`
gemm_sram.xpr                          # Vivado 2024.1 project (target xc7vx485tffg1157-1)
gemm_sram.srcs/sources_1/
    new/
        GEMM.v                         # Top — currently the unmodified MXP TOP
        RMW.v                          # Read-Modify-Write pipeline (HardFloat backend)
    imports/Desktop/MXP/...            # MXP compute RTL (mirrored from ../MXP/, Accumulator_Col modified for IMPLICIT_total)
    imports/Desktop/sram/rtl/...       # SRAM RTL (mirrored from ../sram/)
third_party/berkeley-hardfloat/        # Vendored HardFloat (BSD-3, see VENDORING.md)
tb/                                    # Standalone TBs
sim/                                   # XSim batch run scripts
docs/superpowers/specs/                # Design specs
docs/superpowers/plans/                # Implementation plans
MXP_Tools/                             # Python verification toolkit (project-local copy)
gemm_sram.{sim,cache,hw,ip_user_files} # Vivado scratch — generated, do not edit
\`\`\`
```

(The triple backticks above are escaped because they live inside this markdown plan; in the actual `CLAUDE.md` they should be plain triple backticks.)

- [ ] **Step 3: Add HardFloat note under "Common commands"**

Find the "Common commands" section in `CLAUDE.md`. After the XSim batch description, add:

```markdown
**RMW unit test** (uses HardFloat in `third_party/berkeley-hardfloat/`):
\`\`\`bash
# Generate test vectors first
cd MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0
# Run TB
cd ../sim && bash run_rmw.sh
\`\`\`

Smoke test (HardFloat round-trip only, no RMW): \`bash sim/run_rmw_smoke.sh\`.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): add third_party, sim, tb, and RMW unit-test commands"
```

---

## Done criteria

- `bash sim/run_rmw_smoke.sh` reports `ALL TESTS PASSED`.
- `bash sim/run_rmw.sh` reports `ALL 71 TESTS PASSED` (or whatever N your generator produced).
- `Accumulator_Col.v` (project-local) elaborates without error and subtracts `IMPLICIT_total` from `comb_s{0,1,2,3}`.
- `python -m mxp_tools rmw-gen --help` shows the new sub-command.
- `CLAUDE.md` reflects the new directory layout.
- `bash sim/run_int_to_fp32.sh` and `bash sim/run_fp32_adder.sh` each report `ALL TESTS PASSED`.
- `git log --oneline` shows 9 small commits (one per task).

## Out of scope (future plans)

- External scheduler RTL (column → bank mapping, fire timing, ping-pong swap, K-tile counter, first-tile forwarding, 1RW arbitration).
- Integrating RMW into `GEMM.v` (top-level wiring; currently `GEMM.v` is still the unmodified MXP TOP).
- Full GEMM + SRAM end-to-end TB.
- Timing closure / synthesis result reporting.
