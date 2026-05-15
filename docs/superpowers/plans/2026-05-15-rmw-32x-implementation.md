# RMW 32× 확장 (col-parallel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RMW 를 col-parallel 32 instance 로 확장하고, SRAM wrapper 를 per-bank 독립 port 노출형 (`sram_1rw_banked_mp.v`) 으로 교체. 9 정밀도 모드 모두에서 기존 1-RMW 결과와 bit-exact 일치 유지.

**Architecture:** col j 의 RMW 가 자기 bank j (NB=32) 만 건드림 — 매핑 `bank = flat % 32`, `128 % 32 = 0` 이라 모든 mode 에서 충돌 0. wrapper 는 leaf `sram_1rw` 32 개를 generate-for 로 묶고 각 bank 포트를 flat bus 로 노출. TB 는 single FIFO → 32 per-col FIFO + 32-stream parallel drain.

**Tech Stack:** Verilog-2001 (RTL), SystemVerilog OK in TB, Python 3 + pytest (MXP_Tools), XSim (Vivado 2024.1) batch run via bash scripts.

**Spec:** `docs/superpowers/specs/2026-05-15-rmw-32x-design.md` (커밋 `7e1b794`).

---

## File Structure

### Created
| Path | Responsibility |
|---|---|
| `gemm_sram.srcs/sources_1/new/sram_1rw_banked_mp.v` | per-bank port-exposed 1RW SRAM wrapper (NB × {CEB, WEB, A, D, WMASK, Q} flat bus) |
| `tb/sram_1rw_banked_mp_tb.v` | new wrapper 단위 회귀 — bank 별 독립 write/read |
| `sim/run_sram_mp.sh` | new wrapper TB 의 XSim 배치 스크립트 |

### Modified
| Path | What changes |
|---|---|
| `gemm_sram.srcs/sources_1/new/gemm_sram_top.v` | RMW 1→32, sram_1rw_banked → sram_1rw_banked_mp(NB=32, depth=1024), 포트 32× expand |
| `tb/gemm_sram_top_tb.v` | DUT port wiring, FIFO per-col, capture push-to-col, drain state-machine per-col, INIT zero-prime parallel, DUMP 32 banks |
| `sim/run_top_elab.sh` | file list 에 `sram_1rw_banked_mp.v` 추가 |
| `sim/run_integration_one.sh` | bank list 16→32, compare `--layout` 인자 갱신 |
| `MXP_Tools/mxp_tools/hwio.py` | `interleaved_row_major_32bank` 추가 |
| `MXP_Tools/mxp_tools/cli.py` | `--layout` choices 에 32bank 추가 |
| `MXP_Tools/tests/test_hwio_interleaved.py` | 32-bank round-trip 테스트 추가 |
| `CLAUDE.md` + `docs/next-session-kickoff.md` | "RMW instance" 잠정값 1 → 32 갱신 |

---

## Task 1 — `sram_1rw_banked_mp.v` 작성 + 단위 회귀 PASS

**Files:**
- Create: `gemm_sram.srcs/sources_1/new/sram_1rw_banked_mp.v`
- Create: `tb/sram_1rw_banked_mp_tb.v`
- Create: `sim/run_sram_mp.sh`

### Step 1.1 — TB 먼저 작성 (실패 상태)

- [ ] **TB 작성** (실패하는 테스트가 먼저)

Create `tb/sram_1rw_banked_mp_tb.v`:

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// sram_1rw_banked_mp_tb — per-bank port-exposed 1RW SRAM wrapper 단위 검증.
//
// 검증 목적:
//   sram_1rw_banked_mp 가 NUM_BANKS 개의 독립 1RW bank port 를 노출하고,
//   각 bank 가 같은 cycle 에 서로 다른 word 를 read 또는 write 할 수 있음을
//   확인. 32 RMW × 32 bank conflict-free 매핑의 RTL 기반.
//
// 검증 내용:
//   1) bank 마다 다른 데이터를 동시 write
//   2) 1 cycle 후 동시 read (PIPELINE=0 leaf 의 read latency)
//   3) 각 bank Q 가 자기 write 데이터와 일치 (cross-bank 오염 없음)
//   4) idle bank (CEB=1) 가 active bank 의 동작에 영향 없음
//
// 동작 의도:
//   기존 sram_1rw_banked (single A/D/Q + 내부 mux) 와 달리, 이 wrapper 는
//   bank 별 독립 포트라 동시 다중 R/W 가능. 합성 후 매 cycle bandwidth =
//   NUM_BANKS × (leaf 1RW port). 외부 callsite (gemm_sram_top) 가 bank
//   매핑 책임 (INTERLEAVED/SEQUENTIAL 의미는 호출자가 결정).
//
// 회귀 게이트: `bash sim/run_sram_mp.sh` → "ALL <N> TESTS PASSED".
//////////////////////////////////////////////////////////////////////////////

module sram_1rw_banked_mp_tb;

    // 작은 config (sram repo 의 TB sizing 컨벤션) — fast sim
    localparam DATA_WIDTH = 32;
    localparam NUM_BANKS  = 4;
    localparam BANK_DEPTH = 16;
    localparam BANK_ADDR_WIDTH = 4;     // clog2(16)

    reg CLK = 0;
    always #5 CLK = ~CLK;

    reg  [NUM_BANKS-1:0]                       CEB;
    reg  [NUM_BANKS-1:0]                       WEB;
    reg  [NUM_BANKS*BANK_ADDR_WIDTH-1:0]       A;
    reg  [NUM_BANKS*DATA_WIDTH-1:0]            D;
    reg  [NUM_BANKS*DATA_WIDTH-1:0]            WMASK;
    wire [NUM_BANKS*DATA_WIDTH-1:0]            Q;

    sram_1rw_banked_mp #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_BANKS  (NUM_BANKS),
        .BANK_DEPTH (BANK_DEPTH),
        .PIPELINE   (0)
    ) dut (
        .CLK(CLK), .CEB(CEB), .WEB(WEB),
        .A(A), .D(D), .WMASK(WMASK), .Q(Q)
    );

    integer pass = 0, fail = 0;
    integer b, w;
    reg [DATA_WIDTH-1:0] expected, got;

    // 슬라이스 헬퍼
    task automatic set_bank_A;
        input integer bi;
        input [BANK_ADDR_WIDTH-1:0] addr;
        begin
            A[bi*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH] = addr;
        end
    endtask
    task automatic set_bank_D;
        input integer bi;
        input [DATA_WIDTH-1:0] data;
        begin
            D[bi*DATA_WIDTH +: DATA_WIDTH] = data;
        end
    endtask
    function [DATA_WIDTH-1:0] get_bank_Q;
        input integer bi;
        begin
            get_bank_Q = Q[bi*DATA_WIDTH +: DATA_WIDTH];
        end
    endfunction

    initial begin
        // 초기 idle
        CEB   = {NUM_BANKS{1'b1}};
        WEB   = {NUM_BANKS{1'b1}};
        A     = 0;
        D     = 0;
        WMASK = {NUM_BANKS*DATA_WIDTH{1'b1}};
        @(posedge CLK);

        // ─── Scenario 1: bank 마다 다른 데이터 동시 write ──────────────
        @(negedge CLK);
        for (b = 0; b < NUM_BANKS; b = b + 1) begin
            set_bank_A(b, 4'd0);
            set_bank_D(b, 32'hDEADBE00 + b);  // bank 마다 고유 데이터
        end
        CEB = {NUM_BANKS{1'b0}};
        WEB = {NUM_BANKS{1'b0}};
        @(posedge CLK);

        // ─── Scenario 2: 동시 read 발사 ────────────────────────────────
        @(negedge CLK);
        WEB = {NUM_BANKS{1'b1}};
        for (b = 0; b < NUM_BANKS; b = b + 1) set_bank_A(b, 4'd0);
        @(posedge CLK);

        // ─── Scenario 3: Q 검증 (PIPELINE=0 → 동일 cycle 의 Q 가 유효) ─
        #1;
        for (b = 0; b < NUM_BANKS; b = b + 1) begin
            expected = 32'hDEADBE00 + b;
            got      = get_bank_Q(b);
            if (got === expected) begin
                pass = pass + 1;
                $display("[PASS] bank %0d Q = %h", b, got);
            end else begin
                fail = fail + 1;
                $display("[FAIL] bank %0d Q = %h, expected %h", b, got, expected);
            end
        end

        // ─── Scenario 4: idle bank 영향 없음 ───────────────────────────
        // bank 0 만 active, 나머지 CEB=1. bank 0 에 새 값 write 후
        // 읽었을 때 다른 bank 의 Q 가 변하지 않음.
        @(negedge CLK);
        CEB = {NUM_BANKS{1'b1}}; CEB[0] = 1'b0;
        WEB = {NUM_BANKS{1'b1}}; WEB[0] = 1'b0;
        set_bank_A(0, 4'd1);
        set_bank_D(0, 32'hCAFEBABE);
        @(posedge CLK);
        @(negedge CLK);
        WEB[0] = 1'b1;
        set_bank_A(0, 4'd1);
        @(posedge CLK);
        #1;
        if (get_bank_Q(0) === 32'hCAFEBABE) begin
            pass = pass + 1;
            $display("[PASS] bank 0 isolated write OK");
        end else begin
            fail = fail + 1;
            $display("[FAIL] bank 0 isolated write = %h", get_bank_Q(0));
        end

        // ─── 최종 보고 ────────────────────────────────────────────────
        if (fail == 0) $display("sram_1rw_banked_mp_tb: ALL %0d TESTS PASSED", pass);
        else           $display("sram_1rw_banked_mp_tb: %0d PASS, %0d FAIL", pass, fail);
        $finish;
    end

endmodule
```

- [ ] **`sim/run_sram_mp.sh` 작성**

```bash
#!/bin/bash
# sim/run_sram_mp.sh — sram_1rw_banked_mp 단위 회귀.
set -e
cd "$(dirname "$0")/.."
BUILD=sim/build/sram_mp
mkdir -p "$BUILD"
cd "$BUILD"

SRC_ROOT="../../../gemm_sram.srcs/sources_1"

xvlog -sv \
    $SRC_ROOT/imports/Desktop/sram/rtl/sram_1rw.v \
    $SRC_ROOT/new/sram_1rw_banked_mp.v \
    ../../../tb/sram_1rw_banked_mp_tb.v

xelab -L work sram_1rw_banked_mp_tb -snapshot sram_mp_snap
xsim sram_mp_snap -runall
```

- [ ] **실행해서 실패 확인** (RTL 이 아직 없음)

Run: `chmod +x sim/run_sram_mp.sh && bash sim/run_sram_mp.sh`
Expected: xvlog 또는 xelab 단계에서 fail (`sram_1rw_banked_mp` 모듈 없음).

### Step 1.2 — RTL 작성하여 통과시킴

- [ ] **`gemm_sram.srcs/sources_1/new/sram_1rw_banked_mp.v` 작성**

```verilog
//=============================================================================
// sram_1rw_banked_mp.v - Per-bank port-exposed 1RW SRAM wrapper (Verilog-2001)
//
// `sram_1rw_banked.v` (mux 형) 와 달리, 각 bank 의 CEB/WEB/A/D/WMASK/Q 를
// flat bus 로 외부에 노출. NUM_BANKS 개 독립 1RW port → 매 cycle bandwidth =
// NUM_BANKS × (leaf 1RW port). INTERLEAVED/SEQUENTIAL 의미는 호출자가 결정
// (wrapper 레벨에서 BANK_STRATEGY 없음).
//
// gemm_sram_top 의 col j → bank j 매핑 (col-parallel RMW 32×) 에 사용.
//
// Spec : docs/superpowers/specs/2026-05-15-rmw-32x-design.md
// Style: Pure Verilog-2001 (sram repo 컨벤션 — packed array 안 씀).
//=============================================================================
`timescale 1ns/1ps

module sram_1rw_banked_mp (CLK, CEB, WEB, A, D, WMASK, Q);

    parameter DATA_WIDTH = 32;
    parameter NUM_BANKS  = 32;
    parameter BANK_DEPTH = 1024;
    parameter PIPELINE   = 0;       // leaf PIPELINE 그대로 전달 (0=1cy, 1=2cy)

    // clog2 (sram repo 컨벤션)
    function integer clog2;
        input integer value;
        integer i;
        begin
            clog2 = 0;
            for (i = value - 1; i > 0; i = i >> 1)
                clog2 = clog2 + 1;
        end
    endfunction

    localparam BANK_ADDR_WIDTH = clog2(BANK_DEPTH);

    input                                          CLK;
    input  [NUM_BANKS-1:0]                         CEB;
    input  [NUM_BANKS-1:0]                         WEB;
    input  [NUM_BANKS*BANK_ADDR_WIDTH-1:0]         A;
    input  [NUM_BANKS*DATA_WIDTH-1:0]              D;
    input  [NUM_BANKS*DATA_WIDTH-1:0]              WMASK;
    output [NUM_BANKS*DATA_WIDTH-1:0]              Q;

    // Sanity (sim-only)
    initial begin
        if (NUM_BANKS < 2 || (NUM_BANKS & (NUM_BANKS - 1)) != 0) begin
            $display("ERROR: NUM_BANKS must be a power of 2 >= 2 (got %0d)", NUM_BANKS);
            $finish;
        end
        if (PIPELINE != 0 && PIPELINE != 1) begin
            $display("ERROR: PIPELINE must be 0 or 1 (got %0d)", PIPELINE);
            $finish;
        end
    end

    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : g_bank
            sram_1rw #(
                .DATA_WIDTH (DATA_WIDTH),
                .DEPTH      (BANK_DEPTH),
                .PIPELINE   (PIPELINE),
                .INIT_FILE  ("")
            ) u_bank (
                .CLK   (CLK),
                .CEB   (CEB[i]),
                .WEB   (WEB[i]),
                .A     (A[i*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH]),
                .D     (D[i*DATA_WIDTH +: DATA_WIDTH]),
                .WMASK (WMASK[i*DATA_WIDTH +: DATA_WIDTH]),
                .Q     (Q[i*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

endmodule
```

- [ ] **실행해서 PASS 확인**

Run: `bash sim/run_sram_mp.sh`
Expected (마지막 줄): `sram_1rw_banked_mp_tb: ALL 5 TESTS PASSED`

### Step 1.3 — Commit

- [ ] **Commit**

```bash
git add gemm_sram.srcs/sources_1/new/sram_1rw_banked_mp.v \
        tb/sram_1rw_banked_mp_tb.v \
        sim/run_sram_mp.sh
git commit -m "Add sram_1rw_banked_mp.v — per-bank port-exposed 1RW wrapper

NB 개 독립 1RW bank port (flat bus). 32 RMW × 32 bank conflict-free
매핑의 RTL 기반. 기존 sram_1rw_banked.v (mux 형) 와 공존.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2 — MXP_Tools `interleaved_row_major_32bank` 추가

**Files:**
- Modify: `MXP_Tools/mxp_tools/hwio.py:147` (after `interleaved_row_major_16bank`)
- Modify: `MXP_Tools/mxp_tools/cli.py:118-125, 190` (resolver + argparse)
- Modify: `MXP_Tools/tests/test_hwio_interleaved.py` (add 32-bank tests)

### Step 2.1 — pytest 먼저 (실패 상태)

- [ ] **테스트 케이스 추가**

Modify `MXP_Tools/tests/test_hwio_interleaved.py` — 파일 끝에 추가:

```python
def test_interleaved_row_major_32bank_round_trip():
    """col-parallel RMW design: bank = flat % 32, word = flat // 32."""
    M, N = 128, 128
    flat = lambda m, n: m * N + n
    bank = lambda f: f % 32
    word = lambda f: f // 32

    for m, n in [(0, 0), (0, 31), (0, 32), (1, 0), (127, 127), (63, 31), (5, 96)]:
        f = flat(m, n)
        b = bank(f)
        w = word(f)
        result = hwio.interleaved_row_major_32bank(b, w, M, N)
        assert result == (m, n), f"({m},{n})→bank{b}/word{w}→{result}"


def test_interleaved_row_major_32bank_out_of_range():
    assert hwio.interleaved_row_major_32bank(0, 9999, 128, 128) is None
```

- [ ] **pytest 실행 → 실패 확인**

Run: `cd MXP_Tools && python -m pytest tests/test_hwio_interleaved.py -v`
Expected: 2 FAIL (`AttributeError: module 'mxp_tools.hwio' has no attribute 'interleaved_row_major_32bank'`).

### Step 2.2 — hwio.py 에 함수 추가하여 통과시킴

- [ ] **함수 추가**

Insert into `MXP_Tools/mxp_tools/hwio.py` right after the `interleaved_row_major_16bank` function (after L160):

```python
def interleaved_row_major_32bank(bank_idx, word_offset, M, N):
    """C[m,n] = SRAM[bank=flat%32, word=flat//32] where flat = m*N+n.

    col-parallel RMW (32 instance) 의 매핑. col j 의 RMW 가 자기 bank j
    만 건드림 — 128 % 32 = 0 이므로 모든 mode (A8/A4/A2) 에서 충돌 0.

    Used by: gemm_sram integration testbench (32-RMW phase)
    Spec   : docs/superpowers/specs/2026-05-15-rmw-32x-design.md §4
    """
    flat = word_offset * 32 + bank_idx
    if flat >= M * N:
        return None
    return divmod(flat, N)
```

- [ ] **pytest 재실행 → PASS 확인**

Run: `cd MXP_Tools && python -m pytest tests/test_hwio_interleaved.py -v`
Expected: 4 passed (기존 2 + 신규 2).

### Step 2.3 — CLI 의 --layout choices 갱신

- [ ] **`_resolve_mapping` 에 분기 추가**

Modify `MXP_Tools/mxp_tools/cli.py:118-125`:

```python
def _resolve_mapping(args, M, N):
    if args.layout == "single":
        return hwio.default_single_bank_row_major
    if args.layout == "split-rows":
        return hwio.default_banks_split_rows(args.n_banks)
    if args.layout == "interleaved_row_major_16bank":
        return hwio.interleaved_row_major_16bank
    if args.layout == "interleaved_row_major_32bank":
        return hwio.interleaved_row_major_32bank
    raise ValueError(f"unknown layout {args.layout}")
```

- [ ] **`--layout` choices 갱신**

Modify `MXP_Tools/mxp_tools/cli.py:190`:

```python
    g.add_argument("--layout", choices=("single", "split-rows",
                                        "interleaved_row_major_16bank",
                                        "interleaved_row_major_32bank"),
                   default="single")
```

- [ ] **전체 pytest 재확인**

Run: `cd MXP_Tools && python -m pytest tests/ -q`
Expected: `45 passed` (기존 43 + 신규 2).

### Step 2.4 — Commit

- [ ] **Commit**

```bash
git add MXP_Tools/mxp_tools/hwio.py \
        MXP_Tools/mxp_tools/cli.py \
        MXP_Tools/tests/test_hwio_interleaved.py
git commit -m "MXP_Tools: add interleaved_row_major_32bank layout

32-bank col-parallel RMW 매핑 (bank = flat % 32, word = flat // 32).
기존 16-bank 매핑과 공존. pytest 2 케이스 추가.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3 — `gemm_sram_top.v` 32-RMW + 32-bank wrapper

**Files:**
- Modify: `gemm_sram.srcs/sources_1/new/gemm_sram_top.v` (full rewrite of port list + body)
- Modify: `sim/run_top_elab.sh` (add new wrapper to file list)

### Step 3.1 — `gemm_sram_top.v` 재작성

- [ ] **새 파일 내용**

기존 단일 RMW + 단일 sram_1rw_banked 인스턴스를 32-fold 로 확장. 전체 교체:

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top — GEMM + RMW[32] + sram_1rw_banked_mp 묶음 (pure structural).
// 모든 control 신호는 TB 가 driving (controller 없음).
//
// col j 의 RMW[j] 가 자기 bank[j] 만 건드림 (col-parallel). 매 cycle 최대
// 32 개의 독립 R/W 가능. 매핑 책임은 호출자 (TB).
//
// Spec: docs/superpowers/specs/2026-05-15-rmw-32x-design.md
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top #(
    parameter NUM_BANKS  = 32,
    parameter BANK_DEPTH = 1024
)(
    input  wire        clk,
    input  wire        rst,

    // ───── GEMM stimuli (passthrough — GEMM.v 의 모든 포트) ─────
    input  wire [32-1:0]                     in_a,
    input  wire signed [32*8-1:0]            in_b,
    input  wire [32-1:0]                     in_control,
    input  wire [32-1:0]                     in_loadEN,
    input  wire [1:0]                        in_Station_control,
    input  wire [4*8-1:0]                    in_Scale_Activation,
    input  wire                              in_station_control,
    input  wire                              in_station_loadEN,
    input  wire                              in_start_accumulate,
    input  wire [1:0]                        in_Wcontrol,
    input  wire [7:0]                        in_scale_weight,
    output wire [32*60-1:0]                  out_accumulate,
    output wire [32*36-1:0]                  out_scale,
    output wire [32-1:0]                     out_fire,

    // ───── RMW 32 lane input (TB 가 각 lane 별 직접 발사) ─────
    input  wire [NUM_BANKS*32-1:0]           rmw_in_GEMM,    // flat slice
    input  wire [NUM_BANKS*9-1:0]            rmw_scale,      // flat slice
    output wire [NUM_BANKS*32-1:0]           rmw_out_RMW,    // probe (flat)

    // ───── SRAM 32 bank control (TB 가 각 bank 별 직접 발사) ─────
    input  wire [NUM_BANKS-1:0]                                CEB,
    input  wire [NUM_BANKS-1:0]                                WEB,
    input  wire [NUM_BANKS*$clog2(BANK_DEPTH)-1:0]             A,
    input  wire [NUM_BANKS*32-1:0]                             WMASK,
    input  wire                                                sram_D_use_zero,  // 모든 bank 공통
    output wire [NUM_BANKS*32-1:0]                             Q                 // probe (flat)
);

    // ─── GEMM ──────────────────────────────────────────────────────
    GEMM u_gemm (
        .clk(clk), .rst(rst),
        .in_a(in_a), .in_b(in_b),
        .in_control(in_control), .in_loadEN(in_loadEN),
        .in_Station_control(in_Station_control),
        .in_Scale_Activation(in_Scale_Activation),
        .in_station_control(in_station_control),
        .in_station_loadEN(in_station_loadEN),
        .in_start_accumulate(in_start_accumulate),
        .in_Wcontrol(in_Wcontrol),
        .in_scale_weight(in_scale_weight),
        .out_accumulate(out_accumulate),
        .out_scale(out_scale),
        .out_fire(out_fire)
    );

    // ─── RMW[32] + SRAM bank D-mux per col ─────────────────────────
    wire [NUM_BANKS*32-1:0] sram_D_w;
    genvar c;
    generate
        for (c = 0; c < NUM_BANKS; c = c + 1) begin : g_lane
            wire [31:0] rmw_out_c;
            RMW u_rmw (
                .clk    (clk),
                .rst    (rst),
                .in_SRAM(Q          [c*32 +: 32]),
                .in_GEMM(rmw_in_GEMM[c*32 +: 32]),
                .scale  (rmw_scale  [c*9  +: 9 ]),
                .out_RMW(rmw_out_c)
            );
            assign rmw_out_RMW[c*32 +: 32] = rmw_out_c;
            assign sram_D_w  [c*32 +: 32] = sram_D_use_zero ? 32'h00000000 : rmw_out_c;
        end
    endgenerate

    // ─── SRAM (per-bank port) ──────────────────────────────────────
    sram_1rw_banked_mp #(
        .DATA_WIDTH (32),
        .NUM_BANKS  (NUM_BANKS),
        .BANK_DEPTH (BANK_DEPTH),
        .PIPELINE   (0)
    ) u_sram (
        .CLK   (clk),
        .CEB   (CEB),
        .WEB   (WEB),
        .A     (A),
        .D     (sram_D_w),
        .WMASK (WMASK),
        .Q     (Q)
    );

endmodule
```

### Step 3.2 — elab smoke 스크립트에 새 wrapper 추가

- [ ] **`sim/run_top_elab.sh` 수정**

`$SRC_ROOT/new/RMW.v` 다음 줄 (즉 `$SRC_ROOT/new/GEMM.v` 직전) 에 `$SRC_ROOT/new/sram_1rw_banked_mp.v` 추가:

```bash
xvlog -sv \
    $HF_ROOT/HardFloatBundle.v \
    $SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
    $SRC_ROOT/imports/Desktop/sram/rtl/*.v \
    $SRC_ROOT/new/int_to_fp32.v \
    $SRC_ROOT/new/fp32_adder.v \
    $SRC_ROOT/new/RMW.v \
    $SRC_ROOT/new/sram_1rw_banked_mp.v \
    $SRC_ROOT/new/GEMM.v \
    $SRC_ROOT/new/gemm_sram_top.v
```

- [ ] **elab smoke 실행**

Run: `bash sim/run_top_elab.sh`
Expected: `PASS: gemm_sram_top elaborates cleanly`

### Step 3.3 — Commit

- [ ] **Commit**

```bash
git add gemm_sram.srcs/sources_1/new/gemm_sram_top.v sim/run_top_elab.sh
git commit -m "gemm_sram_top: 32-RMW + sram_1rw_banked_mp (col-parallel)

RMW 1 → 32 (generate-for, col-parallel). SRAM wrapper 를
sram_1rw_banked_mp (NB=32, depth=1024) 으로 교체. 포트 시그니처가
32× flat bus 로 확장됨. elab smoke PASS.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4 — `gemm_sram_top_tb.v` 32-stream capture/drain

이 TB 는 ~870 라인이라 수정 폭이 큼. 변경 부위를 다섯 덩어리로 나눠 순차 수정.

**Files:**
- Modify: `tb/gemm_sram_top_tb.v` (포트 wiring, FIFO, capture, drain, INIT, DUMP)

### Step 4.1 — 헤더 한글 주석 갱신

- [ ] **헤더 갱신**

`tb/gemm_sram_top_tb.v` 의 top comment block 을 새 design 에 맞게 갱신:

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_tb — gemm_sram_top (32-RMW col-parallel) 통합 검증 TB.
//
// 검증 목적:
//   GEMM (MXP 32×32 bit-serial systolic) + RMW[32] (col-parallel) +
//   sram_1rw_banked_mp (32 bank, per-bank port) 의 end-to-end 데이터
//   패스가 9 가지 정밀도 조합 (A∈{2,4,8} × W∈{2,4,8}) 모두에서
//   MXP_Tools 골든 GEMM 과 bit-exact 일치하는지 확인. col j → bank j
//   매핑 (충돌 0) 의 RTL 검증을 겸함.
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8 (Activation 정밀도)
//   B_PREC    : 2 / 4 / 8 (Weight 정밀도, MXP 의 "in_a" 라인)
//   WORK_DIR  : MXP_Tools hex 입력 디렉토리 (필수)
//   DUMP_DIR  : SRAM .mem dump 출력 디렉토리 (필수, 32 파일 출력)
//
// 전체 흐름 (32-RMW 병렬 디스패치):
//   1) INIT     — 리셋 + 32 bank parallel zero-prime (1024 cycle).
//   2) LOAD     — $readmemh (a_input_BS, b_input, a_scale, b_scale).
//   3) CONFIG   — {A_PREC,B_PREC} 으로 모드별 상수/제어 코드 세팅.
//   4) DRIVE    — Stage 2-A, 2-B, 3+4 (MAC), 5 (tail). 캡처 블록이
//                 매 out_fire[c] rising 마다 per-col FIFO[c] 에 push.
//   5) DRAIN    — 32 col 동시 진행. col c 의 always 블록이 자기 FIFO[c]
//                 를 자기 RMW[c] + 자기 bank[c] port 로 흘림 (R→conv→add→W).
//   6) DUMP     — $writememh 로 32 bank 각각 0..511 워드를 .mem 파일로.
//                 외부 compare 가 MXP_Tools golden npz 와 비트 단위 비교.
//
// 회귀 게이트: `bash sim/run_integration_sweep.sh` → "ALL 9 MODES PASSED".
//////////////////////////////////////////////////////////////////////////////
```

### Step 4.2 — DUT 인스턴스 + 포트 wiring 변경

- [ ] **TB 안의 SRAM/RMW 신호 32×, DUT 인스턴스 갱신**

찾아 바꿀 부분 (현재 TB L110-L148 부근):

```verilog
    // RMW (32 col)
    reg  [32*32-1:0]          rmw_in_GEMM       = 0;
    reg  [32*9-1:0]           rmw_scale         = 0;
    wire [32*32-1:0]          rmw_out_RMW;

    // SRAM (32 bank, depth=1024 → AW=10)
    localparam integer NB        = 32;
    localparam integer BD        = 1024;
    localparam integer AW        = 10;       // clog2(1024)

    reg  [NB-1:0]             sram_CEB        = {NB{1'b1}};
    reg  [NB-1:0]             sram_WEB        = {NB{1'b1}};
    reg  [NB*AW-1:0]          sram_A          = 0;
    reg  [NB*32-1:0]          sram_WMASK      = {NB*32{1'b1}};
    reg                       sram_D_use_zero = 1'b1;
    wire [NB*32-1:0]          sram_Q;

    // ─── DUT 인스턴스 ──────────────────────────────────────────────
    gemm_sram_top #(
        .NUM_BANKS  (NB),
        .BANK_DEPTH (BD)
    ) u_top (
        .clk(clk), .rst(rst),
        .in_a(in_a), .in_b(in_b),
        .in_control(in_control), .in_loadEN(in_loadEN),
        .in_Station_control(in_Station_control),
        .in_Scale_Activation(in_Scale_Activation),
        .in_station_control(in_station_control),
        .in_station_loadEN(in_station_loadEN),
        .in_start_accumulate(in_start_accumulate),
        .in_Wcontrol(in_Wcontrol),
        .in_scale_weight(in_scale_weight),
        .out_accumulate(out_accumulate),
        .out_scale(out_scale),
        .out_fire(out_fire),
        .rmw_in_GEMM(rmw_in_GEMM),
        .rmw_scale(rmw_scale),
        .rmw_out_RMW(rmw_out_RMW),
        .CEB(sram_CEB), .WEB(sram_WEB),
        .A(sram_A), .WMASK(sram_WMASK),
        .sram_D_use_zero(sram_D_use_zero),
        .Q(sram_Q)
    );
```

기존 `sram_write_zero` / `sram_idle` task 는 삭제 (drain 안에서 직접 driving).

### Step 4.3 — FIFO 를 per-col 로 분리, capture 블록 push 위치 갱신

- [ ] **FIFO 선언 변경**

현재 (L186-L190):
```verilog
    reg [31:0]   fifo_int   [0:65535];
    reg [8:0]    fifo_scale [0:65535];
    reg [18:0]   fifo_addr  [0:65535];
    integer      fifo_wp;
    integer      fifo_rp;
```

수정:
```verilog
    // ─── Fire 캡처 per-col FIFO ────────────────────────────────────
    // col 당 최대 2048 entry (모든 모드 공통: A8=2048×1, A4=1024×2, A2=512×4).
    localparam integer FIFO_DEPTH = 2048;
    reg [31:0]   fifo_int   [0:31][0:FIFO_DEPTH-1];
    reg [8:0]    fifo_scale [0:31][0:FIFO_DEPTH-1];
    reg [18:0]   fifo_addr  [0:31][0:FIFO_DEPTH-1];
    integer      fifo_wp [0:31];   // per-col write pointer
    integer      fifo_rp [0:31];   // per-col read pointer (drain 시)
```

- [ ] **Capture push 위치 변경**

L221-L334 의 always 블록에서, `fifo_wp` 전역 카운터 사용을 `fifo_wp[ci]` 로 교체. 각 `fifo_int[fifo_wp] <= ...` 를 `fifo_int[ci][fifo_wp[ci]] <= ...` 등으로 변경.

A8 케이스 (L245-L255) 예시:
```verilog
                        8: begin
                            acc_lane_a8 = $signed(out_accumulate[60*ci +: 21]);
                            acc_int32   = {{11{acc_lane_a8[20]}}, acc_lane_a8};
                            sc_lane     = out_scale[36*ci +: 9];
                            ng_dec      = n_t_dec * 32 + ci;
                            flat_dec    = m_g * N_DIM + ng_dec;
                            // 검증 가드: bank = flat % 32 == ci 여야 함
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch A8 col=%0d flat=%0d", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;
                        end
```

A4 (top + bot) 와 A2 (lane0..3) 도 동일 패턴 — 각 lane 의 push 를 `fifo_*[ci][fifo_wp[ci]]` 로, push 직후 `fifo_wp[ci] = fifo_wp[ci] + 1`. assert 가드도 각 lane 마다.

(A4 top 예시:)
```verilog
                        4: begin
                            // top lane (s1_a)
                            acc_lane_a4t = $signed(out_accumulate[60*ci+18 +: 18]);
                            acc_int32    = {{14{acc_lane_a4t[17]}}, acc_lane_a4t};
                            sc_lane      = out_scale[36*ci+9 +: 9];
                            ng_dec       = n_t_dec * 64 + 32 + ci;
                            flat_dec     = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch A4t col=%0d flat=%0d", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;

                            // bot lane (s1_b)
                            acc_lane_a4b = $signed(out_accumulate[60*ci +: 18]);
                            acc_int32    = {{14{acc_lane_a4b[17]}}, acc_lane_a4b};
                            sc_lane      = out_scale[36*ci +: 9];
                            ng_dec       = n_t_dec * 64 + 0 + ci;
                            flat_dec     = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch A4b col=%0d flat=%0d", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;
                        end
```

A2 의 lane0..3 도 동일 패턴 — `fifo_*[ci][fifo_wp[ci]]` + assert.

- [ ] **fifo_wp 초기화 변경**

`initial` 블록에서 `fifo_wp = 0;` 단일 라인이 있다면 (현재 코드 어딘가) 다음으로 교체:

```verilog
    integer init_i;
    initial begin
        for (init_i = 0; init_i < 32; init_i = init_i + 1) begin
            fifo_wp[init_i] = 0;
            fifo_rp[init_i] = 0;
        end
    end
```

### Step 4.4 — INIT (zero-prime) 32-bank parallel write 로 변경

- [ ] **기존 INIT 루프 교체**

기존 (단일 SRAM, 16384 워드 직렬):
```verilog
for (init_addr = 0; init_addr < 16384; init_addr = init_addr + 1) sram_write_zero(init_addr);
```

새로 (32 bank, 1024 word 동시 write — 각 bank 의 word 0..1023):
```verilog
    task init_zero_prime;
        integer w, bi;
        begin
            sram_D_use_zero <= 1'b1;
            sram_WMASK      <= {NB*32{1'b1}};
            for (w = 0; w < BD; w = w + 1) begin
                @(posedge clk);
                sram_CEB <= {NB{1'b0}};       // 모든 bank active
                sram_WEB <= {NB{1'b0}};       // 모든 bank write
                for (bi = 0; bi < NB; bi = bi + 1)
                    sram_A[bi*AW +: AW] <= w[AW-1:0];
            end
            @(posedge clk);
            sram_CEB <= {NB{1'b1}};
            sram_WEB <= {NB{1'b1}};
        end
    endtask
```

(호출 위치는 기존 init 루프 자리에서 `init_zero_prime();` 로 교체.)

### Step 4.5 — DRAIN: 32-stream parallel state machine

- [ ] **기존 `drain_fifo` task 삭제, `drain_prime` 도 삭제**

(per-col always 가 reset 직후부터 IDLE 상태로 명확하므로 prime 불필요. 단 PRIME_CYC 만큼 idle 대기는 유지 — 그 동안 capture 가 끝난 fire 가 settle.)

- [ ] **per-col drain FSM 추가**

DUT 인스턴스 아래에 generate 블록 추가:

```verilog
    // ─── per-col drain FSM (32 stream parallel) ────────────────────
    // State: 0=IDLE, 1=READ, 2=WAIT_R, 3=DRIVE_RMW, 4=WAIT_RMW,
    //        5=WRITE, 6=WRITE_SETTLE
    // FSM 은 fifo_rp[c] < fifo_wp[c] 일 때만 동작. drain_enable 신호로
    // 전체 동시 gating.
    reg drain_enable;
    initial drain_enable = 1'b0;

    reg [3:0]  drain_state [0:31];
    reg [3:0]  drain_wait  [0:31];     // RMW_WAIT_CYC 카운터 (0..7)
    reg [18:0] drain_addr  [0:31];

    initial begin
        for (init_i = 0; init_i < 32; init_i = init_i + 1) begin
            drain_state[init_i] = 4'd0;
            drain_wait [init_i] = 4'd0;
            drain_addr [init_i] = 19'd0;
        end
    end

    // RMW pipeline wait = L_CONV(2) + L_ADD(3) + slack(3) = 8 cy
    // (single-RMW 시절 RMW_WAIT_CYC 와 동일.)
    localparam integer DRAIN_RMW_WAIT = 8;

    genvar dc;
    generate
        for (dc = 0; dc < 32; dc = dc + 1) begin : g_drain
            always @(posedge clk) begin
                if (rst) begin
                    drain_state[dc] <= 4'd0;
                    drain_wait [dc] <= 4'd0;
                end else if (drain_enable) begin
                    case (drain_state[dc])
                        4'd0: begin   // IDLE — fifo 에 데이터 있으면 READ 발사
                            if (fifo_rp[dc] < fifo_wp[dc]) begin
                                drain_addr[dc]                      <= fifo_addr[dc][fifo_rp[dc]];
                                sram_CEB[dc]                        <= 1'b0;
                                sram_WEB[dc]                        <= 1'b1;
                                sram_A[dc*AW +: AW]                 <= fifo_addr[dc][fifo_rp[dc]] >> 5;
                                drain_state[dc]                     <= 4'd1;
                            end else begin
                                sram_CEB[dc]                        <= 1'b1;
                                sram_WEB[dc]                        <= 1'b1;
                            end
                        end
                        4'd1: begin   // READ → WAIT_R (1 cy: PIPELINE=0 leaf)
                            sram_CEB[dc] <= 1'b1;
                            drain_state[dc] <= 4'd2;
                        end
                        4'd2: begin   // WAIT_R — sram_Q[dc] 가 이번 cycle 부터 유효
                            rmw_in_GEMM[dc*32 +: 32] <= fifo_int  [dc][fifo_rp[dc]];
                            rmw_scale  [dc*9  +: 9 ] <= fifo_scale[dc][fifo_rp[dc]];
                            drain_state[dc] <= 4'd3;
                        end
                        4'd3: begin   // DRIVE_RMW (이미 driven, 다음 cycle 부터 wait)
                            drain_wait [dc] <= 4'd0;
                            drain_state[dc] <= 4'd4;
                        end
                        4'd4: begin   // WAIT_RMW (L_CONV+L_ADD+slack = 8 cy)
                            if (drain_wait[dc] == DRAIN_RMW_WAIT[3:0] - 1) begin
                                drain_state[dc] <= 4'd5;
                            end else begin
                                drain_wait[dc] <= drain_wait[dc] + 1;
                            end
                        end
                        4'd5: begin   // WRITE
                            sram_CEB[dc]                        <= 1'b0;
                            sram_WEB[dc]                        <= 1'b0;
                            sram_A[dc*AW +: AW]                 <= drain_addr[dc] >> 5;
                            sram_WMASK[dc*32 +: 32]             <= 32'hFFFFFFFF;
                            sram_D_use_zero                     <= 1'b0;
                            drain_state[dc]                     <= 4'd6;
                        end
                        4'd6: begin   // SETTLE
                            sram_CEB[dc] <= 1'b1;
                            sram_WEB[dc] <= 1'b1;
                            fifo_rp[dc]  <= fifo_rp[dc] + 1;
                            drain_state[dc] <= 4'd0;
                        end
                        default: drain_state[dc] <= 4'd0;
                    endcase
                end
            end
        end
    endgenerate

    // ─── 모든 col drain 완료 대기 ──────────────────────────────────
    task wait_drain_complete;
        integer wc, idle_count, all_idle;
        begin
            idle_count = 0;
            while (idle_count < 10) begin
                @(posedge clk);
                all_idle = 1;
                for (wc = 0; wc < 32; wc = wc + 1) begin
                    if (fifo_rp[wc] < fifo_wp[wc] || drain_state[wc] != 4'd0)
                        all_idle = 0;
                end
                if (all_idle) idle_count = idle_count + 1;
                else          idle_count = 0;
            end
        end
    endtask
```

### Step 4.6 — DUMP 32-bank 로 변경

- [ ] **기존 16-bank dump 루프 → 32 bank 루프**

기존 (16 .mem 파일, 1024 words):
```verilog
    // 16 bank × 1024 word dump (이전 코드)
```

새로 (32 .mem 파일, 512 words — workload 사용 범위):
```verilog
    task dump_banks;
        integer bi, w, fd;
        reg [8*512-1:0] path_str;
        reg [31:0] q_word;
        begin
            for (bi = 0; bi < NB; bi = bi + 1) begin
                $sformat(path_str, "%0s/bank%0d.mem", DUMP_DIR, bi);
                fd = $fopen(path_str, "w");
                if (fd == 0) begin
                    $display("FATAL: cannot open dump file %0s", path_str);
                    $finish;
                end
                for (w = 0; w < 512; w = w + 1) begin
                    // 한 cycle 마다 한 bank 의 한 word 만 read (성능 신경 안 씀)
                    @(posedge clk);
                    sram_CEB[bi] <= 1'b0;
                    sram_WEB[bi] <= 1'b1;
                    sram_A[bi*AW +: AW] <= w[AW-1:0];
                    @(posedge clk);
                    sram_CEB[bi] <= 1'b1;
                    @(posedge clk);  // PIPELINE=0 leaf 의 read latency
                    q_word = sram_Q[bi*32 +: 32];
                    $fwrite(fd, "%08x\n", q_word);
                end
                $fclose(fd);
            end
        end
    endtask
```

또는 더 단순하게 — bank 마다 `$writememh` 직접 못 호출 (sram_1rw 의 mem 배열은 hierarchical path 로 접근). 위 task 가 정공법.

- [ ] **main 시퀀스 갱신**

기존 `drain_fifo()` 호출을 `drain_enable <= 1'b1; wait_drain_complete(); drain_enable <= 1'b0;` 로 교체. PRIME_CYC idle 은 DRIVE 직후, drain_enable 직전에 유지.

```verilog
    // ─── Main test sequence ────────────────────────────────────────
    initial begin
        $display("[INIT] reset + 32-bank zero-prime");
        rst <= 1'b1;
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        init_zero_prime();

        $display("[LOAD] $readmemh");
        // (기존 코드 그대로 — A_PREC/B_PREC 에 따라 hex 파일 load)
        load_hex_files();

        $display("[CONFIG] {A_PREC,B_PREC}=%0d,%0d", A_PREC, B_PREC);
        configure_mode();

        $display("[DRIVE]");
        drive_stage_2a(0, 0);
        drive_stage_2b();
        drive_stage_3_4();
        drive_stage_5_tail();

        // capture 종료 직후 PRIME_CYC idle (in-flight settle)
        repeat (PRIME_CYC) @(posedge clk);

        $display("[DRAIN] 32-stream parallel");
        drain_enable <= 1'b1;
        wait_drain_complete();
        drain_enable <= 1'b0;

        $display("[DUMP] 32 bank .mem files");
        dump_banks();

        $display("[DONE]");
        $finish;
    end
```

(현재 TB 의 main initial 블록을 위와 같이 구조화 — task 이름이 같으면 그대로 호출, 다르면 매치해서 호출. `load_hex_files` / `configure_mode` 는 기존 TB 의 LOAD / CONFIG 블록 코드를 그대로 task 화하면 됨.)

### Step 4.7 — 단일 모드 통합 실행 (A8_B8) 으로 검증

- [ ] **MXP_Tools 입력 생성**

```bash
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8
cd ..
```

- [ ] **`sim/run_integration_one.sh` 의 dump 디렉토리 / compare layout 갱신**

L44 부근 (compare 호출이 있다면) — 기존 `bank{0..15}` → `bank{0..31}`, `--layout interleaved_row_major_16bank` → `--layout interleaved_row_major_32bank`. (sweep.sh 안의 compare 호출도 동일.)

`sim/run_integration_one.sh` 의 마지막 echo 줄도:
```bash
echo "Integration sim ${LABEL} done. Check $DUMP/bank{0..31}.mem"
```

- [ ] **단일 모드 통합 실행**

Run: `bash sim/run_integration_one.sh A8_B8 8 8`
Expected: TB 마지막에 `[DONE]`, `work/A8_B8/hw_out/bank{0..31}.mem` 32 파일 생성.

- [ ] **A8_B8 compare 결과 확인 (sweep 호출하는 compare 명령 단독 실행)**

```bash
cd MXP_Tools
BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..31})
python -m mxp_tools compare \
    --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
    --hw-banks ${BANKS} \
    --layout interleaved_row_major_32bank
cd ..
```
Expected: `max(|HW - SW|) = 0.0` 또는 `bit-exact match`.

### Step 4.8 — Commit

- [ ] **Commit**

```bash
git add tb/gemm_sram_top_tb.v sim/run_integration_one.sh
git commit -m "TB: 32-stream per-col capture/drain + 32-bank dump

FIFO single → per-col[32]. DRAIN 을 32 parallel state machine 으로
교체. INIT zero-prime 도 32-bank parallel write. DUMP 32 .mem files,
bank j 의 word 0..511. A8_B8 모드 bit-exact PASS.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5 — 9-mode sweep PASS + sim time 측정

**Files:**
- Modify: `sim/run_integration_sweep.sh` (16→32 bank list, layout name)

### Step 5.1 — sweep 스크립트 갱신

- [ ] **`sim/run_integration_sweep.sh` 의 bank list 및 layout 갱신**

기존 16-bank 부분을 32-bank 로:

```bash
# 32 bank list builder
BANKS=$(printf "$DUMP/bank%d.mem " $(seq 0 31))

python -m mxp_tools compare \
    --ref "$WORK/sw_ref/C_sw_mxint${B_PREC}_mxint${A_PREC}.npz" \
    --hw-banks ${BANKS} \
    --layout interleaved_row_major_32bank \
    || { echo "compare FAIL for $LABEL"; exit 1; }
```

(파일 안의 16-bank 관련 모든 라인을 grep 으로 확인하고 32 로 교체.)

### Step 5.2 — 9-mode sweep 실행 + sim time 기록

- [ ] **sweep 실행 (time 으로 wall-clock 측정)**

Run: `time bash sim/run_integration_sweep.sh 2>&1 | tee /tmp/sweep_32rmw.log`
Expected: 마지막 줄 `ALL 9 MODES PASSED`.
기록: `real    XXmYYs` 값.

- [ ] **이전 1-RMW sweep time 과 비교**

이전 sweep time 은 ~20-25 min (CLAUDE.md 명시). 새 sweep time 을 비율로 기록 (예: 6 min → 약 3-4× 단축).

### Step 5.3 — Commit

- [ ] **Commit (sweep 스크립트 변경만, time 결과는 doc 으로 별도 commit)**

```bash
git add sim/run_integration_sweep.sh
git commit -m "sweep: 32 bank list + interleaved_row_major_32bank layout

9-mode sweep 마지막 줄 'ALL 9 MODES PASSED' 그대로. compare 인자만
32-bank 매핑으로 갱신.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6 — 문서 갱신 (CLAUDE.md + kickoff)

**Files:**
- Modify: `CLAUDE.md` ("Settled — with re-visit triggers" 표 + "Next session kickoff" 블록)
- Modify: `docs/next-session-kickoff.md` ("완성" 섹션에 phase 3 추가)

### Step 6.1 — CLAUDE.md 의 RMW instance 잠정값 갱신

- [ ] **표 행 수정**

`CLAUDE.md` 의 "Settled — with re-visit triggers" 표 안:

```
| RMW instance count | 32 (col-parallel, per-bank SRAM port) | timing closure 시 합성 코스트 측정 |
```

(기존 `1 (TB-side mode-aware dispatch)` 줄 교체.)

같은 표의 bank strategy 줄:
```
| Bank strategy | INTERLEAVED 의미는 호출자 (TB) 가 결정 | bank-conflict 측정; multi-port 필요성 |
```

NUM_BANKS 항목 (있다면) 16 → 32, BANK_DEPTH 항목 32768 → 1024 갱신.

- [ ] **"Next session kickoff" 블록 갱신**

`CLAUDE.md` 맨 위 kickoff 문단에 phase 3 추가:

```markdown
**Phase 3 완성 (2026-05-15)**: RMW 1 → 32 col-parallel + sram_1rw_banked_mp (32 bank, depth=1024) per-bank port. 9/9 PASS, sweep wall-clock <XX min> (이전 1-RMW <YY min> 대비 <Z>×).
```

(`<XX min>` 과 `<YY min>` 은 Task 5 결과 값으로 채울 것.)

### Step 6.2 — kickoff 문서 갱신

- [ ] **`docs/next-session-kickoff.md` 의 "완성" 섹션에 phase 추가**

```markdown
**완성 — phase 3 (RMW 32x col-parallel, 2026-05-15)**
- `sram_1rw_banked_mp.v` (per-bank port-exposed, NB=32, depth=1024).
- `gemm_sram_top.v` 32-RMW + 32-bank generate.
- `tb/gemm_sram_top_tb.v` 32-stream capture/drain.
- MXP_Tools `interleaved_row_major_32bank` layout.
- 검증: 9/9 PASS, sweep <XX min> (이전 1-RMW <YY min> 대비 <Z>× 단축).
```

(다음 단계 후보 표에서 "1. Throughput 확장" 항목은 "완료" 로 표시하고 새 후보 추가 — 예: "1' SRAM weight 저장소" / "5. RMW 합성 코스트 측정".)

### Step 6.3 — Commit

- [ ] **Commit**

```bash
git add CLAUDE.md docs/next-session-kickoff.md
git commit -m "Docs: phase 3 (RMW 32x col-parallel) + 32-bank tentative values

CLAUDE.md '잠정값' 표의 RMW instance / Bank strategy / NUM_BANKS /
BANK_DEPTH 갱신. kickoff 문서에 phase 3 결과 + sweep time 추가.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 최종 검증 게이트 (모든 task 완료 후)

다음을 순서대로 실행하여 모두 PASS 확인:

```bash
bash sim/run_sram_mp.sh                        # 새 wrapper 단위
bash sim/run_rmw.sh                            # RMW 단위 (회귀)
cd MXP_Tools && python -m pytest tests/ -q && cd ..   # 45/45
bash sim/run_top_elab.sh                       # top elab smoke
bash sim/run_integration_sweep.sh              # 9/9 bit-exact
```

기대값:
- `run_sram_mp.sh`: `ALL 5 TESTS PASSED`
- `run_rmw.sh`: `rmw_tb: ALL 71 TESTS PASSED`
- pytest: `45 passed`
- `run_top_elab.sh`: `PASS: gemm_sram_top elaborates cleanly`
- `run_integration_sweep.sh`: `ALL 9 MODES PASSED`

---

## 잠재 함정 (이 plan 을 실행하다 막힐 가능성)

1. **xelab port width 불일치** — `gemm_sram_top.v` 의 `$clog2(BANK_DEPTH)` 표현이 module port 선언에서 ANSI-style 로 해석 안 되는 시뮬레이터 버전에서 issue 가능. Vivado 2024.1 의 XSim 은 OK 이지만, 만약 fail 하면 `sram_1rw_banked.v` 의 non-ANSI 패턴 (localparam 으로 width 산출 후 port 선언) 으로 wrapper port 도 옮기는 게 안전.

2. **per-col always block 의 state[c] 비결정성** — `generate for` 안의 `always @(posedge clk)` 는 각 iteration 이 독립 always 가 됨. `drain_state[dc]` 가 array element 이므로 multi-driver issue 없음 (인덱스 `dc` 가 generate parameter). 만약 시뮬레이터 경고가 뜨면 array 대신 per-col scalar reg (`reg [3:0] drain_state_0, drain_state_1, ...`) generate 안에서 선언으로 옮길 것.

3. **RMW_WAIT_CYC 8 cycle 의 적정성** — 새 design 의 dispatch interval (col 하나가 다음 fire 까지 대기하는 cycle) 이 워크로드에 따라 짧을 수 있음. fire 가 너무 빨리 들어오면 FSM 이 IDLE 로 돌아가기 전에 다음 fifo entry 가 도착. 시나리오상 col 당 fire 간격이 W_CYC = 2/4/8 cycle (mode 별) 인데 drain 1 entry 처리에 ~11 cycle (IDLE→READ→WAIT_R→DRIVE→WAIT8→WRITE→SETTLE) 걸리므로 backlog 가 자연스레 쌓이고 FIFO 안에서 순차 소화 — 문제 없음. 다만 sim 시간이 길어질 수 있으니 worst-case (A8 = 2048 fires × ~11 cy = ~22K cy/col) 계산 후 expected 와 비교할 것.

4. **TB 의 기존 helper task / function 보존** — `drive_stage_2a/2b/3_4/5_tail`, `drive_prefetch`, `build_in_b`, `build_in_scale_act`, `pack_int4_n_pair`, `pack_int2_n_quad` 는 모두 GEMM driving 쪽이라 변경 없음. capture/drain/INIT/DUMP 만 손댐.

5. **dump_banks 안의 cycle count** — bank 마다 512 word × 3 cy = 1536 cy, 32 bank 직렬 = ~50K cy. 32 bank 를 parallel dump 로 한 cycle 에 한 word 씩 같이 read 하면 ~1500 cy 면 충분. 위 task 는 직렬 형태이므로 sim time 더 짧게 하려면 32-bank parallel dump 로 갱신 (optional).

---

## Self-Review

1. **Spec coverage**: spec § 3 (wrapper) → Task 1, § 4 (매핑 증명) → Task 4 capture assert, § 5 (TB) → Task 4, § 6 (MXP_Tools) → Task 2, § 7 (스크립트) → Task 3/5, § 8 (검증 게이트) → 최종 검증 + Task 5, § 9 (산출물) → Tasks 1-6 의 commit list. **누락 없음.**

2. **Placeholder scan**: `<XX min>`, `<YY min>`, `<Z>` 은 Task 5 실행 결과로 채우는 변수 자리 (의도된 fill-in). 그 외 TODO/TBD 없음.

3. **Type consistency**: `NB`, `BD`, `AW`, `FIFO_DEPTH`, `DRAIN_RMW_WAIT` 등 localparam 이름 일관. `drain_state[c]`, `fifo_wp[c]`, `fifo_int[c][k]` indexing pattern 일관.
