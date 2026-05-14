# GEMM ↔ RMW ↔ SRAM Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 9-mode 검증 (Task 8 onwards) 은 `superpowers:dispatching-parallel-agents` 로 9 subagent 병렬 dispatch 가능.

**Goal:** `RMW.v` 를 `GEMM` 과 `sram_1rw_banked` 사이에 연결하여 9 개 정밀도 조합 (A {2,4,8} × B {2,4,8}) 모두 `MXP_Tools` golden reference 와 100 % bit-exact 일치하는 통합 datapath 를 만든다.

**Architecture:**
- `gemm_sram_top.v` — pure structural wrapper, logic 은 SRAM `D` 입력의 1-bit zero-prime mux 한 줄. 그 외 모든 control 신호는 외부 노출.
- `tb/gemm_sram_top_tb.v` — 단일 TB 가 plusarg 로 1 mode 받음. SRAM zero priming + GEMM driving + RMW dispatch + `$writememh` × 16 bank 덤프 단계로 sequential.
- 검증은 `MXP_Tools compare` 의 bit-exact gate 로. 9 mode 는 directly serial sweep (`run_integration_sweep.sh`) 또는 9-way parallel subagent dispatch 로.

**Tech Stack:** Verilog-2001 RTL (Vivado 합성 호환), Vivado 2024.1 XSim batch (`xvlog`/`xelab`/`xsim`), Python numpy / MXP_Tools.

**Spec:** `docs/superpowers/specs/2026-05-14-integration-design.md`

---

## Resume Protocol (다른 세션에서 이어가기)

이 plan 의 실행이 도중에 중단되거나 새 세션에서 시작할 때:

1. **이 plan + spec 을 먼저 읽음**:
   - `docs/superpowers/plans/2026-05-14-integration-implementation.md` (이 문서)
   - `docs/superpowers/specs/2026-05-14-integration-design.md`
2. **현재 진행 상태 확인**: `git log --oneline -20` 으로 직전 커밋 메시지에서 "Task N" 키워드 찾음.
3. **다음 시작 task**: 마지막으로 커밋된 task 의 다음 번 task 부터. 각 task 끝에 commit 이 있으니 task 단위로 끊김.
4. **9-mode parallel dispatch 단계 (Task 9)** 는 미완료 mode 만 다시 dispatch 가능 — `work/A{i}_B{j}/hw_out/bank0.mem` 의 존재 + `compare` 결과로 확인.
5. **CLAUDE.md** 의 "Next session kickoff" 섹션도 매 phase 끝에 갱신됨 — 보조 navigation 용.

각 task 가 self-contained: 코드 / 명령 / 예상 출력 모두 인-line 으로 기재. 이전 task 의 산출물이 git 에 commit 되어 있으면 그 자체로 다음 task 의 prerequisite 충족.

---

## File Structure (created / modified)

```
gemm_sram/
├── gemm_sram.srcs/sources_1/new/
│   └── gemm_sram_top.v                      [NEW] pure structural wrapper
├── tb/
│   └── gemm_sram_top_tb.v                   [NEW] 1-mode plusarg-driven TB
├── sim/
│   ├── run_integration_smoke.sh             [NEW] smoke (zero-priming + dump)
│   ├── run_integration_one.sh               [NEW] 1 mode (compile + xelab + xsim)
│   ├── run_integration_sweep.sh             [NEW] 9 mode serial sweep wrapper
│   └── run_integration_parallel.sh          [NEW] subagent dispatch helper
├── docs/superpowers/notes/
│   ├── mxp-driving-sequence.md              [NEW] Task 5 산출물: MXP TB driving cycle pattern
│   └── lane-to-c-mapping.md                 [NEW] Task 6 산출물: A4/A2 lane→C[m,n] 매핑
├── MXP_Tools/mxp_tools/
│   ├── hwio.py                              [MODIFY] interleaved_row_major_16bank 추가
│   └── cli.py                               [MODIFY] compare --layout 분기 1줄
├── gemm_sram.xpr                            [MODIFY] top 을 GEMM → gemm_sram_top 으로 (Task 10)
├── CLAUDE.md                                [MODIFY] kickoff 갱신
└── docs/next-session-kickoff.md             [DELETE 또는 MODIFY] 통합 끝나면 정리
```

각 파일의 책임:
- `gemm_sram_top.v` — GEMM + RMW + SRAM instantiate + D 의 zero-prime mux 1 개. 그 외 logic 없음.
- `tb/gemm_sram_top_tb.v` — 단일 TB, plusarg 로 1 mode 만 받음. INIT / LOAD / CONFIG / DRIVE / DRAIN / DUMP 7 단계 sequential.
- `sim/run_integration_smoke.sh` — Task 2-4 의 smoke flow.
- `sim/run_integration_one.sh` — 1 mode 의 build + sim runner.
- `sim/run_integration_sweep.sh` — bash for-loop 으로 9 mode 직렬 sweep.
- `sim/run_integration_parallel.sh` — 9-way 병렬 dispatch 정보 / 가이드. 실제 dispatch 는 Agent 가 수행.
- `mxp-driving-sequence.md` — MXP 의 station chain / acc chain / SystolicArray 입력 발사 시퀀스를 cycle-by-cycle 정리.
- `lane-to-c-mapping.md` — A4 mode 의 2 sub-word, A2 mode 의 4 sub-word 가 C[m,n] 의 어느 위치에 매핑되는지.

---

## Task 1: `gemm_sram_top.v` 작성 + elaboration smoke

**Files:**
- Create: `gemm_sram.srcs/sources_1/new/gemm_sram_top.v`
- (Vivado xpr 의 top 변경은 Task 10 — 지금은 명령줄 xvlog/xelab 만 사용)

- [ ] **Step 1: `gemm_sram_top.v` 작성**

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top — GEMM + RMW + sram_1rw_banked 묶음 (pure structural).
// 모든 control 신호는 TB 가 driving (controller 없음).
// 내부 wiring: 두 줄 (sram_Q ─► rmw.in_SRAM, mux(sram_D_use_zero) ─► sram.D).
//
// Spec: docs/superpowers/specs/2026-05-14-integration-design.md
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top #(
    parameter NUM_BANKS  = 16,
    parameter BANK_DEPTH = 32768
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

    // ───── RMW input (TB 가 직접 발사) ─────
    input  wire [31:0]                       rmw_in_GEMM,
    input  wire [8:0]                        rmw_scale,
    output wire [31:0]                       rmw_out_RMW,        // probe

    // ───── SRAM control (TB 가 직접 발사) ─────
    input  wire                              sram_CEB,
    input  wire                              sram_WEB,
    input  wire [18:0]                       sram_A,             // clog2(16)+clog2(32768)=19
    input  wire [31:0]                       sram_WMASK,
    input  wire                              sram_D_use_zero,    // 1=D 강제 0, 0=RMW 출력
    output wire [31:0]                       sram_Q              // probe
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

    // ─── RMW ───────────────────────────────────────────────────────
    wire [31:0] rmw_out_w;
    RMW u_rmw (
        .clk(clk), .rst(rst),
        .in_SRAM(sram_Q),
        .in_GEMM(rmw_in_GEMM),
        .scale(rmw_scale),
        .out_RMW(rmw_out_w)
    );
    assign rmw_out_RMW = rmw_out_w;

    // ─── SRAM D 입력 mux (zero priming 단계 우회용) ─────────────────
    wire [31:0] sram_D_w = sram_D_use_zero ? 32'h00000000 : rmw_out_w;

    // ─── SRAM banked ───────────────────────────────────────────────
    sram_1rw_banked #(
        .DATA_WIDTH    (32),
        .NUM_BANKS     (NUM_BANKS),
        .BANK_DEPTH    (BANK_DEPTH),
        .BANK_STRATEGY ("INTERLEAVED"),
        .PIPELINE      (0)
    ) u_sram (
        .CLK   (clk),
        .CEB   (sram_CEB),
        .WEB   (sram_WEB),
        .A     (sram_A),
        .D     (sram_D_w),
        .WMASK (sram_WMASK),
        .Q     (sram_Q)
    );

endmodule
```

- [ ] **Step 2: Elaboration smoke (`xvlog` + `xelab`)**

`sim/run_top_elab.sh` 임시로 작성 (Task 7 에서 정식 sim 스크립트로 교체):

```bash
#!/bin/bash
# sim/run_top_elab.sh — gemm_sram_top elaboration smoke (no TB).
set -e
cd "$(dirname "$0")/.."
BUILD=sim/build/top_elab
mkdir -p "$BUILD"
cd "$BUILD"

SRC_ROOT="../../../gemm_sram.srcs/sources_1"
HF_ROOT="../../../third_party/berkeley-hardfloat"

# HardFloat (RMW 가 import) + MXP + SRAM + 통합 top
xvlog -sv \
    $HF_ROOT/HardFloatBundle.v \
    $SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
    $SRC_ROOT/imports/Desktop/sram/rtl/*.v \
    $SRC_ROOT/new/int_to_fp32.v \
    $SRC_ROOT/new/fp32_adder.v \
    $SRC_ROOT/new/RMW.v \
    $SRC_ROOT/new/gemm_sram_top.v

xelab -L work gemm_sram_top -snapshot gemm_sram_top_snap

echo "PASS: gemm_sram_top elaborates cleanly"
```

`chmod +x sim/run_top_elab.sh`.

- [ ] **Step 3: 실행**

```bash
bash sim/run_top_elab.sh
```
Expected output 끝:
```
... (xvlog / xelab 단계별 로그)
PASS: gemm_sram_top elaborates cleanly
```

xelab 에러나 width-mismatch warning 발생 시 stop and fix.

- [ ] **Step 4: Commit**

```bash
git add gemm_sram.srcs/sources_1/new/gemm_sram_top.v sim/run_top_elab.sh
git commit -m "Task 1: add gemm_sram_top.v + elaboration smoke"
```

---

## Task 2: TB skeleton + zero priming smoke

**Files:**
- Create: `tb/gemm_sram_top_tb.v` (skeleton only — INIT + DRAIN + DUMP)
- Create: `sim/run_integration_smoke.sh`

이 task 는 **GEMM driving 없이** TB 가 SRAM 을 zero priming → 즉시 dump 만 검증. 결과: `bank{0..15}.mem` 16 개 파일이 1024 word 의 `00000000` 으로 채워져야 함.

- [ ] **Step 1: `tb/gemm_sram_top_tb.v` skeleton 작성**

```verilog
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_tb — 단일 TB (plusarg 로 1 mode 받음).
// 이 task (Task 2) 에서는 INIT (zero priming) + DUMP 만 동작.
// Task 7 에서 LOAD / CONFIG / DRIVE / DRAIN 단계 추가.
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8                      (Task 2 에선 무시)
//   B_PREC    : 2 / 4 / 8                      (Task 2 에선 무시)
//   TILE_M, TILE_K, TILE_N (default 128 each)
//   WORK_DIR  : MXP_Tools 입력 .hex 디렉토리   (Task 2 에선 무시)
//   DUMP_DIR  : SRAM 덤프 출력 디렉토리        (REQUIRED)
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top_tb;

    // ─── plusarg parsing ───────────────────────────────────────────
    integer A_PREC, B_PREC;
    integer TILE_M, TILE_K, TILE_N;
    reg [8*256-1:0] WORK_DIR;  // string buf
    reg [8*256-1:0] DUMP_DIR;

    initial begin
        if (!$value$plusargs("A_PREC=%d",   A_PREC))    A_PREC   = 8;
        if (!$value$plusargs("B_PREC=%d",   B_PREC))    B_PREC   = 8;
        if (!$value$plusargs("TILE_M=%d",   TILE_M))    TILE_M   = 128;
        if (!$value$plusargs("TILE_K=%d",   TILE_K))    TILE_K   = 128;
        if (!$value$plusargs("TILE_N=%d",   TILE_N))    TILE_N   = 128;
        if (!$value$plusargs("WORK_DIR=%s", WORK_DIR))  WORK_DIR = "work/smoke";
        if (!$value$plusargs("DUMP_DIR=%s", DUMP_DIR))  DUMP_DIR = "work/smoke/hw_out";
    end

    // ─── clock / reset ─────────────────────────────────────────────
    reg clk = 0;
    always #5 clk = ~clk;        // 100 MHz nominal (10 ns period)
    reg rst = 1;

    // ─── DUT 포트 wires ────────────────────────────────────────────
    // GEMM stimuli — Task 2 에선 idle (모두 0)
    reg [32-1:0]              in_a              = 0;
    reg signed [32*8-1:0]     in_b              = 0;
    reg [32-1:0]              in_control        = 0;
    reg [32-1:0]              in_loadEN         = 0;
    reg [1:0]                 in_Station_control= 0;
    reg [4*8-1:0]             in_Scale_Activation= 0;
    reg                       in_station_control = 0;
    reg                       in_station_loadEN  = 0;
    reg                       in_start_accumulate = 0;
    reg [1:0]                 in_Wcontrol        = 0;
    reg [7:0]                 in_scale_weight    = 0;

    wire [32*60-1:0]          out_accumulate;
    wire [32*36-1:0]          out_scale;
    wire [32-1:0]             out_fire;

    // RMW — Task 2 에선 idle
    reg [31:0]                rmw_in_GEMM       = 0;
    reg [8:0]                 rmw_scale         = 0;
    wire [31:0]               rmw_out_RMW;

    // SRAM
    reg                       sram_CEB          = 1;   // idle high
    reg                       sram_WEB          = 1;
    reg [18:0]                sram_A            = 0;
    reg [31:0]                sram_WMASK        = 32'hFFFFFFFF;
    reg                       sram_D_use_zero   = 1;
    wire [31:0]               sram_Q;

    // ─── DUT instance ──────────────────────────────────────────────
    gemm_sram_top #(
        .NUM_BANKS  (16),
        .BANK_DEPTH (32768)
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
        .sram_CEB(sram_CEB), .sram_WEB(sram_WEB),
        .sram_A(sram_A), .sram_WMASK(sram_WMASK),
        .sram_D_use_zero(sram_D_use_zero),
        .sram_Q(sram_Q)
    );

    // ─── helper: 1 word write ─────────────────────────────────────
    task sram_write(input [18:0] addr, input [31:0] data, input use_zero);
        begin
            @(posedge clk);
            sram_CEB        <= 1'b0;
            sram_WEB        <= 1'b0;
            sram_A          <= addr;
            sram_WMASK      <= 32'hFFFFFFFF;
            sram_D_use_zero <= use_zero;
            // 이 task 안에서 rmw_in_GEMM 은 안 만짐 — use_zero=1 일 때 D 가 0 으로 강제됨
        end
    endtask

    task sram_idle;
        begin
            @(posedge clk);
            sram_CEB <= 1'b1;
            sram_WEB <= 1'b1;
        end
    endtask

    // ─── main sequence ─────────────────────────────────────────────
    integer i, b;
    reg [8*512-1:0] dump_path;

    initial begin
        // INIT
        #20;
        rst = 0;
        sram_D_use_zero = 1;

        // SRAM zero priming — flat addr 0..16383
        for (i = 0; i < 16384; i = i + 1) begin
            sram_write(i[18:0], 32'h00000000, 1'b1);
        end
        sram_idle;
        sram_idle;     // settle

        // DUMP — 16 bank 모두 0..1023 word 범위
        for (b = 0; b < 16; b = b + 1) begin
            $sformat(dump_path, "%0s/bank%0d.mem", DUMP_DIR, b);
            case (b)
                0:  $writememh(dump_path, u_top.u_sram.g_bank[0].u_bank.mem,  0, 1023);
                1:  $writememh(dump_path, u_top.u_sram.g_bank[1].u_bank.mem,  0, 1023);
                2:  $writememh(dump_path, u_top.u_sram.g_bank[2].u_bank.mem,  0, 1023);
                3:  $writememh(dump_path, u_top.u_sram.g_bank[3].u_bank.mem,  0, 1023);
                4:  $writememh(dump_path, u_top.u_sram.g_bank[4].u_bank.mem,  0, 1023);
                5:  $writememh(dump_path, u_top.u_sram.g_bank[5].u_bank.mem,  0, 1023);
                6:  $writememh(dump_path, u_top.u_sram.g_bank[6].u_bank.mem,  0, 1023);
                7:  $writememh(dump_path, u_top.u_sram.g_bank[7].u_bank.mem,  0, 1023);
                8:  $writememh(dump_path, u_top.u_sram.g_bank[8].u_bank.mem,  0, 1023);
                9:  $writememh(dump_path, u_top.u_sram.g_bank[9].u_bank.mem,  0, 1023);
                10: $writememh(dump_path, u_top.u_sram.g_bank[10].u_bank.mem, 0, 1023);
                11: $writememh(dump_path, u_top.u_sram.g_bank[11].u_bank.mem, 0, 1023);
                12: $writememh(dump_path, u_top.u_sram.g_bank[12].u_bank.mem, 0, 1023);
                13: $writememh(dump_path, u_top.u_sram.g_bank[13].u_bank.mem, 0, 1023);
                14: $writememh(dump_path, u_top.u_sram.g_bank[14].u_bank.mem, 0, 1023);
                15: $writememh(dump_path, u_top.u_sram.g_bank[15].u_bank.mem, 0, 1023);
            endcase
        end

        $display("INTEGRATION TB: ZERO-PRIME DUMP DONE");
        $finish;
    end

    // ─── timeout safety ────────────────────────────────────────────
    initial begin
        #2_000_000;   // 2 ms = 200000 cycle @ 100 MHz
        $display("ERROR: timeout");
        $finish;
    end

endmodule
```

> NOTE: `$writememh` 의 hierarchical access (`u_top.u_sram.g_bank[i].u_bank.mem`) 는 XSim 의 generate-block 인덱싱이라 case 분기로 풀어씀 — `for (b)` 안에서 `g_bank[b]` 식으로 동적 인덱싱이 안 됨. (Vivado 2024.1 XSim 검증 시 안 되면 Task 2 step 3 에서 fix.)

- [ ] **Step 2: `sim/run_integration_smoke.sh` 작성**

```bash
#!/bin/bash
# sim/run_integration_smoke.sh — Task 2 smoke: SRAM zero priming + dump.
set -e
cd "$(dirname "$0")/.."

LABEL="smoke"
WORK="work/${LABEL}"
DUMP="${WORK}/hw_out"
BUILD="sim/build/${LABEL}"

mkdir -p "$DUMP" "$BUILD"

SRC_ROOT="gemm_sram.srcs/sources_1"
HF_ROOT="third_party/berkeley-hardfloat"

(cd "$BUILD" && \
    xvlog -sv \
        ../../../$HF_ROOT/HardFloatBundle.v \
        ../../../$SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
        ../../../$SRC_ROOT/imports/Desktop/sram/rtl/*.v \
        ../../../$SRC_ROOT/new/int_to_fp32.v \
        ../../../$SRC_ROOT/new/fp32_adder.v \
        ../../../$SRC_ROOT/new/RMW.v \
        ../../../$SRC_ROOT/new/gemm_sram_top.v \
        ../../../tb/gemm_sram_top_tb.v && \
    xelab -L work gemm_sram_top_tb -snapshot gemm_sram_top_tb_snap && \
    xsim work.gemm_sram_top_tb_snap -runall \
        -testplusarg "DUMP_DIR=../../../$DUMP")

echo "Smoke sim done. Check $DUMP/bank{0..15}.mem"
```
`chmod +x sim/run_integration_smoke.sh`.

- [ ] **Step 3: 실행 + 결과 확인**

```bash
bash sim/run_integration_smoke.sh
```
Expected:
```
... (xvlog/xelab/xsim logs)
INTEGRATION TB: ZERO-PRIME DUMP DONE
... $finish called ...
Smoke sim done. Check work/smoke/hw_out/bank{0..15}.mem
```

```bash
ls work/smoke/hw_out/    # 16 files: bank0.mem ~ bank15.mem
head -3 work/smoke/hw_out/bank0.mem      # 모두 00000000 이어야 함
wc -l work/smoke/hw_out/bank0.mem        # 1024 line
```
모두 `00000000` 이면 통과. 일부 미정의 (`x`/`z`) 이면 SRAM init 실패 → fix.

- [ ] **Step 4: Commit**

```bash
git add tb/gemm_sram_top_tb.v sim/run_integration_smoke.sh
git commit -m "Task 2: TB skeleton + zero-priming smoke (16 banks all-zero dump)"
```

---

## Task 3: MXP_Tools 매핑 callable + CLI 분기

**Files:**
- Modify: `MXP_Tools/mxp_tools/hwio.py`
- Modify: `MXP_Tools/mxp_tools/cli.py`

- [ ] **Step 1: `hwio.py` 에 `interleaved_row_major_16bank` 추가**

`MXP_Tools/mxp_tools/hwio.py` 의 `default_banks_split_rows` 함수 바로 아래에 추가:

```python
def interleaved_row_major_16bank(bank_idx, word_offset, M, N):
    """C[m,n] = SRAM[bank=flat%16, word=flat//16] where flat = m*N+n.

    32 col GEMM 결과를 row-major flat 으로 SRAM 에 쌓고
    sram_1rw_banked 의 INTERLEAVED 매핑 (LSB 4비트가 bank_sel) 으로
    16 bank 에 분산되는 layout 의 역매핑.

    Used by: gemm_sram integration testbench
    Spec   : docs/superpowers/specs/2026-05-14-integration-design.md §3
    """
    flat = word_offset * 16 + bank_idx
    if flat >= M * N:
        return None
    return divmod(flat, N)
```

- [ ] **Step 2: `cli.py` 의 `compare` 명령에 `--layout` 분기 추가**

`MXP_Tools/mxp_tools/cli.py` 에서 `compare` subcommand 의 layout 분기 부분 찾기. 패턴 (기존):

```python
if args.layout == "single":
    mapping = hwio.default_single_bank_row_major
elif args.layout == "split_rows":
    mapping = hwio.default_banks_split_rows(len(args.hw_banks))
```

바로 아래에 새 분기 1 줄 추가:

```python
elif args.layout == "interleaved_row_major_16bank":
    mapping = hwio.interleaved_row_major_16bank
```

(만약 `argparse` 의 `choices` 에 layout 목록이 있으면 `"interleaved_row_major_16bank"` 도 그 목록에 추가.)

- [ ] **Step 3: 단위 round-trip 테스트 작성**

`MXP_Tools/tests/test_hwio_interleaved.py` 신규:

```python
"""Unit test: interleaved_row_major_16bank 매핑은
sram_1rw_banked 의 INTERLEAVED 주소 분할의 정확한 역연산이어야 한다.
"""
import numpy as np
from mxp_tools import hwio


def test_interleaved_row_major_round_trip():
    M, N = 128, 128
    # forward: C[m,n] → (bank, word)
    flat = lambda m, n: m * N + n
    bank = lambda f: f % 16
    word = lambda f: f // 16

    # inverse via hwio:
    for m, n in [(0, 0), (0, 15), (0, 16), (1, 0), (127, 127), (63, 31)]:
        f = flat(m, n)
        b = bank(f)
        w = word(f)
        result = hwio.interleaved_row_major_16bank(b, w, M, N)
        assert result == (m, n), f"({m},{n})→bank{b}/word{w}→{result}"


def test_interleaved_row_major_out_of_range():
    assert hwio.interleaved_row_major_16bank(0, 9999, 128, 128) is None
```

```bash
cd MXP_Tools && python -m pytest tests/test_hwio_interleaved.py -v
```
Expected: 2 PASS.

- [ ] **Step 4: Commit**

```bash
git add MXP_Tools/mxp_tools/hwio.py MXP_Tools/mxp_tools/cli.py MXP_Tools/tests/test_hwio_interleaved.py
git commit -m "Task 3: add interleaved_row_major_16bank mapping for gemm_sram integration"
```

---

## Task 4: Round-trip smoke — zero matrix dump → compare PASS

**Files:**
- 이 task 는 코드 추가 없이 Task 2 의 zero-prime dump 결과를 Task 3 의 mapping 으로 reconstruct 해서 zero matrix 가 되는지 확인.

- [ ] **Step 1: 16-bank zero dump 를 numpy array 로 gather**

```bash
cd MXP_Tools
python -c "
from mxp_tools import hwio
import numpy as np

bank_paths = ['../work/smoke/hw_out/bank%d.mem' % i for i in range(16)]
C = hwio.gather_banks(bank_paths, 128, 128, hwio.interleaved_row_major_16bank)
assert C.shape == (128, 128)
assert np.array_equal(C, np.zeros((128, 128), dtype=np.float32))
print('SMOKE COMPARE PASS: 16384 zero words, all bit-exact 0.0')
"
cd ..
```
Expected:
```
SMOKE COMPARE PASS: 16384 zero words, all bit-exact 0.0
```

만약 `gather_banks` 가 `NaN` 잔존 에러로 raise 하면: 매핑이 16 bank 전체를 cover 못 한다는 뜻 → Task 3 의 매핑 함수 fix.

- [ ] **Step 2: Commit (no code change — log 만)**

```bash
git commit --allow-empty -m "Task 4: smoke round-trip PASS (zero matrix → 16-bank → reconstruct → zero)"
```

---

## Task 5: MXP standalone TB 조사 — driving sequence 문서화

**Files:**
- Create: `docs/superpowers/notes/mxp-driving-sequence.md`

이 task 는 코드 추가 없음 — `../MXP/sim/` 또는 `../MXP/` 하위의 standalone TB 가 station chain / accumulator chain / SystolicArray 의 입력 신호를 cycle-by-cycle 어떻게 발사하는지 정리. 통합 TB 의 Task 7 driving sequence 가 이걸 그대로 답습.

- [ ] **Step 1: MXP standalone TB 파일 찾기**

```bash
ls ../MXP/sim/                       # XSim batch scripts
find ../MXP -name "*_tb.v" -o -name "*_tb.sv"   # testbench files
cat ../MXP/CLAUDE.md | head -100     # MXP project convention
```

조사 대상:
- Top-level integration TB (32×32 systolic 전체 driving)
- 한 K-tile 의 driving sequence (cycle-by-cycle)
- station chain / accumulator chain 의 entry timing
- `in_loadEN` / `in_station_loadEN` / `in_start_accumulate` 의 pulse 패턴
- A/B 입력 .hex 파일을 어떻게 `$readmemh` 로 받는지

- [ ] **Step 2: `docs/superpowers/notes/mxp-driving-sequence.md` 작성**

발견한 패턴을 다음 형식으로 정리:

```markdown
# MXP TOP Driving Sequence — Reverse-engineered from `../MXP/sim/`

본 문서는 `gemm_sram` 통합 TB (`tb/gemm_sram_top_tb.v`) 의 Task 7 DRIVE 단계가
참고할 cycle-by-cycle 입력 발사 패턴. 출처는 `../MXP/...`.

## 1. 입력 파일 로딩 (`$readmemh`)

| 파일 | reg 배열 | 의미 |
|---|---|---|
| `a_input_BS_{P}.hex` | [...] | bit-serial A, K_tile × M_tile × 32 row × P bit-slice |
| `b_input_{P}.hex`    | [...] | bit-parallel B, K_tile × N_tile × 32 col |
| `a_scale_{P}.hex`    | [...] | E8M0 scales for A |
| `b_scale_{P}.hex`    | [...] | E8M0 scales for B |

## 2. 한 K-tile 의 driving cycle (예: A8 모드)

| cycle | in_loadEN | in_a | in_b | in_Station_control | ... | 비고 |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

(실제 발견된 패턴으로 채움)

## 3. fire timing

한 K-tile driving 시작 → 각 col j 의 `out_fire` pulse 가 떨어지는 cycle 차이.
(MXP CLAUDE.md "station chain / acc chain" 섹션 참고)
```

이 문서는 plan 의 Task 7 에서 driving code 작성 시 reference 로 사용.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/notes/mxp-driving-sequence.md
git commit -m "Task 5: document MXP standalone TB driving sequence for integration reuse"
```

---

## Task 6: `MXP_Tools/gemm.py` 조사 — A4/A2 lane→C 매핑 문서화

**Files:**
- Create: `docs/superpowers/notes/lane-to-c-mapping.md`

GEMM 의 `Accumulator_Col.out_accumulate[59:0]` 패킹 (A8: 1 lane, A4: 2 lane, A2: 4 lane) 이 SW reference `MXP_Tools/mxp_tools/gemm.py::mxint_gemm_golden` 의 어느 C[m,n] 위치에 매핑되는지. RMW 발사 시 lane 별로 다른 SRAM addr 가 필요 → 이 매핑이 명확해야 함.

- [ ] **Step 1: `MXP_Tools/mxp_tools/gemm.py` 의 `mxint_gemm_golden` 읽고 lane 분해 의미 추출**

```bash
cat MXP_Tools/mxp_tools/gemm.py | head -150
```

확인할 것:
- A4 모드의 두 sub-word 가 "한 8bit lane → 두 4bit lane (high / low)" 인지, "두 인접 column" 인지
- A2 모드의 네 sub-word 의 위치 의미
- SW reference 는 어떻게 4-lane / 2-lane 결과를 누적하는지 (또는 분리해서 저장하는지)

- [ ] **Step 2: `Accumulator_Col.v` 의 패킹과 cross-check**

`Accumulator_Col.v` 의 `out_accumulate` 비트 layout (CLAUDE.md "MXP control surface" 참고):
- A8: `{sign-ext..., s2[20:0]}` — 1 lane
- A4: `{0..0, s1_a[17:0], s1_b[17:0]}` — 2 lane (high half = high 4-bit dot product, low half = low 4-bit)
- A2: `{out_INT2_0, out_INT2_1, out_INT2_2, out_INT2_3}` — 4 lane (각 lane = bit weight 6/4/2/0)

이 lane 들이 C[m,n] 의 어느 위치인지는 SW reference 에서 결정.

- [ ] **Step 3: `docs/superpowers/notes/lane-to-c-mapping.md` 작성**

```markdown
# A4/A2 Mode Lane → C[m,n] Mapping

본 문서는 통합 TB (Task 7+) 에서 RMW 발사 시 어느 lane 의 결과를 어느
SRAM 주소에 write 할지 결정하는 근거.

## A8 모드 (1 lane per col)
- col j 의 fire 시점에 `out_accumulate[j*60 +: 60]` 전체가 1 개 21-bit signed
  (sign-ext to 60-bit) 결과.
- `out_scale[j*36 +: 9]` (가장 lower 9 bit, `comb_s0`) 를 RMW 의 `scale` 로.
- 발사 회수 = 32 (col) × K-step × tile.
- SRAM addr = `flat(m_tile_row + ?, n_tile_start + j)`.
  (정확한 (m, n) 매핑은 `mxint_gemm_golden` 의 output 매트릭스 채우기 순서로 확인)

## A4 모드 (2 lane per col)
- (실제 SW reference 분석 결과로 작성)

## A2 모드 (4 lane per col)
- (실제 SW reference 분석 결과로 작성)
```

이 문서가 명확하지 않으면 Task 7 에서 시뮬을 돌려가며 다시 결정 (debug-iteration).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/notes/lane-to-c-mapping.md
git commit -m "Task 6: document A4/A2 lane → C[m,n] mapping for RMW dispatch"
```

---

## Task 7: A8_B8 1 mode — driving + RMW dispatch + PASS

**Files:**
- Modify: `tb/gemm_sram_top_tb.v` (LOAD / CONFIG / DRIVE / DRAIN 단계 추가)
- Create: `sim/run_integration_one.sh`

가장 큰 task. **A8 모드 + B8 모드** (1 lane per col, scale sub-word 0 만 사용) 만 먼저 동작시킴. 나머지 8 mode 는 Task 8.

- [ ] **Step 1: MXP_Tools 입력 + golden 생성 (A8_B8)**

```bash
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8 --prec-a 8 --prec-b 8
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8
cd ..
ls work/A8_B8/hw_input/
ls work/A8_B8/sw_ref/
```
Expected files:
- `work/A8_B8/hw_input/a_input_BS_8.hex` (또는 비슷한 이름)
- `work/A8_B8/hw_input/b_input_8.hex`
- `work/A8_B8/hw_input/a_scale_8.hex`
- `work/A8_B8/hw_input/b_scale_8.hex`
- `work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz`

(파일 이름이 다르면 `MXP_Tools/mxp_tools/emit.py` 또는 cli 로 확인 후 TB 의 `$readmemh` path 일치시킴.)

- [ ] **Step 2: TB 에 LOAD / CONFIG / DRIVE 단계 추가**

`tb/gemm_sram_top_tb.v` 의 main initial 블록을 다음과 같이 확장.

먼저 reg 메모리 선언 추가 (module 상단):
```verilog
// ─── 입력 메모리 (MXP_Tools 가 emit 한 hex 파일을 $readmemh 로 받음) ──
// 사이즈는 MXP_Tools 가 생성한 양식과 일치. 정확 사이즈는 Task 5 산출물
// (docs/superpowers/notes/mxp-driving-sequence.md) 참고. 아래는 plausible
// upper-bound; 실 데이터는 일부만 사용.
reg [31:0] a_bs    [0:65535];     // bit-serial A
reg [255:0] b_bp   [0:65535];     // bit-parallel B (32×8 bit per row)
reg [7:0]  a_scale [0:8191];      // E8M0 scales for A
reg [7:0]  b_scale [0:8191];      // E8M0 scales for B
```

main initial 의 zero-priming 직후:
```verilog
// ─── LOAD ──────────────────────────────────────────────────────
$readmemh({WORK_DIR, "/hw_input/a_input_BS_8.hex"}, a_bs);
$readmemh({WORK_DIR, "/hw_input/b_input_8.hex"},    b_bp);
$readmemh({WORK_DIR, "/hw_input/a_scale_8.hex"},    a_scale);
$readmemh({WORK_DIR, "/hw_input/b_scale_8.hex"},    b_scale);

// ─── CONFIG ────────────────────────────────────────────────────
sram_D_use_zero <= 1'b0;  // 이제부터 SRAM.D = RMW.out_RMW
in_Station_control <= 2'b01;   // A8 mode (TBD: 정확 값은 Task 5 산출물)
in_Wcontrol        <= 2'b11;   // W8 mode (Accumulator_Col.v § IMPLICIT_total)

// ─── DRIVE — 4×4×4 = 64 K-tile / m_tile / n_tile loop ─────────
// (정확한 cycle pattern 은 docs/superpowers/notes/mxp-driving-sequence.md
//  를 따라 구현. 아래는 skeleton.)
integer k_tile, m_tile, n_tile, t, col;
for (k_tile = 0; k_tile < 4; k_tile = k_tile + 1)
for (m_tile = 0; m_tile < 4; m_tile = m_tile + 1)
for (n_tile = 0; n_tile < 4; n_tile = n_tile + 1) begin

    // (a) station chain 으로 a_scale / station_control / loadEN 발사
    //     — Task 5 산출물의 cycle pattern 따라

    // (b) accumulator chain 으로 b_scale / start_accumulate / Wcontrol 발사

    // (c) SystolicArray 에 in_a (bit-serial 32 cyc) + in_b (8 cyc) 발사
    //     32 cyc 동안 매 cycle a_bs[...] 한 word 발사, in_b 는 8 cyc 만큼.

    // (d) out_fire pulse 감시 — col j 의 fire 가 들어오면:
    //     - rmw_in_GEMM <= sign-ext(out_accumulate[j*60 +: 21])
    //     - rmw_scale   <= out_scale[j*36 +: 9]
    //     - SRAM read 발사 (CEB=0, WEB=1) 시점 = fire - 1 cyc (read latency 1 정렬)
    //     - SRAM write 발사 (CEB=0, WEB=0) 시점 = fire + RMW_LAT (= 5 cyc)
    //     - 주소: addr(m_tile, n_tile, j) = (m_tile*32 + ?) * 128 + (n_tile*32 + j)

    // (TB 정확 구현은 Task 5/6 산출물 + sim waveform iteration 으로 확정.)
end

// ─── DRAIN ─────────────────────────────────────────────────────
repeat (10) @(posedge clk);    // 5 cyc RMW + 여유
sram_CEB <= 1'b1;

// (이미 작성된 DUMP 단계는 그대로 — bank{0..15}.mem 16 개 dump)
```

> 이 task 의 step 2 는 honest 하게 **skeleton 만** 박았음. 실제 cycle-by-cycle 발사는 Task 5/6 산출물을 참고하여 채워야 함. 채우는 과정에서 sim waveform 보며 iterate 함 — RMW unit 의 vector TB (Task 9 of RMW plan, `bash sim/run_rmw.sh`) 가 정확히 그런 식으로 작성됐음 (`docs/superpowers/plans/2026-05-14-rmw-implementation.md` 참고).

- [ ] **Step 3: `sim/run_integration_one.sh` 작성**

```bash
#!/bin/bash
# sim/run_integration_one.sh — 1 mode integration sim.
# Usage: bash sim/run_integration_one.sh <LABEL> <A_PREC> <B_PREC>
set -e
cd "$(dirname "$0")/.."

LABEL="$1"
A_P="$2"
B_P="$3"
WORK="work/${LABEL}"
DUMP="${WORK}/hw_out"
BUILD="sim/build/${LABEL}"

mkdir -p "$DUMP" "$BUILD"

SRC_ROOT="gemm_sram.srcs/sources_1"
HF_ROOT="third_party/berkeley-hardfloat"

(cd "$BUILD" && \
    xvlog -sv \
        ../../../$HF_ROOT/HardFloatBundle.v \
        ../../../$SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
        ../../../$SRC_ROOT/imports/Desktop/sram/rtl/*.v \
        ../../../$SRC_ROOT/new/int_to_fp32.v \
        ../../../$SRC_ROOT/new/fp32_adder.v \
        ../../../$SRC_ROOT/new/RMW.v \
        ../../../$SRC_ROOT/new/gemm_sram_top.v \
        ../../../tb/gemm_sram_top_tb.v && \
    xelab -L work gemm_sram_top_tb -snapshot gemm_sram_top_tb_snap && \
    xsim work.gemm_sram_top_tb_snap -runall \
        -testplusarg "A_PREC=${A_P}" \
        -testplusarg "B_PREC=${B_P}" \
        -testplusarg "WORK_DIR=../../../${WORK}" \
        -testplusarg "DUMP_DIR=../../../${DUMP}")

echo "Sim ${LABEL} done. Dump → ${DUMP}/"
```
`chmod +x sim/run_integration_one.sh`.

- [ ] **Step 4: A8_B8 sim 실행 + compare gate**

```bash
bash sim/run_integration_one.sh A8_B8 8 8

cd MXP_Tools
python -m mxp_tools compare \
    --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
    --hw-banks ../work/A8_B8/hw_out/bank0.mem ../work/A8_B8/hw_out/bank1.mem \
               ../work/A8_B8/hw_out/bank2.mem ../work/A8_B8/hw_out/bank3.mem \
               ../work/A8_B8/hw_out/bank4.mem ../work/A8_B8/hw_out/bank5.mem \
               ../work/A8_B8/hw_out/bank6.mem ../work/A8_B8/hw_out/bank7.mem \
               ../work/A8_B8/hw_out/bank8.mem ../work/A8_B8/hw_out/bank9.mem \
               ../work/A8_B8/hw_out/bank10.mem ../work/A8_B8/hw_out/bank11.mem \
               ../work/A8_B8/hw_out/bank12.mem ../work/A8_B8/hw_out/bank13.mem \
               ../work/A8_B8/hw_out/bank14.mem ../work/A8_B8/hw_out/bank15.mem \
    --layout interleaved_row_major_16bank
cd ..
```
Expected:
```
COMPARE PASS: 128×128 elements, bit-exact
```

- [ ] **Step 5: Mismatch debug iteration (필요 시)**

만약 mismatch 발생:
1. `MXP_Tools/mxp_tools/compare.py` 가 출력하는 첫 mismatch 의 (m, n) 위치 확인
2. 그 (m, n) 의 SRAM addr 계산: `flat = m*128+n, bank=flat%16, word=flat//16`
3. `work/A8_B8/hw_out/bank{bank}.mem` 의 `word` 째 line 의 hex 확인
4. `sw_ref/C_sw_*.npz` 의 같은 (m,n) 값과 bit-pattern 비교
5. RMW unit 자체는 unit test 71/71 PASS → mismatch 는 ① driving sequence, ② lane 디스패치 매핑, ③ first-tile init 셋 중 하나
6. waveform 확인: `xsim -gui` 로 dump 직전의 SRAM mem 내용 직접 봄

각 fix 는 sub-commit. 통합 driving 의 디버깅이라 한 번에 안 끝남 — RMW vector TB 가 같은 패턴으로 iterate 했음 (`docs/superpowers/specs/2026-05-14-rmw-design.md` § 디버깅 노트 참고).

- [ ] **Step 6: PASS 후 commit**

```bash
git add tb/gemm_sram_top_tb.v sim/run_integration_one.sh
git commit -m "Task 7: A8_B8 end-to-end PASS (128×128 bit-exact vs MXP_Tools golden)"
```

---

## Task 8: 나머지 8 mode 확장 (A {2,4,8} × B {2,4,8} \ A8_B8)

**Files:**
- Modify: `tb/gemm_sram_top_tb.v` (A_PREC / B_PREC 분기 추가)

A8_B8 외 8 mode 의 driving 시퀀스 차이:
- **A4 / A2 모드**: `in_Station_control` 다른 값 + lane 디스패치 2 또는 4 회 per col fire (Task 6 산출물의 (m,n) 매핑 사용)
- **B4 / B2 모드**: `in_Wcontrol` 다른 값 (Accumulator_Col.v 의 `impl_w` 분기와 일치)
- 그 외 GEMM 입력 driving cycle 은 동일

- [ ] **Step 1: TB 의 CONFIG 분기 추가**

`tb/gemm_sram_top_tb.v` 의 CONFIG 부분:

```verilog
// A_PREC → in_Station_control
case (A_PREC)
    8: in_Station_control <= 2'b01;  // A8 mode
    4: in_Station_control <= 2'b10;  // A4 mode
    2: in_Station_control <= 2'b11;  // A2 mode
endcase
// (정확한 인코딩은 Accumulator_Col.v 의 in_Mode_oh 디코딩 / Station.v 의 변환 로직 참고)

// B_PREC → in_Wcontrol
case (B_PREC)
    8: in_Wcontrol <= 2'b11;   // W8
    4: in_Wcontrol <= 2'b10;   // W4
    2: in_Wcontrol <= 2'b01;   // W2
endcase

// 입력 .hex 파일 경로도 P 에 따라 분기
case (A_PREC)
    8: $readmemh({WORK_DIR, "/hw_input/a_input_BS_8.hex"}, a_bs);
    4: $readmemh({WORK_DIR, "/hw_input/a_input_BS_4.hex"}, a_bs);
    2: $readmemh({WORK_DIR, "/hw_input/a_input_BS_2.hex"}, a_bs);
endcase
case (B_PREC)
    8: $readmemh({WORK_DIR, "/hw_input/b_input_8.hex"}, b_bp);
    4: $readmemh({WORK_DIR, "/hw_input/b_input_4.hex"}, b_bp);
    2: $readmemh({WORK_DIR, "/hw_input/b_input_2.hex"}, b_bp);
endcase
// scale 도 동일 패턴
```

- [ ] **Step 2: TB 의 DRIVE 안의 lane 디스패치 분기 추가**

각 col j 의 fire pulse 처리 부분:

```verilog
// fire 시 lane 디스패치
case (A_PREC)
    8: begin
        // A8: 1 lane — out_accumulate[j*60 +: 21] sign-extended to 32 bit
        rmw_in_GEMM <= {{11{out_accumulate[j*60+20]}}, out_accumulate[j*60 +: 21]};
        rmw_scale   <= out_scale[j*36 +: 9];                 // comb_s0
        // SRAM addr 1 발사 (lane 1 개)
    end
    4: begin
        // A4: 2 lane — s1_a[17:0], s1_b[17:0]
        // 2 cycle 에 걸쳐 발사 (또는 RMW 1 개라서 시간 분할 직렬화)
        // 정확한 lane→(m,n) 매핑은 Task 6 산출물 참고
        // lane 0: s1_a — rmw_in_GEMM <= {{14{out_accumulate[j*60+35]}}, out_accumulate[j*60+18 +: 18]}
        // lane 1: s1_b — rmw_in_GEMM <= {{14{out_accumulate[j*60+17]}}, out_accumulate[j*60    +: 18]}
        // scale: comb_s0, comb_s1
        // 발사 회수 = 2 per col
    end
    2: begin
        // A2: 4 lane — out_INT2_{0,1,2,3} (15-bit each)
        // 4 cycle 직렬 발사
        // scale: comb_s0..3
        // 발사 회수 = 4 per col
    end
endcase
```

> 실제 인코딩 (2'b01/10/11 → A8/A4/A2 매핑) 은 Task 5 산출물 + `Accumulator_Col.v` 의 `in_Mode_oh` (3-bit one-hot) 디코딩 경로 확인 후 확정.

- [ ] **Step 3: 8 mode 순차 실행 + compare**

```bash
for A_P in 2 4 8; do
  for B_P in 2 4 8; do
    if [ "${A_P}_${B_P}" = "8_8" ]; then continue; fi   # Task 7 에서 이미 PASS
    LABEL="A${A_P}_B${B_P}"
    cd MXP_Tools && \
      python -m mxp_tools gen   --out ../work/${LABEL} -M 128 -K 128 -N 128 --seed 0 && \
      python -m mxp_tools emit  --out ../work/${LABEL} --prec-a ${A_P} --prec-b ${B_P} && \
      python -m mxp_tools ref   --out ../work/${LABEL} --prec-a ${A_P} --prec-b ${B_P} && \
      cd ..
    bash sim/run_integration_one.sh "${LABEL}" "${A_P}" "${B_P}"
    cd MXP_Tools && \
      BANKS=$(printf "../work/${LABEL}/hw_out/bank%d.mem " {0..15}) && \
      python -m mxp_tools compare \
        --ref ../work/${LABEL}/sw_ref/C_sw_mxint${A_P}_mxint${B_P}.npz \
        --hw-banks ${BANKS} \
        --layout interleaved_row_major_16bank
    cd ..
    echo "${LABEL}: PASS"
  done
done
echo "8 additional modes done"
```

각 mode 실패 시 step 5 (Task 7 의 mismatch debug) 같은 iteration.

- [ ] **Step 4: Commit (각 모드 PASS 시 sub-commit, 마지막에 wrap-up commit)**

```bash
# 모든 mode PASS 후
git add tb/gemm_sram_top_tb.v
git commit -m "Task 8: extend TB for all 9 modes (A,B ∈ {2,4,8}) — all PASS sequentially"
```

---

## Task 9: 9-mode serial sweep wrapper

**Files:**
- Create: `sim/run_integration_sweep.sh`

- [ ] **Step 1: `sim/run_integration_sweep.sh` 작성**

```bash
#!/bin/bash
# sim/run_integration_sweep.sh — 9 mode 직렬 sweep + compare gate.
# 한 모드라도 mismatch 면 exit 1.
set -e
cd "$(dirname "$0")/.."

PASSED=()
FAILED=()

for A_P in 2 4 8; do
  for B_P in 2 4 8; do
    LABEL="A${A_P}_B${B_P}"
    echo "=== ${LABEL} ==="

    # 1) MXP_Tools side (gen/emit/ref)
    (cd MXP_Tools && \
      python -m mxp_tools gen   --out ../work/${LABEL} -M 128 -K 128 -N 128 --seed 0 && \
      python -m mxp_tools emit  --out ../work/${LABEL} --prec-a ${A_P} --prec-b ${B_P} && \
      python -m mxp_tools ref   --out ../work/${LABEL} --prec-a ${A_P} --prec-b ${B_P})

    # 2) HW sim
    bash sim/run_integration_one.sh "${LABEL}" "${A_P}" "${B_P}"

    # 3) Compare gate
    BANKS=$(printf "../work/${LABEL}/hw_out/bank%d.mem " {0..15})
    if (cd MXP_Tools && \
        python -m mxp_tools compare \
            --ref ../work/${LABEL}/sw_ref/C_sw_mxint${A_P}_mxint${B_P}.npz \
            --hw-banks ${BANKS} \
            --layout interleaved_row_major_16bank); then
      PASSED+=("${LABEL}")
    else
      FAILED+=("${LABEL}")
    fi
  done
done

echo ""
echo "===================================="
echo "SWEEP RESULT: ${#PASSED[@]}/9 PASS"
echo "PASSED: ${PASSED[*]}"
[ ${#FAILED[@]} -gt 0 ] && echo "FAILED: ${FAILED[*]}" && exit 1
echo "ALL 9 MODES PASSED"
```
`chmod +x sim/run_integration_sweep.sh`.

- [ ] **Step 2: 전체 sweep 실행**

```bash
bash sim/run_integration_sweep.sh
```
Expected 마지막 줄:
```
ALL 9 MODES PASSED
```

- [ ] **Step 3: Commit**

```bash
git add sim/run_integration_sweep.sh
git commit -m "Task 9: serial 9-mode sweep script (gate: ALL 9 MODES PASSED)"
```

---

## Task 10: 9-mode parallel dispatch (optional)

**Files:**
- Create: `sim/run_integration_parallel.sh` (dispatch 정보 + agent template)

Task 9 의 serial sweep 가 잘 굴러간다면 그 자체로 검증 끝. 이 task 는 sim 시간 단축을 위한 **parallel** 옵션.

- [ ] **Step 1: 메모리 / 디스크 footprint 측정**

```bash
ls -lh sim/build/A8_B8/         # xsim work dir 1 개 사이즈
du -sh work/A8_B8/              # MXP_Tools 출력 1 mode 사이즈
```
9 mode 합산 = (xsim build × 9) + (work × 9). 16 GB 머신에서 동시 9 개 실행 가능한지 추정.

- [ ] **Step 2: `sim/run_integration_parallel.sh` 작성 (subagent dispatch 가이드)**

```bash
#!/bin/bash
# sim/run_integration_parallel.sh — 9-way parallel sweep guidance.
#
# 이 스크립트는 직접 9 sim 을 fork 하지 않고, Claude Agent 가
# superpowers:dispatching-parallel-agents 패턴으로 9 subagent 를
# dispatch 할 때의 task template 을 출력한다. 각 subagent 는 동일한
# 입력 (mode label, A_PREC, B_PREC) 을 받고 동일한 절차로 수행.
#
# 사용법 (Agent 측):
#   1) 이 스크립트를 cat 하여 task template 확인
#   2) 9 subagent 동시 dispatch (each receives A_PREC/B_PREC)
#   3) 결과 aggregator: 9/9 PASS 또는 fail list
cat <<'EOF'
=== Per-subagent task template ===

INPUTS:
  - A_PREC ∈ {2, 4, 8}
  - B_PREC ∈ {2, 4, 8}
  - WORKING_DIR = repo root (read-only RTL + isolated work/A{A_PREC}_B{B_PREC}/)

STEPS:
  1. cd MXP_Tools && python -m mxp_tools gen --out ../work/A${A_PREC}_B${B_PREC} ...
  2. python -m mxp_tools emit --out ../work/... --prec-a ${A_PREC} --prec-b ${B_PREC}
  3. python -m mxp_tools ref --out ../work/... --prec-a ${A_PREC} --prec-b ${B_PREC}
  4. cd .. && bash sim/run_integration_one.sh "A${A_PREC}_B${B_PREC}" ${A_PREC} ${B_PREC}
  5. cd MXP_Tools && python -m mxp_tools compare ... (위 task 9 와 동일)
  6. exit 0 if PASS, exit 1 if mismatch (with diagnostic)

ISOLATION:
  - 각 subagent 의 work dir: work/A{A_PREC}_B{B_PREC}/  (mode 별 disjoint)
  - 각 subagent 의 xsim build dir: sim/build/A{A_PREC}_B{B_PREC}/
  - RTL source: read-only — 모든 subagent 가 같은 파일 공유

AGGREGATOR (Agent):
  - 9 task 결과 수집 → "9/9 PASS" 또는 "fail at <list>"
EOF
```

- [ ] **Step 3: 첫 dispatch 실행 (실제)**

Agent 측에서:
1. `superpowers:dispatching-parallel-agents` 스킬 invoke
2. 9 subagent dispatch with `(A_PREC, B_PREC)` ∈ {2,4,8}²
3. 결과 취합

Subagent 가 사용하는 도구: Bash (xvlog/xelab/xsim/python), Read/Edit (디버그용)

- [ ] **Step 4: Commit**

```bash
git add sim/run_integration_parallel.sh
git commit -m "Task 10: parallel dispatch template for 9-mode integration verification"
```

---

## Task 11: Vivado 프로젝트 + 문서 정리

**Files:**
- Modify: `gemm_sram.xpr` (top → `gemm_sram_top`)
- Modify: `CLAUDE.md` (kickoff 갱신)
- Modify or Delete: `docs/next-session-kickoff.md`

- [ ] **Step 1: Vivado xpr 의 top 변경**

```bash
# Vivado 의 xpr 파일을 직접 편집 (XML). top 속성 변경:
#   <Property Name="top" Val="GEMM"/>
#   → <Property Name="top" Val="gemm_sram_top"/>
grep -n 'Val="GEMM"' gemm_sram.xpr
```

발견되면 sed 또는 수동 편집:
```bash
sed -i 's|<Property Name="top" Val="GEMM"/>|<Property Name="top" Val="gemm_sram_top"/>|' gemm_sram.xpr
```

Vivado 에서 열어서 확인:
- Sources 패널: `gemm_sram_top` 이 top icon 으로 표시
- "Run Synthesis" 또는 "Open Elaborated Design" 한 번 → 합성 무에러 (compile-clean 만 확인, timing 은 본 spec scope 아님).

- [ ] **Step 2: `CLAUDE.md` 의 "Next session kickoff" 섹션 갱신**

기존 RMW 단계 끝 메시지 → 통합 단계 끝 메시지로 교체:

```markdown
## Next session kickoff (YYYY-MM-DD, integration done — production / 합성 단계로)

GEMM ↔ RMW ↔ SRAM 통합이 **완성 및 검증**됨 — `bash sim/run_integration_sweep.sh` → ALL 9 MODES PASSED. 기존 `docs/next-session-kickoff.md` 의 4 가지 열린 질문은 모두 해결되거나 잠정값으로 박힘 (재검토 트리거 명시). 다음 단계 후보:

1. **잠정값 재검토** — RMW instance 수 / loop order (spec 의 "잠정 결정값" 표 참고). 합성/timing closure 시작.
2. **Future scope** — SRAM 을 weight input 저장소로 사용. 별도 spec 작성.

이전 통합 단계 흐름은 `docs/superpowers/specs/2026-05-14-integration-design.md` + `docs/superpowers/plans/2026-05-14-integration-implementation.md` 참고.
```

- [ ] **Step 3: `docs/next-session-kickoff.md` 정리**

옵션 A: **delete** — 통합이 끝나 더 이상 안 씀.
옵션 B: **modify** — 통합 후 다음 단계 (합성 / future scope) 의 kickoff 로 재활용.

선택 B 가 자연스러움 — 통합 끝난 시점의 다음 unknown 을 정리:

```markdown
# 다음 세션 kickoff — 통합 후 단계

작성: YYYY-MM-DD, 통합 단계 완료 직후.

## 지금 어디까지 왔나
- RMW unit 완성 (sim/run_rmw.sh → 71/71 PASS)
- GEMM ↔ RMW ↔ SRAM 통합 완성 (sim/run_integration_sweep.sh → 9/9 PASS, all modes bit-exact)
- 잠정값: RMW 1 개 / loop order K-outermost / 워크로드 128³ / INTERLEAVED bank

## 다음 단계 후보 (사용자 선택)

1. **Throughput 확장** — RMW instance 수 늘려서 sim 시간 / 합성 후 cycle count 비교
2. **Timing closure** — Vivado 합성, 250 MHz wrapper / xc7vx485 target 으로 closure 시도 (MXP standalone 의 패턴 답습)
3. **Future scope: SRAM weight storage** — 별도 spec 으로 진행
4. **MXP_Tools 통합 — `compare` 결과를 CI 로 묶기** — 9 mode 자동 회귀 테스트
```

- [ ] **Step 4: Commit**

```bash
git add gemm_sram.xpr CLAUDE.md docs/next-session-kickoff.md
git commit -m "Task 11: Vivado top → gemm_sram_top, refresh CLAUDE.md kickoff"
```

---

## 완료 기준 (Done When)

- ✅ Task 1-9 모두 commit 됨
- ✅ `bash sim/run_integration_sweep.sh` 의 마지막 줄 = `ALL 9 MODES PASSED`
- ✅ 9 mode 각각의 `MXP_Tools compare` bit-exact PASS
- ✅ Vivado 에서 `gemm_sram.xpr` 열고 elaborate → 무에러
- ✅ CLAUDE.md / kickoff 갱신
- (Optional) Task 10 — parallel dispatch 한 번이라도 성공적으로 실행

Task 11 까지 완료되면 본 plan 종료. 다음 단계는 새 spec / plan 으로.

---

## Self-Review Note

- spec § 1 Goals 3 항목 → Task 7-9 의 PASS gate 로 cover
- spec § 2 새 top + 컨트롤러 없음 → Task 1 의 gemm_sram_top.v
- spec § 3 bank 매핑 → Task 3 의 mapping callable
- spec § 4 first-tile init → Task 1 의 `sram_D_use_zero` mux + Task 2 의 zero priming loop
- spec § 5 TB 단계 → Task 2 (skeleton) + Task 7 (full) + Task 8 (9-mode 확장)
- spec § 6 9-mode sweep / parallel → Task 9 / Task 10
- spec § 7 MXP_Tools 연동 → Task 3
- spec § 8 검증 contract → Task 7-9 의 compare gate
- spec § 9 잠정 결정값 → Task 11 의 CLAUDE.md kickoff 에 명시 인계
- spec § 10 open items → Task 5 (driving 시퀀스) + Task 6 (lane→C 매핑) 으로 plan 안에서 해결
- spec § 11 산출물 파일 → Task 1-3, 7, 9, 10 의 [NEW] 파일들과 일치
- spec § 12 검증 단계 → Task 1 (elab) + Task 4 (smoke compare) + Task 7-9 (PASS gates)

Placeholder / TBD 스캔: Task 5/6/7 에 "정확 cycle pattern 은 sim waveform 보며 iterate" 식의 honest-uncertainty 표현 — 통합 sim 의 본질상 첫 시도에 정확 cycle 못 박음. RMW unit plan 의 Task 9 도 같은 패턴 (`docs/superpowers/plans/2026-05-14-rmw-implementation.md` 참고).
