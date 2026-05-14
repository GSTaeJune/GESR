# GEMM ↔ RMW ↔ SRAM 시스템 통합 Design

**Date:** 2026-05-14
**Status:** Design — brainstorming 결과 정리, implementation plan 직전 단계
**Scope:** `GEMM.v` (현재 MXP TOP), `RMW.v` (검증된 단위 산술기), `sram_1rw_banked.v` 세 IP 를 묶어 RMW 연산이 SRAM 위에서 정상 동작하는 datapath 를 구성. **합성 가능한 컨트롤러는 만들지 않는다** — control 은 전부 TB 가 signal level 로 처리. 본 spec 의 산출물은 `gemm_sram_top.v` (pure structural wrapper), `tb/gemm_sram_top_tb.v`, `sim/run_integration_*.sh`, `mxp_tools/hwio.py` 매핑 추가분.

---

## 1. 목적 / Goals / Non-goals

### Goals
1. `RMW.v` 가 시스템 안 (GEMM 출력 + SRAM 누적) 에서도 unit-level 과 bit-exact 동일하게 동작함을 확인.
2. `sram_1rw_banked.v` 의 RW 사이클이 정상 동작함을 확인 (16 bank 모두 활용되는 매핑 사용).
3. **9 개 정밀도 조합 (A {2,4,8} × B {2,4,8}) 전부에서** `MXP_Tools` golden reference 와 **100 % bit-exact 일치** 달성.

### Non-goals (이 spec scope 가 아님)
- Loop order 확정 — 잠정 K-outermost. 재검토 트리거: dataflow 재최적화 단계 진입.
- RMW instance 개수 최적화 — 잠정 **1 개**. 재검토 트리거: throughput / area 목표 도래, 또는 sim 시간 과다.
- Throughput / area / timing closure.
- 합성 가능한 컨트롤러 (FSM / arbiter / scheduler) — TB 에서 signal driving 으로 대체.
- A4 / A2 모드의 lane → C[m,n] 매핑 정확도 — 인터페이스만 미리 노출, 실제 mapping 은 plan 단계에서 `MXP_Tools/gemm.py` 와 cross-check 후 확정.

### Future scope (참고용, 본 spec 에서는 안 함)
- SRAM 을 **input weight (B 행렬) 저장소** 로도 사용 — `$readmemh` 로 SRAM init, GEMM 의 `in_b` 가 SRAM read 결과를 직접 받음. 현재는 A/B 둘 다 TB 가 직접 driving.

---

## 2. 시스템 wiring (current → target)

### 현재 상태
`gemm_sram.xpr` 의 top 은 `GEMM` (= unmodified MXP TOP). `RMW.v` 는 standalone 검증만 됨 (71/71 PASS via `sim/run_rmw.sh`). SRAM 은 RTL 이 import 되어 있으나 어디에도 instantiate 안 됨.

### Target — `gemm_sram_top.v` (NEW, pure structural)

```
                ┌──────────────── gemm_sram_top.v ─────────────────┐
                │                                                   │
       ┌────────┴────┐                                              │
       │  GEMM       │  out_accumulate[32*60]   ─── exposed ───────►│  to TB
       │ (현재 MXP    │  out_scale[32*36]        ─── exposed ───────►│  to TB
       │  TOP, 변경   │  out_fire[32]            ─── exposed ───────►│  to TB
       │  없음)       │                                              │
       └─────────────┘                                              │
                                                                    │
       ┌──────────────┐                                             │
       │  RMW (unit,  │  in_GEMM[32]  ◄── exposed ───── from TB     │
       │  변경 없음)   │  scale[9]     ◄── exposed ───── from TB     │
       │  L_CONV=2,   │  in_SRAM[32]  ◄─────────┐                    │
       │  L_ADD=3,    │  out_RMW[32]  ──────────┼─┐                  │
       │  total 5cyc) │                         │ │                  │
       └──────────────┘                         │ │                  │
                                                │ │                  │
       ┌────────────────────┐                   │ │                  │
       │  sram_1rw_banked   │  Q[32] ───────────┘ │                  │
       │  16×32K×32-bit,    │  D[32] ─────────────┘                  │
       │  INTERLEAVED,      │  A[18], CEB, WEB, WMASK[32]            │
       │  PIPELINE=0        │            ◄── exposed ──── from TB    │
       └────────────────────┘  ──── Q[32] mirror ───────► to TB (probe)│
                                                                    │
       └────────────────────────────────────────────────────────────┘
```

**`gemm_sram_top.v` 내부 logic 양**: 두 줄. 그 외 전부 외부 노출.
- `assign u_rmw.in_SRAM = u_sram.Q;`
- `assign u_sram.D      = u_rmw.out_RMW;`

(WMASK 는 외부에서 받음 — TB 가 매번 `32'hFFFFFFFF` 발사하지만, future 에서 partial write 가능성 열어둠.)

GEMM 의 모든 driving 신호, RMW 의 `in_GEMM` / `scale`, SRAM 의 `A` / `CEB` / `WEB` / `WMASK` 는 top 외부 포트로 노출 → TB 가 직접 driving.

### `gemm_sram_top.v` 포트 시그니처 (요약)

```verilog
module gemm_sram_top (
    input  wire        clk,
    input  wire        rst,

    // ───── GEMM stimuli (현 GEMM.v 포트 그대로 passthrough) ─────
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

    // ───── GEMM probe (TB 가 lane / scale 디스패치 위해 봄) ─────
    output wire [32*60-1:0]                  out_accumulate,
    output wire [32*36-1:0]                  out_scale,
    output wire [32-1:0]                     out_fire,

    // ───── RMW input (TB 가 직접 발사) ─────
    input  wire [31:0]                       rmw_in_GEMM,
    input  wire [8:0]                        rmw_scale,
    output wire [31:0]                       rmw_out_RMW,    // probe

    // ───── SRAM control (TB 가 직접 발사) ─────
    input  wire                              sram_CEB,
    input  wire                              sram_WEB,
    input  wire [18:0]                       sram_A,         // ADDR_WIDTH = clog2(16)+clog2(32768) = 4+15 = 19
    input  wire [31:0]                       sram_WMASK,
    output wire [31:0]                       sram_Q          // probe
);
```

본 spec 의 워크로드 (16384 word) 는 하위 14 bit 만 사용. 포트 width 는 wrapper 의 ADDR_WIDTH = 19 그대로 노출.

---

## 3. SRAM bank-column 매핑

### 파라미터
```verilog
sram_1rw_banked #(
    .DATA_WIDTH    (32),
    .NUM_BANKS     (16),
    .BANK_DEPTH    (32768),
    .BANK_STRATEGY ("INTERLEAVED"),
    .PIPELINE      (0)
) u_sram (...);
```
Read latency = 1 cyc (`bank_sel_d1` 단 1 단).

### C[m,n] → SRAM 주소

**row-major flat + INTERLEAVED**:
```
flat_addr(m, n) = m * N + n           // M = N = 128 → flat ∈ [0, 16384)
bank_sel        = flat_addr[3:0]      // INTERLEAVED → LSB 4비트
within_a        = flat_addr[18:4]
```

```
예시 (M=N=128):
  C[  0,  0] → flat 0     → bank 0  word 0
  C[  0,  1] → flat 1     → bank 1  word 0
  C[  0, 15] → flat 15    → bank 15 word 0
  C[  0, 16] → flat 16    → bank 0  word 1
  C[  1,  0] → flat 128   → bank 0  word 8
  C[127,127] → flat 16383 → bank 15 word 1023
```
Bank 별 사용량 = 16384 / 16 = **1024 word** (BANK_DEPTH 32768 의 3 %). 16 bank 모두 균등 사용 → SRAM banking 동작 확인 가능.

### 잠정 워크로드

| 항목 | 값 |
|---|---|
| M | 128 (MXP_Tools default) |
| K | 128 |
| N | 128 |
| C 사이즈 | 128×128 × 32-bit = 64 KB |
| K-step 수 | K / 32 = 4 |
| (m_tile, n_tile) | 4 × 4 = 16 |
| RMW 발사 횟수 (A8) | K-step × tile × col = 4 × 16 × 32 = **2048** |
| RMW 발사 횟수 (A4) | 2048 × 2 = 4096 |
| RMW 발사 횟수 (A2) | 2048 × 4 = 8192 |

Sim 시간이 과도하면 (예: > 10 분) plan 단계에서 더 작은 형상으로 축소 (예: M=K=N=32, 한 K-tile · 한 output tile).

---

## 4. First-tile init — TB 가 SRAM zero priming

RMW 의 `in_SRAM` 은 SRAM `Q` 에 hardwired → TB 가 zero 강제 mux 를 끼울 수 없음. 대신 sim 시작 시 SRAM 전체 (사용 영역) 를 `32'h00000000` 으로 priming.

```verilog
// TB 의 INIT 단계
initial begin
    rst       = 1; #100; rst = 0;
    sram_CEB  = 1; sram_WEB = 1;
    // ... 모든 GEMM 입력 0 / idle ...

    // C 영역 zero priming
    for (i = 0; i < 16384; i = i + 1) begin
        @(posedge clk);
        sram_CEB <= 1'b0;
        sram_WEB <= 1'b0;
        sram_A   <= i[18:0];
        // sram_D 는 RMW.out_RMW 에 hardwired — 이 시점에는 RMW 출력이 미정의일 수 있어
        // wmask 는 0 으로? 아니면 SRAM 의 D 입력에 mux 가 필요?
    end
end
```

### 문제: `sram_D = rmw_out_RMW` hardwire 와 zero priming 충돌

SRAM 의 D 가 RMW 출력에 hardwired 되어있으면, zero priming 단계에선 D 가 미정의 (RMW 입력이 idle). 해결책 후보:

**옵션 A (선택)**: `gemm_sram_top.v` 의 D wiring 을 mux 로:
```verilog
// 추가 외부 포트:
//   input wire        sram_D_use_zero;
// 내부 wiring:
assign u_sram.D = sram_D_use_zero ? 32'h0 : u_rmw.out_RMW;
```
→ TB 가 priming 단계에서 `sram_D_use_zero = 1`, GEMM driving 단계에서 `= 0`.

옵션 B: 별도 외부 `sram_D_override[32]` + `use_override` 포트.

**채택: 옵션 A** — 1-bit 추가 포트만 필요, top 의 wiring 도 한 줄. "컨트롤러 만들지 마" 의 정신을 살짝 우회하지만 SRAM init 의 부득이한 회로이고 logic 이 mux 1 개로 trivial.

→ 최종 `gemm_sram_top.v` 의 추가 포트:
```verilog
input  wire        sram_D_use_zero;   // 1 = D 강제 0, 0 = RMW 출력 사용
```
→ 내부 wiring 도 한 줄:
```verilog
assign u_sram.D = sram_D_use_zero ? 32'h00000000 : u_rmw.out_RMW;
```

---

## 5. TB 구조

### 파일: `tb/gemm_sram_top_tb.v`

단일 TB 파일. plusarg 로 9 mode 중 1 개 조합 선택:
```
+A_PREC = {2, 4, 8}                  // activation precision
+B_PREC = {2, 4, 8}                  // weight precision
+TILE_M = 128 (default)
+TILE_K = 128
+TILE_N = 128
+WORK_DIR = "work/A8_B8"             // MXP_Tools 가 생성한 입력 .hex 디렉토리
+DUMP_DIR = "work/A8_B8/hw_out"      // SRAM 덤프 출력 디렉토리
```

### 단계 (sequential, `initial` 안에서)

```
1. INIT
   - rst pulse
   - SRAM zero priming (16384 word, sram_D_use_zero = 1)
   - sram_D_use_zero <= 0

2. LOAD
   - $readmemh: a_input_BS_{PREC}.hex, b_input_{PREC}.hex,
                 a_scale_{PREC}.hex, b_scale_{PREC}.hex

3. CONFIG
   - A_PREC → in_Station_control 시퀀스 (A8/A4/A2 모드 set)
   - B_PREC → in_Wcontrol 시퀀스 (W8/W4/W2)

4. DRIVE  (loop: k_tile × m_tile × n_tile)
   for k_tile in [0..3]:
     for m_tile in [0..3]:
       for n_tile in [0..3]:
         a. station chain / acc chain load (precondition)
         b. GEMM 입력 32 cycle 발사 (bit-serial A + bit-parallel B)
         c. out_fire 감시 — 각 col j 의 fire pulse 에 맞춰:
            - lane 디스패치 mux (mode 에 따라 1/2/4 lane 추출)
            - 각 lane 마다:
              * RMW 발사 (rmw_in_GEMM, rmw_scale ← out_accumulate / out_scale 일부)
              * (5 cyc 뒤) SRAM addr 발사 + WEB=0 으로 write
              * 그 사이 (read step) SRAM read 발사 — RMW.in_SRAM 에 prior psum 도착

5. DRAIN  (5 + 1 cyc 여유)
   - RMW pipeline 완전 flush, 마지막 write 완료

6. DUMP
   for b in [0..15]:
     $writememh($sformatf("%s/bank%0d.mem", DUMP_DIR, b),
                u_top.u_sram.g_bank[b].u_bank.mem, 0, 1023);

7. $finish
```

### Driving 디테일은 plan 단계에서

위 step 4 의 lane / scale sub-word 매핑은 MXP 의 dataflow 정확히 파악 후 확정 (CLAUDE.md "MXP control surface" 와 `Accumulator_Col.v` 의 `out_scale` 패킹 참고). Spec 단계에서는 인터페이스만 노출, 실제 driving sequence 는 plan 의 task 로 잡음.

---

## 6. 9-mode sweep + 병렬 dispatch

### 직렬 sweep 옵션 (단순)

`sim/run_integration_sweep.sh`:
```bash
#!/bin/bash
set -e
for A_P in 2 4 8; do
  for B_P in 2 4 8; do
    LABEL="A${A_P}_B${B_P}"
    # 1) MXP_Tools side
    python -m mxp_tools emit --out work/${LABEL} --prec-a ${A_P} --prec-b ${B_P}
    python -m mxp_tools ref  --out work/${LABEL} --prec-a ${A_P} --prec-b ${B_P}
    # 2) HW sim
    bash sim/run_integration_one.sh "${LABEL}" ${A_P} ${B_P}
    # 3) Compare gate
    python -m mxp_tools compare \
      --ref  work/${LABEL}/sw_ref/C_sw_mxint${A_P}_mxint${B_P}.npz \
      --hw-banks work/${LABEL}/hw_out/bank{0..15}.mem \
      --layout interleaved_row_major_16bank
  done
done
echo "ALL 9 MODES PASSED"
```

### 병렬 dispatch (default, 빠름)

`superpowers:dispatching-parallel-agents` 패턴으로 9 subagent 동시 실행:
- 각 agent 작업: 1 mode 의 emit + ref + sim + compare
- 격리: 각 agent 가 `work/A{i}_B{j}/` 디렉토리 + `sim/build/A{i}_B{j}/` xsim work dir 사용
- 공유 상태: read-only RTL (변경 없음)
- 결과 취합: 9/9 PASS 또는 "fail at A{i}_B{j}: <mismatch summary>"

xsim 의 `-work_dir` 또는 `--prj <path>` 인자로 작업 디렉토리 격리. 메모리 부담 시 fallback: 3 wave × 3 mode.

Plan 단계에서 task 9 개로 쪼개고 dispatch 단계 추가.

---

## 7. MXP_Tools 연동

### 7.1 새 mapping callable

`mxp_tools/hwio.py` 에 추가:
```python
def interleaved_row_major_16bank(bank_idx, word_offset, M, N):
    """C[m,n] = SRAM[bank=flat%16, word=flat//16] where flat = m*N+n.

    32 col GEMM 결과를 row-major flat 으로 SRAM 에 쌓고
    sram_1rw_banked 의 INTERLEAVED 매핑 (LSB 4비트가 bank_sel) 으로
    16 bank 에 분산되는 layout 의 역매핑.
    """
    flat = word_offset * 16 + bank_idx
    if flat >= M * N:
        return None
    return divmod(flat, N)
```

### 7.2 CLI `--layout` 분기

`mxp_tools/cli.py` 의 `compare` 명령에 case 1 줄 추가:
```python
elif args.layout == "interleaved_row_major_16bank":
    mapping = hwio.interleaved_row_major_16bank
```

### 7.3 입력 측 변경 없음

A/B input emit (`hw_input/*.hex`) 은 현재 그대로. TB 가 `$readmemh` 로 GEMM input port 에 직접 driving — SRAM 안 거침.

---

## 8. 검증 contract

### Success criterion

```
PASS 조건:
  ∀ (A_PREC, B_PREC) ∈ {2,4,8}²,                       (9 modes)
  ∀ (m, n) ∈ [0, M) × [0, N),                          (M·N elements)
    C_hw[m,n] == C_sw[m,n]   (bit-exact, np.array_equal)

FAIL 시 출력:
  - 어느 mode 인지 (A{i}_B{j})
  - 첫 mismatch 의 (m, n) 위치
  - C_hw, C_sw 의 FP32 hex bit pattern
  - bank{i}.mem 의 해당 word
```

`MXP_Tools/mxp_tools/compare.py` 가 이미 비슷한 출력 포맷 — 그대로 사용.

### Bit-exact 가 의미하는 것

- FP32 IEEE-754 bit pattern 정확 일치 (`np.array_equal(C_hw.view(uint32), C_sw.view(uint32))`).
- `np.allclose` 의 tolerance 비교가 아님 — HardFloat 의 RNE rounding 과 `MXP_Tools/gemm.py::mxint_gemm_golden` 의 FP32 accumulator 결과가 cycle order 까지 같다는 가정.
- 만약 K-step 누적 순서가 SW reference 와 다르면 RNE 차이로 last-bit mismatch 발생 가능 → 그 경우 spec/plan 에서 "K-tile 순서를 SW reference 와 맞춤" 로 추가 결정.

---

## 9. 잠정 결정값 + 재검토 트리거

| 결정 | 잠정값 | 재검토 트리거 |
|---|---|---|
| RMW instance 수 | 1 | throughput / area 목표, sim 시간 과다 |
| Loop order | K-outermost | dataflow 재최적화 단계 |
| 워크로드 (M, K, N) | (128, 128, 128) | sim 시간 > 10 분 |
| Bank strategy | INTERLEAVED | bank-conflict 측정 후 SEQUENTIAL 비교 필요 시 |
| SRAM PIPELINE | 0 | timing closure 단계 |
| RMW L_CONV, L_ADD | 2, 3 (RMW.v default) | timing closure 단계 |

---

## 10. Plan 단계로 미루는 open items

| 항목 | 왜 plan 에서 결정 |
|---|---|
| A4 / A2 모드 lane → C[m,n] 매핑 | `MXP_Tools/gemm.py::mxint_gemm_golden` 정확히 읽고 cross-check 필요 |
| GEMM 입력 시퀀스 (bit-serial A + bit-parallel B 발사 cycle 패턴) | 현 MXP standalone TB (`../MXP/sim/...`) 와 정확히 동일하게 답습 |
| `out_fire` cycle 시점 추적 → RMW 발사 timing | sim waveform 보면서 cycle-by-cycle 맞춤 |
| SRAM read / write 순서 정렬 (read latency 1 + RMW 5 cyc) | timing diagram 으로 정확히 작성 |
| Sim 시간 측정 후 워크로드 축소 결정 | 첫 prototype 돌려봐야 함 |

---

## 11. 산출물 파일 목록

```
gemm_sram.srcs/sources_1/new/
    gemm_sram_top.v                  # NEW — pure structural wrapper

tb/
    gemm_sram_top_tb.v               # NEW — 1 mode plusarg-driven TB

sim/
    run_integration_one.sh           # NEW — 1 mode sim (compile + xelab + xsim)
    run_integration_sweep.sh         # NEW — 9 mode serial sweep wrapper
    (run_integration_parallel.sh)    # OPTIONAL — subagent dispatch wrapper

MXP_Tools/mxp_tools/
    hwio.py                          # MODIFIED — interleaved_row_major_16bank 추가
    cli.py                           # MODIFIED — compare --layout 분기 1줄
```

Vivado 의 top 은 `gemm_sram.xpr` 에서 `GEMM` → `gemm_sram_top` 으로 변경.

---

## 12. 검증 단계 별 통과 기준

| 단계 | 통과 기준 |
|---|---|
| `gemm_sram_top.v` elab | xelab 무에러 (`xvlog` warning 0 / minor warning 만) |
| TB compile | `xvlog`, `xelab` 무에러 |
| Single-mode sim (A8_B8) | `bank{0..15}.mem` 16 개 생성, mismatch 없음 |
| 9-mode sweep | 9/9 PASS |
| 병렬 dispatch (optional) | 9 subagent 동시 실행, 결과 9/9 PASS |

---

## 부록 A — `gemm_sram_top.v` 전체 ports (확정 안)

```verilog
module gemm_sram_top #(
    parameter NUM_BANKS  = 16,
    parameter BANK_DEPTH = 32768
)(
    input  wire        clk,
    input  wire        rst,

    // GEMM (passthrough)
    input  wire [31:0]                       in_a,
    input  wire signed [32*8-1:0]            in_b,
    input  wire [31:0]                       in_control,
    input  wire [31:0]                       in_loadEN,
    input  wire [1:0]                        in_Station_control,
    input  wire [4*8-1:0]                    in_Scale_Activation,
    input  wire                              in_station_control,
    input  wire                              in_station_loadEN,
    input  wire                              in_start_accumulate,
    input  wire [1:0]                        in_Wcontrol,
    input  wire [7:0]                        in_scale_weight,
    output wire [32*60-1:0]                  out_accumulate,
    output wire [32*36-1:0]                  out_scale,
    output wire [31:0]                       out_fire,

    // RMW
    input  wire [31:0]                       rmw_in_GEMM,
    input  wire [8:0]                        rmw_scale,
    output wire [31:0]                       rmw_out_RMW,

    // SRAM
    input  wire                              sram_CEB,
    input  wire                              sram_WEB,
    input  wire [18:0]                       sram_A,         // clog2(16) + clog2(32768) = 19
    input  wire [31:0]                       sram_WMASK,
    input  wire                              sram_D_use_zero,
    output wire [31:0]                       sram_Q
);

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

    wire [31:0] rmw_out_w;
    RMW u_rmw (
        .clk(clk), .rst(rst),
        .in_SRAM(sram_Q),
        .in_GEMM(rmw_in_GEMM),
        .scale(rmw_scale),
        .out_RMW(rmw_out_w)
    );
    assign rmw_out_RMW = rmw_out_w;

    wire [31:0] sram_D_w = sram_D_use_zero ? 32'h0 : rmw_out_w;

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

(이 부록의 신호 width / 패키지 단위는 plan 의 코딩 task 에서 한 번 더 검증.)
