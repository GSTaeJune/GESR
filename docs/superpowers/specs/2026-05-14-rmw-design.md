# RMW Module Design

**Date:** 2026-05-14
**Status:** Design — brainstorming 결과 정리, implementation plan 직전 단계
**Scope:** `RMW.v` 모듈 본체 + `Accumulator_Col.v` (project-local copy) 의 implicit scale 차감 수정. 외부 스케줄러는 별도 spec.

## 목적

GEMM 계산 결과 (INT32 psum + 9-bit signed scale) 를 IEEE-754 FP32 로 dequantize 하고, SRAM 에서 읽은 이전 FP32 partial sum과 더해 FP32 결과를 만들어 SRAM 으로 writeback 하는 데 쓰는 single-lane RMW 산술 모듈. 외부 스케줄러가 `out_fire`, bank 매핑, ping-pong, K-tile counter, first-tile init 을 담당하고 RMW 는 산술만 담당.

## 수식 (확정)

OCP MX Spec §6.3.2 와 `MXP_Tools/mxp_tools/gemm.py::mxint_gemm_golden` 에서 도출:

```
block_int_dot   = sum_{i=0..31} (a_int[i] × b_int[i])              // int32
true_exponent   = (e_a − 127) + (e_b − 127) − IMPLICIT_total       // e_a, e_b: E8M0
                = (e_a + e_b − 254) − IMPLICIT_total
                = (rtl_scale_post_implicit − 127)

FP32_partial    = block_int_dot × 2^(rtl_scale_post_implicit − 127)
FP32_psum_new   = FP32_psum_prior + FP32_partial
```

여기서 `IMPLICIT_total = IMPLICIT[prec_A] + IMPLICIT[prec_W]`, `IMPLICIT[INT8]=6, [INT4]=2, [INT2]=0`. MXP 가 지원하는 5가지 mode 조합 → `IMPLICIT_total ∈ {0, 4, 8, 12}`.

이 spec 에서는 `rtl_scale_post_implicit = (e_a + e_b − 127 − IMPLICIT_total)` 이 RMW 의 입력 `scale[8:0]` 으로 들어옴. RMW 가 `scale` 을 받아 `2^(scale − 127)` 만 적용하면 됨 (mode-agnostic).

## Backend: Berkeley HardFloat (open-source)

FP32 산술은 **Berkeley HardFloat** (UC Berkeley, John Hauser) RTL 을 vendor 해서 씀. 선택 이유:
- IEEE-754 compliant. denormal/NaN/inf/RNE rounding 검증됨.
- Vendor-neutral RTL — Vivado FPGA 합성, Synopsys/Cadence ASIC 합성 양쪽 OK. 추후 PnR (ASIC) 까지 그대로 사용 가능.
- RISC-V (Rocket, BOOM, Hwacha 등) 진영 표준. 학계 인용 다수.
- Pre-generated Verilog 제공. Chisel→Verilog 재생성 불필요.

사용할 module:
| Module | 역할 |
|---|---|
| `iNToRecFN` | signed INT32 → recoded FP32 (33-bit). combinational. |
| `recFNFromFN` | IEEE-754 FP32 (32-bit) → recoded FP32 (33-bit). combinational. |
| `addRecFN` | recoded FP32 + recoded FP32 → recoded FP32. combinational. |
| `fNFromRecFN` | recoded FP32 → IEEE-754 FP32. combinational. |

**Recoded FP32 형식 (33-bit)**: HardFloat 내부 포맷. sign[32] + exp[31:23] (9-bit, bias=256) + mantissa[22:0]. 모든 산술은 recoded 에서 수행, 모듈 진입/탈출에서만 IEEE-754 변환. RMW 외부 (SRAM, scheduler) 는 항상 IEEE-754 (32-bit) 만 봄.

모든 HardFloat module 은 **combinational**. RMW 가 자체적으로 pipeline register 단을 끼워서 timing 맞춤. HardFloat 자체에 reset 없음.

## 모듈 계층 (decomposition)

RMW 는 thin top wrapper. 산술은 두 sub-module 에 캡슐화 — 각각 IEEE-754 boundary 로 통신, HardFloat recoded format 은 sub-module 안에 갇힘. 미래에 backend 교체 (ASIC 단계) 시 sub-module 단위로 swap.

```
RMW.v (top wrapper)
  ├── int_to_fp32.v        (INT32 + 9-bit scale → IEEE-754 FP32)
  │     내부 dataflow: iNToRecFN → recoded exp adjust → fNFromRecFN
  │     latency: parameter L_CONV
  │
  └── fp32_adder.v         (IEEE-754 FP32 + FP32 → FP32)
        내부 dataflow: recFNFromFN × 2 → addRecFN → fNFromRecFN
        latency: parameter L_ADD
```

총 RMW latency = `L_CONV + L_ADD` (외부 스케줄러 cycle 계산용).

### `int_to_fp32.v`

```verilog
module int_to_fp32 #(
    parameter L_CONV = 2        // ≥ 1 cycle. internal pipeline register depth.
)(
    input  wire        clk,
    input  wire        rst,     // unused at module level (HardFloat is comb).
    input  wire [31:0] in_int,  // signed INT32
    input  wire [8:0]  scale,   // signed 9-bit; output = in_int × 2^(scale − 127)
    output wire [31:0] out_fp32 // IEEE-754 FP32
);
```

### `fp32_adder.v`

```verilog
module fp32_adder #(
    parameter L_ADD = 3         // ≥ 1 cycle. internal pipeline register depth.
)(
    input  wire        clk,
    input  wire        rst,     // unused at module level.
    input  wire [31:0] a,       // IEEE-754 FP32
    input  wire [31:0] b,       // IEEE-754 FP32
    output wire [31:0] sum      // IEEE-754 FP32; RNE rounding
);
```

### `RMW.v` (top)

```verilog
module RMW #(
    parameter L_CONV = 2,
    parameter L_ADD  = 3
    // 총 RMW latency = L_CONV + L_ADD (외부 스케줄러 노출)
)(
    input  wire        clk,
    input  wire        rst,     // unused at module level (sub-modules use it
                                // only for their internal pipeline regs, optional).
    input  wire [31:0] in_SRAM, // IEEE-754 FP32. 첫 K-tile 은 외부에서 32'h0.
    input  wire [31:0] in_GEMM, // signed INT32
    input  wire [8:0]  scale,   // 9-bit signed, IMPLICIT_total 차감된 값.
    output wire [31:0] out_RMW  // IEEE-754 FP32
);
```

- **Handshake 없음**. Fully pipelined, 매 cycle 새 입력 가능. 외부 스케줄러는 `L_CONV + L_ADD` cycle 후 결과가 나오는 걸 알고 valid window 결정.
- **Reset 정책 (완화)**: 본 RTL 은 검증용 (FPGA + ASIC PnR 예정). HardFloat 자체는 combinational 이라 reset 무관. Sub-module 들의 pipeline register 는 verification 단계에서 reset 없이 valid window mask 로 처리. PnR 단계에서 silicon init 요구사항 보고 재검토.
- **`in_SRAM` align**: RMW top 안에 `L_CONV` cycle delay register chain 두고 `in_SRAM` 을 정렬 후 `fp32_adder.b` 에 연결. 외부는 `in_GEMM` 과 `in_SRAM` 을 **같은 cycle** 에 보냄.

## Internal Dataflow

```
in_GEMM ──┐
          ├──► int_to_fp32 ──► fp_a ────────────────────────┐
scale ────┘                                                    │
                                                               ▼
                                                          fp32_adder ──► out_RMW
                                                               ▲
in_SRAM ──► [L_CONV-cycle delay chain in RMW top] ─────────────┘
```

### `int_to_fp32` 내부

```
in_int ──► iNToRecFN [comb] ──► [exp adjust on recoded] ──► [pipeline regs ×N] ──► fNFromRecFN [comb] ──► out_fp32
                                       ▲
                          scale ───────┘
```

### `fp32_adder` 내부

```
a ──► recFNFromFN [comb] ──┐
                            ├──► [pipeline regs ×N] ──► addRecFN [comb] ──► [pipeline regs ×N] ──► fNFromRecFN [comb] ──► sum
b ──► recFNFromFN [comb] ──┘
```

모든 HardFloat module 은 combinational. 각 sub-module 이 자체 pipeline register 를 삽입해서 `L_CONV` / `L_ADD` cycle 만큼 latency 분할. register 단 위치는 implementation 자유 (예: `L_CONV=2` → exp adjust 후 1단 + 출력단 1단).

### `int_to_fp32` 본체 (참고용 skeleton)

```verilog
// 1) Int → recoded FP32 (HardFloat)
wire [32:0] recFN_int;
iNToRecFN #(.intWidth(32), .expWidth(8), .sigWidth(24)) u_i2f (
    .control       (`flControl_default),
    .signedIn      (1'b1),
    .in            (in_int),
    .roundingMode  (`round_near_even),
    .out           (recFN_int),
    .exceptionFlags()
);

// 2) Exponent bias on recoded FP exp [31:23]. HardFloat normal-range encoding
//    에서 exp 직접 가산 가능. zero/inf 영역으로 넘어가면 special encoding
//    으로 자연스럽게 처리됨.
wire        sign_bit  = recFN_int[32];
wire [8:0]  exp_field = recFN_int[31:23];
wire [22:0] sig_field = recFN_int[22:0];

wire signed [9:0] exp_ext   = $signed({1'b0, exp_field});
wire signed [9:0] scale_ext = $signed(scale);
wire signed [9:0] new_exp10 = exp_ext + scale_ext - 10'sd127;
wire [8:0]        new_exp   = new_exp10[8:0];

wire [32:0] recFN_scaled = {sign_bit, new_exp, sig_field};

// 3) pipeline reg (L_CONV 만큼 단 삽입; 구체적 배치는 timing closure)
//    예시 L_CONV=2: scaled 직후 1단 + 출력단 1단

// 4) recoded → IEEE-754
fNFromRecFN #(.expWidth(8), .sigWidth(24)) u_out (
    .in  (recFN_scaled_delayed),
    .out (out_fp32_pre)   // 출력단 register 후 → out_fp32
);
```

### `fp32_adder` 본체 (참고용 skeleton)

```verilog
wire [32:0] recFN_a, recFN_b, recFN_sum;
recFNFromFN #(.expWidth(8), .sigWidth(24)) u_in_a (.in (a), .out (recFN_a));
recFNFromFN #(.expWidth(8), .sigWidth(24)) u_in_b (.in (b), .out (recFN_b));

// pipeline reg (L_ADD 만큼 단 삽입)
//   예시 L_ADD=3: in 변환 후 1단 + addRecFN 후 1단 + 출력단 1단

addRecFN #(.expWidth(8), .sigWidth(24)) u_add (
    .control      (`flControl_default),
    .subOp        (1'b0),
    .a            (recFN_a_delayed),
    .b            (recFN_b_delayed),
    .roundingMode (`round_near_even),
    .out          (recFN_sum),
    .exceptionFlags()
);

fNFromRecFN #(.expWidth(8), .sigWidth(24)) u_out (
    .in  (recFN_sum_delayed),
    .out (sum_pre)   // 출력단 register 후 → sum
);
```

### `RMW` (top) 본체

```verilog
wire [31:0] fp_a;
int_to_fp32 #(.L_CONV(L_CONV)) u_conv (
    .clk(clk), .rst(rst),
    .in_int(in_GEMM), .scale(scale),
    .out_fp32(fp_a)
);

// in_SRAM 을 L_CONV cycle 만큼 delay register chain 통과 → fp32_adder 와 정렬
reg [31:0] sram_dly [0:L_CONV-1];
integer di;
always @(posedge clk) begin
    sram_dly[0] <= in_SRAM;
    for (di = 1; di < L_CONV; di = di + 1)
        sram_dly[di] <= sram_dly[di-1];
end
wire [31:0] in_SRAM_aligned = sram_dly[L_CONV-1];

fp32_adder #(.L_ADD(L_ADD)) u_add (
    .clk(clk), .rst(rst),
    .a(fp_a), .b(in_SRAM_aligned),
    .sum(out_RMW)
);
```

- **Rounding mode**: `round_near_even` (IEEE-754 RNE) — `np.float32` 와 bit-exact 호환.
- **NaN**: `in_GEMM` 은 정수라 NaN 안 됨. `in_SRAM` 도 외부 스케줄러가 garbage 안 흘림 (첫 tile = 32'h0, 이후 = 이전 RMW 결과). HardFloat 이 NaN-propagation 자체로 IEEE-754 따름.
- **Subnormal**: HardFloat 가 자체 처리 (`fNFromRecFN` 출력에서 subnormal 생성).
- **L_CONV ≥ 1 제약**: top 의 `in_SRAM` delay chain 이 array indexing 을 쓰므로 L_CONV=0 은 미지원.

## Upstream RTL 수정 (project-local copy 만)

`gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v`:

현재 (line 99-102 부근):
```verilog
wire signed [scale_len:0] comb_s0 = {1'b0,in_scale_act[  7:0]} + {1'b0,in_scale_weight} - 9'sd127;
// ... s1, s2, s3 동일
```

수정 후:
```verilog
// IMPLICIT_total = IMPLICIT[A_prec] + IMPLICIT[W_prec]
// A side: in_Mode_oh = {is_A2, is_A4, is_A8}, W side: in_Wcontrol ∈ {W_INT8(11), W_INT4(10), W_INT2(01), IDLE(00)}
wire [3:0] impl_a = is_A8 ? 4'd6 : is_A4 ? 4'd2 : 4'd0;            // is_A2 / idle → 0
wire [3:0] impl_w = (in_Wcontrol == 2'b11) ? 4'd6 :
                    (in_Wcontrol == 2'b10) ? 4'd2 : 4'd0;          // INT2 / IDLE → 0
wire [4:0] implicit_total = impl_a + impl_w;                       // ∈ {0,4,8,12}

wire signed [scale_len:0] comb_s0 = {1'b0,in_scale_act[  7:0]} + {1'b0,in_scale_weight}
                                  - 9'sd127 - {{4{1'b0}}, implicit_total};
// ... s1, s2, s3 동일
```

원본 `../MXP/` 는 안 건드림. Imports 디렉토리의 copy 만 수정. `CLAUDE.md` 의 "do not edit imports in place" 원칙은 RMW integration 을 위한 의도된 fork 이므로 예외 — 해당 파일 헤더에 이 spec 으로의 링크 주석 추가.

## Testing

### Standalone unit TB (`tb/rmw_tb.v`)

자체 검증 패턴:
1. `(in_GEMM, scale, in_SRAM=0)` → out_RMW 가 순수 dequant FP32 와 일치.
2. `(in_GEMM, scale, in_SRAM=prior_fp32)` → out_RMW = prior_fp32 + dequant.
3. Pipeline back-to-back: 매 cycle 다른 입력 N개 연속, 결과 N개 align 확인.
4. Saturate corner: 극단 scale 로 overflow → +inf, underflow → 0.

Vivado XSim batch 패턴 (`../sram/sim/run_wrapper.sh` 동일):
```bash
xvlog -nolog <rtl + IP wrappers> tb/rmw_tb.v
xelab -nolog -debug typical rmw_tb -s rmw_tb_sim
xsim rmw_tb_sim -nolog -R
```

### MXP_Tools 연동 (`docs/superpowers/specs/...` 미정 — RMW용 sub-command 추가 필요)

데이터 생성:
- `mxp_tools quantize_block_mx` 로 한 32-element block 의 (mantissa[int8 ×32], e8m0[uint8]) 생성
- `block_int_dot = mantissa_a @ mantissa_b` 를 int32 로 계산
- `scale_post_implicit = e_a + e_b − 127 − IMPLICIT_total`
- `expected_fp32 = block_int_dot × 2^(scale_post_implicit − 127)` (Python float 으로 계산 후 FP32 cast)

이걸 TB 가 `$readmemh` 로 읽도록 hex 파일 dump. 한 줄당 (in_GEMM_hex, scale_hex, in_SRAM_hex, expected_out_hex) 4 token 또는 별도 4 파일.

`MXP_Tools` 에 `mxp_tools rmw-gen --out work/` sub-command 추가 (이건 별도 implementation step).

검증:
- TB 가 `out_RMW` 를 dump → Python 이 expected 와 ulp 단위 비교.
- IEEE-754 round-to-nearest-even 차이로 stage 3 FP Add 결과가 Python `np.float32(a) + np.float32(b)` 와 bit-exact 일치하는지 확인. Vivado FP IP 가 IEEE-754 RNE 면 bit-exact.

## 결정 로그

| # | 항목 | 채택 | 이유 |
|---|------|------|------|
| Q1 | FP32 산술 backend | **Berkeley HardFloat** (open-source, vendor-neutral) | IEEE-754 compliant, RISC-V/학계 표준, FPGA + ASIC 양쪽 합성. Vivado FP IP 는 Xilinx 전용이라 ASIC PnR 단계에서 막힘. |
| Q2 | Throughput | Fully pipelined | 1 cycle/input, 외부 스케줄러가 cadence 결정 |
| Q3 | scale 적용 | recoded FP exp 직접 가산 | HardFloat normal-range exp 직접 가산 가능. 별도 multiplier 안 둠. special encoding 영역 (zero/inf) 으로 빠지면 addRecFN 이 자체 처리. |
| Q3a | implicit_total | Accumulator_Col 내부에서 차감 | Mode 신호 fanout replication 회피 (`in_Mode_oh`, `in_Wcontrol` 이미 거기 있음). RMW 는 mode-agnostic. |
| Q4 | First-tile init | 외부 스케줄러 forwarding (in_SRAM=0) | K-tile counter 등 스케줄러 logic 이미 필요 → 같은 곳에서 처리, RMW 내부 mux 절약 |
| Q5 | Latency parameter | Sub-module 별 `L_CONV` / `L_ADD` 노출. 총 RMW latency = L_CONV + L_ADD | 두 sub-module 각자 독립적 timing closure, 외부 스케줄러는 합산값 사용. |
| Q9 | 모듈 분해 | RMW (top) → `int_to_fp32` + `fp32_adder` 두 sub-module | Single responsibility, sub-module unit TB 가능, ASIC backend 교체 시 module-단위 swap. |
| Q6 | Reset 정책 | 검증 단계 IP default ("reset 없음" 가능), PnR 단계 재검토 | HardFloat 자체 combinational. Pipeline reg 의 reset 는 필요 시 sync 권장하나 verification 에서 valid window mask 로 회피 가능. |
| Q7 | `out_RMW` 자료형 | `output wire` | `fNFromRecFN` 출력 직결 또는 한 단 register 후 직결. implementation 결정. |
| Q8 | Handshake | 없음 | Fully pipelined, 외부에서 valid window mask. |

## Out of Scope (별도 spec)

- 외부 스케줄러: column → bank 매핑, fire timing 처리, ping-pong half 전환, K-tile counter, first-tile init forwarding, 1RW bank arbitration.
- SRAM 크기 / banking 토폴로지: ping-pong 32 bank × 4 KB × 32-bit 확정 (CLAUDE.md). `sram_1rw_banked` instantiation 두 개 (ping, pong) 가 자연.
- RMW instantiation 개수: 스케줄러 spec 에서 결정.
- `MXP_Tools` 에 `rmw-gen` sub-command 추가: 별도 implementation step.
- **HardFloat 소스 vendoring**: implementation 첫 step. `third_party/berkeley-hardfloat/` 디렉토리에 pre-generated Verilog 복사 + 라이센스 (BSD-style) 포함. 사용 module: `iNToRecFN`, `addRecFN`, `recFNFromFN`, `fNFromRecFN`, 그리고 dependencies. `HardFloat_consts.vi`, `HardFloat_specialize.vi` 같은 include 파일도 같이 가져옴.
