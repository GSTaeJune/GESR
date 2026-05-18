`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Accumulator_Col — 4 inner Accumulators + adder tree + mode mux per column.
//
// in_Mode_oh: 3-bit one-hot pre-decoded at station {A2, A4, A8}; idle = 3'b000.
//   This replaces the 2-bit in_Acontrol equality comparators that previously
//   sat on the sc_bcast critical path. Bit-exact in IDLE because original
//   semantics (a1/a3 zero-ext, a2 sign-ext, case→default→zero-out) are
//   preserved by the encoding.
//////////////////////////////////////////////////////////////////////////////////


module Accumulator_Col#(
                parameter input_a_len     = 7,
                parameter acc_len         = 15,
                parameter output_len      = 60,
                parameter W_control_len   = 2,
                parameter scale_len       = 8,
                parameter cnt_len         = 3
              )(
                input clk,
                input rst,
                input wire signed [input_a_len-1:0] in_a0,
                input wire signed [input_a_len-1:0] in_a1,
                input wire signed [input_a_len-1:0] in_a2,
                input wire signed [input_a_len-1:0] in_a3,
                input in_start_accumulate,
                input wire [W_control_len-1:0] in_Wcontrol,
                // MAX_FANOUT on Mode_oh broadcast bits was tried at every
                // possible attachment point — upstream Mode_oh1/2 regs (=32
                // and =16), upstream wire `out_local_Mode_oh`, and the
                // receiver-side input port (=32). All regressed by 0.06–0.18 ns.
                // Vivado's natural placement is best for this signal.
                //
                // Routing sign-ext via Lane_ctrl (combinationally decoded from
                // Station_control) instead of Mode_oh (shadow-registered)
                // ADDED a LUT level on the critical path (5 LL vs 4 LL,
                // WNS −0.957 ns). Mode_oh's pre-decoded shadow keeps logic
                // shorter even at the cost of 1 extra fanout consumer.
                input wire [2:0]               in_Mode_oh,        // {A2, A4, A8}; idle=3'b000
                input wire [scale_len-1:0]     in_scale_weight,
                input wire [4*scale_len-1:0]   in_scale_act,
                output reg out_start_accumulate,
                output reg [W_control_len-1:0] out_Wcontrol,
                output reg signed [output_len-1:0] out_accumulate,
                output reg [scale_len-1:0]         out_scale_weight,
                output reg [4*(scale_len+1)-1:0]   out_scale,
                output reg fire
                );

////////////////////////////////////////////////////////////////

// Mode bit aliases (one-hot from station; {A2,A4,A8})
wire is_A8   = in_Mode_oh[0];
wire is_A4   = in_Mode_oh[1];
wire is_A2   = in_Mode_oh[2];
// is_IDLE  = ~|in_Mode_oh  (all zero)

////////////////////////////////////////////////////////////////

wire [acc_len-1:0] out_INT2_0;
wire [acc_len-1:0] out_INT2_1;
wire [acc_len-1:0] out_INT2_2;
wire [acc_len-1:0] out_INT2_3;

wire acc_fire;   // commit pulse from lane 0 (all 4 lanes commit simultaneously)

// Adder tree pipeline registers (bit-position scaled — full MAC reconstruction)
//   Stage 1 (INT4 path): s1_a = (lane0<<2) + lane1   (acc_len+3 bits)
//                        s1_b = (lane2<<2) + lane3
//   Stage 2 (INT8 path): s2   = (s1_a<<4)  + s1_b    (acc_len+6 bits)
//                        ≡ lane0<<6 + lane1<<4 + lane2<<2 + lane3
reg signed [acc_len+2:0] s1_a;    // INT4 high pair (18-bit signed when acc_len=15)
reg signed [acc_len+2:0] s1_b;    // INT4 low pair  (18-bit signed)
reg signed [acc_len+5:0] s2;      // INT8 full sum  (21-bit signed)
reg fire_q1;
reg fire_q2;

// Per-lane sign/zero extension to (input_a_len+1)-bit signed before each Accumulator.
// Mode -> sign-ext lanes:  INT8 {0}, INT4 {0,2}, INT2 {0,1,2,3}.  Others zero-ext.
//
// Encoding via in_Mode_oh:
//   a0  always sign-ext      (lane 0 active in every mode)
//   a1  sign-ext iff A2      → is_A2
//   a2  zero-ext iff A8      → ~is_A8 selects sign-ext (A4, A2, IDLE)
//                              equivalent to original "(Acontrol==A_INT8)?0:msb"
//   a3  sign-ext iff A2      → is_A2
wire [input_a_len:0] a0;
wire [input_a_len:0] a1;
wire [input_a_len:0] a2;
wire [input_a_len:0] a3;

assign a0 = {in_a0[input_a_len-1], in_a0};
assign a1 = {is_A2  ? in_a1[input_a_len-1] : 1'b0, in_a1};
assign a2 = {is_A8  ? 1'b0 : in_a2[input_a_len-1], in_a2};
assign a3 = {is_A2  ? in_a3[input_a_len-1] : 1'b0, in_a3};

// === IMPLICIT_total 보정 (RMW 통합용 추가분) ==================================
//
// 왜 빼는가:
//   MXP의 bit-serial 누산기는 mantissa의 "묵시적(implicit) 비트"를 산술 연산에
//   포함시키지 않은 채 raw INT 도메인에서 누산한다. 결과를 FP 도메인으로
//   되돌릴 때 그 implicit 비트만큼의 자릿수(2^IMPLICIT 배수)를 다시 깎아줘야
//   원래 의도한 실수값과 일치한다. 이 보정을 (act+weight-127) 결합 스케일에
//   합쳐서 한꺼번에 빼버린다.
//
// IMPLICIT 값 (각 변(side)별):
//   INT8 -> 6,   INT4 -> 2,   INT2 -> 0    (IDLE은 0)
//   합쳐서 IMPLICIT_total ∈ {0, 2, 4, 6, 8, 12} (조합에 따라).
//
// 디코딩:
//   A side(activation): in_Mode_oh = {is_A2, is_A4, is_A8} (one-hot, IDLE=000)
//   W side(weight)    : in_Wcontrol = 11(INT8) / 10(INT4) / 01(INT2) / 00(IDLE)
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md (Q3a)
// impl_a 는 의도적으로 pipeline 하지 않음 — 본 프로젝트는 A_PREC 가 layer-fixed
// (sim 시작 시 1회 설정, mid-sim 변경 없음, precision_modes_protocol.md §1 참조)
// 라 is_A2/4/8 가 cycle 마다 상수. 만약 미래에 per-m_in A_PREC mix 를 지원한다면
// impl_w 와 같은 q-chain (impl_a_q1/q2/q3 + mode-aware mux) 도입 필요.
wire [3:0] impl_a = is_A8 ? 4'd6 : is_A4 ? 4'd2 : 4'd0;     // A side IMPLICIT
wire [3:0] impl_w = (in_Wcontrol == 2'b11) ? 4'd6 :         // W side IMPLICIT
                    (in_Wcontrol == 2'b10) ? 4'd2 : 4'd0;

// === Mixed-precision per-m_in W 지원: impl_w 파이프라인 ================
// Added: 2026-05-18, RTL fix #4 (P-Task8 random mixed-W bit-exact)
// Reference: precision_modes_protocol.md §7.3, docs/mixed-precision-tutorial.html §8.5
//
// ─── WHY 이 fix 가 필요한가 ────────────────────────────────────────────
//
// **문제**: mixed-W (per-m_in 별 다른 W_PREC) 시 random TB 가 hw_sw n_diff=
// 16384/16384, max=6.5e4, SNR=-43dB 로 catastrophic fail. uniform W 는 PASS.
//
// **Root cause** (cycle-by-cycle 추적, A=8 W[0]=8, W[1]=4 예시):
//   cycle 18..25: m_in=0 MAC (in_Wcontrol = W[0]=2'b11, impl_w=6).
//   cycle 26:     acc_fire=1, m_in=1 MAC 시작
//                 → TB 가 in_Wcontrol <= W[1]=2'b10 드라이브 (cnt 새 threshold).
//   cycle 27:     fire_q1=1 (acc_fire chain stage 1).
//   cycle 28:     fire_q2=1 (chain stage 2). in_Wcontrol = W[1] (이미 변경됨).
//   cycle 29:     out_scale <= fire_q2 ? comb_s(28) : 0.
//                 ★ comb_s(28) 가 combinational impl_w(in_Wcontrol(28)=W[1])
//                   사용 → m_in=0 의 scale 계산에 W[1]=4 의 impl_w=2 가 들어감
//                   (정답은 W[0]=8 의 impl_w=6). implicit_total 4 차이 →
//                   FP32 변환 시 2^4 = 16× off, 누산되면 max ~ 6e4.
//   uniform W 에선 impl_w 값 자체가 안 바뀜 → bug masked.
//
// **fix**: cnt threshold 제어 (in_Wcontrol 현재값) 와 scale 계산용 impl_w
// (fire chain 으로 캡처되는 m_in 의 W) 의 timing reference 가 다름을 인식.
// impl_w 를 fire chain 깊이 + 1 만큼 reg 지연 → m_in=k 캡처 cycle 에서
// impl_w_eff 가 m_in=k 의 W 를 가리킴.
//
// 왜 +1 인가: out_scale 은 register 라 fire_q2 가 high 인 cycle T 의 값을
// posedge T+1 에서 latch. comb_s(T) 의 impl_w 는 cycle T 의 값. acc_fire 가
// cycle T-2 일 때 (A=8), m_in=k 의 마지막 MAC cycle 은 T-3. drive_w_cyc[k+1]
// 은 cycle T-2 → in_Wcontrol(T) = W[k+1]. m_in=k 의 W[k] 를 얻으려면 cycle
// T-3 (= 3 cycle 이전) 의 impl_w 가 필요 → q3 (3-stage reg).
//
// ─── 호환성 ─────────────────────────────────────────────────────────────
// uniform W (9-mode 회귀) 시 in_Wcontrol 상수 → q1=q2=q3=impl_w → impl_w_eff
// = impl_w 와 동일 → 동작 변경 없음. 검증: 9-mode integration sweep ALL 9
// PASS (Acc behavioral diff = 0).
//
// mode-별 fire chain 깊이 대응 (Accumulator_Col 자체 fire mux 와 일치):
//   - A=2 (out_scale 가 acc_fire 캡처)  → impl_w_q1 (1-cyc lag)
//   - A=4 (out_scale 가 fire_q1  캡처)  → impl_w_q2 (2-cyc lag)
//   - A=8 (out_scale 가 fire_q2  캡처)  → impl_w_q3 (3-cyc lag)
//   - IDLE → impl_w (사용 안 됨, fall-through)
(* INIT = "0" *) reg [3:0] impl_w_q1, impl_w_q2, impl_w_q3;

wire [3:0] impl_w_eff = is_A8 ? impl_w_q3 :
                        is_A4 ? impl_w_q2 :
                        is_A2 ? impl_w_q1 : impl_w;
wire [4:0] implicit_total = impl_a + impl_w_eff;            // 4비트로도 충분(max=12)
                                                            // 하지만 5비트로 슬랙 1비트
                                                            // 둬도 합성에 영향 없음

// 결합 스케일 (조합):
//   comb_s = act_scale_i + weight_scale - 127 - IMPLICIT_total   (9비트 signed)
//   - 127  : FP32 bias 중화
//   - IMPLICIT_total : 위에서 설명한 묵시적 비트 보정
// 4개 sub-word는 A2 모드 시 4 lane 각각의 스케일이고,
// A4/A8 모드에서는 일부만 사용 (아래 mux 분기 참고).
wire signed [scale_len:0] comb_s0 = {1'b0,in_scale_act[  7:0]} + {1'b0,in_scale_weight} - 9'sd127 - {{4{1'b0}}, implicit_total};
wire signed [scale_len:0] comb_s1 = {1'b0,in_scale_act[ 15:8]} + {1'b0,in_scale_weight} - 9'sd127 - {{4{1'b0}}, implicit_total};
wire signed [scale_len:0] comb_s2 = {1'b0,in_scale_act[23:16]} + {1'b0,in_scale_weight} - 9'sd127 - {{4{1'b0}}, implicit_total};
wire signed [scale_len:0] comb_s3 = {1'b0,in_scale_act[31:24]} + {1'b0,in_scale_weight} - 9'sd127 - {{4{1'b0}}, implicit_total};

Accumulator#(
                .input_a_len(input_a_len + 1),
                .output_len(acc_len),
                .control_len(W_control_len),
                .cnt_len(cnt_len)
              ) acc0(
                .clk(clk),
                .rst(rst),
                .in_a(a0),
                .in_start_accumulate(in_start_accumulate),
                .in_control(in_Wcontrol),
                .out_accumulate(out_INT2_0),
                .fire(acc_fire)
                );

Accumulator#(
                .input_a_len(input_a_len + 1),
                .output_len(acc_len),
                .control_len(W_control_len),
                .cnt_len(cnt_len)
              ) acc1(
                .clk(clk),
                .rst(rst),
                .in_a(a1),
                .in_start_accumulate(in_start_accumulate),
                .in_control(in_Wcontrol),
                .out_accumulate(out_INT2_1),
                .fire()
                );

Accumulator#(
                .input_a_len(input_a_len + 1),
                .output_len(acc_len),
                .control_len(W_control_len),
                .cnt_len(cnt_len)
              ) acc2(
                .clk(clk),
                .rst(rst),
                .in_a(a2),
                .in_start_accumulate(in_start_accumulate),
                .in_control(in_Wcontrol),
                .out_accumulate(out_INT2_2),
                .fire()
                );

Accumulator#(
                .input_a_len(input_a_len + 1),
                .output_len(acc_len),
                .control_len(W_control_len),
                .cnt_len(cnt_len)
              ) acc3(
                .clk(clk),
                .rst(rst),
                .in_a(a3),
                .in_start_accumulate(in_start_accumulate),
                .in_control(in_Wcontrol),
                .out_accumulate(out_INT2_3),
                .fire()
                );


// Output mode mux as one-hot if-elseif (synth tools recognize as parallel mux,
// avoiding the 2-bit eq comparator that previously sat on the critical path).
always @(posedge clk or posedge rst) begin
    if (rst) begin
        out_start_accumulate <= 0;
        out_Wcontrol         <= 0;
        out_scale_weight     <= 0;
        s1_a                 <= 0;
        s1_b                 <= 0;
        s2                   <= 0;
        fire_q1              <= 0;
        fire_q2              <= 0;
        impl_w_q1            <= 0;
        impl_w_q2            <= 0;
        impl_w_q3            <= 0;
        out_accumulate       <= 0;
        out_scale            <= 0;
        fire                 <= 0;
    end
    else begin
        // Propagation chain regs
        out_start_accumulate <= in_start_accumulate;
        out_Wcontrol         <= in_Wcontrol;
        out_scale_weight     <= in_scale_weight;

        // Adder tree stage 1: bit-position scaled (lane0/lane2 are upper sub-words)
        //   s1_a = (lane0 << 2) + lane1   (recovers upper 4-bit × weight dot product)
        //   s1_b = (lane2 << 2) + lane3   (recovers lower 4-bit × weight dot product)
        s1_a    <= ($signed(out_INT2_0) <<< 2) + $signed(out_INT2_1);
        s1_b    <= ($signed(out_INT2_2) <<< 2) + $signed(out_INT2_3);
        fire_q1 <= acc_fire;

        // Adder tree stage 2: reuse stage 1 (INT8 = upper_4bit_sum × 16 + lower_4bit_sum)
        //   s2 = (s1_a << 4) + s1_b
        //      ≡ lane0<<6 + lane1<<4 + lane2<<2 + lane3 (full 8-bit × weight MAC)
        s2      <= (s1_a <<< 4) + s1_b;
        fire_q2 <= fire_q1;

        // impl_w 파이프라인: fire chain 과 함께 진행. comb_s 가 캡처되는
        // cycle 에 m_in=k 의 W 와 일치하도록 지연.
        impl_w_q1 <= impl_w;
        impl_w_q2 <= impl_w_q1;
        impl_w_q3 <= impl_w_q2;

        // One-hot mux on Mode_oh — semantics identical to the prior case (in_Acontrol).
        // Bits are mutually exclusive (idle = all zero → default branch via priority).
        if (is_A2) begin
            fire           <= acc_fire;
            out_accumulate <= {out_INT2_0, out_INT2_1, out_INT2_2, out_INT2_3};
            out_scale      <= acc_fire ? {comb_s0, comb_s1, comb_s2, comb_s3}
                                       : {4*(scale_len+1){1'b0}};
        end
        else if (is_A4) begin
            fire           <= fire_q1;
            // INT4: 2 sums × (acc_len+3) bits = 2*(acc_len+3) used, rest zero pad
            out_accumulate <= {{(output_len-2*(acc_len+3)){1'b0}},
                                s1_a[acc_len+2:0], s1_b[acc_len+2:0]};
            out_scale      <= fire_q1 ? {{(2*(scale_len+1)){1'b0}}, comb_s0, comb_s1}
                                      : {4*(scale_len+1){1'b0}};
        end
        else if (is_A8) begin
            fire           <= fire_q2;
            // INT8: 1 sum × (acc_len+6) bits, sign-extend padding
            out_accumulate <= {{(output_len-(acc_len+6)){s2[acc_len+5]}},
                                s2[acc_len+5:0]};
            out_scale      <= fire_q2 ? {{(3*(scale_len+1)){1'b0}}, comb_s0}
                                      : {4*(scale_len+1){1'b0}};
        end
        else begin
            // IDLE — all-zero Mode_oh
            fire           <= 1'b0;
            out_accumulate <= 0;
            out_scale      <= 0;
        end
    end
end

endmodule
