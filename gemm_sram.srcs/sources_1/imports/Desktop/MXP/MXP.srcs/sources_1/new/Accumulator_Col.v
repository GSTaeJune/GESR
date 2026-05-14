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
wire [3:0] impl_a = is_A8 ? 4'd6 : is_A4 ? 4'd2 : 4'd0;     // A side IMPLICIT
wire [3:0] impl_w = (in_Wcontrol == 2'b11) ? 4'd6 :         // W side IMPLICIT
                    (in_Wcontrol == 2'b10) ? 4'd2 : 4'd0;
wire [4:0] implicit_total = impl_a + impl_w;                // 4비트로도 충분(max=12)
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
