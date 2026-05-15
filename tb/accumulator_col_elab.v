`timescale 1ns/1ps
//-----------------------------------------------------------------------------
// accumulator_col_elab.v — Accumulator_Col 의 elaboration-only sanity TB.
//
// 검증 목적:
//   ../MXP 에서 가져온 Accumulator_Col.v 의 프로젝트-로컬 사본을 RMW 작업 중
//   "IMPLICIT_total 보정" 패치한 뒤, xelab 단계에서 파라미터/포트가 깨지지
//   않고 정상적으로 binding 되는지만 본다 (Task 8 의 사전 점검 단계).
//
// 시뮬레이션을 돌리지 않는다:
//   - 모든 입력은 0 으로 묶고 클록도 토글하지 않음.
//   - $display 한 줄로 elab 통과 신호만 남김.
//   - 목표는 "xelab errors=0" 그 자체.
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md (Q3a).
//-----------------------------------------------------------------------------

module accumulator_col_elab;

    // 기본 파라미터로 한 번 인스턴스화 (모든 입력 상수 결선).
    reg                 clk = 1'b0;
    reg                 rst = 1'b1;
    wire signed  [6:0]  in_a0 = 7'sd0;
    wire signed  [6:0]  in_a1 = 7'sd0;
    wire signed  [6:0]  in_a2 = 7'sd0;
    wire signed  [6:0]  in_a3 = 7'sd0;
    wire                in_start_accumulate = 1'b0;
    wire         [1:0]  in_Wcontrol         = 2'b00;
    wire         [2:0]  in_Mode_oh          = 3'b000;
    wire         [7:0]  in_scale_weight     = 8'd0;
    wire         [31:0] in_scale_act        = 32'd0;

    wire                out_start_accumulate;
    wire         [1:0]  out_Wcontrol;
    wire signed  [59:0] out_accumulate;
    wire         [7:0]  out_scale_weight;
    wire         [35:0] out_scale;          // 4*(scale_len+1) = 4*9 = 36
    wire                fire;

    Accumulator_Col dut (
        .clk(clk),
        .rst(rst),
        .in_a0(in_a0),
        .in_a1(in_a1),
        .in_a2(in_a2),
        .in_a3(in_a3),
        .in_start_accumulate(in_start_accumulate),
        .in_Wcontrol(in_Wcontrol),
        .in_Mode_oh(in_Mode_oh),
        .in_scale_weight(in_scale_weight),
        .in_scale_act(in_scale_act),
        .out_start_accumulate(out_start_accumulate),
        .out_Wcontrol(out_Wcontrol),
        .out_accumulate(out_accumulate),
        .out_scale_weight(out_scale_weight),
        .out_scale(out_scale),
        .fire(fire)
    );

    initial begin
        $display("accumulator_col_elab: elaboration sanity TB compiled.");
    end

endmodule
