`timescale 1ns/1ps
//-----------------------------------------------------------------------------
// accumulator_col_elab.v — elaboration-only sanity TB for Accumulator_Col.
//
// Purpose: force xelab to parse + bind the project-local copy of
// Accumulator_Col.v after the IMPLICIT_total subtraction edit (Task 8).
// No simulation run required; we only need elab to succeed with zero errors.
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md (Q3a).
//-----------------------------------------------------------------------------

module accumulator_col_elab;

    // Default-parameter instantiation. All inputs tied to constants.
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
