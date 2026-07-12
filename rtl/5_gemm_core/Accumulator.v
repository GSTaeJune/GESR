`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Bit-serial signed accumulator (single lane). At cnt=0 (the MSB cycle), the
// input is two's-complemented before the add; thereafter it is added straight.
// At a control-defined fire cycle (INT8: cnt=7, INT4: cnt=3, INT2: cnt=1) the
// committed accumulate is forwarded to out_accumulate and the running sum is
// reset to 0 in lock-step with the next MAC starting at cnt=0.
//
// Critical-path note (this revision): both `accumulate` and `out_accumulate`
// are split into their own clocked-only always blocks (no async rst), so
// Vivado maps them to FDRE with a sync R pin instead of FDCE + a D-pin LUT2
// between CARRY4 and the FF. Power-up init is 0 (FPGA default) and the chain
// `in_start_accumulate` immediately re-zeros them post-rst, so functional
// behavior is identical. cnt / fire keep async rst (not on the hot path).
//////////////////////////////////////////////////////////////////////////////////


module Accumulator#(
                parameter input_a_len     = 7,
                parameter output_len      = 10,
                parameter control_len     = 2,
                parameter cnt_len         = 3
              )(
                input clk,
                input rst,
                input wire signed [input_a_len-1:0] in_a,
                input in_start_accumulate,
                input wire [control_len-1:0] in_control,
                output reg signed [output_len-1:0] out_accumulate,
                output reg fire
                );

////////////////////////////////////////////////////////////////

localparam INT8 = 2'b11;
localparam INT4 = 2'b10;
localparam INT2 = 2'b01;
localparam IDLE = 2'b00;

////////////////////////////////////////////////////////////////

reg [cnt_len-1:0] cnt;
(* INIT = "0" *) reg signed [output_len-1:0] accumulate;

wire signed [output_len-1:0] add_operand1;
assign add_operand1 = &(~cnt) ? -in_a : in_a;               //MSB 2's complement

wire signed [output_len-1:0] add_operand2;
assign add_operand2 = accumulate<<<1;

wire signed [output_len-1:0] res_add;
assign res_add = add_operand1 + add_operand2;

// Fire cycle: end of MAC, commits final res_add and resets accumulate.
wire fire_cycle = (in_control == INT8 && cnt == 3'b111) ||
                  (in_control == INT4 && cnt == 3'b011) ||
                  (in_control == INT2 && cnt == 3'b001) ;

// Sync-clear condition for `accumulate`. Identical semantics to the original
// case-mux: zero on rst, in_start_accumulate, IDLE, or end-of-MAC fire cycle.
wire acc_clr = rst || in_start_accumulate || (in_control == IDLE) || fire_cycle;

// `out_accumulate` only loads at the fire cycle when the lane is active.
// All other cycles (rst, in_start, IDLE, mid-MAC) it holds 0.
wire load_out_acc = fire_cycle && (in_control != IDLE) && !in_start_accumulate;
wire out_acc_clr  = rst || !load_out_acc;

////////////////////////////////////////////////////////////////
// `accumulate` — sync-only. Vivado maps to FDRE; CARRY4 drives D directly.
////////////////////////////////////////////////////////////////
always @(posedge clk)
begin
    if (acc_clr) accumulate <= 0;
    else         accumulate <= res_add;
end

////////////////////////////////////////////////////////////////
// `out_accumulate` — sync-only. FDRE again; CARRY4 → D, no D-side LUT2.
////////////////////////////////////////////////////////////////
always @(posedge clk)
begin
    if (out_acc_clr) out_accumulate <= 0;
    else             out_accumulate <= res_add;
end

////////////////////////////////////////////////////////////////
// Counter / fire — async-rst block.
////////////////////////////////////////////////////////////////
always@(posedge clk or posedge rst)
begin
    if(rst)
    begin
        cnt <= 0;
        fire <= 0;
    end
    else begin
        if(in_start_accumulate)
        begin
            cnt <= 0;
            fire <= 0;
        end
        else if(in_control != IDLE)
        begin
            if (fire_cycle)
            begin
                fire <= 1;
                cnt <= 0;
            end
            else begin
                cnt <= cnt + 1;
                fire <= 0;
            end
        end
        else begin
            cnt <= 0;
            fire <= 0;
        end
    end
end
endmodule
