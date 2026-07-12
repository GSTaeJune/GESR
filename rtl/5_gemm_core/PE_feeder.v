`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////////////////


module PE_feeder#(
                parameter input_a_len  = 1,
                parameter input_b_len  = 8,
                parameter output_len   = 7,
                parameter control_len  = 1,
                parameter Lane_ctrl_len = 4              // pre-decoded from station (was Station_control_len=2)
              )(
                input clk,
                input rst,
                input wire [input_a_len-1:0] in_a,
                input wire signed [input_b_len-1:0] in_b,
                input wire [output_len-1:0] psum0_in,
                input wire [output_len-1:0] psum1_in,
                input wire [output_len-1:0] psum2_in,
                input wire [output_len-1:0] psum3_in,
                input wire [control_len-1:0] in_control,
                input wire [Lane_ctrl_len-1:0] in_Lane_ctrl,
                input in_loadEN,
                output reg out_loadEN,
                output reg [control_len-1:0] out_controlLeft,
                output reg [control_len-1:0] out_controlRight,
                output reg [input_a_len-1:0] out_aLeft,
                output reg [input_a_len-1:0] out_aRight,
                output reg signed [input_b_len-1:0] out_b,
                output reg [output_len-1:0] psum0_out,
                output reg [output_len-1:0] psum1_out,
                output reg [output_len-1:0] psum2_out,
                output reg [output_len-1:0] psum3_out
                );

///////////////////////////////////////////////////

reg signed [input_b_len-1:0] station1;      //in_control == 0 load
reg signed [input_b_len-1:0] station2;      //in_control == 1 load

///////////////////////////////////////////////////

wire signed [input_b_len-1:0] station;
assign station = in_control ? station1 : station2;

wire signed [input_b_len-1:0] add_operand;
assign add_operand = in_a ? station : 0;

wire [output_len-1:0] res_add0;
wire [output_len-1:0] res_add1;
wire [output_len-1:0] res_add2;
wire [output_len-1:0] res_add3;

adder_lane lane0(
                    .a(add_operand[7:6]),
                    .b(psum0_in),
                    .ctrl(in_Lane_ctrl[0]),
                    .c(res_add0)
                );

adder_lane lane1(
                    .a(add_operand[5:4]),
                    .b(psum1_in),
                    .ctrl(in_Lane_ctrl[1]),
                    .c(res_add1)
                );

adder_lane lane2(
                    .a(add_operand[3:2]),
                    .b(psum2_in),
                    .ctrl(in_Lane_ctrl[2]),
                    .c(res_add2)
                );

adder_lane lane3(
                    .a(add_operand[1:0]),
                    .b(psum3_in),
                    .ctrl(in_Lane_ctrl[3]),
                    .c(res_add3)
                );

always@(posedge clk or posedge rst)
begin
    if(rst)
    begin
        psum0_out <= 0;
        psum1_out <= 0;
        psum2_out <= 0;
        psum3_out <= 0;
    end
    else begin
        psum0_out <= res_add0;
        psum1_out <= res_add1;
        psum2_out <= res_add2;
        psum3_out <= res_add3;
    end
end

always@(posedge clk or posedge rst)
begin
    if(rst)
    begin
        station1          <= 0;
        station2          <= 0;
        out_aLeft         <= 0;
        out_aRight        <= 0;
        out_b             <= 0;
        out_controlLeft   <= 0;
        out_controlRight  <= 0;
        out_loadEN        <= 0;
    end

    else begin
        out_loadEN  <= in_loadEN;
        out_controlLeft <= in_control;
        out_controlRight <= in_control;
        out_aLeft   <= in_a;
        out_aRight  <= in_a;

        if(in_control)
        begin
            if(in_loadEN)
            begin
                station2 <= in_b;
                out_b    <= station2;
            end
            else begin
                out_b <= in_b;
            end
        end

        else
        begin
            if(in_loadEN)
            begin
                station1 <= in_b;
                out_b    <= station1;
            end
            else begin
                out_b <= in_b;
            end
        end

    end
end

endmodule