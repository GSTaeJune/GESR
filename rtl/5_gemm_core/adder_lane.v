`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////////////////

module adder_lane (
                    input wire [1:0] a,
                    input wire [6:0] b,
                    input wire ctrl,
                    output wire [6:0] c
                );

assign c = ctrl ? {{5{a[1]}},a} + b : {{5{1'b0}},a} + b;        //1일때 signextend 0일때 zero extend

endmodule