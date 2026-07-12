`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////////////////


module SystolicArray#(
                parameter input_a_len          = 1,
                parameter input_b_len          = 8,
                parameter output_len           = 7,
                parameter control_len          = 1,
                parameter Lane_ctrl_len        = 4,
                parameter num_col              = 32,
                parameter num_row              = 32
              )(
                input clk,
                input rst,
                input wire [num_row*input_a_len-1:0] in_a,
                input wire signed [num_row*input_b_len-1:0] in_b,
                input wire [num_row*control_len-1:0] in_control,
                input wire [num_col*Lane_ctrl_len-1:0] in_Lane_ctrl,
                input wire [num_row-1:0] in_loadEN,
                output reg [num_col*4*output_len-1:0] out
               );

wire [input_a_len-1:0] a_in_feed[num_row-1:0];
wire [input_a_len-1:0] a_in[num_row-1:0][num_col:0];
wire signed [input_b_len-1:0] b_in[num_row-1:0][num_col:0];
wire signed [output_len-1:0] psum0[num_row:0][num_col-1:0];
wire signed [output_len-1:0] psum1[num_row:0][num_col-1:0];
wire signed [output_len-1:0] psum2[num_row:0][num_col-1:0];
wire signed [output_len-1:0] psum3[num_row:0][num_col-1:0];
wire [control_len-1:0] control_in_feed[num_row-1:0];
wire [control_len-1:0] control_in[num_row-1:0][num_col:0];
wire loadEN_in[num_row-1:0][num_col:0];

// Per-column Lane_ctrl broadcast slice (column j → all PEs in column j)
wire [Lane_ctrl_len-1:0] lc_per_col[num_col-1:0];

genvar i,j;

generate
    for(j=0;j<num_col;j=j+1) begin : lc_slice
        assign lc_per_col[j] = in_Lane_ctrl[(j+1)*Lane_ctrl_len-1:j*Lane_ctrl_len];
    end
endgenerate

generate
    for(i=0;i<num_row;i=i+1)
    begin : assignfeed
        assign control_in_feed[i] = in_control[(i+1)*control_len-1:i*control_len];
        assign a_in_feed[i] = in_a[(i+1)*input_a_len-1:i*input_a_len];
        assign b_in[i][num_col] = in_b[(i+1)*input_b_len-1:i*input_b_len];
        assign loadEN_in[i][num_col] = in_loadEN[i];
    end
endgenerate

// Each lane accumulates at most 32 contributions of a 2-bit slice; with lane0
// (signed top-pair) ranging [-2,+1] per PE the worst case at any sync node is
// [-62,+31] for 31-PE pre-sync sum, well within 7-bit signed [-64,+63].
// Other lanes are unsigned 0..96, modular-safe in 7 bits. So 7-bit is sufficient.
wire signed [output_len-1:0] psum_sync0[num_col-3:0];
wire signed [output_len-1:0] psum_sync1[num_col-3:0];
wire signed [output_len-1:0] psum_sync2[num_col-3:0];
wire signed [output_len-1:0] psum_sync3[num_col-3:0];

generate
    for(i=0;i<num_col-2;i=i+1) begin : sync_wire
        if(i<num_col/2-1) begin : half
            assign psum_sync0[i] = psum0[0][i+1] + psum0[2*i+3][i+1];
            assign psum_sync1[i] = psum1[0][i+1] + psum1[2*i+3][i+1];
            assign psum_sync2[i] = psum2[0][i+1] + psum2[2*i+3][i+1];
            assign psum_sync3[i] = psum3[0][i+1] + psum3[2*i+3][i+1];
        end
        else begin : rest
            assign psum_sync0[i] = psum0[num_row][i+1] + psum0[2*i-num_col+3][i+1];
            assign psum_sync1[i] = psum1[num_row][i+1] + psum1[2*i-num_col+3][i+1];
            assign psum_sync2[i] = psum2[num_row][i+1] + psum2[2*i-num_col+3][i+1];
            assign psum_sync3[i] = psum3[num_row][i+1] + psum3[2*i-num_col+3][i+1];
        end
    end
endgenerate

generate
    for(i=0;i<num_col;i=i+1)
    begin : zerowire
        if(i<num_col/2)
        begin : half
            assign psum0[i][i] = 0;
            assign psum1[i][i] = 0;
            assign psum2[i][i] = 0;
            assign psum3[i][i] = 0;
        end

        else begin : rest
            assign psum0[i+1][i] = 0;
            assign psum1[i+1][i] = 0;
            assign psum2[i+1][i] = 0;
            assign psum3[i+1][i] = 0;
        end
    end
endgenerate

generate
    for(i=0;i<num_row;i=i+1)
    begin : row
        for(j=0;j<num_col;j=j+1)
        begin : col
////////////////////////////////////////////////////////////////////
            if(i==j) begin : feeder
                if(j<num_col/2) begin : half
                    PE_feeder #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                        ) pe_feed(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in_feed[i]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i][j]),
                            .psum1_in(psum1[i][j]),
                            .psum2_in(psum2[i][j]),
                            .psum3_in(psum3[i][j]),
                            .in_control(control_in_feed[i]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_controlLeft(control_in[i][j]),
                            .out_controlRight(control_in[i][j+1]),
                            .out_aLeft(a_in[i][j]),
                            .out_aRight(a_in[i][j+1]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i+1][j]),
                            .psum1_out(psum1[i+1][j]),
                            .psum2_out(psum2[i+1][j]),
                            .psum3_out(psum3[i+1][j])
                        );
                end

                else begin : rest
                    PE_feeder #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                        ) pe_feed(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in_feed[i]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i+1][j]),
                            .psum1_in(psum1[i+1][j]),
                            .psum2_in(psum2[i+1][j]),
                            .psum3_in(psum3[i+1][j]),
                            .in_control(control_in_feed[i]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_controlLeft(control_in[i][j]),
                            .out_controlRight(control_in[i][j+1]),
                            .out_aLeft(a_in[i][j]),
                            .out_aRight(a_in[i][j+1]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i][j]),
                            .psum1_out(psum1[i][j]),
                            .psum2_out(psum2[i][j]),
                            .psum3_out(psum3[i][j])
                        );
                end
            end

////////////////////////////////////////////////////////////////////
            else begin : normal
                if(j==0) begin : col0
                    PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j+1]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i][j]),
                            .psum1_in(psum1[i][j]),
                            .psum2_in(psum2[i][j]),
                            .psum3_in(psum3[i][j]),
                            .in_control(control_in[i][j+1]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j]),
                            .out_a(a_in[i][j]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i+1][j]),
                            .psum1_out(psum1[i+1][j]),
                            .psum2_out(psum2[i+1][j]),
                            .psum3_out(psum3[i+1][j])
                            );
                end

                else if(j==num_col-1) begin: col31
                    PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i+1][j]),
                            .psum1_in(psum1[i+1][j]),
                            .psum2_in(psum2[i+1][j]),
                            .psum3_in(psum3[i+1][j]),
                            .in_control(control_in[i][j]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j+1]),
                            .out_a(a_in[i][j+1]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i][j]),
                            .psum1_out(psum1[i][j]),
                            .psum2_out(psum2[i][j]),
                            .psum3_out(psum3[i][j])
                            );
                end

                else begin : col1_30
                    if(j<num_col/2) begin : half
                        if(i==2*j+1) begin : sync_half
                            PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j+1]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum_sync0[j-1]),
                            .psum1_in(psum_sync1[j-1]),
                            .psum2_in(psum_sync2[j-1]),
                            .psum3_in(psum_sync3[j-1]),
                            .in_control(control_in[i][j+1]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j]),
                            .out_a(a_in[i][j]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i+1][j]),
                            .psum1_out(psum1[i+1][j]),
                            .psum2_out(psum2[i+1][j]),
                            .psum3_out(psum3[i+1][j])
                            );
                        end

                        else if (i>j) begin : lower
                            PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j+1]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i][j]),
                            .psum1_in(psum1[i][j]),
                            .psum2_in(psum2[i][j]),
                            .psum3_in(psum3[i][j]),
                            .in_control(control_in[i][j+1]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j]),
                            .out_a(a_in[i][j]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i+1][j]),
                            .psum1_out(psum1[i+1][j]),
                            .psum2_out(psum2[i+1][j]),
                            .psum3_out(psum3[i+1][j])
                            );
                        end

                        else begin : upper
                            PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i+1][j]),
                            .psum1_in(psum1[i+1][j]),
                            .psum2_in(psum2[i+1][j]),
                            .psum3_in(psum3[i+1][j]),
                            .in_control(control_in[i][j]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j+1]),
                            .out_a(a_in[i][j+1]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i][j]),
                            .psum1_out(psum1[i][j]),
                            .psum2_out(psum2[i][j]),
                            .psum3_out(psum3[i][j])
                            );
                        end
                    end

                    else begin : rest
                        if(i==2*j-num_col) begin : sync_rest
                            PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum_sync0[j-1]),
                            .psum1_in(psum_sync1[j-1]),
                            .psum2_in(psum_sync2[j-1]),
                            .psum3_in(psum_sync3[j-1]),
                            .in_control(control_in[i][j]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j+1]),
                            .out_a(a_in[i][j+1]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i][j]),
                            .psum1_out(psum1[i][j]),
                            .psum2_out(psum2[i][j]),
                            .psum3_out(psum3[i][j])
                            );
                        end

                        else if (i>j) begin : lower
                            PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j+1]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i][j]),
                            .psum1_in(psum1[i][j]),
                            .psum2_in(psum2[i][j]),
                            .psum3_in(psum3[i][j]),
                            .in_control(control_in[i][j+1]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j]),
                            .out_a(a_in[i][j]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i+1][j]),
                            .psum1_out(psum1[i+1][j]),
                            .psum2_out(psum2[i+1][j]),
                            .psum3_out(psum3[i+1][j])
                            );
                        end

                        else begin : upper
                            PE_naive #(
                            .input_a_len(input_a_len),
                            .input_b_len(input_b_len),
                            .output_len(output_len),
                            .control_len(control_len),
                            .Lane_ctrl_len(Lane_ctrl_len)
                            ) pe(
                            .clk(clk),
                            .rst(rst),
                            .in_a(a_in[i][j]),
                            .in_b(b_in[i][j+1]),
                            .psum0_in(psum0[i+1][j]),
                            .psum1_in(psum1[i+1][j]),
                            .psum2_in(psum2[i+1][j]),
                            .psum3_in(psum3[i+1][j]),
                            .in_control(control_in[i][j]),
                            .in_Lane_ctrl(lc_per_col[j]),
                            .in_loadEN(loadEN_in[i][j+1]),
                            .out_loadEN(loadEN_in[i][j]),
                            .out_control(control_in[i][j+1]),
                            .out_a(a_in[i][j+1]),
                            .out_b(b_in[i][j]),
                            .psum0_out(psum0[i][j]),
                            .psum1_out(psum1[i][j]),
                            .psum2_out(psum2[i][j]),
                            .psum3_out(psum3[i][j])
                            );
                        end
                    end
                end
            end
        end
    end
endgenerate

generate
    for(i=0;i<num_col;i=i+1)
    begin : assignout
        if(i<num_col/2) begin : half
            always @(posedge clk or posedge rst)
            begin
                if(rst)
                    out[(i+1)*4*output_len-1:i*4*output_len] <= 0;
                else
                    out[(i+1)*4*output_len-1:i*4*output_len] <= {psum0[num_row][i], psum1[num_row][i], psum2[num_row][i], psum3[num_row][i]};
            end
        end

        else begin : rest
            always @(posedge clk or posedge rst)
            begin
                if(rst)
                    out[(i+1)*4*output_len-1:i*4*output_len] <= 0;
                else
                    out[(i+1)*4*output_len-1:i*4*output_len] <= {psum0[0][i], psum1[0][i], psum2[0][i], psum3[0][i]};
            end
        end
    end
endgenerate

endmodule
