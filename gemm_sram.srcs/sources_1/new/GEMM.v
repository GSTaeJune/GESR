`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// TOP — wraps SystolicArray + 32 station instances + 32 Accumulator_Col instances.
//
// station chain: col 31 entry, single-direction left propagation (matches in_b
// wave). Each station broadcasts out_local_Lane_ctrl (4-bit pre-decoded) to
// SystolicArray's per-column lane-ctrl bus → PE adder_lane.ctrl direct. Each
// station also broadcasts out_local_Mode_oh (3-bit one-hot) to its
// Accumulator_Col in_Mode_oh and out_local_Scale_Activation to in_scale_act.
//
// Accumulator_Col chain: middle entry at col num_col/2, fanning outward
// (left half: i+1→i, right half: i→i+1) — mirrors reference BitSerial_Systolic
// TOP.v topology for start_accumulate/Wcontrol propagation.
//
// Weight scale chain: same topology as Accumulator_Col chain (entry at num_col/2,
// fanning outward). in_scale_weight propagates alongside start_accumulate.
//////////////////////////////////////////////////////////////////////////////////

module GEMM#(
                parameter input_a_len          = 1,
                parameter input_b_len          = 8,
                parameter output_len           = 7,    // per-lane psum width from SA
                parameter control_len          = 1,
                parameter Station_control_len  = 2,
                parameter scale_len            = 8,
                parameter acc_len              = 15,
                parameter acc_out_len          = 60,
                parameter W_control_len        = 2,
                parameter A_control_len        = 2,
                parameter cnt_len              = 3,
                parameter num_col              = 32,
                parameter num_row              = 32
              )(
                input  clk,
                input  rst,

                // Systolic array stimuli (per row)
                input  wire [num_row*input_a_len-1:0]  in_a,
                input  wire signed [num_row*input_b_len-1:0] in_b,
                input  wire [num_row*control_len-1:0]  in_control,
                input  wire [num_row-1:0]              in_loadEN,

                // Station chain entry (scalar at col 31; propagates leftward)
                input  wire [Station_control_len-1:0]  in_Station_control,
                input  wire [4*scale_len-1:0]          in_Scale_Activation,
                input  wire [control_len-1:0]          in_station_control,
                input  wire                            in_station_loadEN,

                // Accumulator chain entry (scalar at col num_col/2)
                input  wire                            in_start_accumulate,
                input  wire [W_control_len-1:0]        in_Wcontrol,
                input  wire [scale_len-1:0]            in_scale_weight,

                // Per-column accumulator outputs
                output wire [num_col*acc_out_len-1:0]          out_accumulate,
                output wire [num_col*4*(scale_len+1)-1:0]      out_scale,
                output wire [num_col-1:0]                      out_fire
               );

genvar j;

////////////////////////////////////////////////////////////////////
// Station chains:
//   - Data chain  (col 31 entry, leftward):    in_Scale_Activation,
//                                                in_Station_control (Acontrol),
//                                                in_loadEN.
//                  Per-col data delivery aligned with in_b chain.
//   - Selector chain (col num_col/2 entry, fan outward): in_control
//                  (buffer-selector toggle). SYMMETRIC like acc chain so the
//                  broadcast switch wave aligns with each col's fire timing
//                  → enables continuous MAC across K-tiles.
////////////////////////////////////////////////////////////////////

// Data chain wires (asymmetric, col 31 entry leftward)
wire [Station_control_len-1:0] sc_chain   [num_col:0];
wire [4*scale_len-1:0]         sa_chain   [num_col:0];
wire                           sld_chain  [num_col:0];

// Selector chain wires (symmetric, col num_col/2 entry, fan outward)
wire [control_len-1:0]         sctl_chain [num_col:0];

localparam Lane_ctrl_len = 4;   // pre-decoded lane-ctrl width (matches SystolicArray parameter)
localparam Mode_oh_len   = 3;   // pre-decoded one-hot {A2,A4,A8} for Acc_Col

// Broadcast wires (combinational buffer mux from each station — stable per-column
// during MAC since they read latched buffer values, not the propagating chain).
wire [4*scale_len-1:0]          sa_bcast   [num_col-1:0];
wire [Lane_ctrl_len-1:0]        lc_bcast   [num_col-1:0];
wire [Mode_oh_len-1:0]          mo_bcast   [num_col-1:0];

assign sc_chain  [num_col]     = in_Station_control;
assign sa_chain  [num_col]     = in_Scale_Activation;
assign sld_chain [num_col]     = in_station_loadEN;
assign sctl_chain[num_col/2]   = in_station_control;   // sym entry at col num_col/2

generate
    for (j=0; j<num_col; j=j+1) begin : station_gen
        if (j < num_col/2) begin : sta_left
            // Left half: data chain leftward (j+1 → j),
            //            selector chain leftward from col num_col/2 (j+1 → j).
            station #(
                        .control_len(control_len),
                        .Station_control_len(Station_control_len),
                        .scale_len(scale_len)
                      ) sta (
                        .clk                       (clk),
                        .rst                       (rst),
                        .in_Station_control        (sc_chain  [j+1]),
                        .in_Scale_Activation       (sa_chain  [j+1]),
                        .in_control                (sctl_chain[j+1]),
                        .in_loadEN                 (sld_chain [j+1]),
                        .out_control               (sctl_chain[j]),
                        .out_loadEN                (sld_chain [j]),
                        .out_Station_control       (sc_chain  [j]),
                        .out_Scale_Activation      (sa_chain  [j]),
                        .out_local_Station_control (),
                        .out_local_Scale_Activation(sa_bcast  [j]),
                        .out_local_Lane_ctrl       (lc_bcast  [j]),
                        .out_local_Mode_oh         (mo_bcast  [j])
                      );
        end else begin : sta_right
            // Right half: data chain leftward (j+1 → j) — UNCHANGED,
            //             selector chain rightward from col num_col/2 (j → j+1).
            station #(
                        .control_len(control_len),
                        .Station_control_len(Station_control_len),
                        .scale_len(scale_len)
                      ) sta (
                        .clk                       (clk),
                        .rst                       (rst),
                        .in_Station_control        (sc_chain  [j+1]),
                        .in_Scale_Activation       (sa_chain  [j+1]),
                        .in_control                (sctl_chain[j]),
                        .in_loadEN                 (sld_chain [j+1]),
                        .out_control               (sctl_chain[j+1]),
                        .out_loadEN                (sld_chain [j]),
                        .out_Station_control       (sc_chain  [j]),
                        .out_Scale_Activation      (sa_chain  [j]),
                        .out_local_Station_control (),
                        .out_local_Scale_Activation(sa_bcast  [j]),
                        .out_local_Lane_ctrl       (lc_bcast  [j]),
                        .out_local_Mode_oh         (mo_bcast  [j])
                      );
        end
    end
endgenerate

// Pack per-column station pre-decoded Lane_ctrl into the SA's per-column input bus
wire [num_col*Lane_ctrl_len-1:0] sa_in_Lane_ctrl;
generate
    for (j=0; j<num_col; j=j+1) begin : sa_lc_pack
        assign sa_in_Lane_ctrl[(j+1)*Lane_ctrl_len-1 : j*Lane_ctrl_len] = lc_bcast[j];
    end
endgenerate

////////////////////////////////////////////////////////////////////
// Systolic array
////////////////////////////////////////////////////////////////////

wire [num_col*4*output_len-1:0] sa_out;

SystolicArray #(
                    .input_a_len(input_a_len),
                    .input_b_len(input_b_len),
                    .output_len(output_len),
                    .control_len(control_len),
                    .Lane_ctrl_len(Lane_ctrl_len),
                    .num_col(num_col),
                    .num_row(num_row)
                ) sa (
                    .clk               (clk),
                    .rst               (rst),
                    .in_a              (in_a),
                    .in_b              (in_b),
                    .in_control        (in_control),
                    .in_Lane_ctrl      (sa_in_Lane_ctrl),
                    .in_loadEN         (in_loadEN),
                    .out               (sa_out)
                );

////////////////////////////////////////////////////////////////////
// Per-column SA-output → Accumulator_Col input slicing
//   sa_out[(i+1)*28-1 : i*28] = {psum0[7], psum1[7], psum2[7], psum3[7]}
//   MSB = psum0 (lane0), LSB = psum3 (lane3)
////////////////////////////////////////////////////////////////////

wire signed [output_len-1:0] col_a0[num_col-1:0];
wire signed [output_len-1:0] col_a1[num_col-1:0];
wire signed [output_len-1:0] col_a2[num_col-1:0];
wire signed [output_len-1:0] col_a3[num_col-1:0];

generate
    for (j=0; j<num_col; j=j+1) begin : col_slice
        assign col_a0[j] = sa_out[j*4*output_len + 4*output_len - 1 : j*4*output_len + 3*output_len];
        assign col_a1[j] = sa_out[j*4*output_len + 3*output_len - 1 : j*4*output_len + 2*output_len];
        assign col_a2[j] = sa_out[j*4*output_len + 2*output_len - 1 : j*4*output_len + 1*output_len];
        assign col_a3[j] = sa_out[j*4*output_len + 1*output_len - 1 : j*4*output_len + 0*output_len];
    end
endgenerate

////////////////////////////////////////////////////////////////////
// Accumulator_Col chain (col num_col/2 entry, fanning outward)
// Weight scale chain: same topology (entry at num_col/2, fan outward)
////////////////////////////////////////////////////////////////////

wire                       sa_chain_pulse [num_col:0];
wire [W_control_len-1:0]   wc_chain       [num_col:0];
wire [scale_len-1:0]       sw_chain       [num_col:0];   // weight scale propagation

assign sa_chain_pulse[num_col/2] = in_start_accumulate;
assign wc_chain      [num_col/2] = in_Wcontrol;
assign sw_chain      [num_col/2] = in_scale_weight;

wire signed [acc_out_len-1:0]       col_acc  [num_col-1:0];
wire [4*(scale_len+1)-1:0]          col_scale[num_col-1:0];
wire                                col_fire [num_col-1:0];

generate
    // Left half: chain[j+1] → acc[j] → chain[j] (leftward from middle)
    for (j=0; j<num_col/2; j=j+1) begin : acc_left
        Accumulator_Col #(
                    .input_a_len(output_len),
                    .acc_len(acc_len),
                    .output_len(acc_out_len),
                    .W_control_len(W_control_len),
                    .scale_len(scale_len),
                    .cnt_len(cnt_len)
                ) acc (
                    .clk                 (clk),
                    .rst                 (rst),
                    .in_a0               (col_a0[j]),
                    .in_a1               (col_a1[j]),
                    .in_a2               (col_a2[j]),
                    .in_a3               (col_a3[j]),
                    .in_start_accumulate (sa_chain_pulse[j+1]),
                    .in_Wcontrol         (wc_chain      [j+1]),
                    .in_Mode_oh          (mo_bcast[j]),
                    .in_scale_weight     (sw_chain      [j+1]),
                    .in_scale_act        (sa_bcast[j]),
                    .out_start_accumulate(sa_chain_pulse[j]),
                    .out_Wcontrol        (wc_chain      [j]),
                    .out_scale_weight    (sw_chain      [j]),
                    .out_accumulate      (col_acc[j]),
                    .out_scale           (col_scale[j]),
                    .fire                (col_fire[j])
                );
    end

    // Right half: chain[j] → acc[j] → chain[j+1] (rightward from middle)
    for (j=num_col/2; j<num_col; j=j+1) begin : acc_right
        Accumulator_Col #(
                    .input_a_len(output_len),
                    .acc_len(acc_len),
                    .output_len(acc_out_len),
                    .W_control_len(W_control_len),
                    .scale_len(scale_len),
                    .cnt_len(cnt_len)
                ) acc (
                    .clk                 (clk),
                    .rst                 (rst),
                    .in_a0               (col_a0[j]),
                    .in_a1               (col_a1[j]),
                    .in_a2               (col_a2[j]),
                    .in_a3               (col_a3[j]),
                    .in_start_accumulate (sa_chain_pulse[j]),
                    .in_Wcontrol         (wc_chain      [j]),
                    .in_Mode_oh          (mo_bcast[j]),
                    .in_scale_weight     (sw_chain      [j]),
                    .in_scale_act        (sa_bcast[j]),
                    .out_start_accumulate(sa_chain_pulse[j+1]),
                    .out_Wcontrol        (wc_chain      [j+1]),
                    .out_scale_weight    (sw_chain      [j+1]),
                    .out_accumulate      (col_acc[j]),
                    .out_scale           (col_scale[j]),
                    .fire                (col_fire[j])
                );
    end
endgenerate

generate
    for (j=0; j<num_col; j=j+1) begin : pack_out
        assign out_accumulate[(j+1)*acc_out_len-1 : j*acc_out_len]                    = col_acc[j];
        assign out_scale     [(j+1)*4*(scale_len+1)-1 : j*4*(scale_len+1)]            = col_scale[j];
        assign out_fire[j]                                                              = col_fire[j];
    end
endgenerate

endmodule
