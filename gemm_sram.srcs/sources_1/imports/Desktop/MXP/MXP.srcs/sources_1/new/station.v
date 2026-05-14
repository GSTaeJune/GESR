`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// station — per-column ping-pong scale/control buffer.
//
// Pre-decoding strategy (this revision):
//   Lane_ctrl: kept combinational from out_local_Station_control (BASELINE
//              behavior — the PE-side critical path tolerates this and the
//              prior pre-decode-at-buffer attempt regressed routing/placement).
//   Mode_oh:   pre-decoded at buffer-load time and broadcast as 3-bit one-hot
//              {A2, A4, A8}. This drops 1 logic level from the Acc_Col-side
//              eq comparator on the sc_bcast critical path.
//
// Cost: +6 FF / station (Mode_oh1 + Mode_oh2 = 2×3 bits) × 32 stations = +192 FF
//       (vs 68k baseline = +0.28%, well within the ±5% area budget).
//////////////////////////////////////////////////////////////////////////////////


module station #(
                parameter control_len          = 1,
                parameter Station_control_len  = 2,
                parameter scale_len            = 8
                )
                (
                input clk,
                input rst,
                input  [Station_control_len-1:0] in_Station_control,
                input  [4*scale_len-1:0]         in_Scale_Activation,
                input  [control_len-1:0]         in_control,
                input                            in_loadEN,
                output reg [control_len-1:0]         out_control,
                output reg                           out_loadEN,
                output reg [Station_control_len-1:0] out_Station_control,        // chain (registered)
                output reg [4*scale_len-1:0]         out_Scale_Activation,        // chain (registered)
                output     [Station_control_len-1:0] out_local_Station_control,  // broadcast (comb)
                output     [4*scale_len-1:0]         out_local_Scale_Activation, // broadcast (comb)
                output     [3:0]                     out_local_Lane_ctrl,        // pre-decoded (comb)
                output     [2:0]                     out_local_Mode_oh           // pre-decoded one-hot {A2,A4,A8}; idle=000
                );

//////////////////////////////////////////////////////////////////////////////////

localparam A8    = 2'b00;
localparam A4    = 2'b01;
localparam A2    = 2'b10;
localparam idle  = 2'b11;

//////////////////////////////////////////////////////////////////////////////////

reg [4*scale_len-1:0]         Station_Scale1;
reg [4*scale_len-1:0]         Station_Scale2;
// MAX_FANOUT pushes Vivado to replicate these registers when the decoded
// Lane_ctrl drives many PE adder_lane consumers in the column. Without it,
// the lone LUT5 (Lane_ctrl bit decode) ends up with fanout >100 and a 1.5+ ns
// route-delay tail on the PE psum critical path.
(* MAX_FANOUT = "32" *) reg [Station_control_len-1:0] Station_Ctrl1;
(* MAX_FANOUT = "32" *) reg [Station_control_len-1:0] Station_Ctrl2;

// Mode_oh shadow: 3-bit one-hot {A2,A4,A8}. Loaded in sync with Station_Ctrl{1,2}
// so the post-mux broadcast skips the 2-bit eq comparator on the Acc_Col path.
// MAX_FANOUT was tried here (=32 and =16) under both default and Explore impl
// flows — both regressed by 0.06–0.18 ns. The shadow register has narrow,
// localized fanout (one Acc_Col only); replication scatters placement and
// hurts more than it helps. Leave it un-attributed.
reg [2:0] Mode_oh1, Mode_oh2;

function [2:0] decode_mode_oh;
    input [Station_control_len-1:0] sc;
    case (sc)
        A8:      decode_mode_oh = 3'b001;
        A4:      decode_mode_oh = 3'b010;
        A2:      decode_mode_oh = 3'b100;
        default: decode_mode_oh = 3'b000;   // idle
    endcase
endfunction

// Buffer broadcast (combinational mux — ping-pong: read OPPOSITE of currently-loading buffer)
//   in_control=0 (loading Buf1) → broadcast = Buf2  (previously-loaded values, stable during MAC)
//   in_control=1 (loading Buf2) → broadcast = Buf1
// Mirrors PE_naive's `station = in_control ? station1 : station2` pattern.
assign out_local_Scale_Activation = in_control ? Station_Scale1 : Station_Scale2;
assign out_local_Station_control  = in_control ? Station_Ctrl1  : Station_Ctrl2;
assign out_local_Mode_oh          = in_control ? Mode_oh1       : Mode_oh2;

// Pre-decoded 4-bit lane-enable (PE adder_lane.ctrl 직결용) — combinational
// from the broadcast Station_control (matches the prior Acontrol-centralize
// commit; no extra register added on this hot path).
//   A8  → 4'b0001  (lane0 sign-ext)
//   A4  → 4'b0101  (lane0,2)
//   A2  → 4'b1111  (all)
//   IDLE→ 4'b0000
assign out_local_Lane_ctrl = (out_local_Station_control == A8) ? 4'b0001 :
                             (out_local_Station_control == A4) ? 4'b0101 :
                             (out_local_Station_control == A2) ? 4'b1111 :
                                                                  4'b0000 ;

//////////////////////////////////////////////////////////////////////////////////

always @(posedge clk or posedge rst)
begin
    if (rst)
    begin
        Station_Scale1       <= 0;
        Station_Scale2       <= 0;
        Station_Ctrl1        <= 0;                  // = A8 (2'b00)
        Station_Ctrl2        <= 0;                  // = A8
        // Match Station_Ctrl reset semantics so the post-decode broadcast is
        // bit-exact equivalent to the original 2-bit-driven case statement
        // even before the first buffer load.
        Mode_oh1             <= 3'b001;             // = decode_mode_oh(A8)
        Mode_oh2             <= 3'b001;
        out_control          <= 0;
        out_loadEN           <= 0;
        out_Station_control  <= 0;
        out_Scale_Activation <= 0;
    end
    else begin
        out_control <= in_control;
        out_loadEN  <= in_loadEN;

        if (in_control)
        begin
            if (in_loadEN)
            begin
                Station_Scale2       <= in_Scale_Activation;
                Station_Ctrl2        <= in_Station_control;
                Mode_oh2             <= decode_mode_oh(in_Station_control);
                out_Scale_Activation <= Station_Scale2;
                out_Station_control  <= Station_Ctrl2;
            end
            else begin
                out_Scale_Activation <= in_Scale_Activation;
                out_Station_control  <= in_Station_control;
            end
        end

        else
        begin
            if (in_loadEN)
            begin
                Station_Scale1       <= in_Scale_Activation;
                Station_Ctrl1        <= in_Station_control;
                Mode_oh1             <= decode_mode_oh(in_Station_control);
                out_Scale_Activation <= Station_Scale1;
                out_Station_control  <= Station_Ctrl1;
            end
            else begin
                out_Scale_Activation <= in_Scale_Activation;
                out_Station_control  <= in_Station_control;
            end
        end
    end
end

endmodule
