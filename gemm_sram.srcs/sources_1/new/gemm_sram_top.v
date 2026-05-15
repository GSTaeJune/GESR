`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top — GEMM + RMW[32] + sram_1rw_banked_mp 묶음 (pure structural).
// 모든 control 신호는 TB 가 driving (controller 없음).
//
// col j 의 RMW[j] 가 자기 bank[j] 만 건드림 (col-parallel). 매 cycle 최대
// 32 개의 독립 R/W 가능. 매핑 책임은 호출자 (TB).
//
// Spec: docs/superpowers/specs/2026-05-15-rmw-32x-design.md
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top #(
    parameter NUM_BANKS  = 32,
    parameter BANK_DEPTH = 1024
)(
    input  wire        clk,
    input  wire        rst,

    // ───── GEMM stimuli (passthrough — GEMM.v 의 모든 포트) ─────
    input  wire [32-1:0]                     in_a,
    input  wire signed [32*8-1:0]            in_b,
    input  wire [32-1:0]                     in_control,
    input  wire [32-1:0]                     in_loadEN,
    input  wire [1:0]                        in_Station_control,
    input  wire [4*8-1:0]                    in_Scale_Activation,
    input  wire                              in_station_control,
    input  wire                              in_station_loadEN,
    input  wire                              in_start_accumulate,
    input  wire [1:0]                        in_Wcontrol,
    input  wire [7:0]                        in_scale_weight,
    output wire [32*60-1:0]                  out_accumulate,
    output wire [32*36-1:0]                  out_scale,
    output wire [32-1:0]                     out_fire,

    // ───── RMW 32 lane input (TB 가 각 lane 별 직접 발사) ─────
    input  wire [NUM_BANKS*32-1:0]           rmw_in_GEMM,    // flat slice
    input  wire [NUM_BANKS*9-1:0]            rmw_scale,      // flat slice
    output wire [NUM_BANKS*32-1:0]           rmw_out_RMW,    // probe (flat)

    // ───── SRAM 32 bank control (TB 가 각 bank 별 직접 발사) ─────
    input  wire [NUM_BANKS-1:0]                                CEB,
    input  wire [NUM_BANKS-1:0]                                WEB,
    input  wire [NUM_BANKS*$clog2(BANK_DEPTH)-1:0]             A,
    input  wire [NUM_BANKS*32-1:0]                             WMASK,
    input  wire                                                sram_D_use_zero,  // 모든 bank 공통
    output wire [NUM_BANKS*32-1:0]                             Q                 // probe (flat)
);

    // ─── GEMM ──────────────────────────────────────────────────────
    GEMM u_gemm (
        .clk(clk), .rst(rst),
        .in_a(in_a), .in_b(in_b),
        .in_control(in_control), .in_loadEN(in_loadEN),
        .in_Station_control(in_Station_control),
        .in_Scale_Activation(in_Scale_Activation),
        .in_station_control(in_station_control),
        .in_station_loadEN(in_station_loadEN),
        .in_start_accumulate(in_start_accumulate),
        .in_Wcontrol(in_Wcontrol),
        .in_scale_weight(in_scale_weight),
        .out_accumulate(out_accumulate),
        .out_scale(out_scale),
        .out_fire(out_fire)
    );

    // ─── RMW[32] + SRAM bank D-mux per col ─────────────────────────
    wire [NUM_BANKS*32-1:0] sram_D_w;
    genvar c;
    generate
        for (c = 0; c < NUM_BANKS; c = c + 1) begin : g_lane
            wire [31:0] rmw_out_c;
            RMW u_rmw (
                .clk    (clk),
                .rst    (rst),
                .in_SRAM(Q          [c*32 +: 32]),
                .in_GEMM(rmw_in_GEMM[c*32 +: 32]),
                .scale  (rmw_scale  [c*9  +: 9 ]),
                .out_RMW(rmw_out_c)
            );
            assign rmw_out_RMW[c*32 +: 32] = rmw_out_c;
            assign sram_D_w  [c*32 +: 32] = sram_D_use_zero ? 32'h00000000 : rmw_out_c;
        end
    endgenerate

    // ─── SRAM (per-bank port) ──────────────────────────────────────
    sram_1rw_banked_mp #(
        .DATA_WIDTH (32),
        .NUM_BANKS  (NUM_BANKS),
        .BANK_DEPTH (BANK_DEPTH),
        .PIPELINE   (0)
    ) u_sram (
        .CLK   (clk),
        .CEB   (CEB),
        .WEB   (WEB),
        .A     (A),
        .D     (sram_D_w),
        .WMASK (WMASK),
        .Q     (Q)
    );

endmodule
