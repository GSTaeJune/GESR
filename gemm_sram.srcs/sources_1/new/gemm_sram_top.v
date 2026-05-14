`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top — GEMM + RMW + sram_1rw_banked 묶음 (pure structural).
// 모든 control 신호는 TB 가 driving (controller 없음).
// 내부 wiring: 두 줄 (sram_Q ─► rmw.in_SRAM, mux(sram_D_use_zero) ─► sram.D).
//
// Spec: docs/superpowers/specs/2026-05-14-integration-design.md
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top #(
    parameter NUM_BANKS  = 16,
    parameter BANK_DEPTH = 32768
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

    // ───── RMW input (TB 가 직접 발사) ─────
    input  wire [31:0]                       rmw_in_GEMM,
    input  wire [8:0]                        rmw_scale,
    output wire [31:0]                       rmw_out_RMW,        // probe

    // ───── SRAM control (TB 가 직접 발사) ─────
    input  wire                              sram_CEB,
    input  wire                              sram_WEB,
    input  wire [18:0]                       sram_A,             // clog2(16)+clog2(32768)=19
    input  wire [31:0]                       sram_WMASK,
    input  wire                              sram_D_use_zero,    // 1=D 강제 0, 0=RMW 출력
    output wire [31:0]                       sram_Q              // probe
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

    // ─── RMW ───────────────────────────────────────────────────────
    wire [31:0] rmw_out_w;
    RMW u_rmw (
        .clk(clk), .rst(rst),
        .in_SRAM(sram_Q),
        .in_GEMM(rmw_in_GEMM),
        .scale(rmw_scale),
        .out_RMW(rmw_out_w)
    );
    assign rmw_out_RMW = rmw_out_w;

    // ─── SRAM D 입력 mux (zero priming 단계 우회용) ─────────────────
    wire [31:0] sram_D_w = sram_D_use_zero ? 32'h00000000 : rmw_out_w;

    // ─── SRAM banked ───────────────────────────────────────────────
    sram_1rw_banked #(
        .DATA_WIDTH    (32),
        .NUM_BANKS     (NUM_BANKS),
        .BANK_DEPTH    (BANK_DEPTH),
        .BANK_STRATEGY ("INTERLEAVED"),
        .PIPELINE      (0)
    ) u_sram (
        .CLK   (clk),
        .CEB   (sram_CEB),
        .WEB   (sram_WEB),
        .A     (sram_A),
        .D     (sram_D_w),
        .WMASK (sram_WMASK),
        .Q     (sram_Q)
    );

endmodule
