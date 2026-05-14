`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_tb — 단일 TB (plusarg 로 1 mode 받음).
// 이 task (Task 2) 에서는 INIT (zero priming) + DUMP 만 동작.
// Task 7 에서 LOAD / CONFIG / DRIVE / DRAIN 단계 추가.
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8                      (Task 2 에선 무시)
//   B_PREC    : 2 / 4 / 8                      (Task 2 에선 무시)
//   TILE_M, TILE_K, TILE_N (default 128 each)
//   WORK_DIR  : MXP_Tools 입력 .hex 디렉토리   (Task 2 에선 무시)
//   DUMP_DIR  : SRAM 덤프 출력 디렉토리        (REQUIRED)
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top_tb;

    // ─── plusarg parsing ───────────────────────────────────────────
    integer A_PREC, B_PREC;
    integer TILE_M, TILE_K, TILE_N;
    reg [8*256-1:0] WORK_DIR;  // string buf
    reg [8*256-1:0] DUMP_DIR;

    initial begin
        if (!$value$plusargs("A_PREC=%d",   A_PREC))    A_PREC   = 8;
        if (!$value$plusargs("B_PREC=%d",   B_PREC))    B_PREC   = 8;
        if (!$value$plusargs("TILE_M=%d",   TILE_M))    TILE_M   = 128;
        if (!$value$plusargs("TILE_K=%d",   TILE_K))    TILE_K   = 128;
        if (!$value$plusargs("TILE_N=%d",   TILE_N))    TILE_N   = 128;
        if (!$value$plusargs("WORK_DIR=%s", WORK_DIR))  WORK_DIR = "work/smoke";
        if (!$value$plusargs("DUMP_DIR=%s", DUMP_DIR))  DUMP_DIR = "work/smoke/hw_out";
    end

    // ─── clock / reset ─────────────────────────────────────────────
    reg clk = 0;
    always #5 clk = ~clk;        // 100 MHz nominal (10 ns period)
    reg rst = 1;

    // ─── DUT 포트 wires ────────────────────────────────────────────
    // GEMM stimuli — Task 2 에선 idle (모두 0)
    reg [32-1:0]              in_a              = 0;
    reg signed [32*8-1:0]     in_b              = 0;
    reg [32-1:0]              in_control        = 0;
    reg [32-1:0]              in_loadEN         = 0;
    reg [1:0]                 in_Station_control= 0;
    reg [4*8-1:0]             in_Scale_Activation= 0;
    reg                       in_station_control = 0;
    reg                       in_station_loadEN  = 0;
    reg                       in_start_accumulate = 0;
    reg [1:0]                 in_Wcontrol        = 0;
    reg [7:0]                 in_scale_weight    = 0;

    wire [32*60-1:0]          out_accumulate;
    wire [32*36-1:0]          out_scale;
    wire [32-1:0]             out_fire;

    // RMW — Task 2 에선 idle
    reg [31:0]                rmw_in_GEMM       = 0;
    reg [8:0]                 rmw_scale         = 0;
    wire [31:0]               rmw_out_RMW;

    // SRAM
    reg                       sram_CEB          = 1;   // idle high
    reg                       sram_WEB          = 1;
    reg [18:0]                sram_A            = 0;
    reg [31:0]                sram_WMASK        = 32'hFFFFFFFF;
    reg                       sram_D_use_zero   = 1;
    wire [31:0]               sram_Q;

    // ─── DUT instance ──────────────────────────────────────────────
    gemm_sram_top #(
        .NUM_BANKS  (16),
        .BANK_DEPTH (32768)
    ) u_top (
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
        .out_fire(out_fire),
        .rmw_in_GEMM(rmw_in_GEMM),
        .rmw_scale(rmw_scale),
        .rmw_out_RMW(rmw_out_RMW),
        .sram_CEB(sram_CEB), .sram_WEB(sram_WEB),
        .sram_A(sram_A), .sram_WMASK(sram_WMASK),
        .sram_D_use_zero(sram_D_use_zero),
        .sram_Q(sram_Q)
    );

    // ─── helper: 1 word write ─────────────────────────────────────
    task sram_write(input [18:0] addr, input [31:0] data, input use_zero);
        begin
            @(posedge clk);
            sram_CEB        <= 1'b0;
            sram_WEB        <= 1'b0;
            sram_A          <= addr;
            sram_WMASK      <= 32'hFFFFFFFF;
            sram_D_use_zero <= use_zero;
            // 이 task 안에서 rmw_in_GEMM 은 안 만짐 — use_zero=1 일 때 D 가 0 으로 강제됨
        end
    endtask

    task sram_idle;
        begin
            @(posedge clk);
            sram_CEB <= 1'b1;
            sram_WEB <= 1'b1;
        end
    endtask

    // ─── main sequence ─────────────────────────────────────────────
    integer i, b;
    reg [8*512-1:0] dump_path;

    initial begin
        // INIT
        #20;
        rst = 0;
        sram_D_use_zero = 1;

        // SRAM zero priming — flat addr 0..16383
        for (i = 0; i < 16384; i = i + 1) begin
            sram_write(i[18:0], 32'h00000000, 1'b1);
        end
        sram_idle;
        sram_idle;     // settle

        // DUMP — 16 bank 모두 0..1023 word 범위
        for (b = 0; b < 16; b = b + 1) begin
            $sformat(dump_path, "%0s/bank%0d.mem", DUMP_DIR, b);
            case (b)
                0:  $writememh(dump_path, u_top.u_sram.g_bank[0].u_bank.mem,  0, 1023);
                1:  $writememh(dump_path, u_top.u_sram.g_bank[1].u_bank.mem,  0, 1023);
                2:  $writememh(dump_path, u_top.u_sram.g_bank[2].u_bank.mem,  0, 1023);
                3:  $writememh(dump_path, u_top.u_sram.g_bank[3].u_bank.mem,  0, 1023);
                4:  $writememh(dump_path, u_top.u_sram.g_bank[4].u_bank.mem,  0, 1023);
                5:  $writememh(dump_path, u_top.u_sram.g_bank[5].u_bank.mem,  0, 1023);
                6:  $writememh(dump_path, u_top.u_sram.g_bank[6].u_bank.mem,  0, 1023);
                7:  $writememh(dump_path, u_top.u_sram.g_bank[7].u_bank.mem,  0, 1023);
                8:  $writememh(dump_path, u_top.u_sram.g_bank[8].u_bank.mem,  0, 1023);
                9:  $writememh(dump_path, u_top.u_sram.g_bank[9].u_bank.mem,  0, 1023);
                10: $writememh(dump_path, u_top.u_sram.g_bank[10].u_bank.mem, 0, 1023);
                11: $writememh(dump_path, u_top.u_sram.g_bank[11].u_bank.mem, 0, 1023);
                12: $writememh(dump_path, u_top.u_sram.g_bank[12].u_bank.mem, 0, 1023);
                13: $writememh(dump_path, u_top.u_sram.g_bank[13].u_bank.mem, 0, 1023);
                14: $writememh(dump_path, u_top.u_sram.g_bank[14].u_bank.mem, 0, 1023);
                15: $writememh(dump_path, u_top.u_sram.g_bank[15].u_bank.mem, 0, 1023);
            endcase
        end

        $display("INTEGRATION TB: ZERO-PRIME DUMP DONE");
        $finish;
    end

    // ─── timeout safety ────────────────────────────────────────────
    initial begin
        #2_000_000;   // 2 ms = 200000 cycle @ 100 MHz
        $display("ERROR: timeout");
        $finish;
    end

endmodule
