//=============================================================================
// sram_1rw_banked_mp.v - Per-bank port-exposed 1RW SRAM wrapper (Verilog-2001)
//
// `sram_1rw_banked.v` (mux 형) 와 달리, 각 bank 의 CEB/WEB/A/D/WMASK/Q 를
// flat bus 로 외부에 노출. NUM_BANKS 개 독립 1RW port → 매 cycle bandwidth =
// NUM_BANKS × (leaf 1RW port). INTERLEAVED/SEQUENTIAL 의미는 호출자가 결정
// (wrapper 레벨에서 BANK_STRATEGY 없음).
//
// gemm_sram_top 의 col j → bank j 매핑 (col-parallel RMW 32×) 에 사용.
//
// Spec : docs/superpowers/specs/2026-05-15-rmw-32x-design.md
// Style: Pure Verilog-2001 (sram repo 컨벤션 — packed array 안 씀).
//=============================================================================
`timescale 1ns/1ps

module sram_1rw_banked_mp (CLK, CEB, WEB, A, D, WMASK, Q);

    parameter DATA_WIDTH = 32;
    parameter NUM_BANKS  = 32;
    parameter BANK_DEPTH = 1024;
    parameter PIPELINE   = 0;       // leaf PIPELINE 그대로 전달 (0=1cy, 1=2cy)

    // clog2 (sram repo 컨벤션)
    function integer clog2;
        input integer value;
        integer i;
        begin
            clog2 = 0;
            for (i = value - 1; i > 0; i = i >> 1)
                clog2 = clog2 + 1;
        end
    endfunction

    localparam BANK_ADDR_WIDTH = clog2(BANK_DEPTH);

    input                                          CLK;
    input  [NUM_BANKS-1:0]                         CEB;
    input  [NUM_BANKS-1:0]                         WEB;
    input  [NUM_BANKS*BANK_ADDR_WIDTH-1:0]         A;
    input  [NUM_BANKS*DATA_WIDTH-1:0]              D;
    input  [NUM_BANKS*DATA_WIDTH-1:0]              WMASK;
    output [NUM_BANKS*DATA_WIDTH-1:0]              Q;

    // Sanity (sim-only)
    initial begin
        if (NUM_BANKS < 2 || (NUM_BANKS & (NUM_BANKS - 1)) != 0) begin
            $display("ERROR: NUM_BANKS must be a power of 2 >= 2 (got %0d)", NUM_BANKS);
            $finish;
        end
        if (PIPELINE != 0 && PIPELINE != 1) begin
            $display("ERROR: PIPELINE must be 0 or 1 (got %0d)", PIPELINE);
            $finish;
        end
    end

    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : g_bank
            sram_1rw #(
                .DATA_WIDTH (DATA_WIDTH),
                .DEPTH      (BANK_DEPTH),
                .PIPELINE   (PIPELINE),
                .INIT_FILE  ("")
            ) u_bank (
                .CLK   (CLK),
                .CEB   (CEB[i]),
                .WEB   (WEB[i]),
                .A     (A[i*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH]),
                .D     (D[i*DATA_WIDTH +: DATA_WIDTH]),
                .WMASK (WMASK[i*DATA_WIDTH +: DATA_WIDTH]),
                .Q     (Q[i*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

endmodule
