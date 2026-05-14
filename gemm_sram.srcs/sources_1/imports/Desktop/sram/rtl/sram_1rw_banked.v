//=============================================================================
// sram_1rw_banked.v - Banked 1RW SRAM wrapper (Verilog-2001)
//
// Leaf `sram_1rw` 인스턴스를 NUM_BANKS개 묶어 외부에서는 하나의 큰 SRAM처럼
// 보이게 만드는 wrapper. Banking strategy(INTERLEAVED/SEQUENTIAL)와 wrapper-
// level pipeline(PIPELINE=0/1)을 parameter로 선택.
//
// Spec: docs/superpowers/specs/2026-05-13-sram-1rw-banked-design.md
//=============================================================================
`timescale 1ns/1ps

module sram_1rw_banked (CLK, CEB, WEB, A, D, WMASK, Q);

    //-------------------------------------------------------------------------
    // Parameters
    //-------------------------------------------------------------------------
    parameter DATA_WIDTH    = 32;
    parameter NUM_BANKS     = 16;
    parameter BANK_DEPTH    = 32768;
    parameter BANK_STRATEGY = "INTERLEAVED";   // or "SEQUENTIAL"
    parameter PIPELINE      = 0;               // wrapper output stage (0 or 1)

    //-------------------------------------------------------------------------
    // clog2 — Verilog-2001 constant function (leaf bank과 동일)
    //-------------------------------------------------------------------------
    function integer clog2;
        input integer value;
        integer i;
        begin
            clog2 = 0;
            for (i = value - 1; i > 0; i = i >> 1)
                clog2 = clog2 + 1;
        end
    endfunction

    localparam BANK_BITS       = clog2(NUM_BANKS);
    localparam BANK_ADDR_WIDTH = clog2(BANK_DEPTH);
    localparam ADDR_WIDTH      = BANK_BITS + BANK_ADDR_WIDTH;
    localparam TOTAL_DEPTH     = NUM_BANKS * BANK_DEPTH;

    //-------------------------------------------------------------------------
    // Ports (non-ANSI — ADDR_WIDTH는 localparam 이후 평가됨)
    //-------------------------------------------------------------------------
    input                       CLK;
    input                       CEB;
    input                       WEB;
    input  [ADDR_WIDTH-1:0]     A;
    input  [DATA_WIDTH-1:0]     D;
    input  [DATA_WIDTH-1:0]     WMASK;
    output [DATA_WIDTH-1:0]     Q;

    //-------------------------------------------------------------------------
    // Sim-only sanity checks (합성 시 무시되는 initial 블록)
    //-------------------------------------------------------------------------
    initial begin
        // NUM_BANKS는 2의 거듭제곱이고 ≥ 2 (x & (x-1) == 0 트릭)
        if (NUM_BANKS < 2 || (NUM_BANKS & (NUM_BANKS - 1)) != 0) begin
            $display("ERROR: NUM_BANKS must be a power of 2 >= 2 (got %0d)", NUM_BANKS);
            $finish;
        end
        // BANK_STRATEGY는 두 문자열 중 하나
        if (BANK_STRATEGY != "INTERLEAVED" && BANK_STRATEGY != "SEQUENTIAL") begin
            $display("ERROR: BANK_STRATEGY must be \"INTERLEAVED\" or \"SEQUENTIAL\" (got %0s)",
                     BANK_STRATEGY);
            $finish;
        end
        // PIPELINE은 0 또는 1
        if (PIPELINE != 0 && PIPELINE != 1) begin
            $display("ERROR: PIPELINE must be 0 or 1 (got %0d)", PIPELINE);
            $finish;
        end
    end

    //-------------------------------------------------------------------------
    // 주소 분할 — BANK_STRATEGY로 결정
    // (두 분기 함께 작성. INTERLEAVED는 LSB가 bank_sel, SEQUENTIAL은 MSB.)
    //-------------------------------------------------------------------------
    wire [BANK_BITS-1:0]       bank_sel;
    wire [BANK_ADDR_WIDTH-1:0] within_a;

    generate
        if (BANK_STRATEGY == "INTERLEAVED") begin : g_addr_interleaved
            assign bank_sel = A[BANK_BITS-1:0];
            assign within_a = A[ADDR_WIDTH-1:BANK_BITS];
        end else begin : g_addr_sequential
            assign bank_sel = A[ADDR_WIDTH-1:ADDR_WIDTH-BANK_BITS];
            assign within_a = A[ADDR_WIDTH-BANK_BITS-1:0];
        end
    endgenerate

    //-------------------------------------------------------------------------
    // Per-bank instantiation + CEB gating
    //
    // bank_ceb[i] = CEB || (bank_sel != i)
    //   - wrapper CEB가 1이면 모든 bank idle.
    //   - 선택 안 된 bank도 CEB=1로 idle.
    //   - 선택된 bank만 CEB=0으로 active.
    //
    // WEB, within_a, D, WMASK는 모든 bank에 broadcast — idle된 bank는 무시.
    //-------------------------------------------------------------------------
    wire [DATA_WIDTH-1:0] bank_q   [0:NUM_BANKS-1];
    wire                  bank_ceb [0:NUM_BANKS-1];

    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : g_bank
            assign bank_ceb[i] = CEB || (bank_sel != i);

            sram_1rw #(
                .DATA_WIDTH (DATA_WIDTH),
                .DEPTH      (BANK_DEPTH),
                .PIPELINE   (0),           // wrapper output stage에서 처리
                .INIT_FILE  ("")
            ) u_bank (
                .CLK   (CLK),
                .CEB   (bank_ceb[i]),
                .WEB   (WEB),
                .A     (within_a),
                .D     (D),
                .WMASK (WMASK),
                .Q     (bank_q[i])
            );
        end
    endgenerate

    //-------------------------------------------------------------------------
    // bank_sel 1-cycle 지연 — leaf bank의 read latency와 정렬
    //-------------------------------------------------------------------------
    reg [BANK_BITS-1:0] bank_sel_d1;
    always @(posedge CLK) bank_sel_d1 <= bank_sel;

    wire [DATA_WIDTH-1:0] mux_out = bank_q[bank_sel_d1];

    //-------------------------------------------------------------------------
    // Output register (PIPELINE=1일 때만 wrapper 출력단 register 단 1개 추가)
    //-------------------------------------------------------------------------
    generate
        if (PIPELINE == 0) begin : g_out_pass
            assign Q = mux_out;
        end else begin : g_out_reg
            reg [DATA_WIDTH-1:0] Q_reg;
            always @(posedge CLK) Q_reg <= mux_out;
            assign Q = Q_reg;
        end
    endgenerate

endmodule
