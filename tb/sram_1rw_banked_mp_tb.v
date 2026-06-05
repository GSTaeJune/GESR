`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// sram_1rw_banked_mp_tb — per-bank port-exposed 1RW SRAM wrapper 단위 검증.
//
// 검증 목적:
//   sram_1rw_banked_mp 가 NUM_BANKS 개의 독립 1RW bank port 를 노출하고,
//   각 bank 가 같은 cycle 에 서로 다른 word 를 read 또는 write 할 수 있음을
//   확인. 32 RMW × 32 bank conflict-free 매핑의 RTL 기반.
//
// 검증 내용:
//   1) bank 마다 다른 데이터를 동시 write
//   2) 1 cycle 후 동시 read (PIPELINE=0 leaf 의 read latency)
//   3) 각 bank Q 가 자기 write 데이터와 일치 (cross-bank 오염 없음)
//   4) idle bank (CEB=1) 가 active bank 의 동작에 영향 없음
//
// 동작 의도:
//   기존 sram_1rw_banked (single A/D/Q + 내부 mux) 와 달리, 이 wrapper 는
//   bank 별 독립 포트라 동시 다중 R/W 가능. 합성 후 매 cycle bandwidth =
//   NUM_BANKS × (leaf 1RW port). 외부 callsite (gemm_sram_top) 가 bank
//   매핑 책임 (INTERLEAVED/SEQUENTIAL 의미는 호출자가 결정).
//
// 회귀 게이트: `bash sim/run_sram_mp.sh` → "ALL <N> TESTS PASSED".
//////////////////////////////////////////////////////////////////////////////

module sram_1rw_banked_mp_tb;

    // 작은 config (sram repo 의 TB sizing 컨벤션) — fast sim
    localparam DATA_WIDTH = 32;
    localparam NUM_BANKS  = 4;
    localparam BANK_DEPTH = 16;
    localparam BANK_ADDR_WIDTH = 4;     // clog2(16)

    reg CLK = 0;
    always #5 CLK = ~CLK;

    reg  [NUM_BANKS-1:0]                       CEB;
    reg  [NUM_BANKS-1:0]                       WEB;
    reg  [NUM_BANKS*BANK_ADDR_WIDTH-1:0]       A;
    reg  [NUM_BANKS*DATA_WIDTH-1:0]            D;
    reg  [NUM_BANKS*DATA_WIDTH-1:0]            WMASK;
    wire [NUM_BANKS*DATA_WIDTH-1:0]            Q;

    sram_1rw_banked_mp #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_BANKS  (NUM_BANKS),
        .BANK_DEPTH (BANK_DEPTH),
        .PIPELINE   (0)
    ) dut (
        .CLK(CLK), .CEB(CEB), .WEB(WEB),
        .A(A), .D(D), .WMASK(WMASK), .Q(Q)
    );

    integer pass = 0, fail = 0;
    integer b;
    reg [DATA_WIDTH-1:0] expected, got;

    // 슬라이스 헬퍼
    task automatic set_bank_A;
        input integer bi;
        input [BANK_ADDR_WIDTH-1:0] addr;
        begin
            A[bi*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH] = addr;
        end
    endtask
    task automatic set_bank_D;
        input integer bi;
        input [DATA_WIDTH-1:0] data;
        begin
            D[bi*DATA_WIDTH +: DATA_WIDTH] = data;
        end
    endtask
    function [DATA_WIDTH-1:0] get_bank_Q;
        input integer bi;
        begin
            get_bank_Q = Q[bi*DATA_WIDTH +: DATA_WIDTH];
        end
    endfunction

    initial begin
        // 초기 idle
        CEB   = {NUM_BANKS{1'b1}};
        WEB   = {NUM_BANKS{1'b1}};
        A     = 0;
        D     = 0;
        WMASK = {NUM_BANKS*DATA_WIDTH{1'b1}};
        @(posedge CLK);

        // ─── Scenario 1: bank 마다 다른 데이터 동시 write ──────────────
        @(negedge CLK);
        for (b = 0; b < NUM_BANKS; b = b + 1) begin
            set_bank_A(b, 4'd0);
            set_bank_D(b, 32'hDEADBE00 + b);  // bank 마다 고유 데이터
        end
        CEB = {NUM_BANKS{1'b0}};
        WEB = {NUM_BANKS{1'b0}};
        @(posedge CLK);

        // ─── Scenario 2: 동시 read 발사 ────────────────────────────────
        @(negedge CLK);
        WEB = {NUM_BANKS{1'b1}};
        for (b = 0; b < NUM_BANKS; b = b + 1) set_bank_A(b, 4'd0);
        @(posedge CLK);

        // ─── Scenario 3: Q 검증 (PIPELINE=0 → 동일 cycle 의 Q 가 유효) ─
        #1;
        for (b = 0; b < NUM_BANKS; b = b + 1) begin
            expected = 32'hDEADBE00 + b;
            got      = get_bank_Q(b);
            if (got === expected) begin
                pass = pass + 1;
                $display("[PASS] bank %0d Q = %h", b, got);
            end else begin
                fail = fail + 1;
                $display("[FAIL] bank %0d Q = %h, expected %h", b, got, expected);
            end
        end

        // ─── Scenario 4: idle bank 영향 없음 ───────────────────────────
        // bank 0 만 active, 나머지 CEB=1. bank 0 에 새 값 write 후
        // 읽었을 때 다른 bank 의 Q 가 변하지 않음.
        @(negedge CLK);
        CEB = {NUM_BANKS{1'b1}}; CEB[0] = 1'b0;
        WEB = {NUM_BANKS{1'b1}}; WEB[0] = 1'b0;
        set_bank_A(0, 4'd1);
        set_bank_D(0, 32'hCAFEBABE);
        @(posedge CLK);
        @(negedge CLK);
        WEB[0] = 1'b1;
        set_bank_A(0, 4'd1);
        @(posedge CLK);
        #1;
        if (get_bank_Q(0) === 32'hCAFEBABE) begin
            pass = pass + 1;
            $display("[PASS] bank 0 isolated write OK");
        end else begin
            fail = fail + 1;
            $display("[FAIL] bank 0 isolated write = %h", get_bank_Q(0));
        end
        // idle bank (1..N-1) 은 Scenario 2/3 에서 addr 0 의 값을 마지막으로 read
        // 한 뒤 CEB=1 로 유지됐으므로 Q 가 변하면 안 됨 (active bank 0 의 오염 검출).
        for (b = 1; b < NUM_BANKS; b = b + 1) begin
            expected = 32'hDEADBE00 + b;
            got      = get_bank_Q(b);
            if (got === expected) begin
                pass = pass + 1;
                $display("[PASS] idle bank %0d held Q = %h", b, got);
            end else begin
                fail = fail + 1;
                $display("[FAIL] idle bank %0d Q = %h, expected %h (corrupted by active bank)", b, got, expected);
            end
        end

        // ─── 최종 보고 ────────────────────────────────────────────────
        if (fail == 0) $display("sram_1rw_banked_mp_tb: ALL %0d TESTS PASSED", pass);
        else           $display("sram_1rw_banked_mp_tb: %0d PASS, %0d FAIL", pass, fail);
        $finish;
    end

endmodule
