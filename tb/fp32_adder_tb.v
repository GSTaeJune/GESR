`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// fp32_adder_tb — fp32_adder 단위 검증 TB.
//
// 검증 목적:
//   gemm_sram.srcs/sources_1/new/fp32_adder.v 가 L_ADD 사이클 파이프라인을
//   거친 후에도 IEEE-754 FP32 덧셈을 RNE 라운딩으로 정확히 수행하는지 확인.
//   HardFloat AddRecFN 래핑 + recoded FP 변환부의 결합 동작을 직접 자극.
//
// 검증 시나리오 (4 케이스, L_ADD=3 사이클 후 출력 채킹):
//   Case 1: 1.0 + 1.0       → 2.0                (가장 기본, 동일 부호 정상값)
//   Case 2: 1.0 + (-1.0)    → 0.0                (부호 상쇄, exact zero 확인)
//   Case 3: 1.5 + 2.5       → 4.0                (분수 정렬 + 정상 캐리)
//   Case 4: 0.0 + 3.14...   → 3.14...            (zero passthrough)
//
// 각 케이스마다:
//   - negedge 에 a/b 드라이브
//   - L_ADD 사이클 동안 @(posedge clk)
//   - check() 호출 → 기대값과 sum 비교
//////////////////////////////////////////////////////////////////////////////////

module fp32_adder_tb;
    localparam L_ADD      = 3;       // DUT 파이프라인 단수 (RMW.v 와 동일)
    localparam CLK_PERIOD = 10;      // 100 MHz

    reg              clk;
    reg              rst;
    reg  [31:0]      a;
    reg  [31:0]      b;
    wire [31:0]      sum;

    fp32_adder #(.L_ADD(L_ADD)) dut (
        .clk(clk), .rst(rst),
        .a(a), .b(b),
        .sum(sum)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("fp32_adder_tb.vcd"); $dumpvars(0, fp32_adder_tb); end

    integer errors;
    integer i;

    // sum 과 expected 비교 → PASS/FAIL 로그. errors 누적.
    task check;
        input [31:0] expected;
        input [255:0] label;
        begin
            #1;
            if (sum !== expected) begin
                $display("FAIL %0s: got %h expected %h", label, sum, expected);
                errors = errors + 1;
            end else begin
                $display("PASS %0s", label);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst = 1; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Case 1: 1.0 + 1.0 = 2.0 (가장 기본 동작)
        @(negedge clk); a = 32'h3F800000; b = 32'h3F800000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h40000000, "1.0 + 1.0 -> 2.0");

        // Case 2: 1.0 + (-1.0) = 0.0 (부호 상쇄, exact zero)
        @(negedge clk); a = 32'h3F800000; b = 32'hBF800000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h00000000, "1.0 + (-1.0) -> 0.0");

        // Case 3: 1.5 + 2.5 = 4.0 (다른 지수 정렬 + 캐리)
        @(negedge clk); a = 32'h3FC00000; b = 32'h40200000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h40800000, "1.5 + 2.5 -> 4.0");

        // Case 4: 0.0 + 3.14 ≈ 3.14 (zero passthrough)
        @(negedge clk); a = 32'h00000000; b = 32'h4048F5C3;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h4048F5C3, "0.0 + 3.14 -> 3.14");

        if (errors == 0) $display("fp32_adder_tb: ALL TESTS PASSED");
        else              $display("fp32_adder_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
