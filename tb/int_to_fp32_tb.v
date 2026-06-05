`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// int_to_fp32_tb — int_to_fp32 단위 검증 TB.
//
// 검증 목적:
//   gemm_sram.srcs/sources_1/new/int_to_fp32.v 가 다음 변환을 L_CONV 사이클
//   파이프라인을 거친 후 정확히 수행하는지 확인:
//
//       out_fp32 = in_int * 2^(scale - 127)   (RNE)
//
//   특히 다음 두 경계 동작을 강하게 자극:
//     1) 일반 동작: scale ≠ 127 일 때 지수 이동이 IEEE-754 비트로 정확히 표현되는가
//     2) Zero passthrough: in_int = 0 일 때 어떤 scale 이 들어와도 반드시 +0.0
//        - 과거 회귀: int=0 + scale≠127 시 HardFloat recoded zero 인코딩이
//          (scale-127) 더하기로 손상돼 inf/subnormal 이 튀어나오던 버그를
//          int_to_fp32.v 에서 `int_is_zero` 가드로 막음. Case 7-10 이 그 회귀를 잡음.
//
// 검증 시나리오 (11 케이스, L_CONV=2 사이클 후 출력 체킹):
//   일반:
//     Case 1:  int=1   scale=127  → +1.0           (가장 기본, bias 중심)
//     Case 2:  int=-1  scale=127  → -1.0           (부호 비트 처리)
//     Case 3:  int=2   scale=127  → +2.0
//     Case 4:  int=0   scale=127  → +0.0           (zero 정상 경로)
//     Case 5:  int=1   scale=128  → +2.0           (1 × 2^1, exp 이동 up)
//     Case 6:  int=4   scale=125  → +1.0           (4 × 2^-2, exp 이동 down)
//     Case 6b: int=256 scale=-2   → 2^-121         (음수 signed scale, exp 큰 폭 down)
//   Zero passthrough 회귀 (int=0 + scale≠127):
//     Case 7:  int=0   scale=125  → +0.0
//     Case 8:  int=0   scale=128  → +0.0
//     Case 9:  int=0   scale=0    → +0.0           (scale 최저)
//     Case 10: int=0   scale=255  → +0.0           (scale 최고)
//////////////////////////////////////////////////////////////////////////////////

module int_to_fp32_tb;
    localparam L_CONV     = 2;       // DUT 파이프라인 단수 (RMW.v 와 동일)
    localparam CLK_PERIOD = 10;      // 100 MHz

    reg              clk;
    reg              rst;
    reg  [31:0]      in_int;
    reg  [8:0]       scale;
    wire [31:0]      out_fp32;

    int_to_fp32 #(.L_CONV(L_CONV)) dut (
        .clk(clk), .rst(rst),
        .in_int(in_int), .scale(scale),
        .out_fp32(out_fp32)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("int_to_fp32_tb.vcd"); $dumpvars(0, int_to_fp32_tb); end

    integer errors;
    integer i;

    // out_fp32 와 expected 비교 → PASS/FAIL 로그. errors 누적.
    task check;
        input [31:0] expected;
        input [255:0] label;
        begin
            #1;
            if (out_fp32 !== expected) begin
                $display("FAIL %0s: got %h expected %h", label, out_fp32, expected);
                errors = errors + 1;
            end else begin
                $display("PASS %0s", label);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst    = 1; in_int = 0; scale = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Case 1: int=1, scale=127 → +1.0 (bias 중심, 가장 기본)
        @(negedge clk); in_int = 32'sd1;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h3F800000, "int=1 scale=127 -> 1.0");

        // Case 2: int=-1 → -1.0 (부호 비트 처리)
        @(negedge clk); in_int = -32'sd1;  scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'hBF800000, "int=-1 scale=127 -> -1.0");

        // Case 3: int=2 → 2.0
        @(negedge clk); in_int = 32'sd2;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h40000000, "int=2 scale=127 -> 2.0");

        // Case 4: int=0, scale=127 → +0.0 (zero 정상 경로)
        @(negedge clk); in_int = 32'sd0;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h00000000, "int=0 scale=127 -> 0.0");

        // Case 5: int=1, scale=128 → 2.0 (1 × 2^1, exp 이동 up)
        @(negedge clk); in_int = 32'sd1;   scale = 9'sd128;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h40000000, "int=1 scale=128 -> 2.0");

        // Case 6: int=4, scale=125 → 1.0 (4 × 2^-2, exp 이동 down)
        @(negedge clk); in_int = 32'sd4;   scale = 9'sd125;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h3F800000, "int=4 scale=125 -> 1.0");

        // Case 6b: int=256, scale=-2 (9'h1FE) -> 2^8 * 2^-129 = 2^-121 (signed scale, exp down)
        @(negedge clk); in_int = 32'sd256;  scale = 9'h1FE;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h03000000, "int=256 scale=-2 -> 2^-121 (signed)");

        // ───── Zero passthrough 회귀 케이스 (int=0 + scale≠127) ─────
        // 버그 히스토리: int=0 일 때 HardFloat recoded 표현의 특수 zero
        // 인코딩 (exp=0) 에 (scale-127) 을 더하면 정상/subnormal/inf 영역으로
        // 망가짐. int_to_fp32.v 의 `int_is_zero` 가드가 fix. 아래 4 케이스가
        // 그 가드를 보호하는 회귀 테스트.
        // Case 7: int=0, scale=125 → +0.0
        @(negedge clk); in_int = 32'sd0;   scale = 9'sd125;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h00000000, "int=0 scale=125 -> 0.0");

        // Case 8: int=0, scale=128 → +0.0
        @(negedge clk); in_int = 32'sd0;   scale = 9'sd128;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h00000000, "int=0 scale=128 -> 0.0");

        // Case 9: int=0, scale=0 → +0.0 (scale 최저값)
        @(negedge clk); in_int = 32'sd0;   scale = 9'sd0;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h00000000, "int=0 scale=0 -> 0.0");

        // Case 10: int=0, scale=255 → +0.0 (signed 9비트 최대 양수 +255)
        @(negedge clk); in_int = 32'sd0;   scale = 9'sd255;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h00000000, "int=0 scale=255 -> 0.0");

        if (errors == 0) $display("int_to_fp32_tb: ALL TESTS PASSED");
        else              $display("int_to_fp32_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
