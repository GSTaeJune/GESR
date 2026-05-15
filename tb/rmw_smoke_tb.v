`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// rmw_smoke_tb — Berkeley HardFloat 라운드트립 smoke TB.
//
// 검증 목적:
//   third_party/berkeley-hardfloat/ 에 vendored 된 HardFloat 모듈들이
//   xelab 단계에서 제대로 elaborate 되는지, 그리고 INT32 → recoded FP32 →
//   IEEE-754 FP32 의 왕복 변환이 사칙연산 없이도 정상적으로 끝까지 통하는지
//   확인한다. RMW 본체 로직(int_to_fp32, fp32_adder)을 짓기 전에 백엔드
//   라이브러리부터 동작하는지 보는 1차 sanity check.
//
// 검증 내용 (4 케이스, 조합 회로라 클록 불필요):
//   int=1   → 0x3F800000  (+1.0)
//   int=0   → 0x00000000  (+0.0, zero passthrough)
//   int=-1  → 0xBF800000  (-1.0)
//   int=2   → 0x40000000  (+2.0)
//
// 사용 모듈명은 third_party/berkeley-hardfloat/VENDORING.md 기준.
//////////////////////////////////////////////////////////////////////////////////

module rmw_smoke_tb;

    reg  [31:0] int_in;
    wire [32:0] recFN;
    wire [31:0] fp_out;

    // 1) 부호 있는 INT32 → recoded FP32 (조합)
    INToRecFN_i32_e8_s24 u_i2f (
        .io_signedIn       (1'b1),
        .io_in             (int_in),
        .io_roundingMode   (3'd0),       // RNE
        .io_out            (recFN),
        .io_exceptionFlags ()
    );

    // 2) recoded FP32 → IEEE-754 FP32 (조합)
    FNFromRecFN_wrapper u_f2f (
        .in  (recFN),
        .out (fp_out)
    );

    integer errors;

    initial begin
        $dumpfile("rmw_smoke_tb.vcd");
        $dumpvars(0, rmw_smoke_tb);

        errors = 0;

        // Case 1: int=1 → +1.0
        int_in = 32'sd1;  #1;
        if (fp_out !== 32'h3F800000) begin
            $display("FAIL int=1: got %h expected 3F800000", fp_out);
            errors = errors + 1;
        end

        // Case 2: int=0 → +0.0 (zero encoding 그대로 통과)
        int_in = 32'sd0;  #1;
        if (fp_out !== 32'h00000000) begin
            $display("FAIL int=0: got %h expected 00000000", fp_out);
            errors = errors + 1;
        end

        // Case 3: int=-1 → -1.0 (부호 비트 처리 확인)
        int_in = -32'sd1; #1;
        if (fp_out !== 32'hBF800000) begin
            $display("FAIL int=-1: got %h expected BF800000", fp_out);
            errors = errors + 1;
        end

        // Case 4: int=2 → +2.0
        int_in = 32'sd2;  #1;
        if (fp_out !== 32'h40000000) begin
            $display("FAIL int=2: got %h expected 40000000", fp_out);
            errors = errors + 1;
        end

        if (errors == 0) $display("rmw_smoke_tb: ALL TESTS PASSED");
        else              $display("rmw_smoke_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
