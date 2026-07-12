`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// bf16_adder - IEEE bfloat16 덧셈 (RNE). fp32 도메인에서 계산 후 narrow.
//
//   bf16 a,b -> {a,16'h0}/{b,16'h0} 로 fp32 확장 (bf16이 fp32 상위16비트라 exact)
//   -> 기존 검증된 fp32_adder (HardFloat AddRecFN, e8/s24) -> sum_fp32
//   -> fp32_to_bf16_rne -> sum_bf16.
// round_bf16(round_fp32(a+b)) = round_bf16(a+b) = true bf16 add (double-rounding
// 안전: 24 >= 2*8+2). ml_dtypes bf16 add 와 bit 일치.
// 파이프라인 지연 = fp32_adder 의 L_ADD (narrow 는 조합).
//
// 왜 fp32 도메인인가: HardFloat AddRecFN(8,8) 은 elaborate 안 됨 (bf16 폭 미지원).
//   그래서 exact widen → 검증된 fp32 가산 → RNE narrow 우회를 쓴다.
//
// 포트: a,b(bf16 16b) → sum(bf16 16b). rst 미사용.
// 인스턴스: fp32_adder (fp32 번들 AddRecFN) + fp32_to_bf16_rne (로컬).
//   인스턴스되는 곳: RMW.
// 상태: ACTIVE (RMW 덧셈단 본선).
//
// 검증: `bash sim/run_bf16_adder.sh` → 기대 "ALL 200005 TESTS PASSED"
//   (ml_dtypes 크로스체크; NaN 부호는 IEEE 미규정이라 sign-agnostic 비교).
//////////////////////////////////////////////////////////////////////////////////
module bf16_adder #(
    parameter L_ADD = 3
)(
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] sum
);
    wire [31:0] a_fp32 = {a, 16'h0000};   // exact bf16 -> fp32 widen
    wire [31:0] b_fp32 = {b, 16'h0000};
    wire [31:0] sum_fp32;
    fp32_adder #(.L_ADD(L_ADD)) u_add (
        .clk(clk), .rst(rst), .a(a_fp32), .b(b_fp32), .sum(sum_fp32)
    );
    fp32_to_bf16_rne u_narrow (.in(sum_fp32), .out(sum));
endmodule
