`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// fp32_to_bf16_rne - IEEE FP32 -> bfloat16, round-half-to-even (combinational).
//
// bf16 = fp32의 상위 16비트. 버려지는 하위 16비트로 RNE:
//   round bit = in[15], sticky = |in[14:0], lsb = in[16] (bf16 mantissa LSB).
//   round_up = round & (sticky | lsb) 이면 top16 += 1 (carry가 exp/inf로 전파 = 정상).
// NaN -> canonical quiet NaN (sign|0x7FC0), inf -> top16 그대로.
// ml_dtypes 의 fp32->bf16 및 Phase-1 f32_to_bf16_bits_rne 레퍼런스와 bit 일치.
//
// 포트: in(fp32 32b) → out(bf16 16b). clk 없음 (순수 조합).
// 인스턴스: 없음 (leaf). 인스턴스되는 곳: 없음 — 2026-07-13 native 재작성으로
//   int_to_bf16 v3 / bf16_adder v3 가 자체 라운더를 내장하며 데이터패스에서 빠짐.
// 상태: PRESERVED (자체 TB 게이트 70012 로 보존; fp32 복구 라인 참고용).
//
// 검증: `bash sim/run_fp32_to_bf16_rne.sh` → 기대 "ALL 70012 TESTS PASSED"
//   (subnormal 포함 단일 RNE 를 ml_dtypes 로 전수/근사 크로스체크).
//////////////////////////////////////////////////////////////////////////////////
module fp32_to_bf16_rne (
    input  wire [31:0] in,
    output wire [15:0] out
);
    wire        sign      = in[31];
    wire [7:0]  exp       = in[30:23];
    wire        mant_nz   = |in[22:0];
    wire [15:0] top       = in[31:16];      // bf16 layout: {sign, exp[7:0], mant[6:0]}
    wire        lsb       = in[16];
    wire        round_bit = in[15];
    wire        sticky    = |in[14:0];
    wire        round_up  = round_bit & (sticky | lsb);
    wire        is_inf_nan= (exp == 8'hFF);
    wire        is_nan    = is_inf_nan & mant_nz;
    assign out = is_nan     ? {sign, 8'hFF, 7'h40}     // canonical qNaN
               : is_inf_nan ? top                       // +/- inf
               : top + (round_up ? 16'd1 : 16'd0);      // RNE (carry into exp/inf ok)
endmodule
