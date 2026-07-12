`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// fp32_adder — IEEE-754 단정밀(FP32) 덧셈기.
//
// 동작: sum = a + b   (RNE 라운딩, IEEE-754 기본 모드)
//
// 백엔드: Berkeley HardFloat (third_party/berkeley-hardfloat/HardFloatBundle.v).
//   사용 모듈: RecFNFromFN_wrapper (x2) + AddRecFN + FNFromRecFN_wrapper
//
// HardFloat의 "recoded FP" 포맷이란?
//   IEEE-754 표준 32비트 대신 HardFloat 내부에서는 33비트 recoded 형식을 쓴다.
//   {sign[1], exp[9], frac[23]} 구조로, exp 필드를 1비트 늘려서 zero/subnormal/
//   normal/inf/NaN을 통일된 정규화 표현으로 다룰 수 있게 만든 것.
//   덧셈 같은 산술 연산은 recoded 도메인에서 수행하는 게 효율적이라
//   입력에서 한 번 변환(RecFN←FN), 출력에서 한 번 환원(FN←RecFN)한다.
//
// 파이프라인 구조:
//   레지스터단(L_ADD개)을 33비트 recoded 신호 위에 둔다. 그래서 입출력 경계의
//   recode/un-recode는 모두 조합 회로다. 입력값 → 출력은 L_ADD 사이클 지연.
//
// 포트: a,b(fp32 32b) → sum(fp32 32b, RNE). rst 미사용.
// 인스턴스: RecFNFromFN_wrapper(x2) + AddRecFN + FNFromRecFN_wrapper (fp32 번들).
//   인스턴스되는 곳: bf16_adder (bf16 덧셈의 fp32 도메인 코어).
//
// 상태: ACTIVE — bf16_adder 안에서 여전히 데이터패스에 살아 있다. (RMW 최상위
//   에서 직접 인스턴스하던 fp32 시절과 달리, 지금은 bf16_adder 를 경유.)
//
// 검증: `bash sim/run_fp32_adder.sh` → 단위 TB PASS (directed 케이스).
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md
//////////////////////////////////////////////////////////////////////////////////

module fp32_adder #(
    parameter L_ADD = 3            // 파이프라인 단수 (>=1)
)(
    input  wire        clk,
    input  wire        rst,        // 미사용 — FPGA 파워업 시 reg가 0으로 초기화됨
    input  wire [31:0] a,          // 피연산자 a (IEEE-754 FP32)
    input  wire [31:0] b,          // 피연산자 b (IEEE-754 FP32)
    output wire [31:0] sum         // a + b (IEEE-754 FP32, RNE)
);
    localparam REC_W = 33;          // recoded FP32 폭 (IEEE 32 + 1)

    // 1) IEEE-754 FP32 -> recoded FP32 변환 (조합 회로)
    //    각 피연산자에 한 번씩 변환기를 둔다.
    //    RecFNFromFN_wrapper는 RawModule이라 io_ prefix 없이 in/out 포트를 쓴다.
    wire [REC_W-1:0] recFN_a;
    wire [REC_W-1:0] recFN_b;

    RecFNFromFN_wrapper u_rec_a (
        .in  (a),
        .out (recFN_a)
    );
    RecFNFromFN_wrapper u_rec_b (
        .in  (b),
        .out (recFN_b)
    );

    // 2) 두 recoded 신호를 L_ADD단 레지스터 체인으로 지연.
    //    [0]단이 가장 최근, [L_ADD-1]단이 가장 오래된 값.
    reg [REC_W-1:0] recFN_a_dly [0:L_ADD-1];
    reg [REC_W-1:0] recFN_b_dly [0:L_ADD-1];
    integer di;
    always @(posedge clk) begin
        recFN_a_dly[0] <= recFN_a;
        recFN_b_dly[0] <= recFN_b;
        for (di = 1; di < L_ADD; di = di + 1) begin
            recFN_a_dly[di] <= recFN_a_dly[di-1];
            recFN_b_dly[di] <= recFN_b_dly[di-1];
        end
    end

    // 3) recoded 도메인에서 FP32 덧셈 수행 (조합 회로).
    //    io_subOp=0 -> 덧셈 (1이면 뺄셈), roundingMode=0 -> RNE (round-to-nearest-even).
    //    io_detectTininess=0 -> "after rounding" 기준 (IEEE-754 표준 권장 모드).
    wire [REC_W-1:0] recFN_sum;
    AddRecFN u_add (
        .io_subOp          (1'b0),
        .io_a              (recFN_a_dly[L_ADD-1]),
        .io_b              (recFN_b_dly[L_ADD-1]),
        .io_roundingMode   (3'd0),     // RNE
        .io_detectTininess (1'b0),     // IEEE 표준
        .io_out            (recFN_sum),
        .io_exceptionFlags ()           // 사용 안 함 (overflow/inexact 등 5비트 플래그)
    );

    // 4) recoded -> IEEE-754 FP32 환원 (조합 회로, 출력단에서)
    FNFromRecFN_wrapper u_out (
        .in  (recFN_sum),
        .out (sum)
    );
endmodule
