`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// bf16_adder - IEEE bfloat16 덧셈 (RNE). fp32 도메인에서 계산 후 narrow.
//
//   bf16 a,b -> {a,16'h0}/{b,16'h0} 로 fp32 확장 (bf16이 fp32 상위16비트라 exact)
//   -> 기존 검증된 fp32_adder (HardFloat AddRecFN, e8/s24) -> sum_fp32
//   -> fp32_to_bf16_rne -> sum_bf16.
// round_bf16(round_fp32(a+b)) = round_bf16(a+b) = true bf16 add (double-rounding
// 안전: 24 >= 2*8+2). ml_dtypes bf16 add 와 bit 일치.
//
// 왜 fp32 도메인인가: HardFloat AddRecFN(8,8) 은 elaborate 안 됨 (bf16 폭 미지원).
//   그래서 exact widen → 검증된 fp32 가산 → RNE narrow 우회를 쓴다.
//
// 파이프라인 단 파라미터 (Phase 2c 재배치, 전체 지연 = L_IN + L_ADD + L_SUM + L_OUT):
//   L_IN  : bf16 입력단 (a/b 공통 지연). 기본 0.
//   L_ADD : fp32_adder 피연산자단 (recoded). 기본 3.
//   L_SUM : fp32_adder 합(recoded)단. 기본 0.
//   L_OUT : bf16 출력단 (narrow 후). 기본 0.
// L_ADD 의미 변경 주의: 종전에는 "전체 가산 지연" 이었으나 이제는 "fp32_adder
//   피연산자단 단수" 다. 기본값 (L_IN=L_SUM=L_OUT=0) 에서는 전체 지연 = L_ADD 로
//   수치·타이밍 모두 종전과 동일 (기존 인스턴스/TB 주석 그대로 유효).
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
    parameter L_IN  = 0,   // bf16 입력단 레지스터 (a/b 공통 지연, >=0)
    parameter L_ADD = 3,   // fp32_adder 피연산자단 (recoded, >=1) — 종전 "전체 지연" 의미에서 변경
    parameter L_SUM = 0,   // fp32_adder 합(recoded)단 (>=0)
    parameter L_OUT = 0    // bf16 출력단 (narrow 후, >=0)
)(                          // 전체 지연 = L_IN + L_ADD + L_SUM + L_OUT
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] sum
);
    // 0) L_IN단 bf16 입력 레지스터. a/b 를 같은 체인 길이로 지연하므로
    //    두 피연산자의 상호 정렬(pairing)은 불변.
    wire [15:0] a_d;
    wire [15:0] b_d;
    generate
        if (L_IN > 0) begin : g_in_reg
            reg [15:0] a_q [0:L_IN-1];
            reg [15:0] b_q [0:L_IN-1];
            integer ii;
            always @(posedge clk) begin
                a_q[0] <= a;
                b_q[0] <= b;
                for (ii = 1; ii < L_IN; ii = ii + 1) begin
                    a_q[ii] <= a_q[ii-1];
                    b_q[ii] <= b_q[ii-1];
                end
            end
            assign a_d = a_q[L_IN-1];
            assign b_d = b_q[L_IN-1];
        end else begin : g_in_comb
            assign a_d = a;
            assign b_d = b;
        end
    endgenerate

    // 1) exact widen -> 2) fp32 도메인 가산 (L_ADD 피연산자단 + L_SUM 합단)
    wire [31:0] a_fp32 = {a_d, 16'h0000};
    wire [31:0] b_fp32 = {b_d, 16'h0000};
    wire [31:0] sum_fp32;
    fp32_adder #(.L_ADD(L_ADD), .L_SUM(L_SUM)) u_add (
        .clk(clk), .rst(rst), .a(a_fp32), .b(b_fp32), .sum(sum_fp32)
    );

    // 3) 단일 RNE narrow (조합)
    wire [15:0] sum_bf16;
    fp32_to_bf16_rne u_narrow (.in(sum_fp32), .out(sum_bf16));

    // 4) L_OUT단 bf16 출력 레지스터 (RMW 구성에서 out_RMW 를 레지스터 출력으로
    //    만들어 SRAM D 입력까지의 조합 꼬리를 제거한다)
    generate
        if (L_OUT > 0) begin : g_out_reg
            reg [15:0] sum_q [0:L_OUT-1];
            integer oi;
            always @(posedge clk) begin
                sum_q[0] <= sum_bf16;
                for (oi = 1; oi < L_OUT; oi = oi + 1)
                    sum_q[oi] <= sum_q[oi-1];
            end
            assign sum = sum_q[L_OUT-1];
        end else begin : g_out_comb
            assign sum = sum_bf16;
        end
    endgenerate
endmodule
