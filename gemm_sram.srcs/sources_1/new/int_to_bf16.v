`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// int_to_bf16 — INT32 + 9비트 signed 스케일 → IEEE bfloat16 변환기.
//
// 동작: out_bf16 = in_int * 2^(scale - 127)    (RNE 라운딩)
//   - in_int  : 부호 있는 32비트 정수 (GEMM accumulator의 부분합)
//   - scale   : 9비트 signed 결합 스케일. Accumulator_Col에서 만드는
//               (act_scale + weight_scale - 127 - IMPLICIT_total) 값.
//               중심값 127 (= IEEE-754 지수 bias, bf16/fp32 공통) 기준 +면
//               오른쪽, -면 왼쪽으로 지수를 옮긴다.
//
// 백엔드: Berkeley HardFloat (third_party/berkeley-hardfloat/HardFloatBundle_bf16.v).
//   사용 모듈: INToRecFN_i32_e8_s8 (정수→recoded bf16) + FNFromRecFN_bf16_wrapper.
//   INToRecFN_i32_e8_s8 은 io_detectTininess 입력이 추가로 있다 → 1'b0 로 고정.
//
// 구현 트릭: "곱하기 2^k"는 FP에서 지수 더하기로 끝낸다.
//   in_int을 그냥 INToRecFN으로 FP로 바꾼 뒤, recoded 형식의 9비트 지수
//   필드에 (scale-127)을 직접 더해버린다. 별도의 곱셈기 불필요.
//   bf16 과 fp32 는 expWidth=8 을 공유 → recoded 지수 필드가 둘 다 9비트라
//   이 지수-더하기 산술은 폭에 무관 (width-invariant). 가수 폭만 23→7 로 바뀜.
//
// 파이프라인: L_CONV단 (>=1). 레지스터는 지수 보정까지 끝난 recoded 신호 위에
// 위치 → 입력 변환과 출력 환원은 모두 조합 회로.
//
// Spec: docs/superpowers/specs/2026-07-08-rmw-bf16-design.md
//////////////////////////////////////////////////////////////////////////////////

module int_to_bf16 #(
    parameter L_CONV = 2          // 파이프라인 단수 (>=1)
)(
    input  wire        clk,
    input  wire        rst,        // 미사용 — FPGA 파워업 시 reg가 0으로 초기화됨
    input  wire [31:0] in_int,     // 부호 있는 INT32 입력
    input  wire [8:0]  scale,      // 9비트 signed 스케일
    output wire [15:0] out_bf16    // IEEE bfloat16 결과
);
    localparam REC_W = 17;          // recoded bf16 폭 (IEEE 16 + 1)

    // 1) INT32 -> recoded bf16 변환 (조합)
    //    io_signedIn=1 -> 부호 있는 정수로 해석.
    //    io_detectTininess=1'b0 -> tininess 검출 비활성 (RNE 결과엔 영향 없음).
    //    이 결과는 아직 스케일 적용 전 (즉 in_int 값 그대로의 bf16 표현).
    wire [REC_W-1:0] recFN_int;
    INToRecFN_i32_e8_s8 u_i2f (
        .io_signedIn       (1'b1),
        .io_in             (in_int),
        .io_roundingMode   (3'd0),         // RNE
        .io_detectTininess (1'b0),
        .io_out            (recFN_int),
        .io_exceptionFlags ()
    );

    // 2) Recoded FP의 지수 필드에 (scale-127) 더해서 곱하기 2^(scale-127) 효과.
    //
    //    Recoded bf16 비트 레이아웃:
    //      [16]    : 부호 비트
    //      [15:7]  : 9비트 지수 (HardFloat 인코딩)
    //      [6:0]   : 가수
    //
    //    HardFloat의 normal-range 인코딩 안에서는 지수 필드를 직접 더해도
    //    값이 정확히 2^k 배 이동한다. 결과가 표현 범위를 벗어나면 zero/inf
    //    인코딩 영역으로 자연스럽게 빠져서 downstream이 알아서 처리한다
    //    (별도 saturate 회로 불필요).
    //
    //    127을 빼는 이유: scale은 (act+weight-127) 형태로 이미 bias 빠진 값
    //    같지만, RMW 입력 시점의 scale 정의는 "127이 중심"으로 약속돼 있다
    //    (RMW 헤더 참고). 그래서 여기서 한 번 더 빼야 진짜 지수 보정값이 됨.
    wire             sign_bit  = recFN_int[REC_W-1];
    wire [8:0]       exp_field = recFN_int[REC_W-2 -: 9];   // [15:7]
    wire [6:0]       sig_field = recFN_int[6:0];

    // 오버플로우 방지를 위해 10비트로 확장한 뒤 계산
    wire signed [9:0] exp_ext   = $signed({1'b0, exp_field});
    wire signed [9:0] scale_ext = $signed(scale);
    wire signed [9:0] new_exp10 = exp_ext + scale_ext - 10'sd127;
    wire [8:0]        new_exp   = new_exp10[8:0];           // 9비트로 잘라서 사용

    // Zero passthrough: when in_int=0, the recoded representation has exp=0
    // (special "zero" encoding). Adding (scale-127) to exp would corrupt this
    // into a non-zero normal/subnormal/overflow value. Detect in_int=0 and
    // skip the exp manipulation entirely.
    wire int_is_zero = (in_int == 32'h00000000);
    wire [REC_W-1:0] recFN_scaled = int_is_zero ? recFN_int
                                                 : {sign_bit, new_exp, sig_field};

    // 3) L_CONV단 파이프라인 레지스터 체인 (스케일 보정 끝난 recoded 신호 위에)
    reg [REC_W-1:0] recFN_dly [0:L_CONV-1];
    integer di;
    always @(posedge clk) begin
        recFN_dly[0] <= recFN_scaled;
        for (di = 1; di < L_CONV; di = di + 1)
            recFN_dly[di] <= recFN_dly[di-1];
    end

    // 4) recoded -> IEEE bfloat16 환원 (조합, 출력단에서)
    FNFromRecFN_bf16_wrapper u_out (
        .in  (recFN_dly[L_CONV-1]),
        .out (out_bf16)
    );
endmodule
