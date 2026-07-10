`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// int_to_bf16 (v2) — INT32 + 9비트 signed 스케일 → IEEE bfloat16 변환기.
//
// 동작: out_bf16 = bf16_RNE( bf16(in_int) * 2^(scale - 127) )
//   1) INToRecFN_i32_e8_s8 : in_int 을 8-bit significand 로 RNE 라운딩
//      (= golden 의 r8 = bf16(block_int) 와 동일한 첫 라운딩).
//   2) recoded bf16 -> recoded fp32 EXACT widen: {sign, exp9, frac7, 16'b0}.
//      bf16/fp32 는 expWidth=8 공유 -> 9비트 recoded 지수 인코딩이 동일,
//      가수만 0-패딩 (int 입력이라 zero/normal 만 나옴 -> 무손실).
//   3) 지수 필드에 (scale-127) 더하기 = 곱하기 2^(scale-127) (int_to_fp32 트릭).
//   4) FNFromRecFN_wrapper (fp32) : recoded -> IEEE fp32.
//      subnormal 밴드 (new_exp10 in [122,129]) 의 denorm shift 는 0..7 로
//      0-패딩 16비트만 떨어뜨리므로 EXACT (r8*2^E 의 정확한 fp32 표현).
//   5) fp32_to_bf16_rne : 단 한 번의 RNE (subnormal 포함, 2a 근사-전수 검증 블록).
//
// 깊은 언더플로 flush: new_exp10 < 122 (즉 |값| < 2^-134 = bf16 최소 subnormal
//   의 절반) 이면 recoded ±0 으로 직접 flush — RNE 상 정확히 ±0 인 영역.
//   이 flush 가 (i) 음수 new_exp10 의 9비트 절단 wrap (inf/NaN 발산, int_to_fp32
//   에서 물려받은 잠복 버그) 과 (ii) fp32 denorm shifter 의 깊은-지수 aliasing
//   위험을 함께 차단한다. 비교는 10-bit signed 전체값으로 하므로 wrap 없음.
//
// v1 (INToRecFN_s8 -> exp-add -> FNFromRecFN_bf16_wrapper) 폐기 이유:
//   FNFromRecFN 의 denormalization 은 TRUNCATE (round/sticky 없음,
//   HardFloatBundle_bf16.v:179-181) -> subnormal 결과가 golden(RNE) 대비
//   1 ULP 낮게 나옴 (리뷰 라운드1 Critical, 어떤 flush 임계로도 교정 불가).
//   FNFromRecFN_bf16_wrapper 는 이 파일에서 더 이상 인스턴스하지 않음
//   (번들에는 보존).
//
// 파이프라인: L_CONV단 (>=1). 레지스터는 스케일 보정 끝난 recoded fp32 (33b)
// 위에 위치 — 입력 변환과 출력 환원(4,5)은 모두 조합 회로.
//
// Spec: docs/superpowers/specs/2026-07-08-rmw-bf16-design.md (§6.1 carry-over #1)
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
    localparam REC_W = 33;          // recoded fp32 폭 (IEEE 32 + 1)

    // 1) INT32 -> recoded bf16 변환 (조합). 8-bit significand RNE = golden r8.
    wire [16:0] recFN_bf16;
    INToRecFN_i32_e8_s8 u_i2f (
        .io_signedIn       (1'b1),
        .io_in             (in_int),
        .io_roundingMode   (3'd0),         // RNE
        .io_detectTininess (1'b0),
        .io_out            (recFN_bf16),
        .io_exceptionFlags ()
    );

    // 2) recoded bf16 -> recoded fp32 widen (exact).
    wire        sign_bit  = recFN_bf16[16];
    wire [8:0]  exp_field = recFN_bf16[15:7];
    wire [22:0] sig_field = {recFN_bf16[6:0], 16'b0};

    // 3) 지수 더하기. 10비트 signed 는 전 도메인에서 wrap 없음: 0 아닌 int32 의
    //    recoded 지수는 [256, 287], scale 은 [-256, 255] -> new_exp10 in
    //    [-127, 415]. 위쪽 clamp 불필요 — 최대 415 는 inf 인코딩 밴드 [384, 447]
    //    안 (정상 saturate, golden 도 inf), NaN 밴드 (>=448) 도달 불가.
    wire signed [9:0] exp_ext   = $signed({1'b0, exp_field});
    wire signed [9:0] scale_ext = $signed(scale);
    wire signed [9:0] new_exp10 = exp_ext + scale_ext - 10'sd127;
    wire [8:0]        new_exp   = new_exp10[8:0];

    // 깊은 언더플로 flush (헤더 설명 참조): recoded 지수 123 = bf16 최소
    // subnormal 2^-133, 122 밴드 [2^-134, 2^-133) 는 RNE 반올림 대상이라 통과,
    // 그 아래 (<122) 는 정확히 ±0.
    wire underflow_z = (new_exp10 < 10'sd122);

    // Zero passthrough: in_int=0 이면 recoded zero 인코딩을 그대로 widen 해서
    // 통과 (지수 조작 금지 — 조작하면 zero 인코딩이 깨짐).
    wire int_is_zero = (in_int == 32'h00000000);
    wire [REC_W-1:0] recFN_scaled =
          int_is_zero ? {sign_bit, exp_field, sig_field}
        : underflow_z ? {sign_bit, 9'd0, 23'd0}
                      : {sign_bit, new_exp, sig_field};

    // 4) L_CONV단 파이프라인 레지스터 체인 (스케일 보정 끝난 recoded fp32 위에)
    reg [REC_W-1:0] recFN_dly [0:L_CONV-1];
    integer di;
    always @(posedge clk) begin
        recFN_dly[0] <= recFN_scaled;
        for (di = 1; di < L_CONV; di = di + 1)
            recFN_dly[di] <= recFN_dly[di-1];
    end

    // 5) recoded fp32 -> IEEE fp32 (exact) -> 6) bf16 RNE narrow (조합, 출력단)
    wire [31:0] fp32_exact;
    FNFromRecFN_wrapper u_out (
        .in  (recFN_dly[L_CONV-1]),
        .out (fp32_exact)
    );
    fp32_to_bf16_rne u_narrow (
        .in  (fp32_exact),
        .out (out_bf16)
    );
endmodule
