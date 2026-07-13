`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// int_to_bf16 (v3, native) — INT32 + 9비트 signed 스케일 → IEEE bfloat16 변환기.
//
// 하는 일: out_bf16 = bf16_RNE( bf16(in_int) * 2^(scale - 127) )
//   즉 (1) 정수를 8비트 유효숫자로 RNE 라운딩해 r8 을 만들고 (= golden 의
//   bf16(block_int); 일치 도메인은 아래 "라운딩 도메인 주의" 참조), (2) 거기에
//   2의 거듭제곱 스케일을 곱한 값을 bf16 으로 다시 RNE 인코딩한다. 두 번의
//   라운딩이 golden(ml_dtypes) 모델의 의미 그대로다 — 스케일 곱 자체는
//   정확하고, 결과가 subnormal 밴드에 떨어질 때만 두 번째 라운딩
//   (denormalization RNE)이 실제로 값을 바꾼다.
//
// 라운딩 도메인 주의 (full-INT32 포트의 정밀 한계 — 2026-07-13 리뷰 확정):
//   이 파일의 r8 은 int -> 8비트 유효숫자의 "단일" RNE 다 (v2 의 HardFloat
//   INToRecFN_i32_e8_s8 도 동일한 단일 라운딩 — 의미 변화 없음). 반면
//   golden(ml_dtypes)은 int -> fp32 -> bf16 의 "이중" 라운딩으로 r8 을 만든다.
//   두 방식은 int->fp32 가 무손실인 |int| < 2^24 에서 증명적으로 일치하며,
//   golden 자체가 |block_int| < 2^20 을 계약으로 둔다 (MXP_Tools gemm.py) —
//   실현 가능한 K-블록 합(int8*int8*32 ~ 2^19)은 전부 이 안이다.
//   |int| >= 2^25 의 극소수 tie 케이스(~1/200k)에서는 golden 의 이중 라운딩이
//   1 ULP 위로 어긋날 수 있고, 그 경우 correctly-rounded 값은 이쪽(단일)이다.
//   oracle 의 full-range directed 벡터는 LZC32/round-carry 경로의 witness 이지
//   "전 INT32 에서 golden 과 일치"의 게이트가 아니다. 이 차이를 없애겠다고
//   RTL 을 이중 라운딩으로 "고치지" 말 것 (정확도 퇴행).
//
// 왜 native 인가 (2026-07-13, A6):
//   v2 는 HardFloat 조합 (INToRecFN_i32_e8_s8 → recoded 지수 add → FNFromRecFN
//   → fp32_to_bf16_rne) 이었다 — 정확했지만 기계 생성 netlist 라 읽을 수 없고
//   recoded 포맷 우회가 불필요하게 크다. v3 는 같은 수치 계약을 손으로 다시
//   쓴 것: LZC + 시프트 + 8비트 RNE 두 번이 전부다. (v2 는 git history 에;
//   v2 가 고친 v1 의 "denorm TRUNCATE" 결함 이력은 spec §6.2 참조 — v3 의
//   subnormal 라운더가 그 결함 클래스의 witness 벡터들로 게이트된다.)
//
// ── 알고리즘 (두 개의 조합 절반) ──────────────────────────────────────────
//
//   [F] 앞절반 — INT32 → r8 (첫 번째 RNE):
//     1. 절댓값: mag = |in_int| 를 u32 로 (-2^31 → 0x80000000 그대로 = exact).
//     2. LZC32 → lz. norm = mag << lz 로 MSB 를 bit31 에 정렬.
//     3. sig8 = norm[31:24] (hidden 포함 8b), G = norm[23], 나머지는 sticky.
//        RNE: up = G & (LSB | sticky). 캐리(0xFF+1) 는 sig=0x80 에 지수+1 로
//        재정규화. 비편향 지수 E = (31 - lz) + carry, r8 = ±sig8 · 2^(E-7).
//     * in_int == 0 은 플래그로 따로 흘린다 — 0 은 지수/스케일 조작 없이
//       +0 그대로 통과해야 한다 (일반 경로에 태우면 스케일이 큰 경우
//       sig8=0 인 채 inf 인코딩으로 새는 함정이 있다).
//
//   [B] 뒷절반 — 스케일 반영 + bf16 인코딩 (필요시 두 번째 RNE):
//     1. 결과 biased 지수 e_tot = E + signed(scale). 10비트 signed 로 전 구간
//        wrap 없음: E ∈ [0,32], scale ∈ [-256,255] → e_tot ∈ [-256,287].
//        (v2 가 flush 로 막았던 "9비트 지수 wrap" 잠복 버그가 폭 자체로 소멸.)
//     2. e_tot >= 255 → ±inf 포화 (golden 의 fp32 overflow → inf 와 동일).
//        e_tot ∈ [1,254] → normal {sign, e_tot, sig8[6:0]} — 스케일 곱은
//        2의 거듭제곱이라 이 구간에선 재라운딩이 없다 (exact).
//     3. e_tot <= 0 → subnormal 밴드: s = 1 - e_tot 만큼 sig8 을 우측 시프트
//        하며 GRS 로 RNE (bf16_adder 와 동일한 11b 그리드/22b funnel 재사용
//        패턴). 여기가 두 번째 라운딩이며:
//          - f8 = 0x80 으로 올라서면 min normal 로 자연 승격,
//          - f8 = 0 이면 부호 있는 0 — 깊은 언더플로 flush 가 별도 임계값
//            비교 없이 라운더에서 저절로 나온다 (s >= 9 면 G=0 → 항상 0;
//            정확히 2^-134 는 tie → +0(짝수), 1.5·2^-134 는 min subnormal 로
//            올림 — oracle 의 boundary 벡터들이 이 밴드를 조밀하게 고정).
//
// ── 파이프라인 (전체 지연 = L_CONV 단, >= 1) ──────────────────────────────
//   중간 절단점([F] 끝, 25b 페이로드 {sign, zero, sig8, E, scale}) 에 1단,
//   나머지 L_CONV-1 단은 출력 bf16 뒤에 체인. scale 을 페이로드에 함께 태워
//   [B] 가 항상 같은 벡터의 스케일을 보게 한다 (skew 방지 — 외부에서 scale 이
//   in_int 와 같은 사이클에 도착한다는 RMW 계약과 맞물림).
//   RMW 는 L_CONV=1 로 인스턴스 → [F] 가 S1, [B] 가 S2 스테이지가 된다
//   (Phase 2c 재배치 계약 유지; 남은 1단은 bf16_adder 의 L_IN 이 담당).
//
// 포트: in_int(INT32) + scale(s9) → out_bf16(bf16 16b). rst 미사용.
// 인스턴스: 없음 (leaf, HardFloat 미사용). 인스턴스되는 곳: RMW.
// 상태: ACTIVE (v3 native — RMW 변환단 본선). v1/v2 는 git history.
//
// 검증: `bash sim/run_int_to_bf16.sh` → 기대 "ALL 32360 TESTS PASSED"
//   (oracle sim/bf16_vectors.py 와 bit-exact; 음수 scale 언더플로 + subnormal
//    boundary 조밀 벡터 + full-range INT32 directed 포함. 듀얼 DUT 로
//    L_CONV=2/1 동시 검증.)
//
// Spec: docs/superpowers/plans/2026-07-13-bf16-native-datapath.md (§3)
//////////////////////////////////////////////////////////////////////////////////

module int_to_bf16 #(
    parameter L_CONV = 2          // 파이프라인 단수 (>=1)
)(
    input  wire        clk,
    input  wire        rst,        // 미사용 (인터페이스 호환용)
    input  wire [31:0] in_int,     // 부호 있는 INT32 입력
    input  wire [8:0]  scale,      // 9비트 signed 스케일
    output wire [15:0] out_bf16    // IEEE bfloat16 결과
);

    // 32비트 벡터의 leading-zero count (0..31). 상행 루프의 "마지막 대입 승리"
    // 가 최상위 set 비트를 찾는 priority encoder 로 합성된다. x==0 은 호출측
    // 에서 zero 플래그로 우회하므로 반환값이 쓰이지 않는다.
    function [4:0] lzc32;
        input [31:0] x;
        integer k;
        begin
            lzc32 = 5'd31;
            for (k = 0; k <= 31; k = k + 1)
                if (x[k]) lzc32 = 5'd31 - k;
        end
    endfunction

    // ── [F] 앞절반: INT32 → r8 = ±sig8 · 2^(E-7)  (조합) ──────────────────
    wire        sign    = in_int[31];
    wire [31:0] mag     = sign ? (~in_int + 32'd1) : in_int;  // |int|, -2^31 안전
    wire        is_zero = (in_int == 32'd0);

    wire [4:0]  lz   = lzc32(mag);
    wire [31:0] norm = mag << lz;              // MSB 를 bit31 로 (mag != 0 전제)

    // 8비트 유효숫자 RNE: L = sig8_t[0], G = norm[23], sticky = 그 아래 전부.
    wire [7:0]  sig8_t = norm[31:24];
    wire        g_bit  = norm[23];
    wire        stky   = |norm[22:0];
    wire        up_f   = g_bit & (sig8_t[0] | stky);
    wire [8:0]  sig9_f = {1'b0, sig8_t} + {8'd0, up_f};
    wire        rc_f   = sig9_f[8];                    // 0xFF+1 → 재정규화
    wire [7:0]  sig8_f = rc_f ? 8'h80 : sig9_f[7:0];
    wire [5:0]  e_unb  = {1'b0, 5'd31 - lz} + {5'd0, rc_f};   // E ∈ [0, 32]

    // ── 중간 절단점: 1단 레지스터 (페이로드 25b — scale 도 같이 지연) ──────
    localparam PW = 25;
    wire [PW-1:0] f_w = {sign, is_zero, sig8_f, e_unb, scale};
    reg  [PW-1:0] f_q;
    always @(posedge clk)
        f_q <= f_w;

    wire        q_sign  = f_q[24];
    wire        q_zero  = f_q[23];
    wire [7:0]  q_sig8  = f_q[22:15];
    wire [5:0]  q_e     = f_q[14:9];
    wire [8:0]  q_scale = f_q[8:0];

    // ── [B] 뒷절반: 스케일 반영 + 인코딩  (조합) ───────────────────────────

    // 결과 biased 지수. biased(r8) = E + 127 에 (scale - 127) 을 더하면
    // 127 이 상쇄되어 e_tot = E + scale. 10b signed 로 전 구간 표현.
    wire signed [9:0] e_tot = $signed({4'd0, q_e}) + $signed(q_scale);

    // subnormal 라운더: s = 1 - e_tot 만큼 {sig8, G,R,S} 그리드에서 우측
    // 시프트 (bf16_adder 와 동일한 22b funnel 패턴 — 떨어진 비트 전부 S 로
    // OR). f7 필드의 정수 그리드에서 RNE. s > 11 은 11 로 클램프해도 안전
    // (그때 sig8 전체가 sticky → up=0 → ±0 = 깊은 언더플로 flush).
    wire signed [9:0] s_amt = 10'sd1 - e_tot;                 // e_tot<=0 에서 >= 1
    wire [3:0]  sc4   = (s_amt > 10'sd11) ? 4'd11 : s_amt[3:0];
    wire [10:0] body  = {q_sig8, 3'b000};
    wire [21:0] wide  = {body, 11'b0} >> sc4;
    wire [10:0] den   = {wide[21:12], wide[11] | (|wide[10:0])};
    wire        up_b  = den[2] & (den[3] | den[1] | den[0]);  // RNE
    wire [7:0]  f8    = den[10:3] + {7'd0, up_b};   // <= 0x80 (min-normal 승격)

    // 인코딩. zero 플래그가 최우선 (헤더 [F] 의 함정 참조).
    wire [15:0] bf16_w =
          q_zero               ? 16'h0000
        : (e_tot >= 10'sd255)  ? {q_sign, 8'hFF, 7'd0}          // inf 포화
        : (e_tot >= 10'sd1)    ? {q_sign, e_tot[7:0], q_sig8[6:0]}  // normal, exact
        : (f8[7])              ? {q_sign, 8'd1, 7'd0}           // 올림 → min normal
        :                        {q_sign, 8'd0, f8[6:0]};       // subnormal / ±0 flush

    // ── 출력단: 나머지 L_CONV-1 단 레지스터 체인 ───────────────────────────
    generate
        if (L_CONV > 1) begin : g_out_reg
            reg [15:0] o_q [0:L_CONV-2];
            integer oi;
            always @(posedge clk) begin
                o_q[0] <= bf16_w;
                for (oi = 1; oi < L_CONV-1; oi = oi + 1)
                    o_q[oi] <= o_q[oi-1];
            end
            assign out_bf16 = o_q[L_CONV-2];
        end else begin : g_out_comb
            assign out_bf16 = bf16_w;
        end
    endgenerate
endmodule
