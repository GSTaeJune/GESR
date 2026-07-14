`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// bf16_adder - IEEE bfloat16 덧셈기 (RNE), native 구현.
//
// 하는 일: sum = bf16_RNE(a + b). bf16 = {sign 1b, exp 8b (bias 127), frac 7b},
//   subnormal/±0/±inf/NaN 전부 IEEE 규칙대로. ml_dtypes 의 bfloat16 덧셈과
//   bit-exact (NaN 은 부호/payload 만 미규정 — 아래 "NaN 정책" 참조).
//
// 왜 native 인가 (2026-07-13, A6):
//   Phase 2a~2c 구현은 "fp32 도메인 우회" 였다 — {bf16,16'h0} 로 fp32 확장 후
//   HardFloat AddRecFN(24b significand) 으로 더하고 RNE narrow. 정확성 증명은
//   쉬웠지만(24 >= 2*8+2 double-rounding 안전) 8비트 유효숫자 연산에 24비트
//   하드웨어를 쓰는 꼴이라, 정직 타이밍 측정에서 AddRecFN 한 스테이지가
//   est. 12.4ns 로 병목이었다. 이 파일은 같은 수치 계약을 8비트 폭 그대로,
//   단일 라운딩으로 다시 구현한 것. 모든 내부 데이터패스가 11~12비트라
//   스테이지당 로직이 사소해진다. (구 구현은 git history / 태그 fp32-rmw-final.)
//
// ── 알고리즘 (교과서 FP 가산기, 3개 조합 블록) ─────────────────────────────
//
//   표기: 각 피연산자를 sig8 = {hidden, frac7} (8b 유효숫자), ee = 유효 biased
//   지수로 푼다. subnormal(exp field 0) 은 hidden=0, ee=1 로 취급 — subnormal 이
//   min-normal 과 같은 binade(2^-126) 에 살기 때문에 이 규약 하나로 정렬이
//   normal/subnormal 구분 없이 균일해진다.
//
//   [A] 정렬/가감산:
//     1. |a| vs |b| 를 {exp,frac} 15b unsigned 로 비교해 big/small 스왑.
//        (bf16 인코딩은 크기순 = 비트순 이라 이 비교가 곧 크기 비교다.)
//     2. 11비트 연산 그리드에 올린다:  [10:3]=sig8, [2]=G, [1]=R, [0]=S.
//        bit10 의 가중치 = 2^(ee_big-127) (즉 1.0 자리). small 을 d = ee_big -
//        ee_small 만큼 우측 시프트하고, R 아래로 떨어진 비트는 전부 S(sticky)
//        한 비트로 OR 한다 (d>=11 이면 small 전체가 sticky 로 붕괴).
//     3. 부호가 같으면 12b 가산, 다르면 12b 감산 (스왑 덕에 결과 >= 0).
//        sticky 를 감산 "전에" bit0 에 OR 해 두는 것이 고전 GRS 기법의 핵심:
//        진짜 값은 (계산값-1, 계산값) 구간에 있고 계산값의 bit0=1 이
//        "부정확" 마커로 살아남아, 이후 라운딩이 올바른 쪽을 고른다.
//   [B] 정규화:
//     - 가산 캐리(m12[11]): 우측 1 시프트, 밀려나는 두 비트는 S 로 병합, 지수+1.
//     - 그 외: 11b LZC 로 좌측 시프트. 단 지수가 biased 1 밑으로 못 내려가게
//       클램프 (sh = min(lzc, ee_big-1)) — 남으면 그대로 subnormal 결과가 된다.
//       불변식(고전): 2비트 이상의 대량 상쇄는 d<=1 에서만 생기고 그때 정렬은
//       비트를 하나도 안 버렸으므로(G=R=S=0, exact) sticky 가 좌측 시프트로
//       숫자 자리에 끌려 올라오는 일이 없다. d>=2 면 상쇄는 최대 1비트.
//   [C] 라운드/패킹:
//     - RNE: up = G & (L | R | S). 유효숫자 캐리(0xFF+1) 는 sig=0x80, 지수+1.
//     - e >= 255 → ±inf (캐리로 255 에 닿는 순간 진짜 값이 이미 overflow 문턱
//       (2-2^-8)*2^127 이상이라 inf 가 올바른 RNE 결과다).
//     - sig8 MSB=0 → subnormal 인코딩 (exp field 0; 정규화 클램프에 의해 이때
//       지수는 항상 biased 1). sig8=0 → ±0 (이 가산기의 0 결과는 전부 exact —
//       subnormal 그리드는 ±에 닫혀 있고 normal 상쇄의 최소값도 2^-133 이라
//       "0 으로 반올림되는 미세값" 케이스는 존재하지 않는다).
//     - 0 의 부호 (IEEE, RNE): 실질 감산의 exact 0 → +0. 실질 가산의 0 은
//       두 입력이 모두 0 일 때뿐 → big 의 부호 (+0+ +0=+0, -0+ -0=-0).
//
// NaN 정책: 입력 NaN 또는 inf+(-inf) → canonical quiet NaN 0x7FC0.
//   IEEE 는 이 NaN 의 부호를 규정하지 않는다 (ml_dtypes/x86 은 -qNaN). TB 는
//   양쪽 다 NaN 이면 부호/payload 무시 비교.
//
// ── 파이프라인 파라미터 (전체 지연 = L_IN + L_ADD + L_SUM + L_OUT) ─────────
//   각 파라미터 = 해당 절단점의 시프트 레지스터 깊이 (0 = 조합 통과):
//     L_IN  : 입력 a/b (16b x2). a/b 를 같은 깊이로 지연하므로 pairing 불변.
//     L_ADD : 블록 [A] 뒤 (가감산 결과 25b 버스). 기본 3.
//     L_SUM : 블록 [B] 뒤 (정규화 결과 25b 버스).
//     L_OUT : 블록 [C] 뒤 (최종 bf16 16b). RMW 구성에서 out_RMW 를 레지스터
//             출력으로 만들어 SRAM D 까지의 조합 꼬리를 제거한다.
//   기본값 (0,3,0,0) 은 종전과 동일한 전체 지연 3. RMW 는 (1,1,1,1) 로
//   인스턴스 — 블록당 정확히 1단 (Phase 2c 재배치 계약 유지).
//   레지스터 위치는 수치에 영향 없음 (기능은 순수 조합 [A]-[B]-[C] 합성).
//
// 포트: a,b(bf16 16b) → sum(bf16 16b). rst 미사용 (레지스터는 파이프라인
//   워밍업으로 채워짐 — TB 들이 첫 유효 출력 전 구간을 버린다).
// 인스턴스: 없음 (leaf, HardFloat 미사용). 인스턴스되는 곳: RMW.
// 상태: ACTIVE (RMW 덧셈단 본선, native v3).
//
// 검증: `bash sim/run_bf16_adder.sh` → 기대 "ALL 200025 TESTS PASSED"
//   (ml_dtypes 크로스체크 + directed edges: 유한 overflow→inf, min-normal −
//    min-subnormal 경계 상쇄, d=24 sticky 붕괴 ± (round-carry 포함), x+(-x),
//    -0+-0, RNE tie, one-sided inf±finite. 트리플 DUT 로 기본단(0,3,0,0)/
//    RMW 재배치단(1,1,1,1)/전조합(0,0,0,0) 동시 검증.)
//////////////////////////////////////////////////////////////////////////////////

module bf16_adder #(
    parameter L_IN  = 0,   // 입력단 레지스터 깊이 (>=0)
    parameter L_ADD = 3,   // 정렬/가감산 뒤 레지스터 깊이 (>=0). 기본 3 = 종전 지연
    parameter L_SUM = 0,   // 정규화 뒤 레지스터 깊이 (>=0)
    parameter L_OUT = 0    // 최종 bf16 뒤 레지스터 깊이 (>=0)
)(                          // 전체 지연 = L_IN + L_ADD + L_SUM + L_OUT
    input  wire        clk,
    input  wire        rst,   // 미사용 (인터페이스 호환용)
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] sum
);

    // 11비트 벡터의 leading-zero count (0..11). 상행 루프의 "마지막 대입 승리"
    // 가 최상위 set 비트를 찾는 priority encoder 로 합성된다.
    function [3:0] lzc11;
        input [10:0] x;
        integer k;
        begin
            lzc11 = 4'd11;                       // x == 0
            for (k = 0; k <= 10; k = k + 1)
                if (x[k]) lzc11 = 4'd10 - k;
        end
    endfunction

    // ── 0) L_IN 단: 입력 레지스터 ──────────────────────────────────────────
    wire [15:0] a_d, b_d;
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

    // ── [A] 정렬 / 가감산 (조합) ───────────────────────────────────────────

    // 필드 분해 + 특수값 판정.
    wire        sa    = a_d[15];
    wire        sb    = b_d[15];
    wire [7:0]  ea    = a_d[14:7];
    wire [7:0]  eb    = b_d[14:7];
    wire [6:0]  fa    = a_d[6:0];
    wire [6:0]  fb    = b_d[6:0];
    wire        a_nan = (ea == 8'hFF) &  (|fa);
    wire        b_nan = (eb == 8'hFF) &  (|fb);
    wire        a_inf = (ea == 8'hFF) & ~(|fa);
    wire        b_inf = (eb == 8'hFF) & ~(|fb);

    wire        eff_sub = sa ^ sb;             // 실질 감산 여부

    // 특수값 결과 플래그 — 일반 데이터패스와 나란히 파이프라인을 타고 내려가
    // [C] 의 최종 mux 에서 우선 적용된다 (inf 피연산자가 일반 경로에 만드는
    // 쓰레기 값은 그 mux 가 덮는다).
    wire        r_nan   = a_nan | b_nan | (a_inf & b_inf & eff_sub); // inf-inf 포함
    wire        r_inf   = (a_inf | b_inf) & ~r_nan;
    wire        inf_sgn = a_inf ? sa : sb;

    // 크기 비교 & 스왑. {exp,frac} 15b unsigned 비교 = |값| 비교 (bf16 성질).
    // 크기가 같으면 a 를 big 으로 (어느 쪽이든 결과 동일 — 같은 크기는 exact).
    wire        a_ge     = (a_d[14:0] >= b_d[14:0]);
    wire        sign_big = a_ge ? sa : sb;
    wire [7:0]  e_big    = a_ge ? ea : eb;
    wire [6:0]  f_big    = a_ge ? fa : fb;
    wire [7:0]  e_sml    = a_ge ? eb : ea;
    wire [6:0]  f_sml    = a_ge ? fb : fa;

    // hidden bit 복원 + 유효 지수 (subnormal: hidden=0, ee=1 — 헤더 참조).
    wire [7:0]  sig_big  = {(e_big != 8'd0), f_big};
    wire [7:0]  sig_sml  = {(e_sml != 8'd0), f_sml};
    wire [7:0]  ee_big   = (e_big == 8'd0) ? 8'd1 : e_big;
    wire [7:0]  ee_sml   = (e_sml == 8'd0) ? 8'd1 : e_sml;

    // 정렬: small 을 d 만큼 우측 시프트, 떨어진 비트는 전부 S(bit0) 로 OR.
    // 22b funnel — {body,11'b0} 을 min(d,11) 시프트하면 상위 11b 가 정렬값,
    // 하위 11b 가 "떨어진 비트" 로 그대로 남아 sticky reduction 이 한 줄이 된다.
    // d>=11 이면 small 의 MSB 조차 S 자리 아래 → 순수 sticky (클램프가 안전한 이유).
    wire [7:0]  d        = ee_big - ee_sml;             // 스왑 덕에 항상 >= 0
    wire [3:0]  dc       = (d > 8'd11) ? 4'd11 : d[3:0];
    wire [10:0] body_sml = {sig_sml, 3'b000};
    wire [21:0] wide_a   = {body_sml, 11'b0} >> dc;
    wire [10:0] small_op = {wide_a[21:12], wide_a[11] | (|wide_a[10:0])};
    wire [10:0] big_op   = {sig_big, 3'b000};

    // 가감산 (12b). 감산이 음수가 되지 않는 근거: 크기 스왑으로 big >= small 이고,
    // d>=1 이면 big 은 normal(MSB set, >=2^10) 인데 정렬된 small 은 < 2^10.
    // d=0 이면 같은 그리드에서 sig 비교가 곧 크기 비교라 big_op >= small_op.
    wire [11:0] m12 = eff_sub ? ({1'b0, big_op} - {1'b0, small_op})
                              : ({1'b0, big_op} + {1'b0, small_op});

    // ── L_ADD 단: [A] 결과 버스 {m12, ee_big, sign, eff_sub, 특수플래그} ────
    localparam S1W = 25;
    wire [S1W-1:0] s1_w = {m12, ee_big, sign_big, eff_sub, r_nan, r_inf, inf_sgn};
    wire [S1W-1:0] s1;
    generate
        if (L_ADD > 0) begin : g_add_reg
            reg [S1W-1:0] q [0:L_ADD-1];
            integer ai;
            always @(posedge clk) begin
                q[0] <= s1_w;
                for (ai = 1; ai < L_ADD; ai = ai + 1)
                    q[ai] <= q[ai-1];
            end
            assign s1 = q[L_ADD-1];
        end else begin : g_add_comb
            assign s1 = s1_w;
        end
    endgenerate
    wire [11:0] s1_m12     = s1[24:13];
    wire [7:0]  s1_ee_big  = s1[12:5];
    wire        s1_sign    = s1[4];
    wire        s1_eff_sub = s1[3];
    wire        s1_nan     = s1[2];
    wire        s1_inf     = s1[1];
    wire        s1_infsgn  = s1[0];

    // ── [B] 정규화 (조합) ──────────────────────────────────────────────────

    // 가산 캐리: 우측 1 시프트. 새 S = 옛 R|S (sticky 는 어떤 시프트에서도
    // "잃어버린 정보가 있었다" 플래그로 보존돼야 한다). 지수 +1.
    wire        carry = s1_m12[11];
    wire [10:0] m_c   = {s1_m12[11:2], s1_m12[1] | s1_m12[0]};

    // 상쇄 정규화: 좌측 시프트를 지수 바닥(biased 1)에서 클램프. 클램프가
    // 걸리면 MSB 가 bit10 에 못 닿은 채 남고 → [C] 에서 자연히 subnormal 인코딩.
    // (좌측 시프트가 sticky 를 오염시키지 않는 불변식은 헤더 [B] 참조.)
    wire [3:0]  nz     = lzc11(s1_m12[10:0]);
    wire [7:0]  max_sh = s1_ee_big - 8'd1;
    wire [7:0]  shl    = ({4'd0, nz} > max_sh) ? max_sh : {4'd0, nz};
    wire [10:0] m_l    = s1_m12[10:0] << shl[3:0];       // shl <= 11 (nz 상한)

    wire [10:0] m_n = carry ? m_c : m_l;
    wire [8:0]  e_n = carry ? ({1'b0, s1_ee_big} + 9'd1)
                            : ({1'b0, s1_ee_big} - {1'b0, shl});

    // ── L_SUM 단: [B] 결과 버스 {m_n, e_n, sign, eff_sub, 특수플래그} ───────
    localparam S2W = 25;
    wire [S2W-1:0] s2_w = {m_n, e_n, s1_sign, s1_eff_sub, s1_nan, s1_inf, s1_infsgn};
    wire [S2W-1:0] s2;
    generate
        if (L_SUM > 0) begin : g_sum_reg
            reg [S2W-1:0] q [0:L_SUM-1];
            integer si;
            always @(posedge clk) begin
                q[0] <= s2_w;
                for (si = 1; si < L_SUM; si = si + 1)
                    q[si] <= q[si-1];
            end
            assign s2 = q[L_SUM-1];
        end else begin : g_sum_comb
            assign s2 = s2_w;
        end
    endgenerate
    wire [10:0] s2_m       = s2[24:14];
    wire [8:0]  s2_e       = s2[13:5];
    wire        s2_sign    = s2[4];
    wire        s2_eff_sub = s2[3];
    wire        s2_nan     = s2[2];
    wire        s2_inf     = s2[1];
    wire        s2_infsgn  = s2[0];

    // ── [C] 라운드 / 패킹 (조합) ───────────────────────────────────────────

    // RNE: L=유효숫자 LSB, G=guard, R|S=그 아래 전부. "절반 초과면 올림,
    // 정확히 절반이면 LSB 가 홀수일 때만 올림(짝수로)".
    wire       up    = s2_m[2] & (s2_m[3] | s2_m[1] | s2_m[0]);
    wire [8:0] sig9  = {1'b0, s2_m[10:3]} + {8'd0, up};
    wire       rc    = sig9[8];                          // 0xFF+1 → 재정규화
    wire [7:0] sig8  = rc ? 8'h80 : sig9[7:0];
    wire [8:0] e_fin = s2_e + {8'd0, rc};

    // exact-0 의 부호 (헤더 [C] 참조). sig8==0 은 항상 exact 0 케이스다.
    wire       zero_sign = s2_eff_sub ? 1'b0 : s2_sign;

    // 인코딩. 우선순위: NaN > inf(피연산자) > 0 > overflow-inf > subnormal > normal.
    //   - subnormal 가지(sig8 MSB=0)에서 e_fin 은 클램프에 의해 항상 biased 1.
    //   - 라운드 캐리로 subnormal 0x7F→0x80 이 되면 sig8[7]=1 → min normal 로
    //     자연 승격 (e_fin=1 그대로 normal 인코딩).
    wire [15:0] bf16_w =
          s2_nan              ? 16'h7FC0                        // canonical qNaN
        : s2_inf              ? {s2_infsgn, 8'hFF, 7'd0}
        : (sig8 == 8'd0)      ? {zero_sign, 15'd0}
        : (e_fin >= 9'd255)   ? {s2_sign, 8'hFF, 7'd0}          // RNE overflow → inf
        : (~sig8[7])          ? {s2_sign, 8'd0, sig8[6:0]}      // subnormal
        :                       {s2_sign, e_fin[7:0], sig8[6:0]};

    // ── L_OUT 단: 최종 bf16 레지스터 ───────────────────────────────────────
    generate
        if (L_OUT > 0) begin : g_out_reg
            reg [15:0] o_q [0:L_OUT-1];
            integer oi;
            always @(posedge clk) begin
                o_q[0] <= bf16_w;
                for (oi = 1; oi < L_OUT; oi = oi + 1)
                    o_q[oi] <= o_q[oi-1];
            end
            assign sum = o_q[L_OUT-1];
        end else begin : g_out_comb
            assign sum = bf16_w;
        end
    endgenerate
endmodule
