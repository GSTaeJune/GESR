`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// RMW — Read-Modify-Write 최상위 래퍼 (bf16 단위 연산기).
//
// 하는 일: GEMM이 뱉은 INT32 부분합을 bf16으로 변환한 뒤, SRAM에 저장돼 있던
// 이전 부분합(bf16)과 더해서 다시 SRAM에 쓸 값을 만든다.
//
// 데이터 패스 (Phase 2c 재배치 유지 + 2026-07-13 native 재작성 — 블록당 레지스터
// 1단, out_RMW 는 레지스터 출력. 두 하위 유닛 모두 손코딩 bf16 폭, HardFloat-free):
//
//   S1: i2b [F] |int|→LZC32→8b RNE      -> #1  (int_to_bf16 중간 절단, L_CONV-1 = 1단)
//   S2: i2b [B] scale/denorm/인코딩      -> #2  (bf16_adder L_IN 레지스터, a/b 공통 16b)
//   S3: 가산기 [A] 정렬/가감산 (11~12b)  -> #3  (bf16_adder 내부, L_ADD-2 = 1단)
//   S4: 가산기 [B] 정규화 (LZC+클램프)   -> #4  (bf16_adder L_SUM 레지스터, 25b)
//   S5: 가산기 [C] RNE 라운드/패킹       -> #5  (bf16_adder L_OUT 레지스터, 16b)
//   out_RMW = #5 출력 (레지스터) -> D-mux -> SRAM D
//
//   in_GEMM (INT32) ─┐
//                    ├─► int_to_bf16(L_CONV-1) ── fp_a (bf16) ──┐
//   scale (s9)  ─────┘                                          │
//                                                               ├─► bf16_adder ─► out_RMW
//   in_SRAM (bf16) ──► sram_dly[0:L_CONV-2] ────────────────────┘  (L_IN/L_SUM/L_OUT) (bf16)
//                       (L_CONV-1단 시프트 레지스터)
//
// in_SRAM 딜레이 체인이 필요한 이유:
//   int_to_bf16(L_CONV-1) 에서 나오는 fp_a 는 입력보다 L_CONV-1 사이클 늦다.
//   bf16_adder "입력에서" 두 피연산자가 같은 클럭 엣지에 도착하려면 in_SRAM 도
//   똑같이 L_CONV-1 만큼 늦춰야 한다. 이후 bf16_adder 의 L_IN 레지스터가 a/b 를
//   함께 지연하므로 짝(pairing)은 그대로 유지된다. 그래야 같은 K-tile 에 속한
//   GEMM 결과와 SRAM 이전값이 짝지어 더해진다.
//
// 전체 파이프라인 지연: L_CONV + L_ADD 사이클 (입력 → out_RMW까지) — 외부적으로
//   불변. 내부 분배만 재배치: (L_CONV-1) + L_IN(1) + (L_ADD-2) + L_SUM(1) + L_OUT(1)
//   = L_CONV + L_ADD. out_RMW 는 이제 레지스터 출력이라 SRAM D 까지 조합 꼬리가 없다.
//
// 스큐 검증 (load-bearing): rmw_tb 의 back-to-back 스트리밍(1 벡터/사이클, 파이프라인
//   모델 없는 per-index 순수함수 오라클) 이 RMW 내부 a/b 정렬 스큐를 잡는 유일한
//   게이트다 — sram_dly 깊이가 틀리면 in_SRAM[i] 가 엉뚱한 dq[i±1] 과 짝지어져 거의
//   전 벡터가 FAIL 한다. 통합 sweep 은 col drain FSM 이 두 피연산자를 10사이클 창
//   내내 정적으로 고정하므로 스큐를 잡지 못한다. 어떤 "단순화" 에서도 rmw_tb 의
//   스트리밍 케이던스를 약화하지 말 것.
//
// 이 모듈에서 처리하지 않는 것 (호출자가 알아서 해야 함):
//   * Accumulator_Col.out_scale의 9비트 sub-word 중 어느 것을 `scale`로 줄지
//     선택 (A8/A4/A2 모드별로 패킹이 다름 — Accumulator_Col.v 주석 참고).
//   * 첫 K-tile 초기화: 해당 SRAM 주소를 처음 쓸 때는 in_SRAM에 16'h0000을
//     강제로 넣어줘야 한다 (SRAM 초기값이 NaN 등 쓰레기일 수 있음).
//
// 포트:
//   in_SRAM  (bf16 16b)  : SRAM에서 읽어온 이전 부분합.
//   in_GEMM  (INT32 32b) : GEMM accumulator 한 lane 분의 원시 부분합.
//   scale    (s9)        : 9비트 signed 결합 스케일 (act+weight-127). 저장 안 함.
//   out_RMW  (bf16 16b)  : SRAM에 다시 쓸 합산 결과.
//
// 인스턴스: int_to_bf16 (변환) + bf16_adder (덧셈). sram_dly 는 로컬 시프트레지스터.
// 인스턴스되는 곳: gemm_sram_top (col j → RMW[j] → bank[j], 32×).
//
// 상태: ACTIVE (Phase 2b — bf16 데이터패스 본선). fp32 시절 int_to_fp32/fp32_adder
//   조합은 태그 `fp32-rmw-final` 로 보존 (README 참고).
//
// 검증: `bash sim/run_rmw.sh` → 기대 "rmw_tb: ALL 113 TESTS PASSED".
//   (벡터 생성 선행: cd MXP_Tools && python -m mxp_tools rmw-gen ...)
//
// Spec: docs/superpowers/specs/2026-07-08-rmw-bf16-design.md
//////////////////////////////////////////////////////////////////////////////////

module RMW #(
    parameter L_CONV = 2,    // int_to_bf16 파이프라인 단수 (>=1)
    parameter L_ADD  = 3     // bf16_adder  파이프라인 단수 (>=1)
)(
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] in_SRAM,    // SRAM에서 읽어온 이전 부분합 (bf16)
    input  wire [31:0] in_GEMM,    // GEMM accumulator의 INT32 출력 (lane 1개분)
    input  wire [8:0]  scale,      // 9비트 signed 결합 스케일 (act+weight-127)
    output wire [15:0] out_RMW     // SRAM에 다시 쓸 합산 결과 (bf16)
);

    // 파라미터 안전 체크: 재배치 파이프라인은 변환측 >=2, 가산측 >=3 을 요구한다
    // (내부 분배: int_to_bf16 L_CONV-1 / L_IN 1 / 피연산자 L_ADD-2 / L_SUM 1 / L_OUT 1).
    initial begin
        if (L_CONV < 2 || L_ADD < 3) begin
            $display("FATAL RMW: L_CONV=%0d L_ADD=%0d (need L_CONV>=2 && L_ADD>=3)",
                     L_CONV, L_ADD);
            $finish;
        end
    end

    // 1) INT32 -> bf16 디퀀타이즈. 재배치: 변환기 내부 레지스터는 L_CONV-1 단,
    //    남은 1단은 bf16_adder 의 L_IN(입력단) 레지스터가 담당한다.
    wire [15:0] fp_a;
    int_to_bf16 #(.L_CONV(L_CONV-1)) u_conv (
        .clk      (clk),
        .rst      (rst),
        .in_int   (in_GEMM),
        .scale    (scale),
        .out_bf16 (fp_a)
    );

    // 2) in_SRAM 을 L_CONV-1 사이클 늦춰서 fp_a 와 가산기 "입력에서" 정렬시킨다.
    //    (L_IN 레지스터는 a/b 를 함께 지연하므로 여기서는 변환기 내부 단수만 맞춘다.)
    reg [15:0] sram_dly [0:L_CONV-2];
    integer di;
    always @(posedge clk) begin
        sram_dly[0] <= in_SRAM;
        for (di = 1; di < L_CONV-1; di = di + 1)
            sram_dly[di] <= sram_dly[di-1];
    end

    // 3) bf16 덧셈. 재배치 구성: 입력 1 + 피연산자 L_ADD-2 + 합 1 + 출력 1 단.
    //    L_OUT=1 이 out_RMW 를 레지스터 출력으로 만들어 SRAM D 까지의 조합 꼬리를
    //    제거한다 (PNR 시 매크로 D-setup 경로 단축).
    bf16_adder #(.L_IN(1), .L_ADD(L_ADD-2), .L_SUM(1), .L_OUT(1)) u_add (
        .clk (clk),
        .rst (rst),
        .a   (fp_a),
        .b   (sram_dly[L_CONV-2]),
        .sum (out_RMW)
    );
endmodule
