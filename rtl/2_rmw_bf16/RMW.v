`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// RMW — Read-Modify-Write 최상위 래퍼 (bf16 단위 연산기).
//
// 하는 일: GEMM이 뱉은 INT32 부분합을 bf16으로 변환한 뒤, SRAM에 저장돼 있던
// 이전 부분합(bf16)과 더해서 다시 SRAM에 쓸 값을 만든다.
//
// 데이터 패스:
//
//   in_GEMM (INT32) ─┐
//                    ├─► int_to_bf16 ──── fp_a (bf16) ──┐
//   scale (s9)  ─────┘   (L_CONV 사이클)                │
//                                                       ├─► bf16_adder ─► out_RMW
//   in_SRAM (bf16) ──► sram_dly[L_CONV-1:0] ────────────┘   (L_ADD 사이클)   (bf16)
//                       (L_CONV단 시프트 레지스터)
//
// in_SRAM 딜레이 체인이 필요한 이유:
//   int_to_bf16에서 나오는 fp_a는 입력보다 L_CONV 사이클 늦게 나온다.
//   가산기 입력에서 두 피연산자가 같은 클럭 엣지에 도착하려면 in_SRAM도
//   똑같이 L_CONV 사이클만큼 늦춰줘야 한다. 그래야 같은 K-tile에 속한
//   GEMM 결과와 SRAM 이전값이 짝지어 더해진다.
//
// 전체 파이프라인 지연: L_CONV + L_ADD 사이클 (입력 → out_RMW까지).
// out_RMW는 bf16_adder 마지막 레지스터단에서 조합 회로로 바로 뽑힌다.
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

    // 파라미터 안전 체크: 두 sub-module 모두 단수 >=1 을 요구한다.
    // L_CONV=0 이면 sram_dly[0:-1] 라는 불법 배열이 생긴다.
    initial begin
        if (L_CONV < 1 || L_ADD < 1) begin
            $display("FATAL RMW: L_CONV=%0d L_ADD=%0d, 둘 다 >= 1 이어야 함",
                     L_CONV, L_ADD);
            $finish;
        end
    end

    // 1) INT32 -> bf16 디퀀타이즈 (L_CONV단 파이프라인)
    //    실수값 = in_GEMM * 2^(scale - 127), 깊은 언더플로는 ±0 flush
    wire [15:0] fp_a;
    int_to_bf16 #(.L_CONV(L_CONV)) u_conv (
        .clk      (clk),
        .rst      (rst),
        .in_int   (in_GEMM),
        .scale    (scale),
        .out_bf16 (fp_a)
    );

    // 2) in_SRAM을 L_CONV 사이클 늦춰서 fp_a와 가산기 입력에서 정렬시킨다.
    reg [15:0] sram_dly [0:L_CONV-1];
    integer di;
    always @(posedge clk) begin
        sram_dly[0] <= in_SRAM;
        for (di = 1; di < L_CONV; di = di + 1)
            sram_dly[di] <= sram_dly[di-1];
    end

    // 3) bf16 덧셈 (L_ADD단 파이프라인; 내부는 fp32 도메인 가산 + RNE narrow)
    bf16_adder #(.L_ADD(L_ADD)) u_add (
        .clk (clk),
        .rst (rst),
        .a   (fp_a),
        .b   (sram_dly[L_CONV-1]),
        .sum (out_RMW)
    );
endmodule
