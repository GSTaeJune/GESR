`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// RMW — Read-Modify-Write 최상위 래퍼 (FP32 단위 연산기).
//
// 하는 일: GEMM이 뱉은 INT32 부분합을 FP32로 변환한 뒤, SRAM에 저장돼 있던
// 이전 부분합(FP32)과 더해서 다시 SRAM에 쓸 값을 만든다.
//
// 데이터 패스:
//
//   in_GEMM (INT32) ─┐
//                    ├─► int_to_fp32 ──── fp_a (FP32) ──┐
//   scale (s9)  ─────┘   (L_CONV 사이클)                │
//                                                       ├─► fp32_adder ─► out_RMW
//   in_SRAM (FP32) ──► sram_dly[L_CONV-1:0] ────────────┘   (L_ADD 사이클)   (FP32)
//                       (L_CONV단 시프트 레지스터)
//
// in_SRAM 딜레이 체인이 필요한 이유:
//   int_to_fp32에서 나오는 fp_a는 입력보다 L_CONV 사이클 늦게 나온다.
//   가산기 입력에서 두 피연산자가 같은 클럭 엣지에 도착하려면 in_SRAM도
//   똑같이 L_CONV 사이클만큼 늦춰줘야 한다. 그래야 같은 K-tile에 속한
//   GEMM 결과와 SRAM 이전값이 짝지어 더해진다.
//
// 전체 파이프라인 지연: L_CONV + L_ADD 사이클 (입력 → out_RMW까지).
// out_RMW는 fp32_adder 마지막 레지스터단에서 조합 회로로 바로 뽑힌다.
//
// 이 모듈에서 처리하지 않는 것 (호출자가 알아서 해야 함):
//   * Accumulator_Col.out_scale의 9비트 sub-word 중 어느 것을 `scale`로 줄지
//     선택 (A8/A4/A2 모드별로 패킹이 다름 — Accumulator_Col.v 주석 참고).
//   * 첫 K-tile 초기화: 해당 SRAM 주소를 처음 쓸 때는 in_SRAM에 32'h00000000을
//     강제로 넣어줘야 한다 (SRAM 초기값이 NaN 등 쓰레기일 수 있음).
//
// Spec: docs/superpowers/specs/2026-05-14-rmw-design.md
//////////////////////////////////////////////////////////////////////////////////

module RMW #(
    parameter L_CONV = 2,    // int_to_fp32 파이프라인 단수 (>=1)
    parameter L_ADD  = 3     // fp32_adder  파이프라인 단수 (>=1)
)(
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] in_SRAM,    // SRAM에서 읽어온 이전 부분합 (FP32)
    input  wire [31:0] in_GEMM,    // GEMM accumulator의 INT32 출력 (lane 1개분)
    input  wire [8:0]  scale,      // 9비트 signed 결합 스케일 (act+weight-127)
    output wire [31:0] out_RMW     // SRAM에 다시 쓸 합산 결과 (FP32)
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

    // 1) INT32 -> FP32 디퀀타이즈 (L_CONV단 파이프라인)
    //    실수값 = in_GEMM * 2^(scale - 127)
    wire [31:0] fp_a;
    int_to_fp32 #(.L_CONV(L_CONV)) u_conv (
        .clk      (clk),
        .rst      (rst),
        .in_int   (in_GEMM),
        .scale    (scale),
        .out_fp32 (fp_a)
    );

    // 2) in_SRAM을 L_CONV 사이클 늦춰서 fp_a와 가산기 입력에서 정렬시킨다.
    //    sram_dly[0]이 1단째, sram_dly[L_CONV-1]이 L_CONV단째 (가장 오래된 값).
    reg [31:0] sram_dly [0:L_CONV-1];
    integer di;
    always @(posedge clk) begin
        sram_dly[0] <= in_SRAM;
        for (di = 1; di < L_CONV; di = di + 1)
            sram_dly[di] <= sram_dly[di-1];
    end

    // 3) FP32 덧셈 (L_ADD단 파이프라인)
    //    a (변환된 GEMM 결과) + b (정렬된 이전 SRAM 값)
    fp32_adder #(.L_ADD(L_ADD)) u_add (
        .clk (clk),
        .rst (rst),
        .a   (fp_a),
        .b   (sram_dly[L_CONV-1]),
        .sum (out_RMW)
    );
endmodule
