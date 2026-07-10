`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// rmw_tb — RMW 본체 (int_to_bf16 + delay + bf16_adder) 의 벡터 기반 TB.
//
// 검증 목적:
//   gemm_sram.srcs/sources_1/new/RMW.v 가 다음 동작을 L_TOTAL = L_CONV + L_ADD
//   사이클 파이프라인을 거친 후 정확히 수행하는지 확인:
//
//       out_RMW = bf16( bf16(in_GEMM * 2^(scale-127)) + in_SRAM )   (IEEE bfloat16, RNE)
//
//   특히 다음을 동시에 자극:
//     1) INT→bf16 변환 + 지수 보정 (zero passthrough + 깊은 언더플로 ±0 flush + subnormal RNE 포함)
//     2) sram_dly 체인이 in_SRAM 을 L_CONV 만큼 지연시켜 가산기 입력에서 정렬
//     3) bf16 가산기 (fp32 도메인 가산 + RNE narrow) 의 정상값/특수값 동작
//   ⇒ Python 측 (MXP_Tools/mxp_tools/rmw_gen.py) 이 생성한 N개의 무작위
//      벡터를 back-to-back 으로 흘려보내고, 캡처된 out_RMW 가 Python 기대값
//      (expected_out.hex) 과 비트 단위로 일치하는지 본다.
//
// 검증 내용:
//   - N: rmw-gen 의 --n (기본 64). 본 TB 는 MAX_N=128 까지 허용.
//   - 벡터 파일 (모두 hex):
//       N.txt           : 한 줄 정수 N
//       in_GEMM.hex     : INT32 32 비트 × N
//       scale.hex       : 9 비트 scale × N (gen 측에서 0x1FF 마스킹)
//       in_SRAM.hex     : bf16 16비트 × N
//       expected_out.hex: bf16 16비트 × N (ml_dtypes 계산 결과)
//
//   - 자극 패턴: negedge 마다 새 입력 하나씩, posedge 직후 #1 에 out_RMW 캡처.
//     out_idx 가 -(L_TOTAL-1) 에서 시작해 매 사이클 +1, 0..N-1 구간만 저장.
//     (= 파이프라인 깊이만큼 미리 흘려보내고, 첫 유효 출력부터 캡처)
//
//   - 비교: captured_out[i] 와 mem_exp[i] 를 !== 로 비교 (X 도 FAIL).
//     0 mismatch → "ALL N TESTS PASSED" 출력.
//
// VECTOR_DIR override: xvlog -d VECTOR_DIR='"path"' 로 디렉토리 교체 가능.
//////////////////////////////////////////////////////////////////////////////////

`ifndef VECTOR_DIR
`define VECTOR_DIR "../MXP_Tools/work/rmw"
`endif

module rmw_tb;
    localparam L_CONV     = 2;
    localparam L_ADD      = 3;
    localparam L_TOTAL    = L_CONV + L_ADD;
    localparam CLK_PERIOD = 10;
    localparam MAX_N      = 128;

    reg              clk;
    reg              rst;
    reg  [15:0]      in_SRAM;
    reg  [31:0]      in_GEMM;
    reg  [8:0]       scale;
    wire [15:0]      out_RMW;

    RMW #(.L_CONV(L_CONV), .L_ADD(L_ADD)) dut (
        .clk     (clk),
        .rst     (rst),
        .in_SRAM (in_SRAM),
        .in_GEMM (in_GEMM),
        .scale   (scale),
        .out_RMW (out_RMW)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("rmw_tb.vcd"); $dumpvars(0, rmw_tb); end

    // 벡터 저장소.
    reg [31:0] mem_int   [0:MAX_N-1];   // in_GEMM 자극
    reg [8:0]  mem_scale [0:MAX_N-1];   // scale 자극
    reg [15:0] mem_sram  [0:MAX_N-1];   // in_SRAM 자극
    reg [15:0] mem_exp   [0:MAX_N-1];   // 기대 출력 (Python 계산)
    reg [15:0] captured_out [0:MAX_N-1];

    integer N;
    integer fh;
    integer rc;
    integer i;
    integer errors;
    integer out_idx;

    initial begin
        errors = 0;

        // ─── 벡터 로드 ────────────────────────────────────────────────
        fh = $fopen({`VECTOR_DIR, "/N.txt"}, "r");
        if (fh == 0) begin
            $display("FATAL: cannot open %s/N.txt", `VECTOR_DIR);
            $finish;
        end
        rc = $fscanf(fh, "%d", N);
        $fclose(fh);
        if (rc != 1) begin
            $display("FATAL: bad N.txt (fscanf rc=%0d)", rc);
            $finish;
        end
        if (N <= 0 || N > MAX_N) begin
            $display("FATAL: N=%0d out of range (MAX_N=%0d)", N, MAX_N);
            $finish;
        end
        $display("rmw_tb: loading %0d vectors from %s", N, `VECTOR_DIR);

        $readmemh({`VECTOR_DIR, "/in_GEMM.hex"},      mem_int);
        // scale.hex: 한 줄당 3 hex 글자 = 12 비트. rmw_gen.py 가 emit 전에
        // 각 값을 0x1FF 로 마스킹하므로 상위 3 비트는 항상 0, $readmemh 가
        // 9 비트 mem_scale 로 silent truncate 해도 안전.
        $readmemh({`VECTOR_DIR, "/scale.hex"},        mem_scale);
        $readmemh({`VECTOR_DIR, "/in_SRAM.hex"},      mem_sram);
        $readmemh({`VECTOR_DIR, "/expected_out.hex"}, mem_exp);

        // ─── 리셋 ────────────────────────────────────────────────────
        rst     = 1;
        in_SRAM = 16'h0;
        in_GEMM = 32'h0;
        scale   = 9'd0;
        out_idx = -(L_TOTAL - 1);
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        rst = 0;

        // ─── 자극 + 캡처 루프 ────────────────────────────────────────
        // negedge 마다 새 입력 1 개씩 흘림, posedge+#1 에 out_RMW 캡처.
        // out_idx 는 -L_TOTAL+1 에서 시작 → 매 사이클 +1.
        // 0 <= out_idx < N 구간만 captured_out 에 저장 (파이프라인 채움 구간은 버림).
        for (i = 0; i < N + L_TOTAL; i = i + 1) begin
            // N 개를 다 흘렸으면 idle (zero) 로 둠.
            if (i < N) begin
                in_GEMM = mem_int[i];
                scale   = mem_scale[i];
                in_SRAM = mem_sram[i];
            end else begin
                in_GEMM = 32'h0;
                scale   = 9'd0;
                in_SRAM = 16'h0;
            end

            @(posedge clk);
            #1;
            if (out_idx >= 0 && out_idx < N) begin
                captured_out[out_idx] = out_RMW;
            end
            out_idx = out_idx + 1;
            @(negedge clk);
        end

        // ─── 비교 ────────────────────────────────────────────────────
        // captured_out[i] !== mem_exp[i] 면 FAIL (X 도 FAIL).
        for (i = 0; i < N; i = i + 1) begin
            if (captured_out[i] !== mem_exp[i]) begin
                $display("FAIL[%0d]: in_GEMM=%h scale=%h in_SRAM=%h  got=%h  exp=%h",
                         i, mem_int[i], mem_scale[i], mem_sram[i],
                         captured_out[i], mem_exp[i]);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("rmw_tb: ALL %0d TESTS PASSED", N);
        else
            $display("rmw_tb: %0d / %0d FAILURES", errors, N);

        $finish;
    end
endmodule
