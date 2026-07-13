`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// int_to_bf16_tb - int_to_bf16 크로스체크 TB.
//
// 검증 목적:
//   gemm_sram.srcs/sources_1/new/int_to_bf16.v 의 INT32 + 9비트 스케일 -> bf16
//   변환 (v3 native: LZC32 + 8비트 RNE 로 r8 생성 -> 지수+scale -> normal 은
//   exact 인코딩 / subnormal 밴드는 GRS 라운더로 두 번째 RNE — HardFloat
//   미사용, 2026-07-13 A6 재작성) 이 ml_dtypes 기반 오라클과 bit-exact 로
//   일치하는지 확인.
//   오라클 모델: bf16(int -> bf16) * 2^(scale-127) = int->bf16 후 지수 shift
//   (8-sig 첫 라운딩 = golden r8; 스케일 곱은 exact, subnormal 밴드에서만
//    두 번째 RNE — flush/tie/min-subnormal 경계 포함).
//
// 검증 내용 (sim/bf16_vectors.py 생성 work/bf16_vec/int_to_bf16.mem):
//   - realizable block_int 크기 (|.| < 2^20) x 다양한 결합 스케일
//   - directed int (0, +-1, 127, -128, 2^19-1, -(2^19), 255/256/257, 2^15 등)
//     + full-range INT32 (2^31-1, -2^31 등 — 32b LZC/round-carry witness)
//   - subnormal 유발 스케일 밴드 포함.
//   - 듀얼 DUT (Phase 2c 재배치): 기본단 dut (L_CONV=2) + RMW 실배치 dut_l1
//     (L_CONV=1) 를 같은 벡터로 동시 검증 -> RMW 에서 쓰는 L_CONV-1 깊이도
//     32360 전 벡터를 통과. nfail 은 두 DUT 공유(OR), ntot 은 벡터당 1회.
//     주의: 대기는 settle-tolerant (입력 유지 후 샘플) — "기능" 검증이며
//     latency 는 고정하지 않는다 (RMW 총 5cy latency 는 rmw_tb 가 고정).
//
// 동작 의도:
//   free-running clock. 벡터마다 in_int/scale 를 negedge 에 인가하고
//   L_CONV+1 posedge 를 기다린 뒤 (파이프라인 통과) out_bf16 을 샘플,
//   기대 bf16 워드와 !== 로 비교. latency-tolerant serial loop.
//   NaN/inf 는 오라클/DUT 인코딩이 동일하므로 특별처리 불필요.
//////////////////////////////////////////////////////////////////////////////////

module int_to_bf16_tb;
    localparam L_CONV = 2;

    reg         clk;
    reg         rst;
    reg  [31:0] in_int;
    reg  [8:0]  scale;
    wire [15:0] out_bf16;

    int_to_bf16 #(.L_CONV(L_CONV)) dut (
        .clk      (clk),
        .rst      (rst),
        .in_int   (in_int),
        .scale    (scale),
        .out_bf16 (out_bf16)
    );

    // RMW 재배치 구성 (Phase 2c): int_to_bf16 은 L_CONV-1 = 1 단으로 인스턴스된다.
    wire [15:0] out_bf16_l1;
    int_to_bf16 #(.L_CONV(1)) dut_l1 (
        .clk      (clk),
        .rst      (rst),
        .in_int   (in_int),
        .scale    (scale),
        .out_bf16 (out_bf16_l1)
    );

    // free-running clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer fd, r, nfail, ntot, k;
    reg [31:0] in_int_w;
    reg [8:0]  scale_w;
    reg [15:0] exp_w;

    initial begin
        rst    = 1'b1;
        in_int = 32'h0;
        scale  = 9'h0;
        nfail  = 0;
        ntot   = 0;

        // reset once at start
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;

        fd = $fopen("work/bf16_vec/int_to_bf16.mem", "r");
        if (fd == 0) begin
            $display("int_to_bf16_tb: FAIL cannot open vectors");
            $finish;
        end

        r = $fscanf(fd, "%h %h %h\n", in_int_w, scale_w, exp_w);
        while (r == 3) begin
            // drive on negedge for clean setup before the capturing posedge
            @(negedge clk);
            in_int = in_int_w;
            scale  = scale_w;
            // wait for the value to traverse the L_CONV pipeline stages
            for (k = 0; k < L_CONV + 1; k = k + 1)
                @(posedge clk);
            #1;  // let combinational output settle after the last posedge
            if (out_bf16 !== exp_w) begin
                if (nfail < 10)
                    $display("MISMATCH(L2) in_int=%08x scale=%03x got=%04x exp=%04x",
                             in_int_w, scale_w, out_bf16, exp_w);
                nfail = nfail + 1;
            end
            if (out_bf16_l1 !== exp_w) begin
                if (nfail < 10)
                    $display("MISMATCH(L1) in_int=%08x scale=%03x got=%04x exp=%04x",
                             in_int_w, scale_w, out_bf16_l1, exp_w);
                nfail = nfail + 1;
            end
            ntot = ntot + 1;
            r = $fscanf(fd, "%h %h %h\n", in_int_w, scale_w, exp_w);
        end
        $fclose(fd);

        if (nfail == 0)
            $display("int_to_bf16_tb: ALL %0d TESTS PASSED", ntot);
        else
            $display("int_to_bf16_tb: FAIL %0d/%0d", nfail, ntot);
        $finish;
    end
endmodule
