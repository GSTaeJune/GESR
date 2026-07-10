`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// int_to_bf16_tb - int_to_bf16 크로스체크 TB.
//
// 검증 목적:
//   gemm_sram.srcs/sources_1/new/int_to_bf16.v 의 INT32 + 9비트 스케일 -> bf16
//   변환 (INToRecFN_i32_e8_s8 + fp32 도메인 지수-더하기 + FNFromRecFN_wrapper
//   + fp32_to_bf16_rne, v2) 이
//   ml_dtypes 기반 오라클과 bit-exact 로 일치하는지 확인.
//   오라클 모델: bf16(int -> bf16) * 2^(scale-127) = int->bf16 후 지수 shift
//   (INToRecFN 이 8-sig 로 먼저 라운딩 = golden r8; 이후 fp32 도메인 exact shift 후
//    fp32_to_bf16_rne 가 단일 RNE — subnormal 포함).
//
// 검증 내용 (sim/bf16_vectors.py 생성 work/bf16_vec/int_to_bf16.mem):
//   - realizable block_int 크기 (|.| < 2^20) x 다양한 결합 스케일
//   - directed int (0, +-1, 127, -128, 2^19-1, -(2^19), 255/256/257, 2^15 등)
//   - subnormal 유발 스케일 밴드 포함.
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
                    $display("MISMATCH in_int=%08x scale=%03x got=%04x exp=%04x",
                             in_int_w, scale_w, out_bf16, exp_w);
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
