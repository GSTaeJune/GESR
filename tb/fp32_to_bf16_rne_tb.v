`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// fp32_to_bf16_rne_tb - fp32_to_bf16_rne 크로스체크 TB.
//
// 검증 목적:
//   gemm_sram.srcs/sources_1/new/fp32_to_bf16_rne.v 의 조합 FP32->bf16 RNE narrow
//   가 ml_dtypes 의 fp32->bfloat16 변환과 bit-exact 로 일치하는지 확인.
//   (Phase-1 golden 이 HW 의 증명된 모델이 되기 위한 D6 핵심 블록.)
//
// 검증 내용 (near-exhaustive 벡터, sim/bf16_vectors.py 생성):
//   - 다양한 크기의 random normal (50k)
//   - directed: tie(RNE even), subnormal band, overflow band, +/-0, +/-inf
//   - subnormal-inducing uniform band (20k)
//
// 동작 의도:
//   work/bf16_vec/fp32_to_bf16.mem 를 $fopen + $fscanf 루프로 한 줄씩 읽어
//   ("%h %h" = fp32 입력워드, 기대 bf16 출력워드) DUT 에 in 을 인가하고
//   조합 지연(#1) 후 out 을 !== 로 비교. NaN 은 오라클/DUT 양쪽 모두
//   0x7FC0 canonical qNaN 이므로 특별처리 불필요.
//////////////////////////////////////////////////////////////////////////////////

module fp32_to_bf16_rne_tb;
    reg  [31:0] in;
    wire [15:0] out;

    fp32_to_bf16_rne dut (.in(in), .out(out));

    integer fd, r, nfail, ntot;
    reg [31:0] in_w;
    reg [15:0] exp_w;

    initial begin
        fd = $fopen("work/bf16_vec/fp32_to_bf16.mem", "r");
        if (fd == 0) begin
            $display("fp32_to_bf16_rne_tb: FAIL cannot open vectors");
            $finish;
        end
        nfail = 0;
        ntot  = 0;
        r = $fscanf(fd, "%h %h\n", in_w, exp_w);
        while (r == 2) begin
            in = in_w; #1;
            if (out !== exp_w) begin
                if (nfail < 10)
                    $display("MISMATCH in=%08x got=%04x exp=%04x", in_w, out, exp_w);
                nfail = nfail + 1;
            end
            ntot = ntot + 1;
            r = $fscanf(fd, "%h %h\n", in_w, exp_w);
        end
        $fclose(fd);

        if (nfail == 0)
            $display("fp32_to_bf16_rne_tb: ALL %0d TESTS PASSED", ntot);
        else
            $display("fp32_to_bf16_rne_tb: FAIL %0d/%0d", nfail, ntot);
        $finish;
    end
endmodule
