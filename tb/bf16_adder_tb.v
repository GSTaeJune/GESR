`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// bf16_adder_tb - bf16_adder 크로스체크 TB.
//
// 검증 목적:
//   gemm_sram.srcs/sources_1/new/bf16_adder.v 의 bf16 + bf16 -> bf16 덧셈
//   (native 단일-라운딩 구현: 정렬 -> 가감산 -> 정규화 -> RNE, 내부 11~12b
//   그리드, HardFloat 미사용 — 2026-07-13 A6 재작성) 이 ml_dtypes 의
//   bfloat16 덧셈과 bit-exact 로 일치하는지 확인.
//
// 검증 내용 (sim/bf16_vectors.py 생성 work/bf16_vec/bf16_add.mem):
//   - random bf16 x 다양한 크기 (200k)
//   - directed edges: inf+(-inf)->NaN, inf+inf, 1.0+(-1.0), subnormal 합,
//     +0 + (-0) 상쇄 + native 재작성 witness 20종 (유한 overflow->inf,
//     min-normal - min-subnormal 경계 상쇄, d=24 sticky 붕괴 ± (round-carry),
//     x+(-x) 정확 0 부호, -0+-0, RNE tie 짝수/홀수, one-sided inf±finite
//     — inf_sgn 포트 선택 + r_inf mux 의 datapath 덮어쓰기 witness).
//   - 트리플 DUT: 기본단 dut (L_ADD=3) + RMW 실배치 dut_rmw (L_IN=L_ADD=
//     L_SUM=L_OUT=1) + 전조합 dut_comb (0,0,0,0 — 레지스터 배치 불변성의 극단
//     witness) 를 같은 벡터로 동시 검증 -> 세 구성 모두 200025 전 벡터를 통과.
//     nfail 은 세 DUT 공유(OR), ntot 은 벡터당 1회.
//     주의: 대기는 settle-tolerant (입력 유지 후 샘플) — 이 TB 는 dut_rmw 의
//     "기능"을 검증하며 latency 는 고정하지 않는다. RMW 구성의 latency(총 5cy)
//     는 rmw_tb 의 스트리밍 캡처가 고정한다.
//
// 동작 의도:
//   free-running clock. 벡터마다 a/b 를 negedge 에 인가하고 L_ADD+1 posedge 를
//   기다린 뒤 (파이프라인 통과) sum 을 샘플, 기대 bf16 워드와 !== 로 비교.
//   latency-tolerant serial loop.
//
// NaN 부호 처리:
//   IEEE-754 는 inf+(-inf) 로 생기는 NaN 의 부호를 규정하지 않는다. HardFloat
//   AddRecFN 은 +qNaN(0x7FC0..) 을, ml_dtypes/numpy(x86) 는 -qNaN(0xFFC0..) 을
//   낸다 -> 부호만 다른 qNaN. 산술적으로 NaN==NaN 이므로 got/exp 가 둘 다 bf16
//   NaN 이면 일치로 처리(부호/payload 무시). 유한/inf 값은 그대로 !== bit-exact
//   비교라 수치 버그는 여전히 잡힌다.
//////////////////////////////////////////////////////////////////////////////////

module bf16_adder_tb;
    localparam L_ADD = 3;

    // RMW 실배치 구성 (Phase 2c 재배치): dut_rmw 의 L_IN+L_ADD+L_SUM+L_OUT = 1+1+1+1.
    // 주의: 아래 dut_rmw 인스턴스 파라미터를 바꾸면 이 값도 함께 갱신할 것.
    localparam L_RMW  = 4;
    localparam L_WAIT = (L_RMW > L_ADD) ? L_RMW : L_ADD;   // 두 DUT 모두 통과할 대기

    // bf16 NaN 판정: exp 전부 1 (bit14:7 == 0xFF) 이고 mantissa(bit6:0) 비영.
    function is_bf16_nan;
        input [15:0] x;
        is_bf16_nan = (x[14:7] == 8'hFF) && (|x[6:0]);
    endfunction

    reg         clk;
    reg         rst;
    reg  [15:0] a;
    reg  [15:0] b;
    wire [15:0] sum;

    bf16_adder #(.L_ADD(L_ADD)) dut (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .sum (sum)
    );

    wire [15:0] sum_rmw;
    bf16_adder #(.L_IN(1), .L_ADD(1), .L_SUM(1), .L_OUT(1)) dut_rmw (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .sum (sum_rmw)
    );

    // 전조합 구성 (0,0,0,0): [A][B][C] 세 블록이 한 조합 콘으로 붕괴하는 극단.
    // 파라미터 = 절단점 레지스터 깊이일 뿐 기능 불변이라는 계약의 witness.
    wire [15:0] sum_comb;
    bf16_adder #(.L_IN(0), .L_ADD(0), .L_SUM(0), .L_OUT(0)) dut_comb (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .sum (sum_comb)
    );

    // free-running clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer fd, r, nfail, ntot, k;
    reg [15:0] a_w;
    reg [15:0] b_w;
    reg [15:0] exp_w;

    initial begin
        rst   = 1'b1;
        a     = 16'h0;
        b     = 16'h0;
        nfail = 0;
        ntot  = 0;

        // reset once at start
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;

        fd = $fopen("work/bf16_vec/bf16_add.mem", "r");
        if (fd == 0) begin
            $display("bf16_adder_tb: FAIL cannot open vectors");
            $finish;
        end

        r = $fscanf(fd, "%h %h %h\n", a_w, b_w, exp_w);
        while (r == 3) begin
            // drive on negedge for clean setup before the capturing posedge
            @(negedge clk);
            a = a_w;
            b = b_w;
            // wait for the value to traverse both DUTs' pipeline stages
            for (k = 0; k < L_WAIT + 1; k = k + 1)
                @(posedge clk);
            #1;  // let combinational narrow settle after the last posedge
            // 둘 다 NaN 이면 부호/payload 무시하고 통과, 그 외는 bit-exact.
            if (!(is_bf16_nan(sum) && is_bf16_nan(exp_w)) && (sum !== exp_w)) begin
                if (nfail < 10)
                    $display("MISMATCH(dut)     a=%04x b=%04x got=%04x exp=%04x",
                             a_w, b_w, sum, exp_w);
                nfail = nfail + 1;
            end
            if (!(is_bf16_nan(sum_rmw) && is_bf16_nan(exp_w)) && (sum_rmw !== exp_w)) begin
                if (nfail < 10)
                    $display("MISMATCH(dut_rmw) a=%04x b=%04x got=%04x exp=%04x",
                             a_w, b_w, sum_rmw, exp_w);
                nfail = nfail + 1;
            end
            if (!(is_bf16_nan(sum_comb) && is_bf16_nan(exp_w)) && (sum_comb !== exp_w)) begin
                if (nfail < 10)
                    $display("MISMATCH(dut_comb) a=%04x b=%04x got=%04x exp=%04x",
                             a_w, b_w, sum_comb, exp_w);
                nfail = nfail + 1;
            end
            ntot = ntot + 1;
            r = $fscanf(fd, "%h %h %h\n", a_w, b_w, exp_w);
        end
        $fclose(fd);

        if (nfail == 0)
            $display("bf16_adder_tb: ALL %0d TESTS PASSED", ntot);
        else
            $display("bf16_adder_tb: FAIL %0d/%0d", nfail, ntot);
        $finish;
    end
endmodule
