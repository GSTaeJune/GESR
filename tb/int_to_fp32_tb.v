`timescale 1ns/1ps

module int_to_fp32_tb;
    localparam L_CONV     = 2;
    localparam CLK_PERIOD = 10;

    reg              clk;
    reg              rst;
    reg  [31:0]      in_int;
    reg  [8:0]       scale;
    wire [31:0]      out_fp32;

    int_to_fp32 #(.L_CONV(L_CONV)) dut (
        .clk(clk), .rst(rst),
        .in_int(in_int), .scale(scale),
        .out_fp32(out_fp32)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("int_to_fp32_tb.vcd"); $dumpvars(0, int_to_fp32_tb); end

    integer errors;
    integer i;

    task check;
        input [31:0] expected;
        input [255:0] label;
        begin
            #1;
            if (out_fp32 !== expected) begin
                $display("FAIL %0s: got %h expected %h", label, out_fp32, expected);
                errors = errors + 1;
            end else begin
                $display("PASS %0s", label);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst    = 1; in_int = 0; scale = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Case 1: int=1, scale=127 → FP32 1.0
        @(negedge clk); in_int = 32'sd1;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h3F800000, "int=1 scale=127 -> 1.0");

        // Case 2: int=-1 → -1.0
        @(negedge clk); in_int = -32'sd1;  scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'hBF800000, "int=-1 scale=127 -> -1.0");

        // Case 3: int=2 → 2.0
        @(negedge clk); in_int = 32'sd2;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h40000000, "int=2 scale=127 -> 2.0");

        // Case 4: int=0 → 0.0
        @(negedge clk); in_int = 32'sd0;   scale = 9'sd127;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h00000000, "int=0 -> 0.0");

        // Case 5: int=1, scale=128 → 2.0 (1 × 2^1)
        @(negedge clk); in_int = 32'sd1;   scale = 9'sd128;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h40000000, "int=1 scale=128 -> 2.0");

        // Case 6: int=4, scale=125 → 1.0 (4 × 2^-2)
        @(negedge clk); in_int = 32'sd4;   scale = 9'sd125;
        for (i = 0; i < L_CONV; i = i + 1) @(posedge clk);
        check(32'h3F800000, "int=4 scale=125 -> 1.0");

        if (errors == 0) $display("int_to_fp32_tb: ALL TESTS PASSED");
        else              $display("int_to_fp32_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
