`timescale 1ns/1ps

module fp32_adder_tb;
    localparam L_ADD      = 3;
    localparam CLK_PERIOD = 10;

    reg              clk;
    reg              rst;
    reg  [31:0]      a;
    reg  [31:0]      b;
    wire [31:0]      sum;

    fp32_adder #(.L_ADD(L_ADD)) dut (
        .clk(clk), .rst(rst),
        .a(a), .b(b),
        .sum(sum)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    initial begin $dumpfile("fp32_adder_tb.vcd"); $dumpvars(0, fp32_adder_tb); end

    integer errors;
    integer i;

    task check;
        input [31:0] expected;
        input [255:0] label;
        begin
            #1;
            if (sum !== expected) begin
                $display("FAIL %0s: got %h expected %h", label, sum, expected);
                errors = errors + 1;
            end else begin
                $display("PASS %0s", label);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst = 1; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Case 1: 1.0 + 1.0 = 2.0
        @(negedge clk); a = 32'h3F800000; b = 32'h3F800000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h40000000, "1.0 + 1.0 -> 2.0");

        // Case 2: 1.0 + (-1.0) = 0.0
        @(negedge clk); a = 32'h3F800000; b = 32'hBF800000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h00000000, "1.0 + (-1.0) -> 0.0");

        // Case 3: 1.5 + 2.5 = 4.0
        @(negedge clk); a = 32'h3FC00000; b = 32'h40200000;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h40800000, "1.5 + 2.5 -> 4.0");

        // Case 4: 0.0 + 3.14 ~= 3.14
        @(negedge clk); a = 32'h00000000; b = 32'h4048F5C3;
        for (i = 0; i < L_ADD; i = i + 1) @(posedge clk);
        check(32'h4048F5C3, "0.0 + 3.14 -> 3.14");

        if (errors == 0) $display("fp32_adder_tb: ALL TESTS PASSED");
        else              $display("fp32_adder_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
