`timescale 1ns/1ps

// Smoke TB for vendored HardFloat (third_party/berkeley-hardfloat/).
// Verifies INT32 → recoded FP32 → IEEE-754 FP32 round-trip before we build
// any RMW logic on top. Module names per VENDORING.md.

module rmw_smoke_tb;

    reg  [31:0] int_in;
    wire [32:0] recFN;
    wire [31:0] fp_out;

    INToRecFN_i32_e8_s24 u_i2f (
        .io_signedIn       (1'b1),
        .io_in             (int_in),
        .io_roundingMode   (3'd0),
        .io_out            (recFN),
        .io_exceptionFlags ()
    );

    FNFromRecFN_wrapper u_f2f (
        .in  (recFN),
        .out (fp_out)
    );

    integer errors;

    initial begin
        $dumpfile("rmw_smoke_tb.vcd");
        $dumpvars(0, rmw_smoke_tb);

        errors = 0;

        int_in = 32'sd1;  #1;
        if (fp_out !== 32'h3F800000) begin
            $display("FAIL int=1: got %h expected 3F800000", fp_out);
            errors = errors + 1;
        end

        int_in = 32'sd0;  #1;
        if (fp_out !== 32'h00000000) begin
            $display("FAIL int=0: got %h expected 00000000", fp_out);
            errors = errors + 1;
        end

        int_in = -32'sd1; #1;
        if (fp_out !== 32'hBF800000) begin
            $display("FAIL int=-1: got %h expected BF800000", fp_out);
            errors = errors + 1;
        end

        int_in = 32'sd2;  #1;
        if (fp_out !== 32'h40000000) begin
            $display("FAIL int=2: got %h expected 40000000", fp_out);
            errors = errors + 1;
        end

        if (errors == 0) $display("rmw_smoke_tb: ALL TESTS PASSED");
        else              $display("rmw_smoke_tb: %0d FAILURES", errors);
        $finish;
    end
endmodule
