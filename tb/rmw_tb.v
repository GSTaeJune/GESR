`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// rmw_tb — vector-driven TB for RMW top wrapper.
//
// Streams N back-to-back stimulus vectors loaded from $readmemh files generated
// by `python -m mxp_tools rmw-gen` and compares the captured FP32 outputs
// (after L_TOTAL = L_CONV + L_ADD pipeline cycles) against the expected vector.
//
// VECTOR_DIR may be overridden via `xvlog -d VECTOR_DIR='"path"'`.
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
    reg  [31:0]      in_SRAM;
    reg  [31:0]      in_GEMM;
    reg  [8:0]       scale;
    wire [31:0]      out_RMW;

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

    // Vector storage.
    reg [31:0] mem_int   [0:MAX_N-1];
    reg [8:0]  mem_scale [0:MAX_N-1];
    reg [31:0] mem_sram  [0:MAX_N-1];
    reg [31:0] mem_exp   [0:MAX_N-1];
    reg [31:0] captured_out [0:MAX_N-1];

    integer N;
    integer fh;
    integer rc;
    integer i;
    integer errors;
    integer out_idx;

    initial begin
        errors = 0;

        // --- Load vectors --------------------------------------------------
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
        // scale.hex: 3 hex chars per line = 12 bits; rmw_gen.py masks each
        // value with 0x1FF before emit, so the high 3 bits are always 0 and
        // $readmemh's silent truncation into 9-bit mem_scale is safe.
        $readmemh({`VECTOR_DIR, "/scale.hex"},        mem_scale);
        $readmemh({`VECTOR_DIR, "/in_SRAM.hex"},      mem_sram);
        $readmemh({`VECTOR_DIR, "/expected_out.hex"}, mem_exp);

        // --- Reset ---------------------------------------------------------
        rst     = 1;
        in_SRAM = 32'h0;
        in_GEMM = 32'h0;
        scale   = 9'd0;
        out_idx = -(L_TOTAL - 1);
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        rst = 0;

        // --- Stimulus + capture loop --------------------------------------
        // One new input per negedge; capture out_RMW at posedge+#1.
        // out_idx starts at -L_TOTAL and increments each cycle; we only
        // store captured output when 0 <= out_idx < N.
        for (i = 0; i < N + L_TOTAL; i = i + 1) begin
            // Drive next stimulus on this negedge (idle/zero after N).
            if (i < N) begin
                in_GEMM = mem_int[i];
                scale   = mem_scale[i];
                in_SRAM = mem_sram[i];
            end else begin
                in_GEMM = 32'h0;
                scale   = 9'd0;
                in_SRAM = 32'h0;
            end

            @(posedge clk);
            #1;
            if (out_idx >= 0 && out_idx < N) begin
                captured_out[out_idx] = out_RMW;
            end
            out_idx = out_idx + 1;
            @(negedge clk);
        end

        // --- Compare -------------------------------------------------------
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
