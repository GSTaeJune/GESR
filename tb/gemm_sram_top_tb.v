`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_tb — Task 7 (A8_B8 only): end-to-end GEMM + RMW + SRAM TB.
//
// Strategy (single-RMW-at-a-time, post-drive replay):
//   1) INIT     — reset + SRAM zero-prime via TB-driven do_write loop.
//   2) LOAD     — $readmemh hex files (a_input_BS, b_input, a_scale, b_scale).
//   3) CONFIG   — set A8/W8 control codes, prepare driver state.
//   4) DRIVE    — single big initial block does the 5-chain MXP driving per
//                 precision_modes_protocol.md §3 (Stage 2-A / 2-B / 3+4 / 5).
//                 During driving an `always @(posedge clk)` capture-block
//                 records every out_fire[c] rising edge into FIFO arrays:
//                 fifo_int[i], fifo_scale[i], fifo_addr[i] (15-bit flat).
//   5) DRAIN    — sequential RMW dispatch over the captured FIFO. For each
//                 entry: issue SRAM read → wait L_CONV cyc → drive rmw inputs
//                 → wait L_ADD cyc → write back.
//   6) DUMP     — $writememh all 16 banks (0..1023 word range).
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8   (Task 7: A8 only)
//   B_PREC    : 2 / 4 / 8   (Task 7: B8 only)
//   WORK_DIR  : MXP_Tools .hex input directory (REQUIRED for Task 7)
//   DUMP_DIR  : SRAM .mem dump output dir       (REQUIRED)
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top_tb;

    // ─── plusarg parsing ───────────────────────────────────────────
    integer A_PREC, B_PREC;
    reg [8*256-1:0] WORK_DIR;
    reg [8*256-1:0] DUMP_DIR;

    initial begin
        if (!$value$plusargs("A_PREC=%d",   A_PREC))    A_PREC   = 8;
        if (!$value$plusargs("B_PREC=%d",   B_PREC))    B_PREC   = 8;
        if (!$value$plusargs("WORK_DIR=%s", WORK_DIR))  WORK_DIR = "work/A8_B8";
        if (!$value$plusargs("DUMP_DIR=%s", DUMP_DIR))  DUMP_DIR = "work/A8_B8/hw_out";
    end

    // ─── GEMM size constants (128×128×128, K_T=M_T=N_T=4, TILE=32) ─
    localparam integer M_DIM = 128, K_DIM = 128, N_DIM = 128;
    localparam integer TILE_SIZE = 32;
    localparam integer M_T = M_DIM / TILE_SIZE;     // 4
    localparam integer K_T = K_DIM / TILE_SIZE;     // 4
    localparam integer N_T = N_DIM / TILE_SIZE;     // 4 (logical for A8)

    // A8W8-specific constants (from precision_modes_protocol.md §1)
    localparam integer W_CYC          = 8;
    localparam integer A_FIRE_DELAY   = 2;
    localparam integer FIRST_FIRE_GLOBAL = 28;  // col 16
    localparam integer TOGGLE_VAL     = 24;
    localparam integer FIRES_PER_COL  = 2048;   // = N_T * K_T * M_T * TILE_SIZE
    localparam integer N_T_LOGICAL    = 4;      // A8

    localparam [1:0] A_INT8 = 2'b00;
    localparam [1:0] A_IDLE = 2'b11;
    localparam [1:0] W_INT8 = 2'b11;
    localparam [1:0] W_IDLE = 2'b00;

    // ─── clock / reset ─────────────────────────────────────────────
    reg clk = 0;
    always #5 clk = ~clk;        // 100 MHz nominal (10 ns period)
    reg rst = 1;

    // ─── DUT stimuli registers ─────────────────────────────────────
    reg [32-1:0]              in_a              = 0;
    reg signed [32*8-1:0]     in_b              = 0;
    reg [32-1:0]              in_control        = 0;
    reg [32-1:0]              in_loadEN         = 0;
    reg [1:0]                 in_Station_control = A_IDLE;
    reg [4*8-1:0]             in_Scale_Activation = 0;
    reg                       in_station_control  = 0;
    reg                       in_station_loadEN   = 0;
    reg                       in_start_accumulate = 0;
    reg [1:0]                 in_Wcontrol         = W_IDLE;
    reg [7:0]                 in_scale_weight     = 0;

    wire [32*60-1:0]          out_accumulate;
    wire [32*36-1:0]          out_scale;
    wire [32-1:0]             out_fire;

    // RMW
    reg [31:0]                rmw_in_GEMM       = 0;
    reg [8:0]                 rmw_scale         = 0;
    wire [31:0]               rmw_out_RMW;

    // SRAM
    reg                       sram_CEB          = 1;
    reg                       sram_WEB          = 1;
    reg [18:0]                sram_A            = 0;
    reg [31:0]                sram_WMASK        = 32'hFFFFFFFF;
    reg                       sram_D_use_zero   = 1;
    wire [31:0]               sram_Q;

    // ─── DUT instance ──────────────────────────────────────────────
    gemm_sram_top #(
        .NUM_BANKS  (16),
        .BANK_DEPTH (32768)
    ) u_top (
        .clk(clk), .rst(rst),
        .in_a(in_a), .in_b(in_b),
        .in_control(in_control), .in_loadEN(in_loadEN),
        .in_Station_control(in_Station_control),
        .in_Scale_Activation(in_Scale_Activation),
        .in_station_control(in_station_control),
        .in_station_loadEN(in_station_loadEN),
        .in_start_accumulate(in_start_accumulate),
        .in_Wcontrol(in_Wcontrol),
        .in_scale_weight(in_scale_weight),
        .out_accumulate(out_accumulate),
        .out_scale(out_scale),
        .out_fire(out_fire),
        .rmw_in_GEMM(rmw_in_GEMM),
        .rmw_scale(rmw_scale),
        .rmw_out_RMW(rmw_out_RMW),
        .sram_CEB(sram_CEB), .sram_WEB(sram_WEB),
        .sram_A(sram_A), .sram_WMASK(sram_WMASK),
        .sram_D_use_zero(sram_D_use_zero),
        .sram_Q(sram_Q)
    );

    // ─── helper tasks ──────────────────────────────────────────────
    task sram_write_zero(input [18:0] addr);
        begin
            @(posedge clk);
            sram_CEB        <= 1'b0;
            sram_WEB        <= 1'b0;
            sram_A          <= addr;
            sram_WMASK      <= 32'hFFFFFFFF;
            sram_D_use_zero <= 1'b1;
        end
    endtask

    task sram_idle;
        begin
            @(posedge clk);
            sram_CEB <= 1'b1;
            sram_WEB <= 1'b1;
        end
    endtask

    // ─── Input memories ────────────────────────────────────────────
    // sizes: K_T*M_T*TILE*W_CYC = 4*4*32*8 = 4096 lines for a_bs
    //        N_T*K_T*TILE       = 4*4*32   = 512  lines for b_bp / b_scale
    //        K_T*M_T*TILE       = 4*4*32   = 512  lines for a_scale
    reg [31:0]   a_bs    [0:4095];
    reg [255:0]  b_bp    [0:511];
    reg [7:0]    a_scale [0:511];
    reg [7:0]    b_scale [0:511];

    // ─── Fire capture FIFO ──────────────────────────────────────────
    // Total fires expected (A8W8): 32 col × 2048 fires/col = 65536
    reg [31:0]   fifo_int   [0:65535];
    reg [8:0]    fifo_scale [0:65535];
    reg [18:0]   fifo_addr  [0:65535];
    integer      fifo_wp;          // write pointer
    integer      fifo_rp;          // read pointer (for drain)

    // Per-col fire counter (tracks which fire# within column → decodes m_t, m_in, k_t, n_t)
    reg [11:0]   fire_cnt_per_col [0:31];

    // ─── Fire-capture always block ─────────────────────────────────
    //
    // Decode fire_cnt[c] (linear count of fires on col c) → (n_t, k_t, m_t, m_in):
    //   inner loop order in driving:  n_t → k_t → m_t → o (W_CYC × TILE_SIZE inner)
    //   one fire per col per (n_t, k_t, m_t, m_in), m_in ∈ [0, TILE_SIZE)
    //   so fc = n_t*K_T*M_T*TILE + k_t*M_T*TILE + m_t*TILE + m_in
    //
    // m_global = m_t * TILE_SIZE + m_in
    // n_global = n_t * TILE_SIZE + col_idx     (A8: 1 lane / col)
    // flat = m_global * N_DIM + n_global
    integer fc, n_t_dec, k_t_dec, m_t_dec, m_in_dec, m_g, n_g, flat;
    integer ci;
    reg signed [20:0] acc_lane;
    reg signed [31:0] acc_int32;
    reg [8:0]         sc_lane;

    reg capture_en;     // 1 during DRIVE stage
    initial capture_en = 0;

    always @(posedge clk) begin
        if (rst) begin
            for (ci = 0; ci < 32; ci = ci + 1) fire_cnt_per_col[ci] <= 0;
        end else if (capture_en) begin
            for (ci = 0; ci < 32; ci = ci + 1) begin
                // Ignore spurious fires beyond FIRES_PER_COL (extra Accumulator
                // counter rollovers in Stage 5 tail before W_IDLE propagates).
                if (out_fire[ci] && fire_cnt_per_col[ci] < 12'd2048) begin
                    // A8 lane slice: acc[60c+20:60c+0], scale[36c+8:36c+0]
                    acc_lane = $signed(out_accumulate[60*ci +: 21]);
                    acc_int32 = {{11{acc_lane[20]}}, acc_lane};
                    sc_lane = out_scale[36*ci +: 9];

                    fc = fire_cnt_per_col[ci];
                    n_t_dec = fc / (K_T * M_T * TILE_SIZE);                  // /512
                    k_t_dec = (fc / (M_T * TILE_SIZE)) % K_T;                // /128 %4
                    m_t_dec = (fc / TILE_SIZE) % M_T;                        // /32 %4
                    m_in_dec = fc % TILE_SIZE;                               // %32

                    m_g = m_t_dec * TILE_SIZE + m_in_dec;
                    n_g = n_t_dec * TILE_SIZE + ci;
                    flat = m_g * N_DIM + n_g;

                    fifo_int  [fifo_wp] <= acc_int32;
                    fifo_scale[fifo_wp] <= sc_lane;
                    fifo_addr [fifo_wp] <= flat[18:0];
                    fifo_wp = fifo_wp + 1;

                    fire_cnt_per_col[ci] <= fire_cnt_per_col[ci] + 1;
                end
            end
        end
    end

    // ─── DRIVE: 5-chain MXP driving for A8W8 ───────────────────────
    integer n_t, k_t, m_t, o, oo, cc, cyc;
    integer cyc_in_K, cyc_global, m_idx_scale;
    integer b_idx_cur, b_idx_next;     // index into b_bp[] for current and prefetch
    integer s_idx_cur, s_idx_next;     // index into b_scale[] (note: b_scale is 1 byte/col)
    reg     cur_pp;       // PE in_control toggle state (Buf1/Buf2)
    reg     cur_st_pp;    // Station selector toggle state
    reg     drive_started;

    // Compute address into a_bs[] for k_t, m_t, o (o = bit-serial cycle within m-row)
    function [31:0] a_bs_word;
        input integer k_t_i, m_t_i, o_i;
        begin
            a_bs_word = a_bs[k_t_i * M_T * TILE_SIZE * W_CYC + m_t_i * TILE_SIZE * W_CYC + o_i];
        end
    endfunction

    // Pack a 32-bit word with the same byte 4 times (for A8 mode in_Scale_Activation)
    function [31:0] pack_scale_4x;
        input [7:0] sc;
        begin
            pack_scale_4x = {sc, sc, sc, sc};
        end
    endfunction

    // ─── main sequence ─────────────────────────────────────────────
    integer i, b;
    reg [8*512-1:0] dump_path;
    reg [8*512-1:0] in_path_a_bs, in_path_b, in_path_a_sc, in_path_b_sc;

    initial begin
        fifo_wp = 0;
        fifo_rp = 0;
        // PE in_control: 0 during Stage 2-A (load PE_naive.station1 via in_loadEN=1),
        // then flip to 1 before MAC so MAC reads station1 (`station = in_control ?
        // station1 : station2`).
        cur_pp     = 1'b0;
        cur_st_pp  = 1'b0;
        drive_started = 1'b0;

        // ───── INIT ─────
        #20;
        rst = 0;
        sram_D_use_zero = 1'b1;
        // SRAM zero-prime flat 0..16383
        for (i = 0; i < 16384; i = i + 1) begin
            sram_write_zero(i[18:0]);
        end
        sram_idle;
        sram_idle;

        // ───── LOAD ─────
        $sformat(in_path_a_bs, "%0s/hw_input/a_input_BS_mxint%0d.hex", WORK_DIR, A_PREC);
        $sformat(in_path_b,    "%0s/hw_input/b_input_mxint%0d.hex",    WORK_DIR, B_PREC);
        $sformat(in_path_a_sc, "%0s/hw_input/a_scale_mxint%0d.hex",    WORK_DIR, A_PREC);
        $sformat(in_path_b_sc, "%0s/hw_input/b_scale_mxint%0d.hex",    WORK_DIR, B_PREC);
        $readmemh(in_path_a_bs, a_bs);
        $readmemh(in_path_b,    b_bp);
        $readmemh(in_path_a_sc, a_scale);
        $readmemh(in_path_b_sc, b_scale);
        $display("LOAD OK: a_bs[0]=%h b_bp[0]=%h a_scale[0]=%h b_scale[0]=%h",
                 a_bs[0], b_bp[0], a_scale[0], b_scale[0]);

        // ───── CONFIG (idle hold for chain settle) ─────
        in_Station_control  <= A_IDLE;
        in_Wcontrol         <= W_IDLE;
        in_loadEN           <= 0;
        in_station_loadEN   <= 0;
        in_station_control  <= 1'b0;
        in_control          <= 0;
        in_start_accumulate <= 0;
        in_a                <= 0;
        in_b                <= 0;
        in_Scale_Activation <= 0;
        in_scale_weight     <= 0;
        @(posedge clk); @(posedge clk); @(posedge clk); @(posedge clk);

        // ───── DRIVE ─────
        // Stage 2-A: initial B load for (n_t=0, k_t=0). 32 cycles.
        //   in_b cycles through columns of (n_t=0, k_t=0) tile. The b_bp file
        //   layout: outer N_t → K_t → col (32 lines per tile). So for tile
        //   (n_t=0, k_t=0), lines [0..31] are col 31, 30, ... 0 — no, actually
        //   emit_b_input writes col 0..31 in order. But the chain enters at
        //   col 31 leftward, so the first 256-bit word driven goes to col 31,
        //   the last to col 0. We must drive col 31 first → match b_bp index.
        //
        //   Looking at emit_b_input: it writes block[:, col_idx] for col_idx
        //   in 0..31. So b_bp[n_t*K_T*32 + k_t*32 + 0] = col 0 of tile.
        //   But chain delivers in col-31-first order. So we need to drive
        //   col 31 first.
        //
        //   Actually no — the chain has 31-cycle leftward delay. If we drive
        //   in_b = (col 31's data) at TB cyc 0, then col 31 latches at cyc 0,
        //   col 30 at cyc 1, ... col 0 at cyc 31. So in_b at TB cyc t should
        //   be the column intended for col c = 31 - t. Conversely, if we
        //   drive in_b[t] = b_bp[col_idx = 31 - t], then col_idx will arrive
        //   at col c such that (31 - c) = t → c = 31 - t = col_idx. So col c
        //   gets b_bp[col c]. ✓
        //
        //   Wait, this assumes one-to-one col mapping. Let me re-check: the
        //   chain registers between cols add 1 cycle each. in_b at TB cyc 0
        //   reaches the col-31 station at cyc 0 (registered there at cyc 1).
        //   Then it propagates leftward at 1 cy/col. So the *data we put on
        //   in_b at TB cyc 0* eventually reaches col c at cyc (31-c).
        //
        //   The intent: load tile col c's data into station col c. So at TB
        //   cyc t = 31-c, we drive in_b = data for col c. Thus at TB cyc t=0
        //   we drive data for col c=31; t=1 → c=30; ...; t=31 → c=0.
        //
        //   This means: drive b_bp[n_t*128 + k_t*32 + (31 - t)] at TB cyc t.
        //   Same for scale_act.

        // (drive starts here — we'll loop over n_t, k_t with a single shared
        //  driver: Stage 2-A only for first tile, prefetch for subsequent)
        drive_b_loop_init(0, 0);           // Initial B load → Buf1 (Stage 2-A)

        // Stage 2-B settle (~32 cy)
        in_Station_control  <= A_IDLE;
        in_loadEN           <= 0;
        in_station_loadEN   <= 0;
        in_b                <= 0;
        in_Scale_Activation <= 0;
        // 16 cy idle, then toggle station_control to 1 (broadcast Buf1)
        for (i = 0; i < 16; i = i + 1) @(posedge clk);
        in_station_control  <= 1'b1;
        cur_st_pp = 1'b1;
        for (i = 0; i < 16; i = i + 1) @(posedge clk);

        // Stage 3+4: MAC sweep — n_t → k_t → m_t → o
        // start_accumulate must pulse at cyc_global == 17. We track cyc_global.
        cyc_global = 0;

        // Flip PE in_control to 1 so MAC reads PE_naive.station1 (the buffer
        // we loaded during Stage 2-A with in_control=0+in_loadEN=1).
        cur_pp = 1'b1;

        // Enable fire capture only now, AFTER Stage 2-A/2-B finished. Any
        // out_fire pulses during those 64 idle cycles are spurious (the
        // Accumulator hasn't been started yet but the chain may have junk).
        capture_en <= 1'b1;

        for (n_t = 0; n_t < N_T_LOGICAL; n_t = n_t + 1) begin
            for (k_t = 0; k_t < K_T; k_t = k_t + 1) begin
                for (m_t = 0; m_t < M_T; m_t = m_t + 1) begin
                    for (o = 0; o < TILE_SIZE * W_CYC; o = o + 1) begin
                        cyc_in_K = m_t * TILE_SIZE * W_CYC + o;

                        // Weight bit-serial: distribute the 32-bit word as 1 bit per row
                        // a_bs_word(k_t, m_t, o) packs 32 rows into [31:0], bit r = row r's bit.
                        in_a <= a_bs_word(k_t, m_t, o);

                        // start_accumulate pulse (only first n_t, first k_t, cyc_global == 17)
                        if (n_t == 0 && k_t == 0 && cyc_global == 17) begin
                            in_start_accumulate <= 1'b1;
                        end else begin
                            in_start_accumulate <= 1'b0;
                        end

                        // Wcontrol: W_IDLE before cyc_global=17, then W_INT8
                        if (n_t == 0 && k_t == 0 && cyc_global < 17) begin
                            in_Wcontrol <= W_IDLE;
                        end else begin
                            in_Wcontrol <= W_INT8;
                        end

                        // PE in_control toggle at K-tile boundary (cyc_in_K == 0, not first tile)
                        if (cyc_in_K == 0 && !(n_t == 0 && k_t == 0)) begin
                            cur_pp = ~cur_pp;
                        end
                        in_control <= {32{cur_pp}};

                        // Station selector toggle at cyc_in_K == TOGGLE_VAL, not first K-tile
                        if (cyc_in_K == TOGGLE_VAL && !(n_t == 0 && k_t == 0)) begin
                            cur_st_pp = ~cur_st_pp;
                        end
                        in_station_control <= cur_st_pp;

                        // scale_weight drive at cyc_global == FIRST_FIRE + W_CYC * m
                        // m_idx = (cyc_global - FIRST_FIRE) / W_CYC
                        // m_idx wraps mod (K_T * M_T * TILE_SIZE) = 512 for A8W8
                        if (cyc_global >= FIRST_FIRE_GLOBAL &&
                            ((cyc_global - FIRST_FIRE_GLOBAL) % W_CYC) == 0) begin
                            m_idx_scale = ((cyc_global - FIRST_FIRE_GLOBAL) / W_CYC)
                                          % (K_T * M_T * TILE_SIZE);
                            in_scale_weight <= a_scale[m_idx_scale];
                        end

                        // Prefetch (m_t == 1, o < 32): load next (n_t', k_t') tile into opposite buffer
                        if (m_t == 1 && o < TILE_SIZE) begin
                            // Determine next tile (n_t', k_t')
                            // Loop order: n_t outer → k_t inner. So:
                            //   if k_t < K_T-1: next = (n_t, k_t+1)
                            //   elif n_t < N_T_LOGICAL-1: next = (n_t+1, 0)
                            //   else: no more tiles, skip prefetch
                            if (k_t < K_T - 1) begin
                                drive_prefetch(n_t, k_t + 1, o);
                            end else if (n_t < N_T_LOGICAL - 1) begin
                                drive_prefetch(n_t + 1, 0, o);
                            end else begin
                                // no prefetch — idle
                                in_Station_control  <= A_IDLE;
                                in_loadEN           <= 0;
                                in_station_loadEN   <= 0;
                                in_b                <= 0;
                                in_Scale_Activation <= 0;
                            end
                        end else begin
                            // Not in prefetch window
                            in_Station_control  <= A_IDLE;
                            in_loadEN           <= 0;
                            in_station_loadEN   <= 0;
                            in_b                <= 0;
                            in_Scale_Activation <= 0;
                        end

                        @(posedge clk);
                        cyc_global = cyc_global + 1;
                    end
                end
            end
        end

        // ───── Stage 5: tail — hold W_INT8 just long enough to flush last legit fires.
        // Last fire on col c: cyc_global = LAST_DRIVE_INCR + (FIRST_FIRE - 1) + |c-16|
        // For col 0/31: extra (FIRST_FIRE - 1) + 15 = 27 + 15 = 42 cy past MAC end.
        // Use 45 cy to be safe, then immediately go W_IDLE to stop spurious fires.
        in_Wcontrol         <= W_INT8;
        in_a                <= 0;
        in_Station_control  <= A_IDLE;
        in_loadEN           <= 0;
        in_station_loadEN   <= 0;
        in_b                <= 0;
        in_Scale_Activation <= 0;
        for (i = 0; i < 45; i = i + 1) begin
            if (cyc_global >= FIRST_FIRE_GLOBAL &&
                ((cyc_global - FIRST_FIRE_GLOBAL) % W_CYC) == 0) begin
                m_idx_scale = ((cyc_global - FIRST_FIRE_GLOBAL) / W_CYC)
                              % (K_T * M_T * TILE_SIZE);
                in_scale_weight <= a_scale[m_idx_scale];
            end
            @(posedge clk);
            cyc_global = cyc_global + 1;
        end

        // Disable Wcontrol immediately, but let capture continue ~20 cy for
        // the last col 0/31 fires registered into out_fire.
        in_Wcontrol <= W_IDLE;
        for (i = 0; i < 20; i = i + 1) @(posedge clk);
        capture_en <= 1'b0;

        $display("DRIVE DONE: captured %0d fires (expected 65536)", fifo_wp);

        // ───── DRAIN: serial RMW dispatch ─────
        // Pipeline priming: do a few dummy reads of addr 0 (zero-primed) to
        // flush any X-values out of Q_pre, RMW's sram_dly, and fp32_adder's
        // recFN_b_dly chain. Without this, the first RMW's rmw_out_RMW is X
        // because in_SRAM was never driven to a known value during DRIVE.
        rmw_in_GEMM <= 32'h00000000;
        rmw_scale   <= 9'd127;          // scale=127 → 2^0 = 1, so 0 dequant = 0
        for (i = 0; i < 16; i = i + 1) begin
            @(posedge clk);
            sram_CEB <= 1'b0;
            sram_WEB <= 1'b1;
            sram_A   <= 19'd0;
        end
        @(posedge clk);
        sram_CEB <= 1'b1;
        sram_WEB <= 1'b1;

        sram_D_use_zero <= 1'b0;
        for (fifo_rp = 0; fifo_rp < fifo_wp; fifo_rp = fifo_rp + 1) begin
            // Step 1: Issue SRAM READ at cycle T
            @(posedge clk);
            sram_CEB        <= 1'b0;
            sram_WEB        <= 1'b1;   // read
            sram_A          <= fifo_addr[fifo_rp];
            sram_D_use_zero <= 1'b0;

            // Step 2: Drive RMW inputs the SAME cycle (T+1) when sram_Q is valid.
            // RMW's sram_dly chain (L_CONV=2 long) will pipeline in_SRAM to align with fp_a.
            // We need in_GEMM / scale to enter int_to_fp32 at the cycle when in_SRAM
            // also enters sram_dly[0]. RMW samples in_SRAM at the same edge it samples
            // int_to_fp32 internal regs. So: at cycle T+1 (after read issued at T),
            //   - sram_Q = prev_psum (combinational from leaf, PIPELINE=0)
            //   - drive rmw_in_GEMM, rmw_scale
            //   - RMW internal regs sample at T+1 edge
            @(posedge clk);
            sram_CEB        <= 1'b1;   // deassert read
            rmw_in_GEMM     <= fifo_int  [fifo_rp];
            rmw_scale       <= fifo_scale[fifo_rp];

            // Step 3: wait for RMW pipeline (L_CONV+L_ADD=5) with conservative
            // slack. Total 9 @s between step 2's body and the write @.
            @(posedge clk); @(posedge clk); @(posedge clk); @(posedge clk);
            @(posedge clk); @(posedge clk); @(posedge clk); @(posedge clk);

            // Step 4: issue WRITE — sram_D = rmw_out_RMW
            @(posedge clk);
            sram_CEB        <= 1'b0;
            sram_WEB        <= 1'b0;
            sram_A          <= fifo_addr[fifo_rp];
            sram_WMASK      <= 32'hFFFFFFFF;
            sram_D_use_zero <= 1'b0;     // use RMW output

            // Step 5: deassert + 1 extra idle cy for SRAM write to settle.
            @(posedge clk);
            sram_CEB        <= 1'b1;
            sram_WEB        <= 1'b1;
            @(posedge clk);
        end
        sram_idle;
        sram_idle;

        $display("DRAIN DONE: %0d entries processed", fifo_wp);

        // ───── DUMP ─────
        for (b = 0; b < 16; b = b + 1) begin
            $sformat(dump_path, "%0s/bank%0d.mem", DUMP_DIR, b);
            case (b)
                0:  $writememh(dump_path, u_top.u_sram.g_bank[0].u_bank.mem,  0, 1023);
                1:  $writememh(dump_path, u_top.u_sram.g_bank[1].u_bank.mem,  0, 1023);
                2:  $writememh(dump_path, u_top.u_sram.g_bank[2].u_bank.mem,  0, 1023);
                3:  $writememh(dump_path, u_top.u_sram.g_bank[3].u_bank.mem,  0, 1023);
                4:  $writememh(dump_path, u_top.u_sram.g_bank[4].u_bank.mem,  0, 1023);
                5:  $writememh(dump_path, u_top.u_sram.g_bank[5].u_bank.mem,  0, 1023);
                6:  $writememh(dump_path, u_top.u_sram.g_bank[6].u_bank.mem,  0, 1023);
                7:  $writememh(dump_path, u_top.u_sram.g_bank[7].u_bank.mem,  0, 1023);
                8:  $writememh(dump_path, u_top.u_sram.g_bank[8].u_bank.mem,  0, 1023);
                9:  $writememh(dump_path, u_top.u_sram.g_bank[9].u_bank.mem,  0, 1023);
                10: $writememh(dump_path, u_top.u_sram.g_bank[10].u_bank.mem, 0, 1023);
                11: $writememh(dump_path, u_top.u_sram.g_bank[11].u_bank.mem, 0, 1023);
                12: $writememh(dump_path, u_top.u_sram.g_bank[12].u_bank.mem, 0, 1023);
                13: $writememh(dump_path, u_top.u_sram.g_bank[13].u_bank.mem, 0, 1023);
                14: $writememh(dump_path, u_top.u_sram.g_bank[14].u_bank.mem, 0, 1023);
                15: $writememh(dump_path, u_top.u_sram.g_bank[15].u_bank.mem, 0, 1023);
            endcase
        end

        $display("INTEGRATION TB: A%0d_B%0d DONE", A_PREC, B_PREC);
        $finish;
    end

    // ─── helper: drive 32-cy B/scale load with leftward chain compensation ───
    task drive_b_loop_init;
        input integer n_t_arg;
        input integer k_t_arg;
        integer tt, col_target, b_idx, sc_idx;
        reg [255:0] b_word;
        reg [7:0]   sc_byte;
        begin
            // Chain semantics (station.v): with loadEN=1, each station overwrites
            // its Buf1 each cycle from chain input, and forwards OLD Buf1 to
            // downstream. Solving recurrence: buf(c, 31) = TB_drive(c). So drive
            // col c's value at TB cy c (col 0 first, col 31 last).
            for (tt = 0; tt < TILE_SIZE; tt = tt + 1) begin
                col_target = tt;             // col 0 first, col 31 last
                b_idx  = n_t_arg * K_T * TILE_SIZE + k_t_arg * TILE_SIZE + col_target;
                sc_idx = b_idx;              // b_scale has same layout as b_input (one byte/col)
                b_word  = b_bp[b_idx];
                sc_byte = b_scale[sc_idx];

                in_b                <= b_word;
                in_Scale_Activation <= pack_scale_4x(sc_byte);
                in_Station_control  <= A_INT8;
                in_loadEN           <= 32'hFFFFFFFF;
                in_station_loadEN   <= 1'b1;
                in_station_control  <= 1'b0;       // target Buf1
                @(posedge clk);
            end
        end
    endtask

    // ─── helper: prefetch — drive one cycle of the next tile's B / scale ───
    // Called inside MAC loop at m_t==1, o<32. The chain target is determined by
    // current selector state — opposite of in_station_control.
    task drive_prefetch;
        input integer n_t_next;
        input integer k_t_next;
        input integer o_arg;        // 0..31 within the 32-cy prefetch window
        integer col_target, b_idx;
        reg [255:0] b_word;
        reg [7:0]   sc_byte;
        begin
            // Same chain semantics as drive_b_loop_init: col c at TB cy c.
            col_target = o_arg;
            b_idx  = n_t_next * K_T * TILE_SIZE + k_t_next * TILE_SIZE + col_target;
            b_word  = b_bp[b_idx];
            sc_byte = b_scale[b_idx];
            in_b                <= b_word;
            in_Scale_Activation <= pack_scale_4x(sc_byte);
            in_Station_control  <= A_INT8;
            in_loadEN           <= 32'hFFFFFFFF;
            in_station_loadEN   <= 1'b1;
            // in_station_control stays at whatever the MAC loop is driving (cur_st_pp)
            //   → write target = opposite of broadcast (handled inside station.v)
        end
    endtask

    // ─── timeout safety ────────────────────────────────────────────
    initial begin
        #50_000_000;   // 50 ms = 5_000_000 cycles @ 100 MHz
        $display("ERROR: timeout");
        $finish;
    end

endmodule
