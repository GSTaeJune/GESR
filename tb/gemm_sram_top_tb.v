`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_tb — Task 8: end-to-end GEMM + RMW + SRAM TB, all 9 modes.
//
// Supports A_PREC ∈ {2,4,8} × B_PREC ∈ {2,4,8} via plusargs. Per-mode
// constants (W_CYC, A_FIRE_DELAY, FIRST_FIRE_GLOBAL, TOGGLE_VAL, FIRES_PER_COL,
// N_T_LOGICAL, A_CTRL, W_CTRL) are dispatched from a CONFIG-time `case()` on
// {A_PREC,B_PREC} into TB regs (no longer localparam — plusarg-dependent).
//
// Strategy (single-RMW-at-a-time, post-drive replay):
//   1) INIT     — reset + SRAM zero-prime via TB-driven do_write loop.
//   2) LOAD     — $readmemh hex files (a_input_BS, b_input, a_scale, b_scale).
//   3) CONFIG   — set per-mode constants + control codes from {A_PREC,B_PREC}.
//   4) DRIVE    — Stages 2-A, 2-B, 3+4 (MAC), 5 (tail) via named tasks.
//                 During driving an `always @(posedge clk)` capture-block
//                 records every out_fire[c] rising edge into FIFO arrays:
//                 fifo_int[i], fifo_scale[i], fifo_addr[i] (15-bit flat).
//                 Capture decode is mode-aware: A8 → 1 RMW/fire,
//                 A4 → 2 RMW/fire, A2 → 4 RMW/fire (lane-to-c-mapping.md).
//   5) DRAIN    — sequential RMW dispatch over the captured FIFO. For each
//                 entry: issue SRAM read → wait L_CONV cyc → drive rmw inputs
//                 → wait L_ADD cyc → write back.
//   6) DUMP     — $writememh all 16 banks (0..1023 word range).
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8
//   B_PREC    : 2 / 4 / 8
//   WORK_DIR  : MXP_Tools .hex input directory (REQUIRED)
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

    // ─── GEMM size constants (128×128×128, K_T=M_T=4, TILE=32) ─────
    localparam integer M_DIM = 128, K_DIM = 128, N_DIM = 128;
    localparam integer TILE_SIZE = 32;
    localparam integer M_T = M_DIM / TILE_SIZE;     // 4
    localparam integer K_T = K_DIM / TILE_SIZE;     // 4

    // Encodings — match precision_modes_protocol.md §1
    //   A_INT8 = 2'b00, A_INT4 = 2'b01, A_INT2 = 2'b10, A_IDLE = 2'b11
    //   W_INT8 = 2'b11, W_INT4 = 2'b10, W_INT2 = 2'b01, W_IDLE = 2'b00
    localparam [1:0] A_INT8 = 2'b00, A_INT4 = 2'b01, A_INT2 = 2'b10, A_IDLE = 2'b11;
    localparam [1:0] W_INT8 = 2'b11, W_INT4 = 2'b10, W_INT2 = 2'b01, W_IDLE = 2'b00;

    // ─── Per-mode constants (assigned at CONFIG from {A_PREC,B_PREC}) ─
    integer W_CYC, A_FIRE_DELAY, FIRST_FIRE_GLOBAL, TOGGLE_VAL;
    integer FIRES_PER_COL, N_T_LOGICAL;
    reg [1:0] A_CTRL_CODE, W_CTRL_CODE;

    // ─── RMW pipeline timing helpers ───────────────────────────────
    // RMW latency = L_CONV + L_ADD = 2 + 3 = 5 cy. We wait 8 cy between
    // driving rmw inputs and the write-back @(posedge clk):
    //   = L_CONV + L_ADD + 3 cy slack (covers any signal-prop quirks).
    localparam integer RMW_WAIT_CYC = 8;
    // X-prime: number of pipeline-flush cycles before draining the FIFO. Must
    // exceed the deepest pipeline (RMW's sram_dly + fp32_adder's recFN_b_dly).
    localparam integer PRIME_CYC    = 16;
    // Tail length (Stage 5): hold W_INTn long enough to flush in-flight fires.
    // Worst case = symmetric chain delay (15 cy max at col 0/31) +
    // A_fire_delay (≤2) + W_CYC (≤8). 45 cy is generous for all 9 modes.
    localparam integer TAIL_CYC     = 45;
    // Post-tail capture window: lets last col 0/31 fires register through.
    localparam integer POST_TAIL    = 20;

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

    // ─── helper tasks (SRAM) ───────────────────────────────────────
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
    // sizes: K_T*M_T*TILE*W_CYC_MAX = 4*4*32*8 = 4096 lines for a_bs (A8 worst)
    //        N_T_MAX*K_T*TILE       = 4*4*32   = 512  lines for b_bp (A8 worst)
    //        K_T*M_T*TILE           = 4*4*32   = 512  lines for a_scale
    //        N_T_MAX*K_T*TILE       = 4*4*32   = 512  lines for b_scale (A8 worst)
    reg [31:0]   a_bs    [0:4095];
    reg [255:0]  b_bp    [0:511];
    reg [7:0]    a_scale [0:511];
    reg [7:0]    b_scale [0:511];

    // ─── Fire capture FIFO ──────────────────────────────────────────
    // Total RMW entries (mode-independent): 32 col × 2048 RMW/col = 65536
    //   A8: 2048 fires × 1 RMW = 2048/col
    //   A4: 1024 fires × 2 RMW = 2048/col
    //   A2:  512 fires × 4 RMW = 2048/col
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
    // n_global depends on A_PREC:
    //   A8: n_g = n_t * 32 + col_idx           (1 lane)
    //   A4: n_g_top = n_pair*64 + 32 + col,    (2 lanes, n_pair = n_t)
    //       n_g_bot = n_pair*64 +  0 + col
    //   A2: n_g_l0 = 96 + col, n_g_l1 = 64 + col, n_g_l2 = 32 + col, n_g_l3 = col
    // flat = m_global * N_DIM + n_global
    integer fc, n_t_dec, k_t_dec, m_t_dec, m_in_dec, m_g;
    integer ci;
    reg signed [20:0] acc_lane_a8;
    reg signed [17:0] acc_lane_a4t, acc_lane_a4b;
    reg signed [14:0] acc_lane_a2_0, acc_lane_a2_1, acc_lane_a2_2, acc_lane_a2_3;
    reg signed [31:0] acc_int32;
    reg [8:0]         sc_lane;
    integer ng_dec, flat_dec;

    reg capture_en;     // 1 during DRIVE stage
    initial capture_en = 0;

    always @(posedge clk) begin
        if (rst) begin
            for (ci = 0; ci < 32; ci = ci + 1) fire_cnt_per_col[ci] <= 0;
        end else if (capture_en) begin
            for (ci = 0; ci < 32; ci = ci + 1) begin
                // Ignore spurious fires beyond FIRES_PER_COL (extra Accumulator
                // counter rollovers in Stage 5 tail before W_IDLE propagates).
                // FIRES_PER_COL is mode-dependent (A8=2048, A4=1024, A2=512).
                if (out_fire[ci] && (fire_cnt_per_col[ci] < FIRES_PER_COL)) begin
                    fc = fire_cnt_per_col[ci];

                    // Inner counter wraps mod (K_T*M_T*TILE) = 512.
                    // Above that, the "n_t" index advances. For A8 N_T=4 → fc∈[0,2048).
                    // For A4 N_T=2 → fires_per_col=1024, fc∈[0,1024).
                    // For A2 N_T=1 → fires_per_col=512,  fc∈[0,512).
                    n_t_dec  = fc / (K_T * M_T * TILE_SIZE);                 // /512
                    k_t_dec  = (fc / (M_T * TILE_SIZE)) % K_T;               // /128 %4
                    m_t_dec  = (fc / TILE_SIZE) % M_T;                       // /32  %4
                    m_in_dec = fc % TILE_SIZE;                               // %32
                    m_g      = m_t_dec * TILE_SIZE + m_in_dec;

                    case (A_PREC)
                        // ─── A8: 1 lane / col ─────────────────────────
                        8: begin
                            acc_lane_a8 = $signed(out_accumulate[60*ci +: 21]);
                            acc_int32   = {{11{acc_lane_a8[20]}}, acc_lane_a8};
                            sc_lane     = out_scale[36*ci +: 9];
                            ng_dec      = n_t_dec * 32 + ci;
                            flat_dec    = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;
                        end
                        // ─── A4: 2 lanes / col (top/bot) ──────────────
                        4: begin
                            // top (s1_a): acc[60c+35:60c+18], scale[36c+17:36c+9]
                            //   n = n_pair*64 + 32 + j
                            acc_lane_a4t = $signed(out_accumulate[60*ci+18 +: 18]);
                            acc_int32    = {{14{acc_lane_a4t[17]}}, acc_lane_a4t};
                            sc_lane      = out_scale[36*ci+9 +: 9];
                            ng_dec       = n_t_dec * 64 + 32 + ci;
                            flat_dec     = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // bot (s1_b): acc[60c+17:60c+0], scale[36c+8:36c+0]
                            //   n = n_pair*64 + 0 + j
                            acc_lane_a4b = $signed(out_accumulate[60*ci +: 18]);
                            acc_int32    = {{14{acc_lane_a4b[17]}}, acc_lane_a4b};
                            sc_lane      = out_scale[36*ci +: 9];
                            ng_dec       = n_t_dec * 64 + 0 + ci;
                            flat_dec     = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;
                        end
                        // ─── A2: 4 lanes / col ────────────────────────
                        2: begin
                            // lane0 (out_INT2_0): acc[60c+59:60c+45], scale[36c+35:36c+27], n = 96+j
                            acc_lane_a2_0 = $signed(out_accumulate[60*ci+45 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_0[14]}}, acc_lane_a2_0};
                            sc_lane       = out_scale[36*ci+27 +: 9];
                            ng_dec        = 96 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // lane1: acc[60c+44:60c+30], scale[36c+26:36c+18], n = 64+j
                            acc_lane_a2_1 = $signed(out_accumulate[60*ci+30 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_1[14]}}, acc_lane_a2_1};
                            sc_lane       = out_scale[36*ci+18 +: 9];
                            ng_dec        = 64 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // lane2: acc[60c+29:60c+15], scale[36c+17:36c+9], n = 32+j
                            acc_lane_a2_2 = $signed(out_accumulate[60*ci+15 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_2[14]}}, acc_lane_a2_2};
                            sc_lane       = out_scale[36*ci+9 +: 9];
                            ng_dec        = 32 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // lane3: acc[60c+14:60c+0], scale[36c+8:36c+0], n = j
                            acc_lane_a2_3 = $signed(out_accumulate[60*ci +: 15]);
                            acc_int32     = {{17{acc_lane_a2_3[14]}}, acc_lane_a2_3};
                            sc_lane       = out_scale[36*ci +: 9];
                            ng_dec        = 0 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;
                        end
                    endcase

                    fire_cnt_per_col[ci] <= fire_cnt_per_col[ci] + 1;
                end
            end
        end
    end

    // ─── DRIVE-stage scratch regs ──────────────────────────────────
    integer n_t, k_t, m_t, o;
    integer cyc_in_K, cyc_global, m_idx_scale;
    reg     cur_pp;       // PE in_control toggle state (Buf1/Buf2)
    reg     cur_st_pp;    // Station selector toggle state

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

    // V3 pack helper: combine 2 file 256-bit blocks (top + bot INT4 N-cols)
    //   into 1 station 256-bit. Per row r:  byte[r] = {top[r][3:0], bot[r][3:0]}
    // (mirrors MXP/sim_1/new/tb_A4W8.v::pack_int4_n_pair)
    function [255:0] pack_int4_n_pair;
        input [255:0] top_data;   // file (n_t = 2*n_pair+1) col c
        input [255:0] bot_data;   // file (n_t = 2*n_pair+0) col c
        integer rr;
        begin
            pack_int4_n_pair = 0;
            for (rr = 0; rr < 32; rr = rr + 1)
                pack_int4_n_pair[rr*8 +: 8] =
                    {top_data[rr*8 +: 4],
                     bot_data[rr*8 +: 4]};
        end
    endfunction

    // V3 pack helper: combine 4 file 256-bit blocks (n_t=0..3) into 1 station 256-bit.
    //   Per row r: byte[r] = {n3[r][1:0], n2[r][1:0], n1[r][1:0], n0[r][1:0]}
    //   bit pos:  [7:6] [5:4] [3:2] [1:0]
    //   lane:     lane0 lane1 lane2 lane3
    //   N-pos:    N+96  N+64  N+32  N+0   (per SA col c)
    // (mirrors MXP/sim_1/new/tb_A2W8.v::pack_int2_n_quad)
    function [255:0] pack_int2_n_quad;
        input [255:0] data_n3;   // file n_t=3 → lane0 [7:6]
        input [255:0] data_n2;   // file n_t=2 → lane1 [5:4]
        input [255:0] data_n1;   // file n_t=1 → lane2 [3:2]
        input [255:0] data_n0;   // file n_t=0 → lane3 [1:0]
        integer rr;
        begin
            pack_int2_n_quad = 0;
            for (rr = 0; rr < 32; rr = rr + 1)
                pack_int2_n_quad[rr*8 +: 8] =
                    {data_n3[rr*8 +: 2],
                     data_n2[rr*8 +: 2],
                     data_n1[rr*8 +: 2],
                     data_n0[rr*8 +: 2]};
        end
    endfunction

    // Per-mode in_b builder. n_pair is the LOGICAL n-tile index in driving
    // (0..N_T_LOGICAL-1). Maps to file (file_n_t) indices internally:
    //   A8: file_n_t = n_pair (1:1).
    //   A4: file_n_t ∈ {2*n_pair+1, 2*n_pair} (top, bot pair).
    //   A2: file_n_t ∈ {0,1,2,3} (all four, regardless of n_pair which is 0 only).
    function [255:0] build_in_b;
        input integer n_pair;
        input integer k_t_arg;
        input integer col_target;
        integer base;
        begin
            case (A_PREC)
                8: build_in_b = b_bp[n_pair * K_T * TILE_SIZE + k_t_arg * TILE_SIZE + col_target];
                4: begin
                    base = k_t_arg * TILE_SIZE + col_target;
                    build_in_b = pack_int4_n_pair(
                        b_bp[(2*n_pair+1) * K_T * TILE_SIZE + base],
                        b_bp[(2*n_pair+0) * K_T * TILE_SIZE + base]);
                end
                2: begin
                    base = k_t_arg * TILE_SIZE + col_target;
                    build_in_b = pack_int2_n_quad(
                        b_bp[3 * K_T * TILE_SIZE + base],
                        b_bp[2 * K_T * TILE_SIZE + base],
                        b_bp[1 * K_T * TILE_SIZE + base],
                        b_bp[0 * K_T * TILE_SIZE + base]);
                end
                default: build_in_b = 0;
            endcase
        end
    endfunction

    // Per-mode in_Scale_Activation builder.
    //   A8: {sc, sc, sc, sc} (broadcast same byte to all 4 lanes).
    //   A4: {16'b0, scale[2*n_pair+0], scale[2*n_pair+1]}
    //        — byte[7:0]=top (file_n_t=2*n_pair+1, lane0/1 → comb_s0)
    //          byte[15:8]=bot (file_n_t=2*n_pair+0, lane2/3 → comb_s1)
    //   A2: {scale[file_n_t=0], scale[1], scale[2], scale[3]}
    //        — byte[31:24]=file_n_t=0 → lane3 → comb_s3 (N+0)
    //          byte[23:16]=file_n_t=1 → lane2 → comb_s2 (N+32)
    //          byte[15:8] =file_n_t=2 → lane1 → comb_s1 (N+64)
    //          byte[7:0]  =file_n_t=3 → lane0 → comb_s0 (N+96)
    function [31:0] build_in_scale_act;
        input integer n_pair;
        input integer k_t_arg;
        input integer col_target;
        integer base;
        begin
            case (A_PREC)
                8: build_in_scale_act = pack_scale_4x(
                    b_scale[n_pair * K_T * TILE_SIZE + k_t_arg * TILE_SIZE + col_target]);
                4: begin
                    base = k_t_arg * TILE_SIZE + col_target;
                    build_in_scale_act = {
                        16'b0,
                        b_scale[(2*n_pair+0) * K_T * TILE_SIZE + base],  // bot → [15:8]
                        b_scale[(2*n_pair+1) * K_T * TILE_SIZE + base]   // top → [7:0]
                    };
                end
                2: begin
                    base = k_t_arg * TILE_SIZE + col_target;
                    build_in_scale_act = {
                        b_scale[0 * K_T * TILE_SIZE + base],  // [31:24] file_n_t=0 → lane3
                        b_scale[1 * K_T * TILE_SIZE + base],  // [23:16] file_n_t=1 → lane2
                        b_scale[2 * K_T * TILE_SIZE + base],  // [15:8]  file_n_t=2 → lane1
                        b_scale[3 * K_T * TILE_SIZE + base]   // [7:0]   file_n_t=3 → lane0
                    };
                end
                default: build_in_scale_act = 0;
            endcase
        end
    endfunction

    // ─── DRIVE-helper tasks ─────────────────────────────────────────

    // Stage 2-A initial B load (32 cy): drive col c on TB cy c so that, after
    // station chain leftward propagation, each station's Buf1 holds its own col's
    // activation. Same logic for all A modes — file layout (per b_input_mxint{A})
    // already encodes the lane packing in the 256-bit word.
    task drive_stage_2a;
        input integer n_t_arg, k_t_arg;
        integer tt, col_target;
        begin
            for (tt = 0; tt < TILE_SIZE; tt = tt + 1) begin
                col_target = tt;             // col 0 first, col 31 last
                in_b                <= build_in_b(n_t_arg, k_t_arg, col_target);
                in_Scale_Activation <= build_in_scale_act(n_t_arg, k_t_arg, col_target);
                in_Station_control  <= A_CTRL_CODE;
                in_loadEN           <= 32'hFFFFFFFF;
                in_station_loadEN   <= 1'b1;
                in_station_control  <= 1'b0;       // target Buf1
                @(posedge clk);
            end
        end
    endtask

    // Stage 2-B settle (16 cy idle + flip station selector to 1 + 16 cy settle).
    // After this, broadcast = Buf1 on all 32 cols (sym chain delay ≤15 cy).
    task drive_stage_2b;
        integer i;
        begin
            in_Station_control  <= A_IDLE;
            in_loadEN           <= 0;
            in_station_loadEN   <= 0;
            in_b                <= 0;
            in_Scale_Activation <= 0;
            for (i = 0; i < 16; i = i + 1) @(posedge clk);
            in_station_control  <= 1'b1;
            cur_st_pp = 1'b1;
            for (i = 0; i < 16; i = i + 1) @(posedge clk);
        end
    endtask

    // Stage 3+4: MAC sweep (n_t → k_t → m_t → o). Per-cycle drive of 6 signals
    // (weight bit / PE pp / Station pp / start_acc / Wcontrol / scale_weight) +
    // prefetch of next tile during m_t==1, o<32.
    task drive_stage_3_4;
        begin
            cyc_global = 0;
            cur_pp = 1'b1;       // Flip PE in_control so MAC reads station1
            capture_en <= 1'b1;

            for (n_t = 0; n_t < N_T_LOGICAL; n_t = n_t + 1) begin
                for (k_t = 0; k_t < K_T; k_t = k_t + 1) begin
                    for (m_t = 0; m_t < M_T; m_t = m_t + 1) begin
                        for (o = 0; o < TILE_SIZE * W_CYC; o = o + 1) begin
                            cyc_in_K = m_t * TILE_SIZE * W_CYC + o;

                            // Weight bit-serial: 32 rows × 1 bit per cycle
                            in_a <= a_bs_word(k_t, m_t, o);

                            // start_accumulate pulse (only once at cyc_global==17)
                            if (n_t == 0 && k_t == 0 && cyc_global == 17) begin
                                in_start_accumulate <= 1'b1;
                            end else begin
                                in_start_accumulate <= 1'b0;
                            end

                            // Wcontrol: W_IDLE before cyc_global=17, then W_CTRL_CODE
                            if (n_t == 0 && k_t == 0 && cyc_global < 17) begin
                                in_Wcontrol <= W_IDLE;
                            end else begin
                                in_Wcontrol <= W_CTRL_CODE;
                            end

                            // PE in_control toggle at K-tile boundary (cyc_in_K==0,
                            // not first tile)
                            if (cyc_in_K == 0 && !(n_t == 0 && k_t == 0)) begin
                                cur_pp = ~cur_pp;
                            end
                            in_control <= {32{cur_pp}};

                            // Station selector toggle at cyc_in_K == TOGGLE_VAL, not first K-tile
                            if (cyc_in_K == TOGGLE_VAL && !(n_t == 0 && k_t == 0)) begin
                                cur_st_pp = ~cur_st_pp;
                            end
                            in_station_control <= cur_st_pp;

                            // scale_weight drive at cyc_global == FIRST_FIRE + W_CYC*m
                            if (cyc_global >= FIRST_FIRE_GLOBAL &&
                                ((cyc_global - FIRST_FIRE_GLOBAL) % W_CYC) == 0) begin
                                m_idx_scale = ((cyc_global - FIRST_FIRE_GLOBAL) / W_CYC)
                                              % (K_T * M_T * TILE_SIZE);
                                in_scale_weight <= a_scale[m_idx_scale];
                            end

                            // Prefetch (m_t == 1, o < 32): load next tile into
                            // opposite station buffer.
                            if (m_t == 1 && o < TILE_SIZE) begin
                                if (k_t < K_T - 1) begin
                                    drive_prefetch(n_t, k_t + 1, o);
                                end else if (n_t < N_T_LOGICAL - 1) begin
                                    drive_prefetch(n_t + 1, 0, o);
                                end else begin
                                    in_Station_control  <= A_IDLE;
                                    in_loadEN           <= 0;
                                    in_station_loadEN   <= 0;
                                    in_b                <= 0;
                                    in_Scale_Activation <= 0;
                                end
                            end else begin
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
        end
    endtask

    // Stage 5 tail: hold W_CTRL_CODE for TAIL_CYC cy to flush last in-flight
    // fires (col 0/31 last fires arrive ~|c-16|=15 cy after MAC end + W_CYC
    // + A_fire_delay). Then drop to W_IDLE; capture stays open POST_TAIL cy
    // to register the trailing fires before we close it.
    task drive_stage_5_tail;
        integer i;
        begin
            in_Wcontrol         <= W_CTRL_CODE;
            in_a                <= 0;
            in_Station_control  <= A_IDLE;
            in_loadEN           <= 0;
            in_station_loadEN   <= 0;
            in_b                <= 0;
            in_Scale_Activation <= 0;
            for (i = 0; i < TAIL_CYC; i = i + 1) begin
                if (cyc_global >= FIRST_FIRE_GLOBAL &&
                    ((cyc_global - FIRST_FIRE_GLOBAL) % W_CYC) == 0) begin
                    m_idx_scale = ((cyc_global - FIRST_FIRE_GLOBAL) / W_CYC)
                                  % (K_T * M_T * TILE_SIZE);
                    in_scale_weight <= a_scale[m_idx_scale];
                end
                @(posedge clk);
                cyc_global = cyc_global + 1;
            end

            in_Wcontrol <= W_IDLE;
            for (i = 0; i < POST_TAIL; i = i + 1) @(posedge clk);
            capture_en <= 1'b0;
        end
    endtask

    // ─── prefetch — drive one cycle of the next tile's B / scale ───
    // Called inside MAC loop at m_t==1, o<32. The chain target is determined by
    // current selector state — opposite of in_station_control.
    task drive_prefetch;
        input integer n_t_next;
        input integer k_t_next;
        input integer o_arg;        // 0..31 within the 32-cy prefetch window
        integer col_target;
        begin
            col_target = o_arg;
            in_b                <= build_in_b(n_t_next, k_t_next, col_target);
            in_Scale_Activation <= build_in_scale_act(n_t_next, k_t_next, col_target);
            in_Station_control  <= A_CTRL_CODE;
            in_loadEN           <= 32'hFFFFFFFF;
            in_station_loadEN   <= 1'b1;
        end
    endtask

    // ─── DRAIN helper: prime the RMW + SRAM pipelines before real entries ──
    // Without this, the first RMW's rmw_out_RMW is X because RMW's sram_dly
    // and fp32_adder's recFN_b_dly were never driven to a known value during DRIVE.
    task drain_prime;
        integer i;
        begin
            rmw_in_GEMM <= 32'h00000000;
            rmw_scale   <= 9'd127;          // scale=127 → 2^0=1, so 0 dequant = 0
            for (i = 0; i < PRIME_CYC; i = i + 1) begin
                @(posedge clk);
                sram_CEB <= 1'b0;
                sram_WEB <= 1'b1;
                sram_A   <= 19'd0;
            end
            @(posedge clk);
            sram_CEB <= 1'b1;
            sram_WEB <= 1'b1;
        end
    endtask

    // ─── DRAIN body: sequential RMW dispatch over the captured FIFO. ──
    task drain_fifo;
        integer i;
        begin
            sram_D_use_zero <= 1'b0;
            for (fifo_rp = 0; fifo_rp < fifo_wp; fifo_rp = fifo_rp + 1) begin
                // Step 1: Issue SRAM READ
                @(posedge clk);
                sram_CEB        <= 1'b0;
                sram_WEB        <= 1'b1;
                sram_A          <= fifo_addr[fifo_rp];
                sram_D_use_zero <= 1'b0;

                // Step 2: Drive RMW inputs the next cycle (sram_Q valid).
                @(posedge clk);
                sram_CEB        <= 1'b1;
                rmw_in_GEMM     <= fifo_int  [fifo_rp];
                rmw_scale       <= fifo_scale[fifo_rp];
                // Step 3: wait for RMW pipeline (= L_CONV+L_ADD+slack=8 cy)
                for (i = 0; i < RMW_WAIT_CYC; i = i + 1) @(posedge clk);

                // Step 4: issue WRITE
                @(posedge clk);
                sram_CEB        <= 1'b0;
                sram_WEB        <= 1'b0;
                sram_A          <= fifo_addr[fifo_rp];
                sram_WMASK      <= 32'hFFFFFFFF;
                sram_D_use_zero <= 1'b0;

                // Step 5: deassert + 1 extra idle cy for SRAM write to settle
                @(posedge clk);
                sram_CEB        <= 1'b1;
                sram_WEB        <= 1'b1;
                @(posedge clk);
            end
            sram_idle;
            sram_idle;
        end
    endtask

    // ─── main sequence ─────────────────────────────────────────────
    integer i, b;
    reg [8*512-1:0] dump_path;
    reg [8*512-1:0] in_path_a_bs, in_path_b, in_path_a_sc, in_path_b_sc;

    initial begin
        fifo_wp = 0;
        fifo_rp = 0;
        cur_pp     = 1'b0;
        cur_st_pp  = 1'b0;

        // ───── INIT ─────
        #20;
        rst = 0;
        sram_D_use_zero = 1'b1;
        for (i = 0; i < 16384; i = i + 1) begin
            sram_write_zero(i[18:0]);
        end
        sram_idle;
        sram_idle;

        // ───── LOAD ─────
        // Naming convention (matches MXP_Tools emit + MXP reference TBs):
        //   - mem_a (in TB) = WEIGHT (bit-serial, fed to SA `in_a`).
        //     File `a_input_BS_mxint{P}.hex` where P = WEIGHT precision = B_PREC.
        //   - mem_b (in TB) = ACTIVATION (byte-parallel, fed to SA `in_b`).
        //     File `b_input_mxint{P}.hex` where P = ACTIVATION precision = A_PREC.
        //   - a_scale, b_scale follow same convention.
        // (Task 7 worked only because A_PREC == B_PREC == 8 made the swap invisible.)
        $sformat(in_path_a_bs, "%0s/hw_input/a_input_BS_mxint%0d.hex", WORK_DIR, B_PREC);
        $sformat(in_path_b,    "%0s/hw_input/b_input_mxint%0d.hex",    WORK_DIR, A_PREC);
        $sformat(in_path_a_sc, "%0s/hw_input/a_scale_mxint%0d.hex",    WORK_DIR, B_PREC);
        $sformat(in_path_b_sc, "%0s/hw_input/b_scale_mxint%0d.hex",    WORK_DIR, A_PREC);
        $readmemh(in_path_a_bs, a_bs);
        $readmemh(in_path_b,    b_bp);
        $readmemh(in_path_a_sc, a_scale);
        $readmemh(in_path_b_sc, b_scale);
        $display("LOAD OK: a_bs[0]=%h b_bp[0]=%h a_scale[0]=%h b_scale[0]=%h",
                 a_bs[0], b_bp[0], a_scale[0], b_scale[0]);

        // ───── CONFIG: per-mode constants from {A_PREC, B_PREC} ─────
        //
        // Source: precision_modes_protocol.md §1 Mode Matrix (v1.0, validated 2026-05-11).
        // Notation: A∈{8,4,2} → A_INT8/4/2 (Mode_oh in Accumulator_Col),
        //           W∈{8,4,2} → cnt rollover threshold (Accumulator internal).
        //
        // !!! KEEP IN SYNC: any change to these per-mode values must also update
        // precision_modes_protocol.md §1 / §3. Drift between this case and the
        // protocol doc was the source of Task 8's TOGGLE_VAL (A8W4, A4W8) bug.
        // Re-run `bash sim/run_integration_sweep.sh` after any edit — 9/9 PASS
        // is the regression gate.
        case ({A_PREC[3:0], B_PREC[3:0]})
            // A8 row: A_FIRE_DELAY=2, N_T_LOGICAL=4, FIRES_PER_COL=2048
            {4'd8, 4'd8}: begin
                W_CYC=8; A_FIRE_DELAY=2; FIRST_FIRE_GLOBAL=28; TOGGLE_VAL=24;
                FIRES_PER_COL=2048; N_T_LOGICAL=4;
                A_CTRL_CODE=A_INT8; W_CTRL_CODE=W_INT8;
            end
            {4'd8, 4'd4}: begin
                W_CYC=4; A_FIRE_DELAY=2; FIRST_FIRE_GLOBAL=24; TOGGLE_VAL=22;
                FIRES_PER_COL=2048; N_T_LOGICAL=4;
                A_CTRL_CODE=A_INT8; W_CTRL_CODE=W_INT4;
            end
            {4'd8, 4'd2}: begin
                W_CYC=2; A_FIRE_DELAY=2; FIRST_FIRE_GLOBAL=22; TOGGLE_VAL=21;
                FIRES_PER_COL=2048; N_T_LOGICAL=4;
                A_CTRL_CODE=A_INT8; W_CTRL_CODE=W_INT2;
            end
            // A4 row: A_FIRE_DELAY=1, N_T_LOGICAL=2, FIRES_PER_COL=1024
            {4'd4, 4'd8}: begin
                W_CYC=8; A_FIRE_DELAY=1; FIRST_FIRE_GLOBAL=27; TOGGLE_VAL=23;
                FIRES_PER_COL=1024; N_T_LOGICAL=2;
                A_CTRL_CODE=A_INT4; W_CTRL_CODE=W_INT8;
            end
            {4'd4, 4'd4}: begin
                W_CYC=4; A_FIRE_DELAY=1; FIRST_FIRE_GLOBAL=23; TOGGLE_VAL=21;
                FIRES_PER_COL=1024; N_T_LOGICAL=2;
                A_CTRL_CODE=A_INT4; W_CTRL_CODE=W_INT4;
            end
            {4'd4, 4'd2}: begin
                W_CYC=2; A_FIRE_DELAY=1; FIRST_FIRE_GLOBAL=21; TOGGLE_VAL=20;
                FIRES_PER_COL=1024; N_T_LOGICAL=2;
                A_CTRL_CODE=A_INT4; W_CTRL_CODE=W_INT2;
            end
            // A2 row: A_FIRE_DELAY=0, N_T_LOGICAL=1, FIRES_PER_COL=512
            {4'd2, 4'd8}: begin
                W_CYC=8; A_FIRE_DELAY=0; FIRST_FIRE_GLOBAL=26; TOGGLE_VAL=22;
                FIRES_PER_COL=512; N_T_LOGICAL=1;
                A_CTRL_CODE=A_INT2; W_CTRL_CODE=W_INT8;
            end
            {4'd2, 4'd4}: begin
                W_CYC=4; A_FIRE_DELAY=0; FIRST_FIRE_GLOBAL=22; TOGGLE_VAL=20;
                FIRES_PER_COL=512; N_T_LOGICAL=1;
                A_CTRL_CODE=A_INT2; W_CTRL_CODE=W_INT4;
            end
            {4'd2, 4'd2}: begin
                W_CYC=2; A_FIRE_DELAY=0; FIRST_FIRE_GLOBAL=20; TOGGLE_VAL=19;
                FIRES_PER_COL=512; N_T_LOGICAL=1;
                A_CTRL_CODE=A_INT2; W_CTRL_CODE=W_INT2;
            end
            default: begin
                $display("ERROR: unsupported (A_PREC=%0d, B_PREC=%0d)", A_PREC, B_PREC);
                $finish;
            end
        endcase

        $display("CONFIG: A=%0d W=%0d  W_CYC=%0d A_FIRE_DELAY=%0d FIRST_FIRE=%0d TOGGLE=%0d FIRES_PER_COL=%0d N_T_LOGICAL=%0d",
                 A_PREC, B_PREC, W_CYC, A_FIRE_DELAY, FIRST_FIRE_GLOBAL, TOGGLE_VAL,
                 FIRES_PER_COL, N_T_LOGICAL);

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
        drive_stage_2a(0, 0);          // Stage 2-A: initial B load (n_t=0, k_t=0)
        drive_stage_2b;                // Stage 2-B: settle + Buf1 toggle
        drive_stage_3_4;               // Stage 3+4: MAC sweep (also enables capture_en)
        drive_stage_5_tail;            // Stage 5: tail + close capture

        $display("DRIVE DONE: captured %0d fires (expected up to 65536)", fifo_wp);

        // ───── DRAIN ─────
        drain_prime;
        drain_fifo;

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

    // ─── timeout safety ────────────────────────────────────────────
    initial begin
        #50_000_000;   // 50 ms = 5_000_000 cycles @ 100 MHz
        $display("ERROR: timeout");
        $finish;
    end

endmodule
