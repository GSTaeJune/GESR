`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_tb — gemm_sram_top 통합 검증 TB (9 모드 전부).
//
// 검증 목적:
//   GEMM (MXP 32×32 bit-serial systolic) + RMW (INT→FP32 + FP32 add) +
//   sram_1rw_banked (16 bank INTERLEAVED, FP32 저장) 의 end-to-end 데이터
//   패스가 9 가지 정밀도 조합 (A∈{2,4,8} × W∈{2,4,8}) 모두에서 MXP_Tools
//   골든 GEMM 과 bit-exact 일치하는지 확인. 128×128×128 행렬 누적 결과
//   16384 element 를 16 bank .mem dump 로 뱉어서 외부 Python (`compare`)
//   이 FP32 비트 단위 비교.
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8 (Activation 정밀도)
//   B_PREC    : 2 / 4 / 8 (Weight 정밀도, MXP 의 "in_a" 라인)
//   WORK_DIR  : MXP_Tools hex 입력 디렉토리 (필수)
//   DUMP_DIR  : SRAM .mem dump 출력 디렉토리 (필수)
//
// 모드별 상수 (W_CYC, A_FIRE_DELAY, FIRST_FIRE_GLOBAL, TOGGLE_VAL,
// FIRES_PER_COL, N_T_LOGICAL, A_CTRL, W_CTRL) 는 CONFIG 단계의 case() 로
// {A_PREC,B_PREC} 에 따라 TB reg 에 디스패치 (plusarg 의존이라 localparam X).
//
// 전체 흐름 (single-RMW 직렬 디스패치 방식, drive→replay 후 처리):
//   1) INIT     — 리셋 + SRAM 16384 워드 0 초기화 (do_write 루프).
//   2) LOAD     — $readmemh (a_input_BS, b_input, a_scale, b_scale).
//   3) CONFIG   — {A_PREC,B_PREC} 으로 모드별 상수/제어 코드 세팅.
//   4) DRIVE    — Stage 2-A, 2-B, 3+4 (MAC), 5 (tail) 을 named task 로 발사.
//                 발사 중 `always @(posedge clk)` 캡처 블록이 매 out_fire[c]
//                 rising 을 잡아서 FIFO (fifo_int/scale/addr) 로 적재.
//                 lane 디코딩은 모드 의존 (A8 → 1 RMW/fire, A4 → 2,
//                 A2 → 4). 자세한 슬라이스는 docs/superpowers/notes/
//                 lane-to-c-mapping.md.
//   5) DRAIN    — FIFO 를 순차로 한 개씩 RMW 로 흘림. 매 entry 마다:
//                 SRAM read 발사 → L_CONV 대기 → rmw 입력 드라이브 →
//                 L_ADD 대기 → write-back.
//   6) DUMP     — $writememh 로 16 bank 각각 0..1023 워드를 .mem 파일로.
//                 외부 compare 가 MXP_Tools golden npz 와 비트 단위 비교.
//
// 회귀 게이트: `bash sim/run_integration_sweep.sh` → "ALL 9 MODES PASSED".
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

    // 제어 코드 인코딩 — precision_modes_protocol.md §1 과 일치.
    //   A_INT8 = 2'b00, A_INT4 = 2'b01, A_INT2 = 2'b10, A_IDLE = 2'b11
    //   W_INT8 = 2'b11, W_INT4 = 2'b10, W_INT2 = 2'b01, W_IDLE = 2'b00
    localparam [1:0] A_INT8 = 2'b00, A_INT4 = 2'b01, A_INT2 = 2'b10, A_IDLE = 2'b11;
    localparam [1:0] W_INT8 = 2'b11, W_INT4 = 2'b10, W_INT2 = 2'b01, W_IDLE = 2'b00;

    // ─── 모드별 상수 (CONFIG 에서 {A_PREC,B_PREC} 으로 case 디스패치) ─
    integer W_CYC, A_FIRE_DELAY, FIRST_FIRE_GLOBAL, TOGGLE_VAL;
    integer FIRES_PER_COL, N_T_LOGICAL;
    reg [1:0] A_CTRL_CODE, W_CTRL_CODE;

    // ─── RMW 파이프라인 타이밍 보조 상수 ──────────────────────────
    // RMW latency = L_CONV + L_ADD = 2 + 3 = 5 cy. RMW 입력 드라이브 후
    // write-back 까지 8 cy 대기:
    //   = L_CONV + L_ADD + 3 cy slack (신호 전파 여유분).
    localparam integer RMW_WAIT_CYC = 8;
    // X-prime: drain 시작 전 파이프라인 X 채움 cycle. 가장 깊은 파이프라인
    // (RMW.sram_dly + fp32_adder.recFN_b_dly) 보다 길어야 함.
    localparam integer PRIME_CYC    = 16;
    // Stage 5 tail 길이: in-flight fire 들을 모두 캡처할 때까지 W_INTn 유지.
    // 최악 = symmetric chain delay (col 0/31 에서 15 cy 최대) +
    // A_fire_delay (≤2) + W_CYC (≤8). 45 cy 면 9 모드 모두 여유 있음.
    localparam integer TAIL_CYC     = 45;
    // Stage 5 후 capture 유지 cycle: 마지막 col 0/31 fire 가 reg 에 들어올 때까지.
    localparam integer POST_TAIL    = 20;

    // ─── 클록 / 리셋 ───────────────────────────────────────────────
    reg clk = 0;
    always #5 clk = ~clk;        // 100 MHz 기준 (10 ns 주기)
    reg rst = 1;

    // ─── DUT 자극 레지스터 ─────────────────────────────────────────
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

    // ─── DUT 인스턴스 ──────────────────────────────────────────────
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

    // ─── SRAM 보조 task ────────────────────────────────────────────
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

    // ─── 입력 hex 메모리 ───────────────────────────────────────────
    // 크기 산정 (A8 worst case 기준):
    //   K_T*M_T*TILE*W_CYC_MAX = 4*4*32*8 = 4096 lines  → a_bs   (weight bit-serial)
    //   N_T_MAX*K_T*TILE       = 4*4*32   = 512  lines  → b_bp   (activation, 256-bit)
    //   K_T*M_T*TILE           = 4*4*32   = 512  lines  → a_scale (weight scale, E8M0)
    //   N_T_MAX*K_T*TILE       = 4*4*32   = 512  lines  → b_scale (activation scale, E8M0)
    reg [31:0]   a_bs    [0:4095];
    reg [255:0]  b_bp    [0:511];
    reg [7:0]    a_scale [0:511];
    reg [7:0]    b_scale [0:511];

    // ─── Fire 캡처 FIFO ────────────────────────────────────────────
    // 총 RMW entry 수 (모드 무관, 32 col × 2048 RMW/col = 65536):
    //   A8: 2048 fires × 1 RMW = 2048/col
    //   A4: 1024 fires × 2 RMW = 2048/col
    //   A2:  512 fires × 4 RMW = 2048/col
    reg [31:0]   fifo_int   [0:65535];
    reg [8:0]    fifo_scale [0:65535];
    reg [18:0]   fifo_addr  [0:65535];
    integer      fifo_wp;          // write pointer (capture 시 증가)
    integer      fifo_rp;          // read pointer  (drain 시 증가)

    // col 별 누적 fire 카운터 (col 안에서 몇 번째 fire 인지 → m_t/m_in/k_t/n_t 디코딩에 사용)
    reg [11:0]   fire_cnt_per_col [0:31];

    // ─── Fire 캡처 always 블록 ─────────────────────────────────────
    //
    // fire_cnt[c] (col c 에서의 누적 fire 번호) → (n_t, k_t, m_t, m_in) 디코딩:
    //   driving 의 inner 루프 순서:  n_t → k_t → m_t → o (W_CYC × TILE_SIZE)
    //   col 별 fire 는 (n_t, k_t, m_t, m_in) 한 조합당 1 회 발생, m_in ∈ [0, TILE_SIZE)
    //   따라서 fc = n_t*K_T*M_T*TILE + k_t*M_T*TILE + m_t*TILE + m_in
    //
    // m_global = m_t * TILE_SIZE + m_in
    // n_global 은 A_PREC 의존:
    //   A8: n_g = n_t * 32 + col_idx           (1 lane)
    //   A4: n_g_top = n_pair*64 + 32 + col,    (2 lanes, n_pair = n_t)
    //       n_g_bot = n_pair*64 +  0 + col
    //   A2: n_g_l0 = 96 + col, n_g_l1 = 64 + col, n_g_l2 = 32 + col, n_g_l3 = col
    // flat = m_global * N_DIM + n_global   (FP32 word 의 SRAM 평탄 주소)
    integer fc, n_t_dec, k_t_dec, m_t_dec, m_in_dec, m_g;
    integer ci;
    reg signed [20:0] acc_lane_a8;
    reg signed [17:0] acc_lane_a4t, acc_lane_a4b;
    reg signed [14:0] acc_lane_a2_0, acc_lane_a2_1, acc_lane_a2_2, acc_lane_a2_3;
    reg signed [31:0] acc_int32;
    reg [8:0]         sc_lane;
    integer ng_dec, flat_dec;

    reg capture_en;     // DRIVE stage 동안만 1
    initial capture_en = 0;

    always @(posedge clk) begin
        if (rst) begin
            for (ci = 0; ci < 32; ci = ci + 1) fire_cnt_per_col[ci] <= 0;
        end else if (capture_en) begin
            for (ci = 0; ci < 32; ci = ci + 1) begin
                // FIRES_PER_COL 초과 fire 는 무시. Stage 5 tail 에서 W_IDLE 이
                // 전파되기 전 Accumulator 의 counter 가 한두 번 더 rollover 해서
                // 가짜 fire 가 뜨는 경우가 있음. FIRES_PER_COL 은 모드 의존
                // (A8=2048, A4=1024, A2=512).
                if (out_fire[ci] && (fire_cnt_per_col[ci] < FIRES_PER_COL)) begin
                    fc = fire_cnt_per_col[ci];

                    // 내부 카운터는 mod (K_T*M_T*TILE) = 512 로 wrap.
                    // 그 이상부터 "n_t" 가 증가. A8 (N_T=4) 에서 fc ∈ [0,2048),
                    // A4 (N_T=2) 에서 fires_per_col=1024 → fc ∈ [0,1024),
                    // A2 (N_T=1) 에서 fires_per_col=512  → fc ∈ [0,512).
                    n_t_dec  = fc / (K_T * M_T * TILE_SIZE);                 // /512
                    k_t_dec  = (fc / (M_T * TILE_SIZE)) % K_T;               // /128 %4
                    m_t_dec  = (fc / TILE_SIZE) % M_T;                       // /32  %4
                    m_in_dec = fc % TILE_SIZE;                               // %32
                    m_g      = m_t_dec * TILE_SIZE + m_in_dec;

                    case (A_PREC)
                        // ─── A8: col 당 1 lane ──────────────────────────
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
                        // ─── A4: col 당 2 lane (top/bot) ────────────────
                        4: begin
                            // top lane (s1_a): acc[60c+35:60c+18], scale[36c+17:36c+9]
                            //   n 위치 = n_pair*64 + 32 + col_idx
                            acc_lane_a4t = $signed(out_accumulate[60*ci+18 +: 18]);
                            acc_int32    = {{14{acc_lane_a4t[17]}}, acc_lane_a4t};
                            sc_lane      = out_scale[36*ci+9 +: 9];
                            ng_dec       = n_t_dec * 64 + 32 + ci;
                            flat_dec     = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // bot lane (s1_b): acc[60c+17:60c+0], scale[36c+8:36c+0]
                            //   n 위치 = n_pair*64 + 0 + col_idx
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
                        // ─── A2: col 당 4 lane ──────────────────────────
                        2: begin
                            // lane0 (out_INT2_0): acc[60c+59:60c+45], scale[36c+35:36c+27], n 위치 = 96+col_idx
                            acc_lane_a2_0 = $signed(out_accumulate[60*ci+45 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_0[14]}}, acc_lane_a2_0};
                            sc_lane       = out_scale[36*ci+27 +: 9];
                            ng_dec        = 96 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // lane1: acc[60c+44:60c+30], scale[36c+26:36c+18], n 위치 = 64+col_idx
                            acc_lane_a2_1 = $signed(out_accumulate[60*ci+30 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_1[14]}}, acc_lane_a2_1};
                            sc_lane       = out_scale[36*ci+18 +: 9];
                            ng_dec        = 64 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // lane2: acc[60c+29:60c+15], scale[36c+17:36c+9], n 위치 = 32+col_idx
                            acc_lane_a2_2 = $signed(out_accumulate[60*ci+15 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_2[14]}}, acc_lane_a2_2};
                            sc_lane       = out_scale[36*ci+9 +: 9];
                            ng_dec        = 32 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            fifo_int  [fifo_wp] <= acc_int32;
                            fifo_scale[fifo_wp] <= sc_lane;
                            fifo_addr [fifo_wp] <= flat_dec[18:0];
                            fifo_wp = fifo_wp + 1;

                            // lane3: acc[60c+14:60c+0], scale[36c+8:36c+0], n 위치 = col_idx
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

    // ─── DRIVE 스크래치 reg ────────────────────────────────────────
    integer n_t, k_t, m_t, o;
    integer cyc_in_K, cyc_global, m_idx_scale;
    reg     cur_pp;       // PE in_control 토글 상태 (Buf1/Buf2)
    reg     cur_st_pp;    // Station selector 토글 상태

    // a_bs[] 에서 (k_t, m_t, o) 위치 워드 추출 (o = m-row 안의 bit-serial cycle)
    function [31:0] a_bs_word;
        input integer k_t_i, m_t_i, o_i;
        begin
            a_bs_word = a_bs[k_t_i * M_T * TILE_SIZE * W_CYC + m_t_i * TILE_SIZE * W_CYC + o_i];
        end
    endfunction

    // 32-bit 워드에 같은 바이트를 4 번 복제 (A8 모드의 in_Scale_Activation 용)
    function [31:0] pack_scale_4x;
        input [7:0] sc;
        begin
            pack_scale_4x = {sc, sc, sc, sc};
        end
    endfunction

    // V3 pack: file 256-bit 블록 2 개 (top + bot INT4 N-col) 를 station 256-bit
    // 1 개로 합침. row r 당: byte[r] = {top[r][3:0], bot[r][3:0]}
    // (MXP/sim_1/new/tb_A4W8.v::pack_int4_n_pair 와 동일 동작)
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

    // V3 pack: file 256-bit 블록 4 개 (n_t=0..3) 를 station 256-bit 1 개로 합침.
    //   row r 당: byte[r] = {n3[r][1:0], n2[r][1:0], n1[r][1:0], n0[r][1:0]}
    //   bit pos:  [7:6] [5:4] [3:2] [1:0]
    //   lane:     lane0 lane1 lane2 lane3
    //   N-pos:    N+96  N+64  N+32  N+0   (per SA col c)
    // (MXP/sim_1/new/tb_A2W8.v::pack_int2_n_quad 와 동일 동작)
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

    // 모드별 in_b 빌더. n_pair 는 driving 의 LOGICAL n-tile 인덱스
    // (0..N_T_LOGICAL-1). 내부에서 file (file_n_t) 인덱스로 매핑:
    //   A8: file_n_t = n_pair (1:1)
    //   A4: file_n_t ∈ {2*n_pair+1, 2*n_pair} (top, bot pair 묶음)
    //   A2: file_n_t ∈ {0,1,2,3} (n_pair=0 만 있고, 4 lane 모두 채움)
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

    // 모드별 in_Scale_Activation 빌더.
    //   A8: {sc, sc, sc, sc} (4 lane 모두에 같은 바이트 broadcast)
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

    // ─── DRIVE 보조 task ────────────────────────────────────────────

    // Stage 2-A 초기 B load (32 cy): TB cy c 에 col c 를 드라이브.
    // station chain 이 왼쪽으로 전파되므로, 끝나면 각 col 의 station Buf1 에
    // 자기 col 의 activation 이 들어가 있게 됨. A 모드별 차이는 file layout
    // (b_input_mxint{A_PREC}.hex) 의 256-bit lane packing 으로 흡수됨.
    task drive_stage_2a;
        input integer n_t_arg, k_t_arg;
        integer tt, col_target;
        begin
            for (tt = 0; tt < TILE_SIZE; tt = tt + 1) begin
                col_target = tt;             // col 0 먼저, col 31 마지막
                in_b                <= build_in_b(n_t_arg, k_t_arg, col_target);
                in_Scale_Activation <= build_in_scale_act(n_t_arg, k_t_arg, col_target);
                in_Station_control  <= A_CTRL_CODE;
                in_loadEN           <= 32'hFFFFFFFF;
                in_station_loadEN   <= 1'b1;
                in_station_control  <= 1'b0;       // Buf1 타겟
                @(posedge clk);
            end
        end
    endtask

    // Stage 2-B settle (16 cy idle + station selector 를 1 로 flip + 16 cy settle).
    // 끝나면 32 col 모두 Buf1 broadcast 상태 (sym chain delay ≤ 15 cy 흡수).
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

    // Stage 3+4: MAC sweep (n_t → k_t → m_t → o). 매 사이클 6 개 신호 드라이브
    // (weight bit / PE pp / Station pp / start_acc / Wcontrol / scale_weight) +
    // m_t==1, o<32 구간에서 다음 tile B/scale prefetch.
    task drive_stage_3_4;
        begin
            cyc_global = 0;
            cur_pp = 1'b1;       // PE in_control flip → MAC 이 station1 을 읽도록
            capture_en <= 1'b1;

            for (n_t = 0; n_t < N_T_LOGICAL; n_t = n_t + 1) begin
                for (k_t = 0; k_t < K_T; k_t = k_t + 1) begin
                    for (m_t = 0; m_t < M_T; m_t = m_t + 1) begin
                        for (o = 0; o < TILE_SIZE * W_CYC; o = o + 1) begin
                            cyc_in_K = m_t * TILE_SIZE * W_CYC + o;

                            // Weight bit-serial: 매 cycle 32 row × 1 bit
                            in_a <= a_bs_word(k_t, m_t, o);

                            // start_accumulate 펄스 (cyc_global==17 에서 단 1 번)
                            if (n_t == 0 && k_t == 0 && cyc_global == 17) begin
                                in_start_accumulate <= 1'b1;
                            end else begin
                                in_start_accumulate <= 1'b0;
                            end

                            // Wcontrol: cyc_global=17 전엔 W_IDLE, 이후 W_CTRL_CODE
                            if (n_t == 0 && k_t == 0 && cyc_global < 17) begin
                                in_Wcontrol <= W_IDLE;
                            end else begin
                                in_Wcontrol <= W_CTRL_CODE;
                            end

                            // PE in_control toggle: K-tile 경계 (cyc_in_K==0) 에서 flip.
                            // 단, 첫 K-tile 은 제외.
                            if (cyc_in_K == 0 && !(n_t == 0 && k_t == 0)) begin
                                cur_pp = ~cur_pp;
                            end
                            in_control <= {32{cur_pp}};

                            // Station selector toggle: cyc_in_K == TOGGLE_VAL 에서 flip
                            // (첫 K-tile 은 이미 Stage 2-B 에서 설정했으므로 제외).
                            if (cyc_in_K == TOGGLE_VAL && !(n_t == 0 && k_t == 0)) begin
                                cur_st_pp = ~cur_st_pp;
                            end
                            in_station_control <= cur_st_pp;

                            // scale_weight 드라이브: cyc_global == FIRST_FIRE + W_CYC*m 인 cycle 마다
                            if (cyc_global >= FIRST_FIRE_GLOBAL &&
                                ((cyc_global - FIRST_FIRE_GLOBAL) % W_CYC) == 0) begin
                                m_idx_scale = ((cyc_global - FIRST_FIRE_GLOBAL) / W_CYC)
                                              % (K_T * M_T * TILE_SIZE);
                                in_scale_weight <= a_scale[m_idx_scale];
                            end

                            // Prefetch (m_t == 1, o < 32 구간): 다음 tile B/scale 을
                            // 반대편 station buffer 로 미리 load.
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

    // Stage 5 tail: W_CTRL_CODE 를 TAIL_CYC cy 동안 유지해서 마지막 in-flight
    // fire 들을 모두 비워냄 (col 0/31 의 마지막 fire 는 MAC 종료 후
    // ~|c-16|=15 cy + W_CYC + A_fire_delay 뒤에 도착).
    // 그 다음 W_IDLE 로 떨어뜨리고, capture 는 POST_TAIL cy 더 열어둬서
    // 마지막 trailing fire 들을 reg 에 들어오게 함.
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

    // ─── prefetch — 다음 tile B/scale 의 한 cycle 분 드라이브 ───────
    // MAC 루프의 m_t==1, o<32 구간에서 호출. 타겟 chain 은 현재 selector
    // 상태와 반대편 (in_station_control 반대편) buffer.
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

    // ─── DRAIN 보조: 실제 entry 전 RMW + SRAM 파이프라인 X 채움 ────
    // 이걸 안 하면 첫 RMW 의 rmw_out_RMW 가 X. DRIVE 중에는 RMW.sram_dly 와
    // fp32_adder.recFN_b_dly 가 한 번도 알려진 값으로 채워진 적이 없기 때문.
    task drain_prime;
        integer i;
        begin
            rmw_in_GEMM <= 32'h00000000;
            rmw_scale   <= 9'd127;          // scale=127 → 2^0=1 → 0 dequant 결과 = 0
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

    // ─── DRAIN 본체: FIFO 를 한 entry 씩 순차 RMW 디스패치 ─────────
    task drain_fifo;
        integer i;
        begin
            sram_D_use_zero <= 1'b0;
            for (fifo_rp = 0; fifo_rp < fifo_wp; fifo_rp = fifo_rp + 1) begin
                // Step 1: SRAM READ 발사
                @(posedge clk);
                sram_CEB        <= 1'b0;
                sram_WEB        <= 1'b1;
                sram_A          <= fifo_addr[fifo_rp];
                sram_D_use_zero <= 1'b0;

                // Step 2: 다음 cycle 에 RMW 입력 드라이브 (sram_Q 가 유효해진 시점)
                @(posedge clk);
                sram_CEB        <= 1'b1;
                rmw_in_GEMM     <= fifo_int  [fifo_rp];
                rmw_scale       <= fifo_scale[fifo_rp];
                // Step 3: RMW 파이프라인 대기 (L_CONV+L_ADD+slack = 8 cy)
                for (i = 0; i < RMW_WAIT_CYC; i = i + 1) @(posedge clk);

                // Step 4: WRITE 발사 (rmw_out_w → sram_D 경로로 그대로 들어감)
                @(posedge clk);
                sram_CEB        <= 1'b0;
                sram_WEB        <= 1'b0;
                sram_A          <= fifo_addr[fifo_rp];
                sram_WMASK      <= 32'hFFFFFFFF;
                sram_D_use_zero <= 1'b0;

                // Step 5: deassert + SRAM write settle 용 idle cy 1 개
                @(posedge clk);
                sram_CEB        <= 1'b1;
                sram_WEB        <= 1'b1;
                @(posedge clk);
            end
            sram_idle;
            sram_idle;
        end
    endtask

    // ─── 메인 시퀀스 (INIT → LOAD → CONFIG → DRIVE → DRAIN → DUMP) ──
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
        // 파일명 규칙 (MXP_Tools emit + MXP 레퍼런스 TB 와 일치):
        //   - mem_a (TB 내) = WEIGHT (bit-serial, SA `in_a` 라인).
        //     파일 `a_input_BS_mxint{P}.hex`, P = WEIGHT 정밀도 = B_PREC.
        //   - mem_b (TB 내) = ACTIVATION (byte-parallel, SA `in_b` 라인).
        //     파일 `b_input_mxint{P}.hex`, P = ACTIVATION 정밀도 = A_PREC.
        //   - a_scale, b_scale 도 같은 컨벤션.
        // (Task 7 시 A_PREC == B_PREC == 8 라서 swap 버그가 보이지 않았던 부분.)
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

        // ───── CONFIG: {A_PREC, B_PREC} 으로 모드별 상수 디스패치 ─────
        //
        // 출처: precision_modes_protocol.md §1 Mode Matrix (v1.0, 2026-05-11 검증).
        // 기호: A∈{8,4,2} → A_INT8/4/2 (Accumulator_Col 의 Mode_oh),
        //       W∈{8,4,2} → Accumulator 내부 cnt rollover threshold.
        //
        // !!! 동기화 주의: 이 case 값과 precision_modes_protocol.md §1/§3 은
        // 반드시 같아야 함. Task 8 에서 둘 사이 drift 때문에 TOGGLE_VAL
        // (A8W4, A4W8) 버그가 났었음. 값을 손대면
        // `bash sim/run_integration_sweep.sh` → "ALL 9 MODES PASSED" 가
        // 회귀 게이트.
        case ({A_PREC[3:0], B_PREC[3:0]})
            // A8 행: A_FIRE_DELAY=2, N_T_LOGICAL=4, FIRES_PER_COL=2048
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
            // A4 행: A_FIRE_DELAY=1, N_T_LOGICAL=2, FIRES_PER_COL=1024
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
            // A2 행: A_FIRE_DELAY=0, N_T_LOGICAL=1, FIRES_PER_COL=512
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

        // ───── CONFIG 후 idle 유지 (chain settle) ─────
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
        drive_stage_2a(0, 0);          // Stage 2-A: 초기 B load (n_t=0, k_t=0)
        drive_stage_2b;                // Stage 2-B: settle + Buf1 toggle
        drive_stage_3_4;               // Stage 3+4: MAC sweep (capture_en 도 켬)
        drive_stage_5_tail;            // Stage 5: tail + capture 종료

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

    // ─── timeout 안전장치 ──────────────────────────────────────────
    initial begin
        #50_000_000;   // 50 ms = 5_000_000 cycles @ 100 MHz (정상 sim 의 ~10 배)
        $display("ERROR: timeout");
        $finish;
    end

endmodule
