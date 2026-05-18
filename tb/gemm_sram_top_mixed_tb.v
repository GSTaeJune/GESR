`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_mixed_tb — mixed-precision (per-block W_PREC) 검증 TB.
//
// 검증 목적:
//   weight 의 (M-row, K-block) 단위 random W_PREC ∈ {2,4,8} mix 가 RTL
//   변경 없이 GEMM+RMW+SRAM end-to-end 에서 FP32 bit-exact 동작함을
//   검증. 9-mode sweep TB 는 행렬 전체 균일 mode 만 검증하므로 본 TB
//   가 cycle 단위 in_Wcontrol 갱신 + Accumulator cnt 의 mixed-mode
//   fire 정합성을 보강.
//
// 검증 시나리오:
//   A_PREC ∈ {2,4,8} 각각 × seed=0 random W_PREC map.
//   bit-exact gate = MXP_Tools compare vs sim/gen_mixed.py 산출 골든.
//
// 동작 의도:
//   DRIVE 의 inner-loop 만 mixed-aware (m_in 마다 W_now lookup).
//   capture/decode/drain/dump 는 기존 9-mode TB 와 동일 — A_PREC 만
//   의존하고 W_PREC 는 fire 횟수에 영향 없음.
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8 (layer-wide activation 정밀도)
//   WORK_DIR  : sim/gen_mixed.py 의 --out 디렉토리 (필수)
//   DUMP_DIR  : SRAM .mem dump 출력 디렉토리 (필수)
//
// 회귀 게이트: `bash sim/run_mixed_sweep.sh` → "ALL 3 MIXED SCENARIOS PASSED".
//////////////////////////////////////////////////////////////////////////////

module gemm_sram_top_mixed_tb;

    // ─── plusarg parsing ───────────────────────────────────────────
    integer A_PREC;
    reg [8*256-1:0] WORK_DIR;
    reg [8*256-1:0] DUMP_DIR;

    initial begin
        if (!$value$plusargs("A_PREC=%d",   A_PREC))    A_PREC   = 8;
        if (!$value$plusargs("WORK_DIR=%s", WORK_DIR))  WORK_DIR = "work/mixed_A8";
        if (!$value$plusargs("DUMP_DIR=%s", DUMP_DIR))  DUMP_DIR = "work/mixed_A8/hw_out";
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

    // ─── 모드별 상수 (CONFIG 에서 A_PREC 으로 case 디스패치) ────────
    // W_CYC / FIRST_FIRE_GLOBAL / TOGGLE_VAL / W_CTRL_CODE 는
    // drive_stage_3_4_mixed 의 iteration 마다 동적 계산.
    integer A_FIRE_DELAY, FIRES_PER_COL, N_T_LOGICAL;
    reg [1:0] A_CTRL_CODE;

    // ─── PASS/FAIL 구조적 검증용 카운터 ───────────────────────────
    // EXPECTED_TOTAL 은 mode 무관 — 32 col × 2048 RMW/col 곱.
    //   A8: 2048 fires × 1 lane = 2048 RMW/col
    //   A4: 1024 fires × 2 lane = 2048
    //   A2:  512 fires × 4 lane = 2048
    //   → 32 × 2048 = 65536 (모든 모드 동일)
    localparam integer EXPECTED_TOTAL = 32 * 2048;
    integer total_captured;
    integer total_drained;

    // ─── RMW 파이프라인 타이밍 보조 상수 ──────────────────────────
    // RMW latency = L_CONV + L_ADD = 2 + 3 = 5 cy. RMW 입력 드라이브 후
    // write-back 까지 8 cy 대기.
    localparam integer RMW_WAIT_CYC = 8;
    // X-prime: drain 시작 전 파이프라인 X 채움 cycle.
    localparam integer PRIME_CYC    = 16;
    // Stage 5 tail 길이: in-flight fire 들을 모두 캡처할 때까지 유지.
    // 최악 = symmetric chain delay (col 0/31 에서 15 cy 최대) +
    // A_fire_delay (≤2) + W_CYC_MAX (≤8). 45 cy 면 모든 모드 여유 있음.
    localparam integer TAIL_CYC     = 45;
    // Stage 5 후 capture 유지 cycle.
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

    // RMW (32 col)
    reg  [32*32-1:0]          rmw_in_GEMM       = 0;
    reg  [32*9-1:0]           rmw_scale         = 0;
    wire [32*32-1:0]          rmw_out_RMW;

    // SRAM (32 bank, depth=1024 → AW=10)
    localparam integer NB        = 32;
    localparam integer BD        = 1024;
    localparam integer AW        = 10;       // clog2(1024)

    reg  [NB-1:0]             sram_CEB        = {NB{1'b1}};
    reg  [NB-1:0]             sram_WEB        = {NB{1'b1}};
    reg  [NB*AW-1:0]          sram_A          = 0;
    reg  [NB*32-1:0]          sram_WMASK      = {NB*32{1'b1}};
    reg                       sram_D_use_zero = 1'b1;
    wire [NB*32-1:0]          sram_Q;

    // ─── DUT 인스턴스 ──────────────────────────────────────────────
    gemm_sram_top #(
        .NUM_BANKS  (NB),
        .BANK_DEPTH (BD)
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
        .CEB(sram_CEB), .WEB(sram_WEB),
        .A(sram_A), .WMASK(sram_WMASK),
        .sram_D_use_zero(sram_D_use_zero),
        .Q(sram_Q)
    );

    // ─── 입력 hex 메모리 ───────────────────────────────────────────
    // 크기 산정 (A8 worst case 기준):
    //   K_T*M_T*TILE*W_PAD = 4*4*32*8 = 4096 lines  → a_bs   (weight bit-serial)
    //   N_T_MAX*K_T*TILE   = 4*4*32   = 512  lines  → b_bp   (activation, 256-bit)
    //   K_T*M_T*TILE       = 4*4*32   = 512  lines  → a_scale (weight scale, E8M0)
    //   N_T_MAX*K_T*TILE   = 4*4*32   = 512  lines  → b_scale (activation scale, E8M0)
    reg [31:0]   a_bs    [0:4095];
    reg [255:0]  b_bp    [0:511];
    reg [7:0]    a_scale [0:511];
    reg [7:0]    b_scale [0:511];

    // w_prec_packed: m row 별 K_T K-block W_CTRL_CODE packed.
    // bit[2*k+1:2*k] = W_CTRL_CODE for (m_global, k_t).
    // 128 rows total (M_DIM).
    reg [7:0]    w_prec_packed [0:127];

    // ─── Fire 캡처 per-col FIFO ────────────────────────────────────
    // col 당 최대 2048 entry (모든 모드 공통: A8=2048×1, A4=1024×2, A2=512×4).
    localparam integer FIFO_DEPTH = 2048;
    reg [31:0]   fifo_int   [0:31][0:FIFO_DEPTH-1];
    reg [8:0]    fifo_scale [0:31][0:FIFO_DEPTH-1];
    reg [18:0]   fifo_addr  [0:31][0:FIFO_DEPTH-1];
    integer      fifo_wp [0:31];
    integer      fifo_rp [0:31];

    integer init_i;
    initial begin
        for (init_i = 0; init_i < 32; init_i = init_i + 1) begin
            fifo_wp[init_i] = 0;
            fifo_rp[init_i] = 0;
        end
    end

    // col 별 누적 fire 카운터 (col 안에서 몇 번째 fire 인지 → m_t/m_in/k_t/n_t 디코딩에 사용)
    reg [11:0]   fire_cnt_per_col [0:31];

    // ─── Fire 캡처 always 블록 ─────────────────────────────────────
    //
    // fire_cnt[c] (col c 에서의 누적 fire 번호) → (n_t, k_t, m_t, m_in) 디코딩:
    //   driving 의 inner 루프 순서:  n_t → k_t → m_t → m_in
    //   col 별 fire 는 (n_t, k_t, m_t, m_in) 한 조합당 1 회 발생
    //   fc = n_t*K_T*M_T*TILE + k_t*M_T*TILE + m_t*TILE + m_in
    //
    // m_global = m_t * TILE_SIZE + m_in
    // n_global 은 A_PREC 의존:
    //   A8: n_g = n_t * 32 + col_idx           (1 lane)
    //   A4: n_g_top = n_pair*64 + 32 + col,    (2 lanes)
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

    reg capture_en;     // DRIVE stage 동안만 1
    initial capture_en = 0;

    always @(posedge clk) begin
        if (rst) begin
            for (ci = 0; ci < 32; ci = ci + 1) fire_cnt_per_col[ci] <= 0;
        end else if (capture_en) begin
            for (ci = 0; ci < 32; ci = ci + 1) begin
                // FIRES_PER_COL 초과 fire 는 무시 (tail 에서 가짜 fire 방어).
                if (out_fire[ci] && (fire_cnt_per_col[ci] < FIRES_PER_COL)) begin
                    if (fifo_wp[ci] >= FIFO_DEPTH) begin
                        $display("FATAL: FIFO overflow col=%0d wp=%0d depth=%0d fire_cnt=%0d",
                                 ci, fifo_wp[ci], FIFO_DEPTH, fire_cnt_per_col[ci]);
                        $finish;
                    end
                    fc = fire_cnt_per_col[ci];

                    n_t_dec  = fc / (K_T * M_T * TILE_SIZE);
                    k_t_dec  = (fc / (M_T * TILE_SIZE)) % K_T;
                    m_t_dec  = (fc / TILE_SIZE) % M_T;
                    m_in_dec = fc % TILE_SIZE;
                    m_g      = m_t_dec * TILE_SIZE + m_in_dec;

                    case (A_PREC)
                        // ─── A8: col 당 1 lane ──────────────────────────
                        8: begin
                            acc_lane_a8 = $signed(out_accumulate[60*ci +: 21]);
                            acc_int32   = {{11{acc_lane_a8[20]}}, acc_lane_a8};
                            sc_lane     = out_scale[36*ci +: 9];
                            ng_dec      = n_t_dec * 32 + ci;
                            flat_dec    = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch col=%0d flat=%0d at lane A8", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;
                        end
                        // ─── A4: col 당 2 lane (top/bot) ────────────────
                        4: begin
                            acc_lane_a4t = $signed(out_accumulate[60*ci+18 +: 18]);
                            acc_int32    = {{14{acc_lane_a4t[17]}}, acc_lane_a4t};
                            sc_lane      = out_scale[36*ci+9 +: 9];
                            ng_dec       = n_t_dec * 64 + 32 + ci;
                            flat_dec     = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch col=%0d flat=%0d at lane A4t", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;

                            acc_lane_a4b = $signed(out_accumulate[60*ci +: 18]);
                            acc_int32    = {{14{acc_lane_a4b[17]}}, acc_lane_a4b};
                            sc_lane      = out_scale[36*ci +: 9];
                            ng_dec       = n_t_dec * 64 + 0 + ci;
                            flat_dec     = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch col=%0d flat=%0d at lane A4b", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;
                        end
                        // ─── A2: col 당 4 lane ──────────────────────────
                        2: begin
                            acc_lane_a2_0 = $signed(out_accumulate[60*ci+45 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_0[14]}}, acc_lane_a2_0};
                            sc_lane       = out_scale[36*ci+27 +: 9];
                            ng_dec        = 96 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch col=%0d flat=%0d at lane A2_0", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;

                            acc_lane_a2_1 = $signed(out_accumulate[60*ci+30 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_1[14]}}, acc_lane_a2_1};
                            sc_lane       = out_scale[36*ci+18 +: 9];
                            ng_dec        = 64 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch col=%0d flat=%0d at lane A2_1", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;

                            acc_lane_a2_2 = $signed(out_accumulate[60*ci+15 +: 15]);
                            acc_int32     = {{17{acc_lane_a2_2[14]}}, acc_lane_a2_2};
                            sc_lane       = out_scale[36*ci+9 +: 9];
                            ng_dec        = 32 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch col=%0d flat=%0d at lane A2_2", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;

                            acc_lane_a2_3 = $signed(out_accumulate[60*ci +: 15]);
                            acc_int32     = {{17{acc_lane_a2_3[14]}}, acc_lane_a2_3};
                            sc_lane       = out_scale[36*ci +: 9];
                            ng_dec        = 0 + ci;
                            flat_dec      = m_g * N_DIM + ng_dec;
                            if ((flat_dec & 5'h1F) != ci[4:0]) begin
                                $display("FATAL: bank-col mismatch col=%0d flat=%0d at lane A2_3", ci, flat_dec);
                                $finish;
                            end
                            fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                            fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                            fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                            fifo_wp[ci] = fifo_wp[ci] + 1;
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
    integer k_t_start_cycle;       // mixed drive 에서 현재 K-tile 의 cyc_global 기준점
    reg     cur_pp;       // PE in_control 토글 상태 (Buf1/Buf2)
    reg     cur_st_pp;    // Station selector 토글 상태

    // scale_weight 드라이브 schedule — fire_q2 capture 시점에 맞춰
    // a_scale[k] 가 visible 되도록 cycle 정합.
    // drive_cyc[k] = 18 + A_FIRE_DELAY + Σ W_PREC[0..k]  (cum_W_inclusive[k])
    // — uniform W=8, A=8 일 때 28+8k 로 reduce (= 9-mode FIRST_FIRE+W_CYC*k).
    integer drive_cyc_target;     // 다음 a_scale[] 드라이브 cyc_global
    integer drive_idx_target;     // 다음 드라이브할 a_scale[] 인덱스 (0..K_T*M_T*TILE-1)

    // in_Wcontrol 드라이브 schedule — col 16 의 m_in=k MAC 시작 cycle (= cnt=0)
    // 에 맞춰 in_Wcontrol = W[k] 가 settle 되도록.
    // drive_w_cyc[k] = 18 + Σ W_PREC[0..k-1]  (cum_W_exclusive[k]).
    // col 16 의 m_in=k fire (= P_(18+cum_W[k])) pre-edge = TB iter 17+cum_W[k] = W[k] ✓.
    // 단순 iter 인덱스 기반 (loop's m_in_idx) 드라이브와 다르게, SA pipeline
    // depth 만큼 offset 된 schedule — K-tile 경계에서 col 16 의 마지막 fire 가
    // 잘못된 in_control 로 fail 하는 것을 막음.
    integer drive_w_cyc_target;
    integer drive_w_idx_target;

    // 인덱스 → W (cycle 수) 매핑. drive_idx_target 의 m_in 위치 W_PREC lookup.
    function integer lookup_w_for_idx;
        input integer idx;
        integer kt_v, mt_v, mi_v;
        reg [1:0] w_ctrl_v;
        begin
            kt_v = idx / (M_T * TILE_SIZE);
            mt_v = (idx / TILE_SIZE) % M_T;
            mi_v = idx % TILE_SIZE;
            w_ctrl_v = w_prec_packed[mt_v*TILE_SIZE + mi_v][2*kt_v +: 2];
            case (w_ctrl_v)
                W_INT8:  lookup_w_for_idx = 8;
                W_INT4:  lookup_w_for_idx = 4;
                W_INT2:  lookup_w_for_idx = 2;
                default: lookup_w_for_idx = 8;
            endcase
        end
    endfunction

    // 인덱스 → W_CTRL_CODE 매핑. in_Wcontrol 드라이브용.
    function [1:0] lookup_wctrl_for_idx;
        input integer idx;
        integer kt_v, mt_v, mi_v;
        begin
            kt_v = idx / (M_T * TILE_SIZE);
            mt_v = (idx / TILE_SIZE) % M_T;
            mi_v = idx % TILE_SIZE;
            lookup_wctrl_for_idx = w_prec_packed[mt_v*TILE_SIZE + mi_v][2*kt_v +: 2];
        end
    endfunction

    // a_bs[] 에서 (k_t, m_t, o) 위치 워드 추출.
    // mixed TB 에서 o = m_in_idx*8 + (7-bp) — gen_mixed.py W_PAD=8 padding 에 맞춤.
    // W_CYC 가 없으므로 a_bs_word 의 stride 를 8 (W_PAD) 로 고정.
    function [31:0] a_bs_word;
        input integer k_t_i, m_t_i, o_i;
        begin
            a_bs_word = a_bs[k_t_i * M_T * TILE_SIZE * 8 + m_t_i * TILE_SIZE * 8 + o_i];
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
    function [255:0] pack_int4_n_pair;
        input [255:0] top_data;
        input [255:0] bot_data;
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
    function [255:0] pack_int2_n_quad;
        input [255:0] data_n3;
        input [255:0] data_n2;
        input [255:0] data_n1;
        input [255:0] data_n0;
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

    // 모드별 in_b 빌더.
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
                        b_scale[(2*n_pair+0) * K_T * TILE_SIZE + base],
                        b_scale[(2*n_pair+1) * K_T * TILE_SIZE + base]
                    };
                end
                2: begin
                    base = k_t_arg * TILE_SIZE + col_target;
                    build_in_scale_act = {
                        b_scale[0 * K_T * TILE_SIZE + base],
                        b_scale[1 * K_T * TILE_SIZE + base],
                        b_scale[2 * K_T * TILE_SIZE + base],
                        b_scale[3 * K_T * TILE_SIZE + base]
                    };
                end
                default: build_in_scale_act = 0;
            endcase
        end
    endfunction

    // ─── DRIVE 보조 task ────────────────────────────────────────────

    // Stage 2-A 초기 B load (32 cy).
    task drive_stage_2a;
        input integer n_t_arg, k_t_arg;
        integer tt, col_target;
        begin
            for (tt = 0; tt < TILE_SIZE; tt = tt + 1) begin
                col_target = tt;
                in_b                <= build_in_b(n_t_arg, k_t_arg, col_target);
                in_Scale_Activation <= build_in_scale_act(n_t_arg, k_t_arg, col_target);
                in_Station_control  <= A_CTRL_CODE;
                in_loadEN           <= 32'hFFFFFFFF;
                in_station_loadEN   <= 1'b1;
                in_station_control  <= 1'b0;
                @(posedge clk);
            end
        end
    endtask

    // Stage 2-B settle.
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

    // prefetch — 다음 tile B/scale 의 한 cycle 분 드라이브.
    task drive_prefetch;
        input integer n_t_next;
        input integer k_t_next;
        input integer o_arg;
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

    // ─── drive_stage_3_4_mixed — mixed-aware MAC sweep ─────────────
    // inner-loop: n_t → k_t → m_t → m_in → bit_pos
    // 각 m_in 진입마다 w_prec_packed[m_t*32+m_in][2*k_t +: 2] lookup → w_now / w_ctrl_now.
    // TOGGLE_VAL 은 worst-case (W=8) 기준 보수 24 고정.
    task drive_stage_3_4_mixed;
        integer m_in_idx;
        reg [1:0] w_ctrl_now;
        integer w_now;
        integer bp;
        begin
            cyc_global = 0;
            k_t_start_cycle = 0;
            cur_pp = 1'b1;
            capture_en <= 1'b1;

            // scale_weight schedule 초기화 — m_in=0 의 fire 가 도착할 cyc.
            drive_idx_target = 0;
            drive_cyc_target = 18 + A_FIRE_DELAY + lookup_w_for_idx(0);

            // in_Wcontrol schedule 초기화 — col 16 의 m_in=0 MAC cnt 시작 cyc.
            // drive_cyc[k] = 18 + cum_W_exclusive[k]. m_in=0 → iter 18 (cum=0).
            drive_w_idx_target = 0;
            drive_w_cyc_target = 18;

            for (n_t = 0; n_t < N_T_LOGICAL; n_t = n_t + 1) begin
                for (k_t = 0; k_t < K_T; k_t = k_t + 1) begin
                    for (m_t = 0; m_t < M_T; m_t = m_t + 1) begin
                        for (m_in_idx = 0; m_in_idx < TILE_SIZE; m_in_idx = m_in_idx + 1) begin
                            // (m, k_t) 의 W_CTRL_CODE lookup
                            w_ctrl_now = w_prec_packed[m_t*TILE_SIZE + m_in_idx][2*k_t +: 2];
                            case (w_ctrl_now)
                                W_INT8: w_now = 8;
                                W_INT4: w_now = 4;
                                W_INT2: w_now = 2;
                                default: begin
                                    $display("FATAL: invalid W_CTRL %b at m=%0d k_t=%0d",
                                             w_ctrl_now, m_t*TILE_SIZE+m_in_idx, k_t);
                                    $finish;
                                end
                            endcase

                            for (bp = w_now - 1; bp >= 0; bp = bp - 1) begin
                                cyc_in_K = cyc_global - k_t_start_cycle;

                                // weight bit-serial: a_bs[] 인덱싱은 W=8 padding
                                // (gen_mixed.py 의 _emit_a_input_bs 와 일치).
                                // 각 m_in 의 8 bit_pos 슬롯 중 m_in*8+(7-bp) — MSB-first.
                                in_a <= a_bs_word(k_t, m_t, m_in_idx*8 + (7 - bp));

                                // start_accumulate 펄스: 첫 K-tile 의 cycle 17 한 번.
                                if (n_t == 0 && k_t == 0 && cyc_global == 17) begin
                                    in_start_accumulate <= 1'b1;
                                end else begin
                                    in_start_accumulate <= 1'b0;
                                end

                                // in_Wcontrol 드라이브 — schedule 기반.
                                // drive_w_cyc_target = 18 + cum_W_exclusive[k] = col 16 의
                                // m_in=k MAC cnt 시작 cyc. iter 0..17 동안은 schedule 아직
                                // 안 fire → in_Wcontrol = W_IDLE (CONFIG 초기값).
                                if (cyc_global == drive_w_cyc_target) begin
                                    in_Wcontrol <= lookup_wctrl_for_idx(drive_w_idx_target);
                                    drive_w_cyc_target = drive_w_cyc_target
                                                         + lookup_w_for_idx(drive_w_idx_target);
                                    drive_w_idx_target = drive_w_idx_target + 1;
                                    if (drive_w_idx_target >= K_T*M_T*TILE_SIZE)
                                        drive_w_idx_target = 0;
                                end

                                // PE in_control toggle: K-tile 진입 첫 cycle.
                                if (cyc_in_K == 0 && !(n_t == 0 && k_t == 0)) begin
                                    cur_pp = ~cur_pp;
                                end
                                in_control <= {32{cur_pp}};

                                // Station selector toggle: K-tile 의 W 따라 가변.
                                // 9-mode 의 TOGGLE_VAL = 18 + A_FIRE_DELAY + W/2.
                                // W=8 → 24, W=4 → 22, W=2 → 21 (A=8 기준).
                                // 현재 K-tile 의 W = W[m_in=0 of this K-tile] (K-tile granular 일 땐
                                // 전체 K-tile 동일; random mixed 일 땐 첫 m_in 의 W proxy).
                                if (cyc_in_K == (18 + A_FIRE_DELAY +
                                                 (lookup_w_for_idx(k_t * M_T * TILE_SIZE) / 2))
                                    && !(n_t == 0 && k_t == 0)) begin
                                    cur_st_pp = ~cur_st_pp;
                                end
                                in_station_control <= cur_st_pp;

                                // scale_weight 드라이브 — schedule 기반.
                                // drive_cyc_target = 18 + A_FIRE_DELAY + cum_W_inclusive[k],
                                // 다음 m_in 의 W 만큼 advance. 한 n_t 의 a_scale[] 다 쓰면 wrap.
                                if (cyc_global == drive_cyc_target) begin
                                    in_scale_weight <= a_scale[drive_idx_target];
                                    drive_idx_target = drive_idx_target + 1;
                                    if (drive_idx_target >= K_T*M_T*TILE_SIZE)
                                        drive_idx_target = 0;
                                    drive_cyc_target = drive_cyc_target
                                                       + lookup_w_for_idx(drive_idx_target);
                                end

                                // Prefetch: m_t=1 의 매 m_in 첫 bit 시점에 한 col 분 driving.
                                if (m_t == 1 && bp == w_now - 1) begin
                                    if (k_t < K_T - 1) begin
                                        drive_prefetch(n_t, k_t + 1, m_in_idx);
                                    end else if (n_t < N_T_LOGICAL - 1) begin
                                        drive_prefetch(n_t + 1, 0, m_in_idx);
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
                            end // bp loop
                        end // m_in loop
                    end // m_t loop
                    k_t_start_cycle = cyc_global; // 다음 K-tile 의 cyc_in_K 기준점
                end // k_t loop
            end // n_t loop
        end
    endtask

    // Stage 5 tail: mixed-prec 모드에서 W_CYC 단일 값이 없으므로
    // 마지막 iteration 의 w_ctrl 을 그대로 유지하고 idle 입력을 흘림.
    task drive_stage_5_tail;
        integer i;
        begin
            // mixed-prec: W_CTRL_CODE 는 fixed 가 없으므로, 마지막 iteration 의 w_ctrl
            // 을 그대로 유지하는 식으로 idle. in_Wcontrol 변경 안 함.
            in_a                <= 0;
            in_Station_control  <= A_IDLE;
            in_loadEN           <= 0;
            in_station_loadEN   <= 0;
            in_b                <= 0;
            in_Scale_Activation <= 0;
            for (i = 0; i < TAIL_CYC; i = i + 1) begin
                // tail 동안에도 scale_weight + in_Wcontrol schedule 이어감 — in-flight
                // fire 의 scale 및 wControl capture.
                if (cyc_global == drive_cyc_target) begin
                    in_scale_weight <= a_scale[drive_idx_target];
                    drive_idx_target = drive_idx_target + 1;
                    if (drive_idx_target >= K_T*M_T*TILE_SIZE)
                        drive_idx_target = 0;
                    drive_cyc_target = drive_cyc_target
                                       + lookup_w_for_idx(drive_idx_target);
                end
                if (cyc_global == drive_w_cyc_target) begin
                    in_Wcontrol <= lookup_wctrl_for_idx(drive_w_idx_target);
                    drive_w_cyc_target = drive_w_cyc_target
                                         + lookup_w_for_idx(drive_w_idx_target);
                    drive_w_idx_target = drive_w_idx_target + 1;
                    if (drive_w_idx_target >= K_T*M_T*TILE_SIZE)
                        drive_w_idx_target = 0;
                end
                @(posedge clk);
                cyc_global = cyc_global + 1;
            end

            in_Wcontrol <= W_IDLE;
            for (i = 0; i < POST_TAIL; i = i + 1) @(posedge clk);
            capture_en <= 1'b0;
        end
    endtask

    // ─── INIT: 32 bank parallel zero-prime (1024 cycle) ────────────
    task init_zero_prime;
        integer w, bi;
        begin
            sram_D_use_zero <= 1'b1;
            sram_WMASK      <= {NB*32{1'b1}};
            for (w = 0; w < BD; w = w + 1) begin
                @(posedge clk);
                sram_CEB <= {NB{1'b0}};
                sram_WEB <= {NB{1'b0}};
                for (bi = 0; bi < NB; bi = bi + 1)
                    sram_A[bi*AW +: AW] <= w[AW-1:0];
            end
            @(posedge clk);
            sram_CEB <= {NB{1'b1}};
            sram_WEB <= {NB{1'b1}};
        end
    endtask

    // ─── per-col drain FSM (32 stream parallel) ────────────────────
    // State: 0=IDLE, 1=READ, 2=WAIT_R, 3=DRIVE_RMW, 4=WAIT_RMW,
    //        5=WRITE, 6=WRITE_SETTLE
    reg drain_enable;
    initial drain_enable = 1'b0;

    reg [3:0]  drain_state [0:31];
    reg [4:0]  drain_wait  [0:31];
    reg [18:0] drain_addr  [0:31];

    initial begin
        for (init_i = 0; init_i < 32; init_i = init_i + 1) begin
            drain_state[init_i] = 4'd0;
            drain_wait [init_i] = 5'd0;
            drain_addr [init_i] = 19'd0;
        end
    end

    localparam integer DRAIN_RMW_WAIT = 8;

    genvar dc;
    generate
        for (dc = 0; dc < 32; dc = dc + 1) begin : g_drain
            always @(posedge clk) begin
                if (rst) begin
                    drain_state[dc] <= 4'd0;
                    drain_wait [dc] <= 5'd0;
                    drain_addr [dc] <= 19'd0;
                end else if (drain_enable) begin
                    case (drain_state[dc])
                        4'd0: begin
                            if (fifo_rp[dc] < fifo_wp[dc]) begin
                                drain_addr[dc]                      <= fifo_addr[dc][fifo_rp[dc]];
                                sram_CEB[dc]                        <= 1'b0;
                                sram_WEB[dc]                        <= 1'b1;
                                sram_A[dc*AW +: AW]                 <= fifo_addr[dc][fifo_rp[dc]] >> 5;
                                drain_state[dc]                     <= 4'd1;
                            end else begin
                                sram_CEB[dc]                        <= 1'b1;
                                sram_WEB[dc]                        <= 1'b1;
                            end
                        end
                        4'd1: begin
                            sram_CEB[dc] <= 1'b1;
                            drain_state[dc] <= 4'd2;
                        end
                        4'd2: begin
                            rmw_in_GEMM[dc*32 +: 32] <= fifo_int  [dc][fifo_rp[dc]];
                            rmw_scale  [dc*9  +: 9 ] <= fifo_scale[dc][fifo_rp[dc]];
                            drain_state[dc] <= 4'd3;
                        end
                        4'd3: begin
                            drain_wait [dc] <= 5'd0;
                            drain_state[dc] <= 4'd4;
                        end
                        4'd4: begin
                            if (drain_wait[dc] == DRAIN_RMW_WAIT - 1) begin
                                drain_state[dc] <= 4'd5;
                            end else begin
                                drain_wait[dc] <= drain_wait[dc] + 1;
                            end
                        end
                        4'd5: begin
                            sram_CEB[dc]                        <= 1'b0;
                            sram_WEB[dc]                        <= 1'b0;
                            sram_A[dc*AW +: AW]                 <= drain_addr[dc] >> 5;
                            sram_WMASK[dc*32 +: 32]             <= 32'hFFFFFFFF;
                            drain_state[dc]                     <= 4'd6;
                        end
                        4'd6: begin
                            sram_CEB[dc] <= 1'b1;
                            sram_WEB[dc] <= 1'b1;
                            fifo_rp[dc]  <= fifo_rp[dc] + 1;
                            drain_state[dc] <= 4'd0;
                        end
                        default: drain_state[dc] <= 4'd0;
                    endcase
                end
            end
        end
    endgenerate

    task wait_drain_complete;
        integer wc, idle_count, all_idle;
        begin
            idle_count = 0;
            while (idle_count < 10) begin
                @(posedge clk);
                all_idle = 1;
                for (wc = 0; wc < 32; wc = wc + 1) begin
                    if (fifo_rp[wc] < fifo_wp[wc] || drain_state[wc] != 4'd0)
                        all_idle = 0;
                end
                if (all_idle) idle_count = idle_count + 1;
                else          idle_count = 0;
            end
        end
    endtask

    // ─── DUMP: 32 bank × 512 words, port-based read → $fwrite ──────
    task dump_banks;
        integer bi, w, fd;
        reg [8*512-1:0] path_str;
        reg [31:0] q_word;
        begin
            for (bi = 0; bi < NB; bi = bi + 1) begin
                $sformat(path_str, "%0s/bank%0d.mem", DUMP_DIR, bi);
                fd = $fopen(path_str, "w");
                if (fd == 0) begin
                    $display("FATAL: cannot open dump file %0s", path_str);
                    $finish;
                end
                for (w = 0; w < 512; w = w + 1) begin
                    @(posedge clk);
                    sram_CEB[bi]                    <= 1'b0;
                    sram_WEB[bi]                    <= 1'b1;
                    sram_A[bi*AW +: AW]             <= w[AW-1:0];
                    @(posedge clk);
                    sram_CEB[bi]                    <= 1'b1;
                    @(posedge clk);
                    q_word = sram_Q[bi*32 +: 32];
                    $fwrite(fd, "%08x\n", q_word);
                end
                $fclose(fd);
            end
        end
    endtask

    // ─── 메인 시퀀스 (INIT → LOAD → CONFIG → DRIVE → DRAIN → DUMP) ──
    reg [8*512-1:0] in_path_a_bs, in_path_b, in_path_a_sc, in_path_b_sc;
    reg [8*512-1:0] in_path_w_prec;

    initial begin
        cur_pp     = 1'b0;
        cur_st_pp  = 1'b0;
        k_t_start_cycle = 0;

        // ───── INIT ─────
        $display("[INIT] reset + 32-bank parallel zero-prime");
        #20;
        rst = 0;
        @(posedge clk);
        @(posedge clk);
        init_zero_prime;
        @(posedge clk);
        @(posedge clk);

        // ───── LOAD ─────
        $display("[LOAD] $readmemh inputs (mixed-precision)");
        // mixed TB 용 파일명 규칙 (gen_mixed.py --out 기준):
        //   a_input_BS_mixed.hex      — weight bit-serial (W_PAD=8 고정, 전 row)
        //   b_input_mixed_A{P}.hex    — activation (A_PREC 의존)
        //   a_scale_mixed.hex         — weight scale
        //   b_scale_mixed.hex         — activation scale
        //   w_prec_per_block.hex      — per-row per-K-block W_CTRL_CODE packed
        $sformat(in_path_a_bs,   "%0s/hw_input/a_input_BS_mixed.hex",   WORK_DIR);
        $sformat(in_path_b,      "%0s/hw_input/b_input_mixed_A%0d.hex", WORK_DIR, A_PREC);
        $sformat(in_path_a_sc,   "%0s/hw_input/a_scale_mixed.hex",      WORK_DIR);
        $sformat(in_path_b_sc,   "%0s/hw_input/b_scale_mixed.hex",      WORK_DIR);
        $sformat(in_path_w_prec, "%0s/hw_input/w_prec_per_block.hex",   WORK_DIR);
        $readmemh(in_path_a_bs,   a_bs);
        $readmemh(in_path_b,      b_bp);
        $readmemh(in_path_a_sc,   a_scale);
        $readmemh(in_path_b_sc,   b_scale);
        $readmemh(in_path_w_prec, w_prec_packed);
        $display("LOAD OK: a_bs[0]=%h b_bp[0]=%h a_scale[0]=%h b_scale[0]=%h w_prec_packed[0]=%h",
                 a_bs[0], b_bp[0], a_scale[0], b_scale[0], w_prec_packed[0]);

        // ───── CONFIG: A_PREC 으로 모드별 상수 디스패치 ─────
        $display("[CONFIG] dispatching A_PREC-only constants (W is per-iteration)");
        case (A_PREC)
            8: begin
                A_FIRE_DELAY=2; FIRES_PER_COL=2048; N_T_LOGICAL=4;
                A_CTRL_CODE=A_INT8;
            end
            4: begin
                A_FIRE_DELAY=1; FIRES_PER_COL=1024; N_T_LOGICAL=2;
                A_CTRL_CODE=A_INT4;
            end
            2: begin
                A_FIRE_DELAY=0; FIRES_PER_COL=512;  N_T_LOGICAL=1;
                A_CTRL_CODE=A_INT2;
            end
            default: begin
                $display("ERROR: unsupported A_PREC=%0d", A_PREC); $finish;
            end
        endcase

        $display("CONFIG: A=%0d  A_FIRE_DELAY=%0d  FIRES_PER_COL=%0d  N_T_LOGICAL=%0d",
                 A_PREC, A_FIRE_DELAY, FIRES_PER_COL, N_T_LOGICAL);

        // ───── CONFIG 후 idle 유지 ─────
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
        $display("[DRIVE] stage 2-A / 2-B / 3+4 mixed / 5-tail");
        drive_stage_2a(0, 0);
        drive_stage_2b;
        drive_stage_3_4_mixed;
        drive_stage_5_tail;

        total_captured = 0;
        begin : dbg_drive_count
            integer dbg_c;
            for (dbg_c = 0; dbg_c < 32; dbg_c = dbg_c + 1)
                total_captured = total_captured + fifo_wp[dbg_c];
        end
        $display("DRIVE DONE: captured %0d fires (expected %0d)", total_captured, EXPECTED_TOTAL);

        // ───── PRIME ─────
        repeat (PRIME_CYC) @(posedge clk);

        // ───── DRAIN ─────
        sram_D_use_zero <= 1'b0;
        @(posedge clk);
        $display("[DRAIN] 32-stream per-col R-M-W");
        drain_enable <= 1'b1;
        wait_drain_complete;
        drain_enable <= 1'b0;

        total_drained = 0;
        begin : dbg_drain_count
            integer dbg_c;
            for (dbg_c = 0; dbg_c < 32; dbg_c = dbg_c + 1)
                total_drained = total_drained + fifo_rp[dbg_c];
        end
        $display("DRAIN DONE: %0d entries processed (expected %0d)", total_drained, EXPECTED_TOTAL);

        // ───── DUMP ─────
        $display("[DUMP] writing 32 bank .mem files to %0s", DUMP_DIR);
        dump_banks;

        // ───── PASS/FAIL 배너 ─────
        if (total_captured == EXPECTED_TOTAL && total_drained == EXPECTED_TOTAL) begin
            $display("");
            $display("============================================================");
            $display("  [PASS] MIXED TB A%0d — structural checks OK", A_PREC);
            $display("         captured=%0d  drained=%0d  expected=%0d",
                     total_captured, total_drained, EXPECTED_TOTAL);
            $display("         (bit-exact 검증은 외부 compare 실행)");
            $display("============================================================");
        end else begin
            $display("");
            $display("============================================================");
            $display("  [FAIL] MIXED TB A%0d — count mismatch", A_PREC);
            $display("         captured=%0d  drained=%0d  expected=%0d",
                     total_captured, total_drained, EXPECTED_TOTAL);
            $display("============================================================");
        end

        $display("[DONE] MIXED TB: A%0d", A_PREC);
        $finish;
    end

    // ─── timeout 안전장치 ──────────────────────────────────────────
    initial begin
        #50_000_000;   // 50 ms = 5_000_000 cycles @ 100 MHz
        $display("ERROR: timeout");
        $finish;
    end

endmodule
