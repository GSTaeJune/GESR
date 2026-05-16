`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////
// gemm_sram_top_tb — gemm_sram_top (32-RMW col-parallel) 통합 검증 TB.
//
// 검증 목적:
//   GEMM (MXP 32×32 bit-serial systolic) + RMW[32] (col-parallel) +
//   sram_1rw_banked_mp (32 bank, per-bank port) 의 end-to-end 데이터
//   패스가 9 가지 정밀도 조합 (A∈{2,4,8} × W∈{2,4,8}) 모두에서
//   MXP_Tools 골든 GEMM 과 bit-exact 일치하는지 확인. col j → bank j
//   매핑 (충돌 0) 의 RTL 검증을 겸함.
//
// Plusargs (xsim -testplusarg "KEY=VAL"):
//   A_PREC    : 2 / 4 / 8 (Activation 정밀도)
//   B_PREC    : 2 / 4 / 8 (Weight 정밀도, MXP 의 "in_a" 라인)
//   WORK_DIR  : MXP_Tools hex 입력 디렉토리 (필수)
//   DUMP_DIR  : SRAM .mem dump 출력 디렉토리 (필수, 32 파일 출력)
//
// 전체 흐름 (32-RMW 병렬 디스패치):
//   1) INIT     — 리셋 + 32 bank parallel zero-prime (1024 cycle).
//   2) LOAD     — $readmemh (a_input_BS, b_input, a_scale, b_scale).
//   3) CONFIG   — {A_PREC,B_PREC} 으로 모드별 상수/제어 코드 세팅.
//   4) DRIVE    — Stage 2-A, 2-B, 3+4 (MAC), 5 (tail). 캡처 블록이
//                 매 out_fire[c] rising 마다 per-col FIFO[c] 에 push.
//   5) DRAIN    — 32 col 동시 진행. col c 의 always 블록이 자기 FIFO[c]
//                 를 자기 RMW[c] + 자기 bank[c] port 로 흘림 (R→conv→add→W).
//   6) DUMP     — $writememh 로 32 bank 각각 0..511 워드를 .mem 파일로.
//                 외부 compare 가 MXP_Tools golden npz 와 비트 단위 비교.
//
// 회귀 게이트: `bash sim/run_integration_sweep.sh` → "ALL 9 MODES PASSED".
//
// ─── Vivado GUI 에서 돌리는 절차 ────────────────────────────────────────
//   sim top: `gemm_sram_top_tb` (Vivado 가 .xpr 의 sim_1 fileset 에서 자동 인식).
//
//   1) Hex 입력 미리 준비 (project root 에서):
//        cd MXP_Tools
//        python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
//        python -m mxp_tools emit  --out ../work/A8_B8
//        python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8
//        mkdir -p ../work/A8_B8/hw_out
//   2) Vivado 에서 `gemm_sram.xpr` 열기.
//   3) Flow Navigator → Simulation → Simulation Settings → Simulation tab:
//        xsim.simulate.xsim.more_options 에 plusarg 추가
//        예) -testplusarg "A_PREC=8" -testplusarg "B_PREC=8"
//            -testplusarg "WORK_DIR=../../../../work/A8_B8"
//            -testplusarg "DUMP_DIR=../../../../work/A8_B8/hw_out"
//        (path 는 Vivado sim 스크래치 디렉토리
//         `gemm_sram.sim/sim_1/behav/xsim/` 기준 상대경로.)
//   4) Flow Navigator → Run Simulation → Run Behavioral Simulation.
//   5) Tcl Console 에서 PASS/FAIL 배너 확인 (sim 종료 직전 출력).
//      비트 정확도 검증은 sim 종료 후 외부 compare 실행:
//        cd MXP_Tools
//        BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..31})
//        python -m mxp_tools compare \
//          --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
//          --hw-banks ${BANKS} --layout interleaved_row_major_32bank
//
// PASS/FAIL 배너 의미:
//   TB 자체는 구조적 invariant 만 in-sim 검증:
//     - 캡처 총 push 수 == 드레인 총 pop 수 == EXPECTED (65536)
//     - bank-col 매핑 assert 가 한 번도 $finish FATAL 트리거 안 함
//     - 32 bank dump 파일이 모두 정상 open
//   비트 정확도 (FP32 값 일치) 는 외부 Python compare 가 담당.
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
    //   K_T*M_T*TILE*W_CYC_MAX = 4*4*32*8 = 4096 lines  → a_bs   (weight bit-serial)
    //   N_T_MAX*K_T*TILE       = 4*4*32   = 512  lines  → b_bp   (activation, 256-bit)
    //   K_T*M_T*TILE           = 4*4*32   = 512  lines  → a_scale (weight scale, E8M0)
    //   N_T_MAX*K_T*TILE       = 4*4*32   = 512  lines  → b_scale (activation scale, E8M0)
    reg [31:0]   a_bs    [0:4095];
    reg [255:0]  b_bp    [0:511];
    reg [7:0]    a_scale [0:511];
    reg [7:0]    b_scale [0:511];

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
                    // FIFO overflow guard — exact-fit bound (2048 = A2 worst case).
                    // FIRES_PER_COL × lanes_per_fire = 2048 모든 모드 동일이라 정상
                    // 동작에선 안 터지지만, 워크로드/FIRES_PER_COL 변경 시 silent
                    // memory corruption 방지용 safety net.
                    if (fifo_wp[ci] >= FIFO_DEPTH) begin
                        $display("FATAL: FIFO overflow col=%0d wp=%0d depth=%0d fire_cnt=%0d",
                                 ci, fifo_wp[ci], FIFO_DEPTH, fire_cnt_per_col[ci]);
                        $finish;
                    end
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
                            // top lane (s1_a): acc[60c+35:60c+18], scale[36c+17:36c+9]
                            //   n 위치 = n_pair*64 + 32 + col_idx
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

                            // bot lane (s1_b): acc[60c+17:60c+0], scale[36c+8:36c+0]
                            //   n 위치 = n_pair*64 + 0 + col_idx
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
                            // lane0 (out_INT2_0): acc[60c+59:60c+45], scale[36c+35:36c+27], n 위치 = 96+col_idx
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

                            // lane1: acc[60c+44:60c+30], scale[36c+26:36c+18], n 위치 = 64+col_idx
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

                            // lane2: acc[60c+29:60c+15], scale[36c+17:36c+9], n 위치 = 32+col_idx
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

                            // lane3: acc[60c+14:60c+0], scale[36c+8:36c+0], n 위치 = col_idx
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
    // drain_wait 폭은 DRAIN_RMW_WAIT (현재 8) 보다 충분히 큰 5-bit 사용. 4-bit 면
    // 16 까지밖에 표현 못 해서 후일 wait 값을 16 이상으로 늘릴 때 wrap-around
    // 위험. width-fragile foot-gun 차단.
    reg [4:0]  drain_wait  [0:31];
    reg [18:0] drain_addr  [0:31];

    initial begin
        for (init_i = 0; init_i < 32; init_i = init_i + 1) begin
            drain_state[init_i] = 4'd0;
            drain_wait [init_i] = 5'd0;
            drain_addr [init_i] = 19'd0;
        end
    end

    // RMW pipeline 대기. FSM cycle budget:
    //   state2 (in_GEMM/scale 드라이브) → state5 (WRITE 발사) 까지 실측 = 10 cy
    //   = 1 (state2→3) + 1 (state3→4) + DRAIN_RMW_WAIT (state4 루프).
    // RMW latency L_CONV(2)+L_ADD(3) = 5 cy 이므로 DRAIN_RMW_WAIT=8 은 약 5 cy
    // slack. 단순화하려면 state3 (degenerate reset) 을 state2 로 합쳐서 9 cy
    // 로 줄일 수 있으나 plan 호환 위해 state3 유지.
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
                            // DRAIN_RMW_WAIT integer literal 을 5-bit drain_wait 와
                            // 직접 비교 — Verilog 가 자동 sign-extension. [3:0] slice
                            // 같은 width-narrowing 의 silent truncation 없음.
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
                            // sram_D_use_zero 은 단일 reg 라 multi-driver 회피 위해
                            // 메인 시퀀스에서 drain 시작 전에 한 번 0 으로 설정함.
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

    initial begin
        cur_pp     = 1'b0;
        cur_st_pp  = 1'b0;

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
        $display("[LOAD] $readmemh inputs");
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
        $display("[CONFIG] dispatching mode constants");
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
        $display("[DRIVE] stage 2-A / 2-B / 3+4 / 5-tail");
        drive_stage_2a(0, 0);          // Stage 2-A: 초기 B load (n_t=0, k_t=0)
        drive_stage_2b;                // Stage 2-B: settle + Buf1 toggle
        drive_stage_3_4;               // Stage 3+4: MAC sweep (capture_en 도 켬)
        drive_stage_5_tail;            // Stage 5: tail + capture 종료

        // capture FIFO 의 총 push 수 (per-col 합) — 구조적 검증의 첫 invariant
        total_captured = 0;
        begin : dbg_drive_count
            integer dbg_c;
            for (dbg_c = 0; dbg_c < 32; dbg_c = dbg_c + 1)
                total_captured = total_captured + fifo_wp[dbg_c];
        end
        $display("DRIVE DONE: captured %0d fires (expected %0d)", total_captured, EXPECTED_TOTAL);

        // ───── PRIME: drain 시작 전 in-flight fire 들이 reg 에 들어올 여유 ─────
        repeat (PRIME_CYC) @(posedge clk);

        // ───── DRAIN: 32 col 동시 진행 ─────
        // drain 시작 전 sram_D_use_zero 를 0 으로 (RMW.out_RMW 가 D 로 mux 되도록).
        // generate-for 내부에서 32 always 가 같은 단일 reg 를 driving 하면
        // multi-driver 가 되므로 여기서 한 번만 설정.
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

        // ───── DUMP: 32 bank × 512 words via port-based read → $fwrite ─────
        $display("[DUMP] writing 32 bank .mem files to %0s", DUMP_DIR);
        dump_banks;

        // ───── PASS/FAIL 배너 (in-sim 구조적 검증) ─────────────────────────
        // 검증 항목:
        //   1) capture 총 push 수 == EXPECTED (mode 무관 65536)
        //   2) drain 총 pop 수    == EXPECTED
        //   3) 모든 bank-col assert 가 sim 진행 중 한 번도 $finish FATAL 안 함
        //      → 여기까지 도달한 것 자체가 증거 (FATAL 시 즉시 종료됨)
        //   4) dump task 가 32 파일 모두 fopen 성공 (실패 시 $finish FATAL)
        // 비트 정확도는 외부 Python compare 가 담당 (TB header §Vivado 절차 참고).
        if (total_captured == EXPECTED_TOTAL && total_drained == EXPECTED_TOTAL) begin
            $display("");
            $display("============================================================");
            $display("  [PASS] INTEGRATION TB A%0d_B%0d — structural checks OK", A_PREC, B_PREC);
            $display("         captured=%0d  drained=%0d  expected=%0d",
                     total_captured, total_drained, EXPECTED_TOTAL);
            $display("         (bit-exact 검증은 외부 compare 실행)");
            $display("============================================================");
        end else begin
            $display("");
            $display("============================================================");
            $display("  [FAIL] INTEGRATION TB A%0d_B%0d — count mismatch", A_PREC, B_PREC);
            $display("         captured=%0d  drained=%0d  expected=%0d",
                     total_captured, total_drained, EXPECTED_TOTAL);
            $display("============================================================");
        end

        $display("[DONE] INTEGRATION TB: A%0d_B%0d", A_PREC, B_PREC);
        $finish;
    end

    // ─── timeout 안전장치 ──────────────────────────────────────────
    initial begin
        #50_000_000;   // 50 ms = 5_000_000 cycles @ 100 MHz (정상 sim 의 ~10 배)
        $display("ERROR: timeout");
        $finish;
    end

endmodule
