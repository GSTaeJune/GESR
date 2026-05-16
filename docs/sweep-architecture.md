# Integration TB 동작 원리 — end-to-end

`tb/gemm_sram_top_tb.v` 가 어떻게 외부 입력을 받아 6 단계로 GEMM-RMW-SRAM
데이터패스를 굴리고, 무엇을 검증하고, 외부 compare 와 어떻게 연결되는지
전체 기록.

**TB 는 한 파일.** 9 정밀도 mode (A,B ∈ {2,4,8}) 모두 같은 TB 가 plusarg
로 mode 받아 처리. sweep 은 그 TB 를 9 번 invoke 하는 외부 bash wrapper.

---

## 1. 한눈에 보는 흐름

```
┌──────────────── 외부 stage (bash + Python) ──────────────────────────────┐
│                                                                          │
│ [INPUT PREP]   MXP_Tools gen / emit / ref                                │
│                  work/<LABEL>/{hw_input/*.hex, sw_ref/C_sw_*.npz}        │
│                                                                          │
│ [SIM INVOKE]   run_integration_one.sh <LABEL> <A_PREC> <B_PREC>          │
│                  xvlog + xelab + xsim ── plusarg ──► TB                  │
│                                                                          │
│ [HW DUMP]      work/<LABEL>/hw_out/bank{0..31}.mem                       │
│                                                                          │
│ [VERDICT]      MXP_Tools compare (--layout interleaved_row_major_32bank) │
│                  hw_sw n_diff = 0  →  bit-exact PASS                     │
│                                                                          │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ plusargs (A_PREC, B_PREC, WORK_DIR, DUMP_DIR)
                                   ▼
┌──────── 내부 stage (gemm_sram_top_tb.v — Verilog 한 파일) ────────────────┐
│                                                                          │
│ 1) INIT     reset 4 cy + 32-bank parallel zero-prime (1024 cy)           │
│ 2) LOAD     $readmemh 4 파일 (a_bs, b_bp, a_scale, b_scale)              │
│ 3) CONFIG   {A_PREC,B_PREC} case → 8 mode-specific 상수                  │
│ 4) DRIVE    Stage 2-A / 2-B / 3+4 (MAC) / 5-tail                         │
│              │ (capture always block 이 동시 진행)                       │
│              └─► per-col FIFO[32] push (mode 별 1/2/4 lane)              │
│ 5) PRIME    PRIME_CYC=16 cy idle (in-flight settle)                      │
│ 6) DRAIN    32 col 동시 — per-col FSM (7-state) R-CONV-ADD-W             │
│ 7) DUMP     32 bank × 512 word port-read → $fwrite bank{i}.mem           │
│ 8) BANNER   [PASS]/[FAIL] 4-invariant 구조적 검증                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

검증 두 갈래:
- **TB in-sim 구조적 검증** — 8단계 끝의 PASS/FAIL 배너 (count + assert + fopen 생존)
- **외부 비트 정확도 검증** — sim 종료 후 Python compare (HW dump vs golden NPZ)

둘 다 PASS 여야 회귀 OK.

---

## 2. 외부 wrapper — plusarg 주입 + 입출력 디렉토리

### 2.1 입력 준비 (MXP_Tools)

```bash
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8                              # 9 hex 파일
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8        # 1 npz
```

산출:
- `work/A8_B8/hw_input/a_input_BS_mxint{2,4,8}.hex` — weight bit-serial
- `work/A8_B8/hw_input/b_input_BP_mxint{2,4,8}.hex` — activation byte-parallel
- `work/A8_B8/hw_input/a_scale.hex`, `b_scale.hex` — E8M0 scale
- `work/A8_B8/sw_ref/C_sw_mxint{B}_mxint{A}.npz` — golden FP32 행렬

**Naming gotcha**: MXP_Tools 의 `--prec-a` = WEIGHT 정밀도 = 우리의 B_PREC.
`--prec-b` = ACTIVATION = A_PREC. npz 파일명 slot 순서는 `C_sw_mxint{B}_mxint{A}.npz`.

### 2.2 Sim invocation

```bash
bash sim/run_integration_one.sh A8_B8 8 8
```

내부적으로:
```bash
xvlog -sv <HardFloat + MXP + SRAM + RMW + sram_1rw_banked_mp + GEMM + top + tb>
xelab -L work gemm_sram_top_tb -snapshot gemm_sram_top_tb_snap
xsim gemm_sram_top_tb_snap -runall \
    -testplusarg "A_PREC=8" -testplusarg "B_PREC=8" \
    -testplusarg "WORK_DIR=../../../work/A8_B8" \
    -testplusarg "DUMP_DIR=../../../work/A8_B8/hw_out"
```

각 mode 가 자기 build dir `sim/build/A8_B8/` 사용 → mode 끼리 간섭 없음.

### 2.3 9-mode sweep (`sim/run_integration_sweep.sh`)

`run_integration_one.sh` 를 9 번 invoke + 매번 compare 게이트 통과 확인.
최종 출력 `ALL 9 MODES PASSED`.

자세한 sweep loop / 격리 / 병렬화 가능성은 §11.

---

## 3. TB phase 1 — INIT (reset + zero-prime)

```verilog
rst <= 1'b1;
repeat (4) @(posedge clk);   // 4 cy reset
rst <= 1'b0;
init_zero_prime;             // 32 bank parallel zero-write × 1024 word = 1024 cy
```

**`init_zero_prime` 의 동작**:
- 32 bank CEB=0, WEB=0, WMASK=0xFFFFFFFF, `sram_D_use_zero=1` (D mux 가 0 forced)
- 매 cycle word_addr 1 씩 증가 → 1024 cycle 안에 BANK_DEPTH 전체를 0 으로 채움
- 32 bank 가 독립 port 라서 동시 write — 직렬이면 32×1024 cycle 걸릴 일을 1024 cycle 에 끝냄

**왜 zero-prime 필요**: 첫 K-tile fire 시 RMW 가 prior partial sum 으로 SRAM read.
초기값이 X 면 `addRecFN(NaN, anything) = NaN` 으로 영구 오염. 0 (FP32 `0x00000000`) 으로
초기화하면 `0.0 + GEMM_dequant = GEMM_dequant` 로 첫 누적이 깔끔.

**검증**: 별도 in-sim 가드 없음. 후속 phase 의 dump 결과가 bit-exact 회귀 통과하면
zero-prime 도 OK 였다는 결론.

---

## 4. TB phase 2 — LOAD ($readmemh)

```verilog
$sformat(path, "%0s/hw_input/a_input_BS_mxint%0d.hex", WORK_DIR, B_PREC);
$readmemh(path, a_bs);
$sformat(path, "%0s/hw_input/b_input_BP_mxint%0d.hex", WORK_DIR, A_PREC);
$readmemh(path, b_bp);
$readmemh("%0s/hw_input/a_scale.hex", a_scale);
$readmemh("%0s/hw_input/b_scale.hex", b_scale);
```

배열 크기 (`tb/gemm_sram_top_tb.v:158-161`):
| 배열 | 타입 | 크기 | 의미 |
|---|---|---|---|
| `a_bs`    | `reg [31:0]`  | 4096 | weight bit-serial — `K_T × M_T × TILE × W_CYC_MAX = 4×4×32×8` |
| `b_bp`    | `reg [255:0]` | 512  | activation byte-parallel — `N_T_MAX × K_T × TILE = 4×4×32` |
| `a_scale` | `reg [7:0]`   | 512  | weight E8M0 scale |
| `b_scale` | `reg [7:0]`   | 512  | activation E8M0 scale |

**검증**: `$readmemh` 가 파일 못 찾으면 XSim warning + 배열이 X 값. 후속 DRIVE 가
X 값으로 GEMM 자극 → out_fire 도 X → capture FIFO 빈 채로 phase 8 의 PASS/FAIL
배너에서 `captured=0` 으로 [FAIL] 잡힘. silent corruption 안 됨.

`$display("LOAD OK: a_bs[0]=%h ...")` 한 줄 출력으로 사용자도 즉시 X 여부 눈으로 확인.

---

## 5. TB phase 3 — CONFIG (mode dispatch)

`{A_PREC, B_PREC}` 으로 9-way `case` 분기. 8 개 상수를 reg 에 적재:

| 상수 | 의미 | A_PREC 의존 | B_PREC 의존 |
|---|---|---|---|
| `W_CYC` | weight bit-serial cycle (1 fire 당) | × | ✓ (B8=8, B4=4, B2=2) |
| `A_FIRE_DELAY` | activation chain delay 보정 | ✓ (A8=2, A4=2, A2=0) | × |
| `FIRST_FIRE_GLOBAL` | 첫 fire 의 global cycle (start_acc pulse) | × | ✓ (B8=28, B4=20, B2=16) |
| `TOGGLE_VAL` | station Buf1/Buf2 toggle cycle (cyc_in_K) | ✓ | ✓ |
| `FIRES_PER_COL` | col 당 fire 횟수 (capture 가드) | ✓ (A8=2048, A4=1024, A2=512) | × |
| `N_T_LOGICAL` | logical N-tile loop 횟수 | ✓ (A8=4, A4=2, A2=1) | × |
| `A_CTRL_CODE` | Station 정밀도 2비트 | ✓ (A_INT8/A_INT4/A_INT2) | × |
| `W_CTRL_CODE` | Wcontrol 정밀도 2비트 | × | ✓ (W_INT8/W_INT4/W_INT2) |

**EXPECTED_TOTAL (PASS 게이트 기준)** = `32 × 2048 = 65536` — mode 무관:
- A8: 2048 fires × 1 lane = 2048 RMW/col
- A4: 1024 fires × 2 lane = 2048 RMW/col
- A2:  512 fires × 4 lane = 2048 RMW/col

**검증**: default 케이스에서 `$display("ERROR: unsupported (A_PREC=%d, B_PREC=%d)")
+ $finish`. 지원 안 되는 조합 invoke 시 즉시 종료.

---

## 6. TB phase 4 — DRIVE (Stage 2-A / 2-B / 3+4 / 5-tail)

GEMM 자극 driving. 4 단계 task 호출:

### Stage 2-A: 초기 B station load (32 cycle)
- 매 cycle col `c` (0..31) 에 활성화 + scale broadcast
- station data chain (col 31→0 좌향 전파) 으로 끝나면 모든 col 의 Station Buf1 에 자기 col 의 activation 적재
- mode 별 packing: A8 = `b_bp[n*K_T*TILE + k*TILE + c]` 그대로, A4 = top/bot byte 결합, A2 = 4-lane bit packing

### Stage 2-B: settle + station selector flip (16+16 cy)
- 16 cy idle (chain delay ≤ 15 cy 흡수)
- `in_station_control=1` 로 flip → 32 col 이 Buf1 broadcast 상태
- 16 cy 추가 settle

### Stage 3+4: MAC sweep (메인 phase)
- 4중 루프 `n_t → k_t → m_t → o` (K-outermost)
- 매 cycle 6 신호 driving: `in_a` (weight bit), `in_control` (PE pp toggle),
  `in_station_control` (Station pp toggle), `in_start_accumulate` (pulse 1회),
  `in_Wcontrol` (W_IDLE → W_CTRL), `in_scale_weight` (a_scale[m_idx])
- `m_t==1 && o<32` 구간: 다음 tile B/scale prefetch (반대 buffer 로)
- `capture_en <= 1` 으로 capture always block 켜기

### Stage 5: tail (TAIL_CYC + POST_TAIL)
- `W_CTRL_CODE` 45 cy 유지 (in-flight fire 처리용)
- `W_IDLE` 전환 + POST_TAIL=20 cy
- `capture_en <= 0` 으로 종료

**검증**: DRIVE 자체에 inline assert 없음. 대신 다음 phase 의 capture 가 정상 fire
받는지로 간접 검증 (잘못 driving 시 fire 누락 → count mismatch [FAIL] 배너).

---

## 7. TB phase 5 — Capture (DRIVE 와 동시 진행)

`always @(posedge clk)` block, `capture_en` 동안만 동작. 매 cycle 32 col 검사.

```verilog
for (ci = 0; ci < 32; ci = ci + 1) begin
    if (out_fire[ci] && (fire_cnt_per_col[ci] < FIRES_PER_COL)) begin
        // (1) FIFO overflow guard
        if (fifo_wp[ci] >= FIFO_DEPTH) begin
            $display("FATAL: FIFO overflow col=%0d wp=%0d depth=%0d fire_cnt=%0d", ...);
            $finish;
        end

        // (2) fire 인덱스 → (n_t, k_t, m_t, m_in) 디코딩
        fc = fire_cnt_per_col[ci];
        n_t_dec  = fc / (K_T * M_T * TILE_SIZE);
        k_t_dec  = (fc / (M_T * TILE_SIZE)) % K_T;
        m_t_dec  = (fc / TILE_SIZE) % M_T;
        m_in_dec = fc % TILE_SIZE;
        m_g      = m_t_dec * TILE_SIZE + m_in_dec;

        // (3) mode 별 lane 디코딩 + FIFO push
        case (A_PREC)
            8: begin  // 1 lane per fire
                acc_lane_a8 = $signed(out_accumulate[60*ci +: 21]);
                acc_int32   = {{11{acc_lane_a8[20]}}, acc_lane_a8};
                sc_lane     = out_scale[36*ci +: 9];
                ng_dec      = n_t_dec * 32 + ci;
                flat_dec    = m_g * N_DIM + ng_dec;
                // (4) bank-col 매핑 assert
                if ((flat_dec & 5'h1F) != ci[4:0]) $finish FATAL ...
                fifo_int  [ci][fifo_wp[ci]] <= acc_int32;
                fifo_scale[ci][fifo_wp[ci]] <= sc_lane;
                fifo_addr [ci][fifo_wp[ci]] <= flat_dec[18:0];
                fifo_wp[ci] = fifo_wp[ci] + 1;
            end
            4: begin /* 2 lane: top (n_pair*64+32+ci) + bot (n_pair*64+ci) */ end
            2: begin /* 4 lane: 96+ci / 64+ci / 32+ci / 0+ci */ end
        endcase
        fire_cnt_per_col[ci] <= fire_cnt_per_col[ci] + 1;
    end
end
```

### Capture 가 검증하는 4 가지

| 검증 | 위치 | 실패 시 |
|---|---|---|
| FIFO overflow | guard 진입점 | $display FATAL + $finish |
| FIRES_PER_COL 초과 fire | outer `if` 조건 | 조용히 drop (Stage 5 tail 의 가짜 fire 보호) |
| bank-col 매핑 정확 (`flat & 0x1F == ci`) | mode 별 7개 assert (A8 1 + A4 2 + A2 4) | $display FATAL + $finish |
| lane slice 값 stable | `$signed(out_accumulate[60*ci +: N])` | X 값 propagation (sim warn) |

**bank-col assert 의 의미**: spec § 4 의 수학적 증명 (`128 % 32 = 0`, `bank = n_pos % 32 = ci`)
이 RTL 에서도 holds 함을 매 fire 마다 확증. 한 번이라도 어긋나면 즉시 FATAL — silent
data corruption 차단. 9/9 sweep PASS 가 곧 16384 element × 9 mode = 147456 lane-event
모두 매핑 정확했다는 증거.

---

## 8. TB phase 6 — PRIME (in-flight settle)

```verilog
repeat (PRIME_CYC) @(posedge clk);   // PRIME_CYC = 16
```

DRIVE 종료 직후 in-flight pipeline (MXP Accumulator chain + Station chain) 의 마지막
fire 들이 `out_fire`/`out_accumulate` reg 에 도착하기까지의 여유. capture always block
은 이 동안에도 깨어 있어서 늦은 fire 들 잡음.

**검증**: 없음. PRIME 이 부족하면 capture 가 일부 fire 놓침 → count mismatch [FAIL] 배너.

---

## 9. TB phase 7 — DRAIN (32-stream parallel FSM)

`generate for (dc = 0; dc < 32) begin : g_drain always @(posedge clk) ... endgenerate` —
32 개의 독립 always block. 각 col 의 RMW + bank port 만 건드림.

### 7-state FSM (state[dc] 4-bit reg)

| State | 동작 | 다음 |
|---|---|---|
| 0 IDLE | `fifo_rp < fifo_wp` 면 READ 발사: CEB=0, WEB=1, A=addr>>5 | 1 |
| 1 READ | CEB=1 (read 1 cycle 만) | 2 |
| 2 WAIT_R | sram_Q 유효해진 시점 → RMW.in_GEMM/scale 드라이브 | 3 |
| 3 SET_WAIT | drain_wait=0 초기화 (degenerate state) | 4 |
| 4 WAIT_RMW | drain_wait++ 까지 DRAIN_RMW_WAIT-1 (= 7) | 5 |
| 5 WRITE | CEB=0, WEB=0, A=addr>>5, WMASK=0xFFFFFFFF | 6 |
| 6 SETTLE | CEB=1, WEB=1, fifo_rp++ | 0 |

**총 cycle**: state 0 (1 cy entry) → 5 (WRITE 발사) 까지 11 cy. RMW pipeline 5 cy →
약 6 cy slack. 충분히 여유.

### Drain 의 cycle 동기화 포인트

| 시점 | 무엇이 일어남 |
|---|---|
| state 0 → 1 의 edge | bank[dc].A, CEB, WEB 가 새 read 발사 |
| state 1 → 2 의 edge | bank[dc].Q 가 sram leaf 의 PIPELINE=0 read latency 통과 후 유효 |
| state 2 → 3 의 edge | RMW[dc].in_GEMM, scale 드라이브 (NBA) |
| state 4 → 5 의 edge | RMW.out_RMW 가 (L_CONV+L_ADD=5 cy 후) 유효 — gemm_sram_top.v 의 mux 가 sram[dc].D 에 라우팅 |
| state 5 → 6 의 edge | bank[dc] 가 RMW 결과를 write |

### 동기화 게이트

```verilog
task wait_drain_complete;
    // 모든 col 이 IDLE && fifo_rp == fifo_wp 인 상태가 10 cycle 연속 유지될 때까지 대기
endtask
```

10 cycle 의 보수적 polling — 어느 col 이라도 mid-flight 면 ` idle_count` reset.

### Drain 이 검증하는 것

| 검증 | 위치 | 실패 시 |
|---|---|---|
| 32 col 동시 진행 — bank port 충돌 0 | 매핑 자체 (spec § 4) | 매핑 어긋남 → capture assert FATAL (drain 전에 잡힘) |
| 매 RMW dispatch 가 IDLE 로 복귀 | wait_drain_complete | 영원히 안 끝남 → timeout (`#50_000_000`) FATAL |
| 모든 FIFO 가 drain (fifo_rp == fifo_wp) | wait_drain_complete | timeout |

---

## 10. TB phase 8 — DUMP ($writememh per bank)

```verilog
task dump_banks;
    for (bi = 0; bi < NB; bi = bi + 1) begin
        $sformat(path, "%0s/bank%0d.mem", DUMP_DIR, bi);
        fd = $fopen(path, "w");
        if (fd == 0) $finish FATAL ...
        for (w = 0; w < 512; w = w + 1) begin
            sram_CEB[bi] <= 0; sram_WEB[bi] <= 1; sram_A[bi*AW +: AW] <= w;
            @(posedge clk); sram_CEB[bi] <= 1;
            @(posedge clk);
            q_word = sram_Q[bi*32 +: 32];
            $fwrite(fd, "%08x\n", q_word);
        end
        $fclose(fd);
    end
endtask
```

512 word/bank — 128×128 workload 의 사용 범위 (flat = m*128 + n, 16384 / 32 = 512).
나머지 512..1023 은 zero-prime 그대로 0 으로 남음.

**검증**: $fopen 실패 시 즉시 FATAL. dump 형식은 `%08x\n` (FP32 비트 패턴 hex, 한 줄에
한 word) — MXP_Tools 의 `read_writememh_fp32` 가 같은 형식 기대.

---

## 11. TB phase 9 — PASS/FAIL 배너

```verilog
if (total_captured == EXPECTED_TOTAL && total_drained == EXPECTED_TOTAL) begin
    $display("============================================================");
    $display("  [PASS] INTEGRATION TB A%0d_B%0d — structural checks OK", A_PREC, B_PREC);
    $display("         captured=%0d  drained=%0d  expected=%0d", ...);
    $display("         (bit-exact 검증은 외부 compare 실행)");
    $display("============================================================");
end else begin
    $display("  [FAIL] ... count mismatch");
end
```

### 4 가지 invariant

| Invariant | 어떻게 검증되나 | 실패 시 |
|---|---|---|
| ① capture 총 push 수 == 65536 | `total_captured` 누적 합 | [FAIL] 배너 |
| ② drain 총 pop 수 == 65536 | `total_drained` 누적 합 | [FAIL] 배너 |
| ③ assert 가 한 번도 FATAL 안 함 | 배너 도달 자체가 증거 | 도달 전 $finish — 배너 안 찍힘 |
| ④ dump fopen 모두 성공 | dump_banks 내 가드 | 도달 전 $finish |

①②는 명시적 check + [FAIL] 배너.
③④는 sim 이 phase 9 까지 도달하지 못해서 배너 자체가 출력 안 됨 (FATAL 빠른 종료).

---

## 12. TB 의 안전장치 — Timeout

```verilog
initial begin
    #50_000_000;   // 50 ms = 5_000_000 cycle @ 100 MHz
    $display("ERROR: timeout");
    $finish;
end
```

정상 sim 은 A8_B8 기준 ~1 ms (953,985 ns). 50 ms 면 정상의 50 배 — deadlock /
영원한 wait_drain_complete 등 무한 루프 보호.

---

## 13. 외부 compare — 비트 정확도 검증

TB 가 dump 한 32 .mem 파일을 `python -m mxp_tools compare` 가 처리.

```python
# MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_32bank
def interleaved_row_major_32bank(bank_idx, word_offset, M, N):
    flat = word_offset * 32 + bank_idx     # forward: bank=flat%32, word=flat//32 의 역
    if flat >= M * N: return None
    return divmod(flat, N)                  # (m, n)
```

`gather_banks` 가 32 .mem 을 읽어서 `(M, N)` FP32 행렬 재구성. 모든 (m, n) position
이 정확히 한 번 커버되는지 (duplicate / missing) 같이 확인.

이후 `diff_3way(C_hw, C_sw, C_fp32)` 가 세 쌍의 diff 통계:

```
pair                   max          rmse      mean_abs      snr_dB    n_diff
hw_sw            0.000e+00     0.000e+00     0.000e+00        inf         0
sw_fp32          6.587e-01     1.348e-01     1.068e-01      38.51     16384
hw_fp32          6.587e-01     1.348e-01     1.068e-01      38.51     16384
```

**bit-exact PASS 조건**: `hw_sw n_diff == 0`. `max == 0` (어떤 element 도 다른 비트
없음). `snr_dB == inf` (rmse=0 의 결과).

`sw_fp32` 와 `hw_fp32` 는 SW golden 과 reference FP32 의 diff — 양자화 오차이므로
n_diff > 0 이 정상 (16384 = M*N 모든 element 가 양자화로 변함). 우리 관심사 아님.

---

## 14. 모든 검증 invariant 한 곳에

| # | Invariant | 검증 위치 | 실패 형태 |
|---|---|---|---|
| 1 | Plusarg 정상 주입 | initial $value$plusargs | 기본값 fallback (A8_B8) |
| 2 | Hex 입력 파일 존재 | $readmemh | 배열이 X — 후속 capture 가 빈 결과 |
| 3 | Mode 조합 supported | CONFIG case default | $display ERROR + $finish |
| 4 | FIFO 미오버플로 | capture 진입점 | $display FATAL + $finish |
| 5 | col j → bank j 매핑 | capture 안 7 개 assert | $display FATAL + $finish |
| 6 | Drain 완전 처리 | wait_drain_complete | timeout (50 ms) |
| 7 | Dump 파일 open | dump_banks | $display FATAL + $finish |
| 8 | Capture count == EXPECTED | PASS/FAIL banner | [FAIL] 배너 |
| 9 | Drain count == EXPECTED | PASS/FAIL banner | [FAIL] 배너 |
| 10 | Sim timeout 미달 | initial #50_000_000 | $finish (deadlock 보호) |
| 11 | HW dump bit-exact vs SW | 외부 Python compare | n_diff > 0 → sweep exit 1 |
| 12 | 모든 (m, n) 정확히 한 번 커버 | gather_banks 내부 | duplicate / missing 예외 |

---

## 15. 데이터패스 wiring (col j → bank j)

```
                ┌──────────────── gemm_sram_top.v (pure structural) ──────────────┐
                │                                                                  │
   GEMM ───►   │  out_fire[31:0], out_accumulate[32*60-1:0], out_scale[32*36-1:0] │
   stimuli      │      │ (per col j, 동시 진행)                                   │
                │      ▼                                                           │
                │  ┌─────────────────────── generate for (c=0..31) ────────────┐  │
                │  │                                                            │  │
                │  │   RMW[c]:                                                  │  │
                │  │     in_GEMM  ◄── rmw_in_GEMM[c*32 +: 32]   (TB drives)    │  │
                │  │     scale    ◄── rmw_scale  [c*9  +: 9 ]   (TB drives)    │  │
                │  │     in_SRAM  ◄── Q[c*32 +: 32]             (bank[c] 출력) │  │
                │  │     out_RMW  ──► rmw_out_c                                 │  │
                │  │                       │                                    │  │
                │  │     sram_D_w[c*32 +: 32] = sram_D_use_zero ? 0 : rmw_out_c │  │
                │  │                       │                                    │  │
                │  │   sram_1rw_banked_mp.bank[c]:                              │  │
                │  │     D     ◄── sram_D_w[c*32 +: 32]                        │  │
                │  │     A     ◄── A[c*10 +: 10]                (TB drives)    │  │
                │  │     CEB   ◄── CEB[c]                       (TB drives)    │  │
                │  │     WEB   ◄── WEB[c]                       (TB drives)    │  │
                │  │     WMASK ◄── WMASK[c*32 +: 32]            (TB drives)    │  │
                │  │     Q     ──► RMW[c].in_SRAM                              │  │
                │  │                                                            │  │
                │  └────────────────────────────────────────────────────────────┘  │
                └──────────────────────────────────────────────────────────────────┘
```

col j 의 RMW[j] 는 오직 bank[j] 만 read/write. 32 col 모두 동시 처리 가능. 단,
실제 TB drain 은 capture/replay 모델이라 DRIVE 후 DRAIN — 진정한 real-time
parallel 은 추후 phase (kickoff §1 후보).

---

## 16. col j → bank j 매핑 — 모든 mode 의 수학

`bank = flat % 32`, `flat = m × 128 + n_pos`. `128 % 32 = 0` 이라 `bank = n_pos % 32`.

| Mode | n_pos | bank | word (bank j 내부) |
|---|---|---|---|
| A8 | `n_t*32 + j` | `j` | `m*4 + n_t` |
| A4 top | `n_pair*64 + 32 + j` | `j` | `m*4 + n_pair*2 + 1` |
| A4 bot | `n_pair*64 + j` | `j` | `m*4 + n_pair*2` |
| A2 lane 0 | `96 + j` | `j` | `m*4 + 3` |
| A2 lane 1 | `64 + j` | `j` | `m*4 + 2` |
| A2 lane 2 | `32 + j` | `j` | `m*4 + 1` |
| A2 lane 3 | `j` | `j` | `m*4 + 0` |

→ 모든 mode 의 모든 lane 에서 `bank = j`. **충돌 0**.

bank j 내부 word range: m ∈ [0..127] × (n_t / n_pair / lane ∈ [0..3]) = word ∈ [0..511].

---

## 17. Sweep — 9-mode loop 그 자체

```bash
# sim/run_integration_sweep.sh 핵심
for A_P in 2 4 8; do
  for B_P in 2 4 8; do
    LABEL="A${A_P}_B${B_P}"
    # (1) MXP_Tools input prep
    (cd MXP_Tools && python -m mxp_tools gen/emit/ref ...)
    # (2) HW sim → 32 .mem dump
    bash sim/run_integration_one.sh "${LABEL}" "${A_P}" "${B_P}"
    # (3) compare gate
    (cd MXP_Tools && python -m mxp_tools compare \
        --ref ../work/${LABEL}/sw_ref/C_sw_mxint${B_P}_mxint${A_P}.npz \
        --hw-banks bank{0..31}.mem \
        --layout interleaved_row_major_32bank)
  done
done
```

### Mode 간 격리

각 mode 가 자기 디렉토리 사용 — 동시/순차 실행 모두 안전:
- `sim/build/<LABEL>/` — xvlog/xelab/xsim 산출물
- `work/<LABEL>/hw_input/` — 입력 hex
- `work/<LABEL>/sw_ref/` — golden npz
- `work/<LABEL>/hw_out/` — HW dump (32 .mem)
- xsim process 가 매 invoke 새로 시작 → state 0 부터

### 최종 게이트

```
SWEEP RESULT: 9/9 PASS
PASSED: A2_B2 A2_B4 A2_B8 A4_B2 A4_B4 A4_B8 A8_B2 A8_B4 A8_B8
ALL 9 MODES PASSED
```

`ALL 9 MODES PASSED` 가 최종 회귀 OK 신호. 한 개라도 FAIL 시 `FAILED: <LABEL...>`
+ sweep exit 1.

병렬 실행 가이드: `sim/run_integration_parallel.sh` (subagent 9-way dispatch
template 출력).

---

## 18. Vivado GUI 에서 돌리기

### 절차
1. **입력 stage** — bash 에서 MXP_Tools 미리 실행 (§2.1).
2. **`gemm_sram.xpr` 열기** — sim top = `gemm_sram_top_tb` 자동 인식.
3. **Simulation Settings** (Flow Navigator → Simulation → Simulation Settings):
   ```
   Simulation > Simulator
   xsim.simulate.xsim.more_options:
       -testplusarg "A_PREC=8" -testplusarg "B_PREC=8"
       -testplusarg "WORK_DIR=../../../../work/A8_B8"
       -testplusarg "DUMP_DIR=../../../../work/A8_B8/hw_out"
   ```
   path 는 Vivado sim 스크래치 `gemm_sram.sim/sim_1/behav/xsim/` 기준 상대경로.
4. **Run Simulation → Run Behavioral Simulation**.
5. **Tcl Console** 에서 출력 확인:
   - `LOAD OK: a_bs[0]=...` (입력 정상 로드)
   - `CONFIG: A=8 W=8 ...` (mode 적용)
   - `DRIVE DONE: captured 65536 fires (expected 65536)`
   - `DRAIN DONE: 65536 entries processed`
   - `[PASS] INTEGRATION TB A8_B8 — structural checks OK` ← 최종 in-sim 결과
6. **외부 compare** (bash) — bit-exact 검증:
   ```bash
   cd MXP_Tools
   BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..31})
   python -m mxp_tools compare \
       --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
       --hw-banks ${BANKS} --layout interleaved_row_major_32bank
   ```
   `hw_sw n_diff = 0` 이면 비트 정확도 OK.

Vivado GUI 의 장점: waveform viewer 로 fire timing / drain FSM 의 cycle-by-cycle
관찰 가능. 디버깅 시 유용.

---

## 19. 단일 모드 실행 (회귀 빠르게)

전체 sweep ~11분, 단일 mode ~2분. A8_B8 만 빠르게:

```bash
# 입력 생성
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8
cd ..

# HW sim — TB 의 in-sim 검증 (PASS/FAIL 배너)
bash sim/run_integration_one.sh A8_B8 8 8

# 외부 비트 정확도 검증
cd MXP_Tools
BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..31})
python -m mxp_tools compare \
    --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
    --hw-banks ${BANKS} \
    --layout interleaved_row_major_32bank
```

---

## 20. 새 mode 추가 절차 (참고)

가령 A1 (1-bit activation) 추가 시:

1. **MXP_Tools 측**: `mxp_tools/quant.py` 의 quant 함수에 1-bit 분기 추가.
2. **MXP RTL 측**: Station / Accumulator 의 precision 디코더에 A_INT1 케이스 추가 (`../MXP/`).
3. **TB CONFIG case**: `tb/gemm_sram_top_tb.v` 의 CONFIG 9-way case 에 새 mode 의 8 상수 추가.
4. **Capture lane 디코딩**: A_PREC=1 일 때 col 당 몇 lane 인지 결정 → case 안 새 분기 + bank-col assert.
5. **EXPECTED_TOTAL**: 새 mode 도 `32 × 2048 = 65536` 으로 떨어지는지 확인. 다르면 mode 별 EXPECTED 로 분기.
6. **Sweep loop**: `run_integration_sweep.sh` 의 `for A_P in 2 4 8` 에 `1` 추가.

TB 의 plusarg 진입점과 mode-aware 디스패치 구조 자체는 그대로 — 새 mode 도 자동
같은 TB 가 처리.

---

## 21. 관련 파일 인덱스

### RTL
- `gemm_sram.srcs/sources_1/new/gemm_sram_top.v` — 32-RMW + 32-bank wrapper top.
- `gemm_sram.srcs/sources_1/new/sram_1rw_banked_mp.v` — per-bank port SRAM wrapper.
- `gemm_sram.srcs/sources_1/new/RMW.v` — INT→FP32 + FP32 add (5 cy pipeline).
- `gemm_sram.srcs/sources_1/new/int_to_fp32.v`, `fp32_adder.v` — RMW 의 구성요소.
- `gemm_sram.srcs/sources_1/new/GEMM.v` — MXP TOP wrapper.
- `gemm_sram.srcs/sources_1/imports/Desktop/sram/rtl/sram_1rw.v` — leaf bank.
- `gemm_sram.srcs/sources_1/imports/Desktop/MXP/...` — MXP 컴퓨트 IP.

### TB
- `tb/gemm_sram_top_tb.v` — 통합 TB (본 doc 의 주된 대상).
- `tb/sram_1rw_banked_mp_tb.v` — 새 wrapper 단위 TB.
- `tb/rmw_tb.v`, `int_to_fp32_tb.v`, `fp32_adder_tb.v`, `rmw_smoke_tb.v` — 단위 TB.

### Sim
- `sim/run_integration_one.sh` — 단일 mode (xvlog/xelab/xsim).
- `sim/run_integration_sweep.sh` — 9-mode loop + compare.
- `sim/run_integration_parallel.sh` — 9-way 병렬 dispatch 가이드.
- `sim/run_top_elab.sh` — top elab smoke.
- `sim/run_sram_mp.sh` — 새 wrapper 단위 회귀.
- `sim/run_rmw*.sh`, `run_int_to_fp32.sh`, `run_fp32_adder.sh` — 산술기 단위.

### MXP_Tools
- `MXP_Tools/mxp_tools/cli.py` — `gen / emit / ref / compare / viz / rmw-gen` 서브커맨드.
- `MXP_Tools/mxp_tools/gemm.py` — golden GEMM (NumPy matmul).
- `MXP_Tools/mxp_tools/quant.py` — MX quant.
- `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_32bank` — 32-bank 역매핑.
- `MXP_Tools/mxp_tools/compare.py::diff_3way` — 비트 정확도 비교.
- `MXP_Tools/tests/test_hwio_interleaved.py` — 매핑 round-trip 검증.

### Spec / 참고
- `precision_modes_protocol.md` — MXP 의 driving protocol 원본 spec.
- `docs/superpowers/notes/mxp-driving-sequence.md` — gemm_sram TB 관점의 driving 시퀀스.
- `docs/superpowers/notes/lane-to-c-mapping.md` — A4/A2 lane → C[m,n] 매핑 표.
- `docs/superpowers/specs/2026-05-14-integration-design.md` — phase 2 통합 spec.
- `docs/superpowers/specs/2026-05-15-rmw-32x-design.md` — phase 3 32× 확장 spec.
