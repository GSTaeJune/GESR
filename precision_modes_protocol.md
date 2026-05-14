# MXP Precision Modes Driver Protocol

**Version**: 1.0 (2026-05-11)
**Validated**: 9 modes (A∈{8,4,2} × W∈{8,4,2}), full GEMM 128×128×128 INT MAC + Scale 100% PASS.
**Architecture**: 32×32 systolic array, bit-serial weight, byte-parallel activation, mixed-precision via N-direction lane packing.

이 문서는 MXP TOP 모듈을 다른 AI accelerator 프로젝트의 GEMM 코어로 instantiate 할 때 필요한 driver protocol 을 정리한다. 9가지 (A, W) precision 조합 전부에서 실측 검증됨.

---

## 1. Mode Matrix

| Mode | A_ctrl (in_Station_control) | W_ctrl (in_Wcontrol) | A_fire_delay | W_CYC | N_T_logical | SA capacity | W cy/K-tile | FIRES_PER_COL | Validated cycle |
|---|---|---|---|---|---|---|---|---|---|
| A8W8 | A_INT8 (2'b00) | W_INT8 (2'b11) | 2 (fire_q2) | 8 | 4 | 32×32 | 1024 | 2048 | **16486** |
| A8W4 | A_INT8 (2'b00) | W_INT4 (2'b10) | 2 | 4 | 4 | 32×32 | 512 | 2048 | **8294** |
| A8W2 | A_INT8 (2'b00) | W_INT2 (2'b01) | 2 | 2 | 4 | 32×32 | 256 | 2048 | **4191** |
| A4W8 | A_INT4 (2'b01) | W_INT8 (2'b11) | 1 (fire_q1) | 8 | 2 | 32×64 | 1024 | 1024 | **8293** |
| A4W4 | A_INT4 (2'b01) | W_INT4 (2'b10) | 1 | 4 | 2 | 32×64 | 512 | 1024 | **4197** |
| A4W2 | A_INT4 (2'b01) | W_INT2 (2'b01) | 1 | 2 | 2 | 32×64 | 256 | 1024 | **2142** |
| A2W8 | A_INT2 (2'b10) | W_INT8 (2'b11) | 0 (acc_fire) | 8 | 1 | 32×128 | 1024 | 512 | **4192** |
| A2W4 | A_INT2 (2'b10) | W_INT4 (2'b10) | 0 | 4 | 1 | 32×128 | 512 | 512 | **2142** |
| A2W2 | A_INT2 (2'b10) | W_INT2 (2'b01) | 0 | 2 | 1 | 32×128 | 256 | 512 | **1119** |

**Notes:**
- `A_IDLE = 2'b11`, `W_IDLE = 2'b00`. Drive these between MAC bursts.
- `A_fire_delay`: out_fire register stages relative to inner acc_fire. A8 = 2-stage adder tree, A4 = 1-stage, A2 = 0 (direct).
- `W_CYC`: cnt cycles per fire (Accumulator internal counter rolls over to fire).
- `N_T_logical`: number of activation snapshots required to cover N=128. A_INT2 packs 4 N-positions per byte, A_INT4 packs 2, A_INT8 packs 1.
- `FIRES_PER_COL` = `N_T_logical * K_T * M_T * TILE_SIZE` (with K_T=M_T=4, TILE_SIZE=32).
- **Cycle count includes**: Stage 2 setup (32 cy B load + 32 cy settle) + Stage 3+4 MAC sweep + last-fire latency.

**Architecture invariant — A_ctrl 와 W_ctrl 직교**: `Accumulator_Col` 의 mode mux 는 A_ctrl(`Mode_oh` one-hot) 만 보고 lane sign-ext 와 out_accumulate 폭을 결정. `in_Wcontrol` 는 내부 4개 `Accumulator` 의 cnt rollover threshold 만 결정. 두 신호는 **runtime-independent** — K-tile boundary 정렬 보장 하에 동적 변경 가능 (단, K-tile mid 변경 시 partial sum 깨짐).

---

## 2. Lane → N-Position Packing

### A_INT8 (1 lane / col)
- 1 station byte = 1 INT8 activation per col → 1 N-position per fire
- Layout: byte[7:0] = INT8 act for N = (n_t × 32) + c, where c = SA col index
- out_accumulate slot: bits [60n+20:60n+0] = s2 (21-bit signed full 8b dot product)
- out_scale slot: bits [36n+8:36n+0] = comb_s0

### A_INT4 (2 lanes / col, "V3 N-pair packing")
- 1 station byte = 2 INT4 activations (top + bot nibble), 2 N-positions per fire
- Layout: byte[7:4] = top → lanes 0+1 → s1_a path (N = n_pair × 64 + 32 + c)
- Layout: byte[3:0] = bot → lanes 2+3 → s1_b path (N = n_pair × 64 + 0 + c)
- out_accumulate slot: bits [60n+35:60n+18] = s1_a (top), [60n+17:60n+0] = s1_b (bot). Each 18-bit signed.
- out_scale slot: bits [36n+17:36n+9] = comb_s0 (top), [36n+8:36n+0] = comb_s1 (bot)
- File mapping: `b_input_mxint4.hex` 의 n_t index — top file = `2*n_pair + 1`, bot file = `2*n_pair + 0`.

### A_INT2 (4 lanes / col, "V3 4-way packing")
- 1 station byte = 4 INT2 activations, 4 N-positions per fire (covers N=128 in 1 snapshot)
- Layout:
  - byte[7:6] = lane0 → out_INT2_0 / comb_s0 → N = 96 + c
  - byte[5:4] = lane1 → out_INT2_1 / comb_s1 → N = 64 + c
  - byte[3:2] = lane2 → out_INT2_2 / comb_s2 → N = 32 + c
  - byte[1:0] = lane3 → out_INT2_3 / comb_s3 → N =  0 + c
- out_accumulate slot: bits [60n+59:60n+45] = lane0, [60n+44:60n+30] = lane1, [60n+29:60n+15] = lane2, [60n+14:60n+0] = lane3. Each 15-bit signed (= acc_len).
- out_scale slot: bits [36n+35:36n+27] = comb_s0, [36n+26:36n+18] = comb_s1, [36n+17:36n+9] = comb_s2, [36n+8:36n+0] = comb_s3.
- File mapping: `b_input_mxint2.hex` 의 n_t index — lane0/1/2/3 ← file n_t = 3/2/1/0.

**Scale formula** (모든 A/W 조합 공통):
```
comb_s = act_scale[i] + weight_scale - 127   (E8M0, 9-bit signed)
```

---

## 3. Driver Protocol — 5 Input Chains

### Chain topology (TOP.v)

```
in_a (weight bit-serial)
    → SA row entry, NUM_ROW independent rows. No chain — each row drives its leftmost PE directly.

in_b + station data chain (in_b, in_Scale_Activation, in_Station_control, in_loadEN, in_station_loadEN)
    → col 31 entry, single-direction leftward propagate (col 31 → col 0).
    → Chain delay per col c = (31 - c) cycles.
    → b_in and station data deliver to PE row 0 of col c at cycle (TB_drive_cyc + 31 - c).

in_station_control (Buf1↔Buf2 broadcast selector)
    → col num_col/2 = 16 entry, SYMMETRIC outward (col 16 → cols 15/17 → 14/18 → ... → 0/31).
    → Chain delay per col c = |c - 16| cycles, max 15.
    → Symmetric topology guarantees toggle wave aligns with each col's fire timing for continuous MAC across K-tiles.

in_start_accumulate + in_Wcontrol + in_scale_weight (accumulator chain)
    → col num_col/2 = 16 entry, SYMMETRIC outward (same as in_station_control).
    → Chain delay per col c = |c - 16| cycles, max 15.
    → All cols receive identical weight scale stream (broadcast).
```

### Sequencing rules

**Stage 0 — Reset**
- `rst = 1` for 4+ cycles. Then `rst = 0`. Wait 4+ cycles for settle.

**Stage 1 — Memory load (host responsibility)**
- Load weight (bit-serial), activation, scale_act, scale_weight, ref_c from host memory.

**Stage 2-A — Initial B load (n_t=0, k_t=0)**
- For 32 cycles (TILE_SIZE):
  - `in_b` = packed activation byte per row (A_INT8: 1×8, A_INT4: 2×4 nibbles, A_INT2: 4×2-bit)
  - `in_Scale_Activation[31:0]` = packed scale bytes (per lane)
  - `in_Station_control` = A_INT8 / A_INT4 / A_INT2 per mode
  - `in_loadEN` = 32'hFFFFFFFF (all rows)
  - `in_station_loadEN` = 1
  - `in_station_control` = 0 (Buf1 write target)
- 32 cycles deliver one column-set to each station (col 31 to col 0 leftward).

**Stage 2-B — Settle + initial Buf1 toggle**
- For ~32 cycles: hold A_IDLE, loadEN=0.
- Sub-sequence (16 cy idle + 1 cy toggle + 14 cy settle): split to allow `in_station_control = 1` to propagate via symmetric chain (max delay 15) so broadcast = Buf1 on all 32 cols BEFORE Stage 3 starts.

**Stage 3+4 — MAC sweep** (nested loop: n_t → k_t → m_t → o)
- Inner loop o = 0 .. (TILE_SIZE × W_CYC − 1), 1 weight bit per cycle.
- `cyc_in_K` = `m_t × (TILE_SIZE × W_CYC) + o`, range 0 .. (M_T × TILE_SIZE × W_CYC − 1).
- `cyc_global` = absolute TB cycle since MAC start.

**Per-cycle drive (loop body):**
1. **Weight bit** `in_a[NUM_ROW-1:0]` = `mem_a[k_t × M_T × TILE_SIZE × W_CYC + m_t × TILE_SIZE × W_CYC + o]`
2. **PE in_control toggle** at K-tile boundary (`cyc_in_K == 0`, `k_t > 0 || n_t > 0`): `cur_pp = ~cur_pp`. Drive `in_control = {NUM_ROW{cur_pp}}`.
3. **Station in_control toggle** at K-tile boundary (`cyc_in_K == TOGGLE`, `k_t > 0 || n_t > 0`): `cur_st_pp = ~cur_st_pp`. Drive `in_station_control = cur_st_pp`.
   - TOGGLE per mode (between K-tile last fire and next K-tile first fire of col 16, exact mid):
     ```
     TOGGLE = W_CYC × M_T × TILE_SIZE + first_fire − (W_CYC × M_T × TILE_SIZE) ≈ varies by mode
     ```
   - **Measured values**: A8W8=24, A8W4=24 (TBD verify), A8W2=21, A4W8=24 (TBD), A4W4=21, A4W2=20, A2W8=22, A2W4=20, A2W2=19.
4. **start_accumulate**: 1 cycle pulse at `cyc_global == 17` (only once for first K-tile of first n_t).
5. **Wcontrol**: `cyc_global < 17` → W_IDLE, else W_INT{8,4,2} per mode. Hold throughout the entire 16 K-tile × N_T sweep.
6. **scale_weight drive**: at `cyc_global == FIRST_FIRE + W_CYC × m`, m = 0..(FIRES_PER_COL−1):
   ```
   m_idx = (cyc_global − FIRST_FIRE) / W_CYC
   in_scale_weight = mem_a_scale[m_idx % (K_T × M_T × TILE_SIZE)]
   ```
   - FIRST_FIRE formula: `17 + 1 + W_CYC + A_fire_delay`
     - A8W8: 28, A8W4: 24, A8W2: 22
     - A4W8: 27, A4W4: 23, A4W2: 21
     - A2W8: 26, A2W4: 22, A2W2: 20
7. **Prefetch** (m_t == 1, o < 32): drive next (n_t', k_t') tile's `in_b` + `in_Scale_Activation` into station opposite buffer.

**Stage 5 — Tail**
- After main loop exits, continue 60+ cycles to flush in-flight fires.
- `in_Wcontrol = W_INT{N}` until `cyc_global > LAST_DRIVE + slack` (e.g., 4200 for A8W2). Then W_IDLE.
- `in_scale_weight` continue drive at `FIRST_FIRE + W_CYC × m` until last m reached.

### Fire capture (host responsibility)
- Each cycle, sample `out_fire[c]` per col c. If high:
  - `fc = fire_cnt[c]++`, decode (n_t, k_t, m_t, m_in) from fc
  - Slice `out_accumulate[60c + 60*lane:...]` per A_ctrl mode (see §2 layout)
  - Accumulate into mem_c[m_t × 32 + m_in][N_pos] (signed addition across K-tiles)
  - Capture `out_scale[36c + 9*lane:...]` per fire

---

## 4. File Naming Convention (TransformSerial.py output)

| File | Content | Precision | Layout / tile order |
|---|---|---|---|
| `a_input_BS_mxint{W}.hex` | bit-serial weight (MSB first) | W ∈ {2,4,8} | K_T × M_T × 32 rows × W bits = N lines |
| `a_input_BP_mxint{W}.hex` | bit-parallel weight (256-bit / row) | W | K_T × M_T × 32 rows |
| `b_input_mxint{A}.hex` | activation (256-bit / col-tile) | A ∈ {2,4,8} | N_T × K_T × 32 cols |
| `a_scale_mxint{W}.hex` | E8M0 weight scale (8-bit) | W | K_T × M_T × 32 rows |
| `b_scale_mxint{A}.hex` | E8M0 activation scale (8-bit) | A | N_T × K_T × 32 cols |
| `c_mxint{W}_mxint{A}.hex` | reference INT GEMM result | both | (M, N) row-major, 32-bit signed |

**Convention**:
- `mem_a` in TB = WEIGHT (bit-serial). File `a_input_BS_mxint{W}.hex` is read by `mem_a`.
- `mem_b` in TB = ACTIVATION. File `b_input_mxint{A}.hex` is read by `mem_b`.
- `c_mxint{W}_mxint{A}.hex`: first name = weight precision, second = activation precision.

**Generation**:
```bash
cd <repo_root>
python TransformSerial.py    # → ./generated/ (3×4 + 9 = 21 hex files for M=K=N=128)
```

`TransformSerial.py` produces all 9 reference C files in one run (line 211-224, nested loop over (a_prec, b_prec)).

---

## 5. Regression Cycle Counts (validated 2026-05-11)

| Mode | cycle_cnt | base TB used |
|---|---|---|
| A8W8 | **16486** | tb_A8W8.v |
| A8W4 | **8294** | tb_A8W4.v |
| A8W2 | **4191** | tb_A8W2.v (new) |
| A4W8 | **8293** | tb_A4W8.v |
| A4W4 | **4197** | tb_A4W4.v |
| A4W2 | **2142** | tb_A4W2.v (new) |
| A2W8 | **4192** | tb_A2W8.v (new) |
| A2W4 | **2142** | tb_A2W4.v (new) |
| A2W2 | **1119** | tb_A2W2.v |

**Verification baseline**: 모든 TB INT MAC PASS=16384/FAIL=0, Scale PASS=lanes×32×FIRES_PER_COL/FAIL=0.
- Lanes_per_mode: A8=1, A4=2, A2=4. e.g. A2W2: 4×32×512 = 65536 scale slots.

**Regression runner**: `scripts/run_all_tbs.ps1` (PowerShell, Windows-native Vivado xsim CLI).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all_tbs.ps1
```

---

## 6. Integration Wrapper Guide (다음 AI Accelerator GEMM 통합 시)

이 코어 (`TOP.v` + 7개 sub-module) 를 새 프로젝트 의 GEMM 모듈로 instantiate 시 검토할 점:

### 6.1 인스턴스화

```verilog
TOP #(
  .input_a_len(1), .input_b_len(8), .output_len(7),
  .control_len(1), .Station_control_len(2), .scale_len(8),
  .acc_len(15), .acc_out_len(60), .W_control_len(2),
  .A_control_len(2), .cnt_len(3),
  .num_col(32), .num_row(32)
) gemm_core (
  .clk(...), .rst(...),
  .in_a(weight_bitserial),         // [31:0]
  .in_b(activation_bytes),         // signed [255:0]
  .in_control(pe_pp),              // [31:0], PE Buf1/2 selector
  .in_loadEN(act_load_en),         // [31:0]
  .in_Station_control(a_ctrl),     // [1:0] A_INT8/4/2/IDLE
  .in_Scale_Activation(act_scale), // [31:0] 4 byte
  .in_station_control(st_pp),      // station Buf1/2 selector
  .in_station_loadEN(st_load_en),
  .in_start_accumulate(start),
  .in_Wcontrol(w_ctrl),            // [1:0] W_INT8/4/2/IDLE
  .in_scale_weight(w_scale),       // [7:0]
  .out_accumulate(acc_out),        // [32*60-1:0]
  .out_scale(scale_out),           // [32*36-1:0]
  .out_fire(fire)                  // [31:0]
);
```

### 6.2 외부 컨트롤러 책임 (이 코어에 없음 — 새로 작성)

1. **DMA / memory controller** — DRAM/BRAM 에서 weight, activation, scale 을 fetch
2. **Mode dispatch** — runtime A_ctrl/W_ctrl 결정 (precision schedule)
3. **Chain sequencer** — 5 chain 의 cyc-by-cyc 시퀀싱 (§3 의 protocol). 특히:
   - `cyc_in_K` 카운터 (K-tile 내 위치)
   - `cyc_global` 카운터
   - PE in_control toggle (K-tile boundary)
   - Station in_control toggle (mode-specific TOGGLE 값)
   - scale_weight drive at `FIRST_FIRE + W_CYC × m`
4. **Fire capture & accumulate** — `out_fire[c]` 감지 시 `out_accumulate` 슬라이스, mem_c 업데이트
5. **Mode boundary alignment** — A_ctrl 또는 W_ctrl 변경 시 K-tile boundary 에서만 변경 (mid-K 변경하면 partial sum 깨짐)

### 6.3 주의사항

- **Fire latency 가 A_ctrl 따라 다름** (A8: 2, A4: 1, A2: 0). out_fire 캡처 로직이 mode-aware 해야 함.
- **out_accumulate / out_scale 슬라이스 mode-aware** — §2 layout 참조.
- **station broadcast 의 sym chain delay 15** — `in_station_control` toggle 후 14 cy 이상 settle 보장 필요.
- **acc chain 의 sym delay 15** — `in_start_accumulate`, `in_Wcontrol`, `in_scale_weight` 모두 col 16 entry 기준 ±15 cy 지연. 첫 fire 가 col 0/31 에서는 col 16 보다 15 cy 늦게.
- **PE in_control vs station in_control 의 의미 다름** — PE: 자기 station1/2 read selector. Station: Buf1/Buf2 write/broadcast selector. 둘 다 K-tile boundary 에서 toggle 하지만 시점 다름 (PE: cyc_in_K=0, Station: cyc_in_K=TOGGLE).

### 6.4 TB reference

각 mode 의 정확한 driver sequence reference 는 sim TB 참조:
- `MXP.srcs/sim_1/new/tb_A{a}W{w}.v` (9 files, 모두 PASS validated)
- 통합 wrapper RTL 작성 시 이 TB 의 sequencing 그대로 옮기면 됨.

### 6.5 Sim 환경 (이 프로젝트에서)

Windows 에서 Vivado xsim CLI 사용 시:
- `C:\Xilinx\Vivado\2024.1\bin` 을 **User PATH 의 맨 앞**에 두기 (mingw64 DLL 충돌 방지)
- 또는 `vivado.bat` 통해 launch (settings64.bat 자동 sourcing)
- regression runner: `scripts/run_all_tbs.ps1`

---

## 부록 A. RTL 구조 요약

```
TOP
├── SystolicArray (32×32)
│   ├── PE_feeder (diagonal, 32 instances)
│   └── PE_naive (non-diagonal) → uses adder_lane
├── station × 32 (per-col, Buf1/Buf2 ping-pong)
└── Accumulator_Col × 32 (per-col)
    └── Accumulator × 4 (per-lane within each col)
```

8 RTL 파일, 1,640 lines total. 외부 IP 의존 0, 순수 Verilog 2001.

## 부록 B. Validation summary

이 protocol 의 모든 수치는 다음 sim 으로 검증됨:
- 9 TB (tb_A{8,4,2}W{8,4,2}.v) full 128×128×128 GEMM
- TransformSerial.py 의 FP32 random matrix → MXINT quant → reference C
- Sim: Vivado 2024.1 xsim, 333 MHz clock (CLK_PERIOD=3 ns)
- 모든 TB INT MAC PASS=16384/FAIL=0 AND Scale FAIL=0 (lanes×col×fires)
- 기존 5 baseline cycle (16486/8294/8293/4197/1119) 정확히 보존
- 새 4 mode cycle (4191/2142/4192/2142) 첫 시도 PASS (timing iteration 0회)
