# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Next session kickoff (2026-05-18, mixed-prec TB scale_weight cadence 부분 fix, random mixed 미해결)

**진행 상태**: 2026-05-17 의 P-Task8 BLOCKER (random mixed bit-exact mismatch) 를 디버그해서 **root cause 부분 확정 + uniform W=8 isolation PASS** 까지 도달. random mixed (per-m_in W 변경) 는 여전히 fail. 변경 사항 **미커밋** (사용자 결정 대기).

### 확정 #1 (수정 완료, uniform W=8 isolation 으로 검증): scale_weight cadence misalignment

**증상**: mixed TB 가 `bp == w_now - 1` (m_in 진입 cycle) 에서 `in_scale_weight <= a_scale[m_idx]` 드라이브. 그러나 9-mode TB 는 `cyc_global ≥ FIRST_FIRE_GLOBAL && (cyc - FIRST_FIRE) % W_CYC == 0` cadence — Accumulator_Col 의 fire_q2 가 comb_s 를 capture 하는 cycle 직전 (1 cycle 차) 에 in_scale_weight 가 settle 되도록 calibrated. 둘이 약 28 cycle (W=8, A=8 기준) 어긋남 → fire 시점에 잘못된 m_in 의 scale 값 capture.

**Fix (일반화 공식)**:
```
drive_cyc[k] = 18 + A_FIRE_DELAY + Σ_{j=0..k} W[j]
```
- uniform W=8, A=8 일 때 = 28 + 8k = 9-mode 의 FIRST_FIRE+W_CYC*k 와 정확히 일치.
- A_FIRE_DELAY: A8=2, A4=1, A2=0 (Accumulator_Col 의 fire_qN 단계 수).
- W[j]: j-번째 m_in 의 W_PREC (cumulative_W_inclusive).

**구현**: `tb/gemm_sram_top_mixed_tb.v` 에 `drive_cyc_target` / `drive_idx_target` schedule 변수 + `lookup_w_for_idx(idx)` function. drive_stage_3_4_mixed 진입 시 schedule 초기화 (`drive_cyc_target = 18 + A_FIRE_DELAY + lookup_w_for_idx(0)`). bp 루프 안에서 `cyc_global == drive_cyc_target` 시 드라이브 + idx 증분 + cyc_target += W[next idx]. drive_stage_5_tail 도 같은 schedule 이어감 (in-flight fire scale capture).

**검증**: `work/isol_W8/` (env `MIXED_W_UNIFORM=8` 로 build_w_prec_map 균일 W=8 fix) 에 mixed TB 돌린 결과 vs mixed golden bit-exact PASS (hw_sw: max=0, n_diff=0/16384, SNR=inf). emit hex / golden 도 `compute_golden_mixed` == `mxint_gemm_golden` byte-identical 로 확인 완료.

### 미해결 #2: random mixed (per-m_in W 변경) 여전히 fail

isolation 에 fix 적용 후도 `bash sim/run_mixed_one.sh 8` (random W_PREC map) 은 hw_sw max=6.5e4, n_diff=16384/16384, SNR=-43dB 로 **여전히 fail**. cadence 외에 다른 mixed-prec 정합 요인이 있다는 뜻.

**가설 / 미해결 질문**:
1. **chain topology mismatch**: GEMM.v line 211: `wc_chain[num_col/2] = in_Wcontrol` — Wcontrol 은 **col 16 진입 symmetric chain** (|c-16|/hop). 반면 weight `in_a[r]` 의 PE_feeder 는 **각 row 의 col=r 위치 feeder** 에서 좌우 propagate (col c at row r = TB + |c-r|). uniform W 에선 Wcontrol 안 바뀌어서 무관, mixed W 에선 col 별 in_Wcontrol 도착 시점이 col 별 weight bit 도착 시점과 비대칭 → cnt rollover threshold 가 잘못된 cycle 에 적용될 가능성.
2. **prefetch cadence 차이**: mixed TB 는 `m_t == 1 && bp == w_now-1` 에서 sparse prefetch (32 m_in × W 분산). 9-mode 는 `m_t == 1 && o < TILE_SIZE` 에서 dense (32 consecutive cycles). station chain settle 시점이 mixed-W 에서 어긋날 가능성 (TOGGLE_VAL=24 worst-case 고정).
3. **TOGGLE_VAL fixed-24 의 mixed-W invariant**: 9-mode 는 (A, W) 마다 TOGGLE_VAL 다름 (19~24). mixed TB 는 TOGGLE_VAL=24 worst-case 보수 고정. mixed-W 에선 K-tile 마다 cyc_in_K 가 가변이라 station toggle 시점이 다른 K-tile 의 Wcontrol settle 과 어긋날 가능성.

**사용자 의도 확인 필요 (2026-05-18 conversation 중 명시)**:
- 사용자: "in_Wcontrol 도 weight 처럼 전파될수있기때문에 적절한 타이밍 (weight block 이 바뀌어서 출력되는 타이밍) 에 그거에 맞는 신호를 전파시키면 block 이 타일내에서 바뀌더라도 카운터 조절이 될수있는데".
- 즉 RTL 의 design intent 는 per-(M-row, K-block) W mix 를 지원하고, TB 가 in_Wcontrol 을 "weight block 출력 타이밍에 맞게" 드라이브해야 한다는 입장.
- 그러나 RTL 의 wc_chain 은 col 16 진입 symmetric — weight feeder 가 row 마다 다른 col 위치 에 있는 것과 chain 구조 다름. **TB 만으로 col 별 도착 시점 보정이 가능한지 추가 확인 필요.**
- precision_modes_protocol.md §1 마지막 줄 "K-tile mid 변경 시 partial sum 깨짐" 는 (a) mid-MAC (한 m_in 안에서) 만 가리키는 보수적 표현일 수도 있고, (b) RTL 의 실제 한계를 정확히 반영한 것일 수도 — 본 세션에서 결론 못 냄.

### 다음 세션 protocol

1. **사용자 결정 먼저 물어보기**:
   - (a) 현재 TB 변경 (scale_weight schedule 일반화) 만 일단 commit 하고 random mixed 디버그는 별도 작업으로?
   - (b) random mixed 디버그 계속? (RTL chain timing 분석 필요)
   - (c) spec 단순화 (prec_map shape `(K_T,)` 만 = K-tile 단위 mix) 로 RTL 손 안대고 working 으로?

2. **(b) 진행 시 다음 step 후보**:
   - GEMM.v / SystolicArray.v 의 wc_chain (Wcontrol propagation) vs PE_feeder 의 in_a chain (weight propagation) 의 col 별 도착 시점 cycle 도식화. SA 가 mixed-W per-m_in 에서 어떤 col 에서 어떤 cyc 에 in_Wcontrol 이 어떤 값이어야 하는지 정확히 specify.
   - mixed TB 의 in_Wcontrol 드라이브 시점 (현재 m_in_idx 진입 cycle) 을 RTL 정합 시점으로 보정. 가능하면 isolation 처럼 small-case (e.g., 8 m_in 만 random W) 로 debug 좁히기.

3. **(c) 진행 시**: `sim/gen_mixed.py::build_w_prec_map` 의 shape 를 `(M, K_T)` → `(K_T,)` 로 변경. TB 의 `lookup_w_for_idx` 도 m_in 무관하게 K-tile 단위 lookup 으로 단순화. spec/plan 문서도 업데이트.

### 미커밋 변경 (working tree, status)

- `sim/gen_mixed.py`: `build_w_prec_map` 에 `MIXED_W_UNIFORM` env override 추가 (isolation test helper). 값 ∈ {2,4,8} set 시 random 대신 균일 W. 디버그용이라 keep or revert 둘 다 OK.
- `tb/gemm_sram_top_mixed_tb.v`:
  - module-level: `integer drive_cyc_target, drive_idx_target;` + `function integer lookup_w_for_idx(input integer idx);`.
  - drive_stage_3_4_mixed: schedule 초기화 + bp 루프 안에서 `if (cyc_global == drive_cyc_target) ...` 드라이브.
  - drive_stage_5_tail: 같은 schedule 이어감.
  - 기존 `if (bp == w_now - 1) in_scale_weight <= a_scale[...]` 블록은 새 schedule 로 교체됨 (제거).

### 본 세션 이미 commit 된 산출물 (2026-05-17)

(직전 commit 들. 본 디버그 세션에서는 추가 commit 없음)
- spec/plan: `docs/superpowers/{specs,plans}/2026-05-17-mixed-precision-tb*.md`
- gen_mixed.py + 10 pytest: `sim/gen_mixed.py`, `sim/tests/test_gen_mixed.py` (commits `68a8542` ~ `a77bd12`)
- 신규 TB: `tb/gemm_sram_top_mixed_tb.v` (commit `6106d7a`)
- 신규 sim 스크립트: `sim/run_mixed_one.sh` (commit `a77bd12`)
- `sim/run_mixed_sweep.sh` 는 아직 미작성 (P-Task9). P-Task8 완전 해결 후.

### 별도 작업 — mxp_tools.compare 의 fail-gate

`mxp_tools/cli.py::cmd_compare` 가 mismatch 시 stats print 만 하고 `sys.exit(1)` 안 함 → run_mixed_one.sh / run_integration_sweep.sh 모두 false-positive PASS 가능. `n_nonzero_diff > 0` 시 exit 1 추가 필요. 본 디버그 작업과 독립 — 작업 시 9-mode sweep 회귀 확인 필수.

### 핵심 RTL 매핑 사실 (참조)

SA 의 row=K, col=N, cycle=M 진행. 자세한 표는 본 CLAUDE.md `## MXP control surface` 의 첫 subsection. mixed-prec 디버그 시 row=M 으로 단정 금지 (메모리 `feedback_sa_dimension_mapping.md` 참조).

---

## 기존 status (2026-05-15, integration done + MXP_Tools upstream synced)

GEMM ↔ RMW ↔ sram_1rw_banked **시스템 통합 완성 및 검증**됨 — `bash sim/run_integration_sweep.sh` → `ALL 9 MODES PASSED` (9 precision combinations A,B ∈ {2,4,8}, 각각 128×128 = 16384 element 모두 bit-exact 일치 vs MXP_Tools golden). 단위 검증도 그대로 유효: `bash sim/run_rmw.sh` → 71/71 PASS.

**MXP_Tools 업스트림 동기화 (2026-05-15)** — `MXP_Tools/` 는 `~/Desktop/Desktop/MXP_Tools` 의 fork. 업스트림 버그 픽스 (NaN/Inf-aware compare, `@addr` writememh 파서, gather_banks duplicate-write detection, `_require_block_multiple` enforcement, LF-only newline) 가 머지된 상태. 프로젝트 전용 추가분 (`rmw_gen.py`, cli 의 `rmw-gen`, hwio 의 `interleaved_row_major_16bank` + `interleaved_row_major_32bank`) 은 그대로 보존. pytest 슈트 (45 cases) 도 함께 도입 — `cd MXP_Tools && python -m pytest tests/ -q` 로 0.3 초 만에 단위 검증 가능. 메모리 `reference_mxp_tools_upstream.md` 에 동기화 절차 명시.

TB 6 개 (`tb/*.v`) 는 한글 헤더에 **검증 목적 / 검증 내용 / 동작 의도**를 명시한 상태 (commit `92d6b1e`). 새 TB 작성 시에도 동일 컨벤션 유지 — 헤더만 봐도 그 TB 가 뭘 검증하는지 즉시 파악 가능해야 함.

**RMW 32× col-parallel 완성 (2026-05-15)** — `sram_1rw_banked_mp.v` (32-bank per-bank port 노출) + `gemm_sram_top.v` 32-RMW generate + TB 32-stream capture/drain 으로 재구성. `bash sim/run_integration_sweep.sh` → `ALL 9 MODES PASSED` wall-clock **633s ≈ 10.5 min** (이전 1-RMW sweep ~20–25 min 대비 약 2× 단축). 산출 commit chain: f102756 → c4ae7fc → 3d21544 → 20f60e6 → 76bb597 → 326ce1f.

다음 세션 시작 protocol — `docs/next-session-kickoff.md` 를 먼저 읽고, 사용자가 어떤 방향으로 가고 싶은지에 따라:

1. **잠정값 재검토** (throughput / area) — 통합 spec § 9 의 잠정 결정값 표 (RMW instance 수 32개 col-parallel / loop order K-outermost / 워크로드 128³ / per-bank port) 의 추가 재검토 트리거가 발생한 시점. 새 spec 필요.
2. **Timing closure / 합성** — Vivado 합성, 250 MHz wrapper / xc7vx485 target. MXP standalone 의 closure 패턴 답습.
3. **Future scope: SRAM weight storage** — 통합 spec § 1 "Future scope" 항목. 별도 spec 필요.
4. **9-mode 자동 회귀 CI** — `sim/run_integration_sweep.sh` 를 CI 로 묶기.

이전 단계 흐름:
- RMW unit: `docs/superpowers/specs/2026-05-14-rmw-design.md` + `docs/superpowers/plans/2026-05-14-rmw-implementation.md`
- 통합: `docs/superpowers/specs/2026-05-14-integration-design.md` + `docs/superpowers/plans/2026-05-14-integration-implementation.md`

---

## Project Goal

`gemm_sram` = **integration** of the MXP mixed-precision bit-serial systolic array (GEMM compute) with the parameterized 1RW SRAM banks. The end state is a **Read-Modify-Write (RMW)** datapath: a column's accumulated GEMM result is added to the prior partial sum held in SRAM, then written back to the same address so the SRAM stores the running tile-wise sum.

This is a Vivado 2024.1 project (`gemm_sram.xpr`, target `xc7vx485tffg1157-1`) that **imports** RTL from two sibling sister projects rather than maintaining its own copies. Both upstream repos are the source of truth — do not mutate imported sources without understanding the upstream contract.

## Source provenance (do not edit imports in place)

| Module | Imported from | Role |
|---|---|---|
| `GEMM.v` (top), `RMW.v` | `gemm_sram.srcs/sources_1/new/` | **Project-local** — RMW glue lives here. Edit freely. |
| `Accumulator.v`, `Accumulator_Col.v`, `PE_feeder.v`, `PE_naive.v`, `SystolicArray.v`, `adder_lane.v`, `station.v` | `../../MXP/MXP.srcs/sources_1/new/` | MXP compute engine (32×32 bit-serial systolic). |
| `sram_1rw.v`, `sram_1rw_banked.v` | `../../sram/rtl/` | Parameterized 1RW SRAM (leaf + banked wrapper). |

Vivado's import keeps copies under `gemm_sram.srcs/sources_1/imports/Desktop/{MXP,sram}/...`. If you change algorithm/microarchitecture, fix it upstream (`../MXP/` or `../sram/`) and re-import; do not patch the local copy and forget. Refer to `../MXP/CLAUDE.md` and `../sram/CLAUDE.md` for upstream conventions and gotchas (Verilog-2001 only in SRAM, `in_a`=weight/`in_b`=activation naming gotcha in MXP, etc.).

## Top-level wiring

Top module is `gemm_sram_top` (set in `gemm_sram.xpr`; sim target is `gemm_sram_top_tb`). The wrapper is **pure structural** — GEMM + RMW + sram_1rw_banked instantiated with just two glue lines (Q→RMW.in_SRAM hardwire, sram_D_use_zero mux between zero and RMW.out_RMW). All controller responsibility lives in the TB (`tb/gemm_sram_top_tb.v`) — no synthesizable FSM in the wrapper.

Datapath:

```
GEMM (INT psum + 9-bit scale) ──► RMW (INT→FP32, FP32 add) ──► sram_1rw_banked (FP32)
                                       ▲                              │
                                       └────── FP32 prior psum ◄──────┘
```

Verified end-to-end via `bash sim/run_integration_sweep.sh` (9/9 modes bit-exact PASS vs MXP_Tools golden, 128×128 = 16384 elements per mode).

### RMW contract (from `RMW.v` header — committed decisions)

```verilog
module RMW (
    input  wire        clk, rst,
    input  wire [31:0] in_SRAM,   // FP32 prior partial sum read from SRAM
    input  wire [31:0] in_GEMM,   // INT32 from GEMM accumulator (one lane's worth)
    input  wire [8:0]  scale,     // combined 9-bit scale (matches one sub-word
                                  // of Accumulator_Col `out_scale`, = scale_len+1)
    output reg  [31:0] out_RMW    // FP32 sum to write back to SRAM
);
```

This pins down **two** of the previously-open questions:

1. **Bit-width reconciliation**: dequantize INT→**FP32 at the RMW boundary**. Both SRAM storage and the FP32 add live in IEEE-754 single precision — 32-bit words, no `DATA_WIDTH` widening, no INT-domain accumulation in SRAM. This also lines up with `MXP_Tools/hwio.py::read_writememh_fp32` which already assumes FP32 bit patterns in the `$writememh` dump.
2. **Scale handling**: the 9-bit `scale` is the per-sub-word combined `(act_scale + weight_scale − 127)` that `Accumulator_Col` produces (note: 9-bit signed, `{1'b0, act_scale} + {1'b0, weight_scale} - 9'sd127`). RMW consumes it during the INT→FP32 conversion; it is **not** stored alongside the psum.

The RMW controller (whether inside RMW or wrapping it) must still:
1. On `out_fire[col]` rising, latch the lane's INT32 psum and the corresponding 9-bit scale sub-word.
2. Issue a READ to the SRAM address holding that lane's prior FP32 psum.
3. Wait for the read return (1 cycle if leaf `PIPELINE=0`, 2 cycles if `PIPELINE=1`).
4. Convert INT→FP32 (using `scale`), add to `in_SRAM`, write back.

### Resolved (during integration) — committed values

- **Granularity**: **32 RMW instances** (col-parallel; each col j has its own RMW + dedicated bank j port). Lane decode (A8 = 1, A4 = 2, A2 = 4 dispatches per col fire) lives in `tb/gemm_sram_top_tb.v` per-col FIFO + drain state machine. Re-visit trigger: timing closure 시 합성 코스트 측정.
- **Latency budget**: `L_CONV = 2`, `L_ADD = 3`, total = **5 cycles**. Hand-written via vendored HardFloat (`third_party/berkeley-hardfloat/`).
- **FP32 adder implementation**: **HardFloat-based** (`fp32_adder.v` wraps `addRecFN`). Not Vivado FP IP.
- **First-tile init**: **SRAM zero-prime via TB loop** at sim start (16384 words written through the `sram_D_use_zero=1` mux), then `sram_D_use_zero=0` for the rest. No NaN risk.
- **Bank addressing**: `NUM_BANKS = 32`, `BANK_DEPTH = 1024`, `PIPELINE = 0`. Per-bank port 노출 (`sram_1rw_banked_mp.v`); col j → bank j (충돌 0). `C[m,n] → flat=m*N+n → bank=flat%32, word=flat//32`. Mapping callable in `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_32bank`. 16384 / 32 = 512 words/bank for the 128×128 workload. (기존 `sram_1rw_banked.v` 16-bank wrapper 는 다른 caller 용으로 별도 보존.)

## MXP control surface (you will need this)

### SA 차원 매핑 — 가장 자주 헷갈리는 부분 (2026-05-17 명시)

본 SA 의 차원 매핑을 처음 보면 row=M 으로 가정하기 쉬운데 **틀림**. RTL (`SystolicArray.v`, `PE_feeder.v`, `PE_naive.v`) 검증 결과:

| SA 축 | 행렬 차원 | 근거 |
|---|---|---|
| **row** (`in_a[r]`, `in_b[r]`) | **K-axis** | `in_a[r]` (1-bit) = K=r 의 weight bit. `in_b[r]` (8-bit) = activation[K=r, N=col]. PE_feeder 가 row=col 대각선에 위치, in_a 가 좌우 col 방향 propagate (= K=r 의 weight 가 모든 N-col 에 전파) |
| **col** (Accumulator_Col[c]) | **N-axis** | station chain (col 31 entry, leftward), Accumulator chain (col 16 entry, outward) 모두 col 별 결과 = N-col 별 dot product. `out_fire[c]` = N-col c 완료 펄스 |
| **cycle** | **M-axis 진행** | 한 M-iteration = W_PREC cycle (bit-serial weight). 32 M = 한 m_t tile = 32W cycle |

**한 SA tile (32 K × 32 N) 처리** = 32W cycle 동안 32 M-row 의 dot product 산출. 한 PE(r=K, c=N) 의 station = activation[K=r, N=c].

이 매핑이 의미하는 것:
- 한 cycle 의 `in_a` (32-bit) = 32 K-element 의 같은 bit-position (= 같은 K-block 안). 32 K 가 자연스럽게 같은 K-block 의 W_PREC 공유 — **K-block 단위 mixed-precision 이 dataflow 변경 없이 가능**.
- M 진행마다 `in_Wcontrol` 갱신 → (M-row, K-block) 마다 다른 W_PREC. 128 M × 4 K-block = 512 block 의 W_PREC 자유 설정 가능.
- row 차원이 K 라서 "row 별 다른 W_PREC" 라는 표현 자체가 어색 — 같은 cycle 의 32 row 는 모두 같은 K-block 안의 32 K-element 라 W_PREC 가 자연스레 동일.

(잘못된 가정 사례 — 다른 세션에서도 빠지지 말 것: in_a 의 32-bit 가 32 M-row 의 weight bit 라 가정 → "M 별 W_PREC 다르면 dataflow 깨짐" 잘못된 결론. **틀림**. RTL 의 row 는 K, M 은 cycle 진행이다.)

### 기존 control surface (chain 방향)

`GEMM` is driven by two distinct chains — getting these wrong silently produces dead lanes. From `../MXP/CLAUDE.md`:

- **Station data chain**: enters at col 31, propagates **leftward** (col 31 → col 0). Carries `in_Scale_Activation`, `in_Station_control` (precision: A8/A4/A2/idle), `in_loadEN`. Aligned with `in_b` (activation) chain.
- **Station selector chain** (`in_station_control`): enters at col `num_col/2=16`, fans **outward symmetrically**. Toggles Buf1↔Buf2 ping-pong. Same topology as the Accumulator chain so the toggle wave aligns with the fire window.
- **Accumulator chain**: `in_start_accumulate`, `in_Wcontrol`, `in_scale_weight` enter at col 16, fan outward.

`out_fire` is **per-column**, not global — column fire timings are offset by the chain delays. Any SRAM scheduler must honor this, e.g., a per-column FIFO or 32 parallel RMW engines.

Per-column lane slicing of `sa_out` (in GEMM.v at `col_slice`): MSB-first lane order `{psum0, psum1, psum2, psum3}` — `psum0` carries the top weight-bit pair, `psum3` the bottom. The downstream `Accumulator_Col` combines these via a `<<2/<<4` adder tree depending on precision mode (`Mode_oh` one-hot {A2,A4,A8}).

Precision mode → `out_accumulate` layout (from `Accumulator_Col.v`):
- A8: single 21-bit signed value, sign-extended into 60 bits.
- A4: two 18-bit signed values `{0..0, s1_a, s1_b}` packed low.
- A2: four 15-bit raw lane accumulators `{out_INT2_0..3}` packed.

The RMW datapath must respect this layout when adding the prior SRAM psum.

## SRAM control surface

From `../sram/CLAUDE.md` — **active-low** `CEB` (chip enable bar) and `WEB` (write enable bar), **active-high** `WMASK` (bit-write enable, 1 = write). Read latency is 1 cycle when `PIPELINE=0`, 2 cycles when `PIPELINE=1`. The leaf bank has no reset on `mem[]` — power-up state is whatever simulation initialized to. **`INIT_FILE` exists only on the leaf**; the banked wrapper has no `INIT_FILE` parameter (deliberate — Q5 in `../sram/docs/banking-wrapper-decisions.md`). For tests, initialize via a `do_write` loop instead of file preload.

The banked wrapper additionally has a 1-cycle `bank_sel_d1` register inside, so its effective read latency is unchanged from the leaf's `PIPELINE` setting — but mux-select alignment must use `bank_sel_d1`, not raw `bank_sel`. Important if you ever bypass the wrapper. The wrapper's output mux is `wire [DW-1:0] bank_q [0:NB-1]` with indexed access (`bank_q[bank_sel_d1]`) — clean parametric pattern; reuse it if you write a custom mux.

**Polarity inversion at PNR.** Foundry macros usually expose `BWEB` (active-low bit-write) instead of our active-high `WMASK`. `../sram/rtl/sram_1rw_macro_wrap.v.example` is the template — `assign BWEB = ~WMASK;` plus a vendor macro instance. When this project moves to PNR/silicon, copy that template and swap the leaf instance; wrapper RTL stays unchanged.

## Common commands

Two ways to run sim:

1. **Vivado GUI** — open `gemm_sram.xpr`. Synth top = `gemm_sram_top`; sim top = `gemm_sram_top_tb`. Run Behavioral Simulation.
2. **XSim batch** — `sim/run_*.sh` scripts:

   ```bash
   # Unit-level
   bash sim/run_rmw_smoke.sh         # HardFloat round-trip smoke (no DUT logic)
   bash sim/run_int_to_fp32.sh       # int_to_fp32 unit TB (6 directed cases)
   bash sim/run_fp32_adder.sh        # fp32_adder unit TB (4 directed cases)
   bash sim/run_rmw.sh               # full RMW vector TB (71 cases)
   bash sim/run_top_elab.sh          # gemm_sram_top elab-only smoke
   bash sim/run_integration_smoke.sh # TB zero-prime + dump (no GEMM driving)

   # MXP_Tools Python 단위 검증 (compare / hwio / quant / gemm)
   cd MXP_Tools && python -m pytest tests/ -q   # 45 cases, ~0.3 s

   # Integration (end-to-end vs MXP_Tools golden)
   bash sim/run_integration_one.sh <LABEL> <A_PREC> <B_PREC>   # 1 mode (e.g. "A8_B8" 8 8)
   bash sim/run_integration_sweep.sh                            # all 9 modes serial
   bash sim/run_integration_parallel.sh                         # print parallel dispatch guide
   ```

   **Full RMW vector test** requires `MXP_Tools` to generate the vector files first:

   ```bash
   cd MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0 && cd ..
   bash sim/run_rmw.sh             # expect: rmw_tb: ALL 71 TESTS PASSED
   ```

   **Integration sweep** is self-contained — invokes `gen / emit / ref` internally then `run_integration_one.sh` + `compare`. Last line on success: `ALL 9 MODES PASSED`. Runtime **633s ≈ 10.5 min** for the full 9-mode sweep (32-RMW col-parallel; 이전 1-RMW sweep ~20–25 min 대비 약 2× 단축).

   To author a new TB, copy the pattern from any existing `sim/run_*.sh`. Required directive that **must** appear in both RTL and TB or XSim errors out: `` `timescale 1ns/1ps ``.

   **Gotcha — `-testplusarg "K=V"` under Git Bash**: the `=` is stripped before xsim sees it. Wrap with `cmd //c "xsim ... -testplusarg \"K=V\""` — the integration scripts already do this.

When parameter overrides are needed at sim time, **do not use `xelab -generic_top "P=V"`** — there's a cmd.exe quoting bug under Git Bash on Windows that strips the `=`. Use `` `ifdef `` in the source + `xvlog -d FLAG` instead. The SRAM TBs already use this pattern:

- Leaf TB (`sram_1rw_tb.v`): `USE_PIPELINE`, `RUN_INIT_TEST`.
- Banked TB (`sram_1rw_banked_tb.v`): `USE_PIPELINE`, `USE_SEQUENTIAL` (selects `BANK_STRATEGY="SEQUENTIAL"` vs default `"INTERLEAVED"`).

**TB sizing convention.** Use a small config for fast sim (banked TB uses 4 banks × 64 depth × 32-bit), full-size parameters only via the elab sweep (`../sram/sim/run_wrapper_elab.sh`). When writing this project's TBs, follow the same pattern — do not run a 2 MB sim by default.

## Verification helper: `MXP_Tools/` (Python, local subdir)

**Provenance**: fork of upstream `~/Desktop/Desktop/MXP_Tools`. 사용자가 업스트림에서 버그 픽스를 하면 프로젝트 사본으로 옮겨야 함 (절차는 메모리 `reference_mxp_tools_upstream.md` 참고). 프로젝트 전용 추가분 — `mxp_tools/rmw_gen.py`, cli 의 `rmw-gen` 서브커맨드, hwio 의 `interleaved_row_major_16bank` + `interleaved_row_major_32bank` mapping, `tests/test_hwio_interleaved.py` — 는 머지 시 반드시 보존.

**pytest 슈트** (`tests/`, 45 cases) 는 단위 동작 (NaN/Inf 처리, `@addr` 파싱, gather_banks duplicate detection, emit shape validation, MX quant edge cases) 을 0.3 초 만에 검증. RTL sim 전에 Python tool 변경 검증용으로 우선 돌릴 것.

Generates HW inputs and SW golden GEMM. Typical invocation (the sweep script already wraps these):

```bash
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8                        # emits all 3 precs
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8   # SW golden
# run HW → produces work/A8_B8/hw_out/bank{0..31}.mem ($writememh)
BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..31})
python -m mxp_tools compare --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
                            --hw-banks ${BANKS} \
                            --layout interleaved_row_major_32bank
```

**Naming gotcha (was a Task 8 silent-bug source)**: MXP_Tools' `--prec-a` = WEIGHT precision = our plusarg `B_PREC` (weight is the bit-serial first operand in `mxint_gemm_golden(A, ..., prec_A=weight)`). `--prec-b` = ACTIVATION precision = our `A_PREC`. The resulting `.npz` filename slot order is `C_sw_mxint{prec_a}_mxint{prec_b}.npz` = `C_sw_mxint{B_PREC}_mxint{A_PREC}.npz`. Symmetric modes (A2_B2, A4_B4, A8_B8) hide the bug; asymmetric modes catch it as all-zero HW dumps. The sweep + per-mode scripts already use the corrected convention.

HW dump = one `$writememh` file per SRAM bank (32 files). Mapping callable: `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_32bank`. Numerical contract: HW output words are interpreted as IEEE-754 FP32 bit patterns (RMW dequantizes INT→FP32 before storing).

## Settled — with re-visit triggers

All design questions from the original spec § 9 are now committed with concrete values (see "Resolved" subsection above + the integration spec at `docs/superpowers/specs/2026-05-14-integration-design.md` § 9). Trigger list for re-opening any of them:

| Decision | Committed value | Re-visit trigger |
|---|---|---|
| RMW instance count | 32 (col-parallel, per-bank SRAM port) | timing closure 시 합성 코스트 측정 |
| Loop order | K-outermost | dataflow re-optimization phase |
| Workload (M, K, N) | (128, 128, 128) | sim time > 10 min |
| Bank strategy | per-bank port 노출 (col j → bank j, 충돌 0) | bank-conflict 측정; multi-port 필요성 |
| NUM_BANKS / BANK_DEPTH | 32 / 1024 (128×128 workload) | workload 변경; 더 큰 tile |
| SRAM PIPELINE | 0 | timing closure phase |
| RMW (L_CONV, L_ADD) | (2, 3), total 5 cyc | timing closure phase |
| First-tile init | TB zero-prime via mux | future hardware controller |
| Bank-to-column mapping | row-major flat → bank=flat%32, word=flat//32 | non-128² workload |

Future scope (not in current spec): use SRAM also as **B/weight input storage** (currently TB drives `in_b` directly via `$readmemh`). Separate spec needed.

## File layout

```
gemm_sram.xpr                          # Vivado 2024.1 project (target xc7vx485tffg1157-1)
                                       # synth top = gemm_sram_top; sim top = gemm_sram_top_tb
gemm_sram.srcs/sources_1/
    new/
        gemm_sram_top.v                # Integration wrapper (GEMM + RMW + SRAM, pure structural)
        GEMM.v                         # MXP TOP wrapper (instantiated inside gemm_sram_top)
        RMW.v                          # FP32 RMW unit (int_to_fp32 + delay + fp32_adder)
        int_to_fp32.v                  # INT32 + 9-bit signed scale -> IEEE-754 FP32
                                       # (Task 7 fix: zero-passthrough for in_int=0 + scale≠127)
        fp32_adder.v                   # IEEE-754 FP32 adder (HardFloat-based)
    imports/Desktop/MXP/...            # MXP compute RTL (Accumulator_Col.v has IMPLICIT_total patch)
    imports/Desktop/sram/rtl/...       # SRAM RTL (sram_1rw + sram_1rw_banked)
tb/
    gemm_sram_top_tb.v                 # Integration TB (9-mode plusarg-driven)
    rmw_tb.v, int_to_fp32_tb.v,
    fp32_adder_tb.v, rmw_smoke_tb.v,
    accumulator_col_elab.v             # Unit testbenches
sim/
    run_top_elab.sh                    # gemm_sram_top elab smoke
    run_integration_smoke.sh           # TB zero-prime + dump (no GEMM driving)
    run_integration_one.sh             # 1 mode end-to-end (<LABEL> <A_PREC> <B_PREC>)
    run_integration_sweep.sh           # 9-mode serial sweep + compare gate
    run_integration_parallel.sh        # 9-mode parallel dispatch guide (print template)
    run_rmw*.sh, run_int_to_fp32.sh, run_fp32_adder.sh, clean.sh
third_party/berkeley-hardfloat/        # Vendored HardFloat (HardFloatBundle.v + VENDORING.md)
MXP_Tools/                             # Python verification toolkit (fork of ~/Desktop/Desktop/MXP_Tools)
    pyproject.toml                     # mxp-tools entry point + pytest config
    mxp_tools/                         # gen / emit / ref / compare / viz / rmw-gen subcommands
    tests/                             # 45 pytest cases (compare/gemm/hwio/quant + interleaved 16/32)
        test_compare.py                # NaN/Inf-aware diff_3way (upstream)
        test_gemm.py                   # mxint_gemm_golden (upstream)
        test_hwio.py                   # @addr writememh, dup-write, emit shape, LF-only (upstream)
        test_quant.py                  # MX quant edge cases + NaN/Inf rejection (upstream)
        test_hwio_interleaved.py       # interleaved_row_major_{16,32}bank round-trip (project-only)
docs/
    next-session-kickoff.md            # Phase-handoff doc (updated each major phase end)
    hardfloat-setup.md                 # HardFloat re-vendoring guide (sbt + Chisel)
    superpowers/
        specs/2026-05-14-rmw-design.md
        specs/2026-05-14-integration-design.md
        plans/2026-05-14-rmw-implementation.md
        plans/2026-05-14-integration-implementation.md
        notes/mxp-driving-sequence.md         # Task 5 — gemm_sram TB perspective on driving
        notes/lane-to-c-mapping.md            # Task 6 — A4/A2 lane → C[m,n] dispatch table
precision_modes_protocol.md            # MXP driving protocol (v1.0, validated 2026-05-11)
gemm_sram.{sim,cache,hw,ip_user_files} # Vivado scratch — generated, do not edit
work/<LABEL>/                          # Integration sim work dirs (gitignored, generated by sweep)
```
