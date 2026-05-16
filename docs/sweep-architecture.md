# 9-mode integration sweep — 동작 원리

`bash sim/run_integration_sweep.sh` 가 9 가지 정밀도 조합 (A,B ∈ {2,4,8})
을 한 번에 회귀 검증하는 메커니즘 설명.

**TB 는 한 파일, 한 번도 수정 안 함.** 모든 mode 가 같은 TB 를 plusarg
로 mode-aware 로 굴림.

---

## 1. 전체 구조

```
┌────────────────────────── 외부 (bash 스크립트) ──────────────────────────┐
│                                                                          │
│  run_integration_sweep.sh                                                │
│   └─ for A_P in 2 4 8 / for B_P in 2 4 8 (9 iteration)                  │
│        ├─ MXP_Tools gen/emit/ref  ──► work/<LABEL>/{hw_input,sw_ref}/   │
│        ├─ run_integration_one.sh <LABEL> <A_P> <B_P>                     │
│        │    └─ xvlog + xelab + xsim                                      │
│        │         └─ -testplusarg "A_PREC=$A_P" "B_PREC=$B_P"             │
│        │                          "WORK_DIR=..." "DUMP_DIR=..."          │
│        └─ MXP_Tools compare  ──► PASS / FAIL                             │
│                                                                          │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ plusargs
                                   ▼
┌──────────── 내부 (Verilog, gemm_sram_top_tb.v 한 파일) ───────────────────┐
│                                                                          │
│  initial begin                                                           │
│    $value$plusargs("A_PREC=%d", A_PREC)   ◄── 외부 주입                  │
│    $value$plusargs("B_PREC=%d", B_PREC)                                  │
│    $value$plusargs("WORK_DIR=%s", WORK_DIR)                              │
│    $value$plusargs("DUMP_DIR=%s", DUMP_DIR)                              │
│  end                                                                     │
│                                                                          │
│  [CONFIG] 단계 — {A_PREC,B_PREC} 으로 case 디스패치:                     │
│    W_CYC, FIRES_PER_COL, N_T_LOGICAL,                                    │
│    A_CTRL_CODE, W_CTRL_CODE, TOGGLE_VAL,                                 │
│    A_FIRE_DELAY, FIRST_FIRE_GLOBAL                                       │
│                                                                          │
│  [LOAD]  $readmemh("WORK_DIR/a_input_BS_mxint*.hex", ...)                │
│  [DRIVE] mode-aware MAC sweep                                            │
│  [DRAIN] 32-stream per-col R-M-W                                         │
│  [DUMP]  $writememh per-bank → DUMP_DIR/bank{0..31}.mem                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. TB 의 plusarg 진입점

`tb/gemm_sram_top_tb.v` 의 module 상단 (라인 ~85):

```verilog
integer A_PREC, B_PREC;
reg [8*256-1:0] WORK_DIR;
reg [8*256-1:0] DUMP_DIR;

initial begin
    if (!$value$plusargs("A_PREC=%d",   A_PREC))    A_PREC   = 8;
    if (!$value$plusargs("B_PREC=%d",   B_PREC))    B_PREC   = 8;
    if (!$value$plusargs("WORK_DIR=%s", WORK_DIR))  WORK_DIR = "work/A8_B8";
    if (!$value$plusargs("DUMP_DIR=%s", DUMP_DIR))  DUMP_DIR = "work/A8_B8/hw_out";
end
```

plusarg 가 안 주어지면 A_PREC=B_PREC=8, WORK/DUMP_DIR = `work/A8_B8/...` 가
기본값. Vivado GUI 에서 plusarg 없이 그냥 Run Behavioral Simulation 해도
A8_B8 mode 가 자동으로 돌게 설계.

---

## 3. TB 의 mode 디스패치

CONFIG 단계에서 `{A_PREC, B_PREC}` 으로 9 가지 case 분기. 각 분기가
mode-specific 상수 8 개를 reg 에 적재:

| 상수 | 의미 | A8 값 예시 | A4 값 예시 | A2 값 예시 |
|---|---|---|---|---|
| `W_CYC` | weight bit-serial cycle 수 (B_PREC 의존) | B8: 8 / B4: 4 / B2: 2 | 동일 | 동일 |
| `A_FIRE_DELAY` | activation chain delay 보정 | 2 | 2 | 0 |
| `FIRST_FIRE_GLOBAL` | 첫 fire 의 global cycle | 28 (B8) | mode 의존 | mode 의존 |
| `TOGGLE_VAL` | station Buf1/Buf2 toggle cycle | 24 (B8) | mode 의존 | mode 의존 |
| `FIRES_PER_COL` | col 당 fire 횟수 (capture 가드) | 2048 | 1024 | 512 |
| `N_T_LOGICAL` | logical N-tile loop 횟수 | 4 | 2 | 1 |
| `A_CTRL_CODE` | Station precision 2비트 | A_INT8 | A_INT4 | A_INT2 |
| `W_CTRL_CODE` | Wcontrol precision 2비트 | W_INT8 | W_INT4 | W_INT2 |

이 상수들이 DRIVE 단계의 `drive_stage_3_4` 루프 boundary, capture 의 lane
디코딩 (A8=1 RMW/fire, A4=2, A2=4), drain 의 entry 수 등을 결정.

**총 RMW dispatch 수 = mode 무관 65536**:
- A8: 2048 fire × 1 lane = 2048/col × 32 col = 65536
- A4: 1024 fire × 2 lane = 2048/col × 32 col = 65536
- A2:  512 fire × 4 lane = 2048/col × 32 col = 65536

PASS 배너의 `EXPECTED_TOTAL=65536` 이 9 mode 공통인 이유.

---

## 4. sweep 스크립트의 9-iteration loop

`sim/run_integration_sweep.sh` (요약):

```bash
for A_P in 2 4 8; do
  for B_P in 2 4 8; do
    LABEL="A${A_P}_B${B_P}"

    # 1) MXP_Tools 가 mode 별 입력 hex + golden npz 생성
    (cd MXP_Tools && \
      python -m mxp_tools gen   --out ../work/${LABEL} -M 128 -K 128 -N 128 --seed 0 && \
      python -m mxp_tools emit  --out ../work/${LABEL} && \
      python -m mxp_tools ref   --out ../work/${LABEL} --prec-a ${B_P} --prec-b ${A_P})
      # NOTE arg swap: MXP_Tools --prec-a = WEIGHT = 우리 B_P
      #                MXP_Tools --prec-b = ACTIV  = 우리 A_P

    # 2) HW sim — TB 에 plusarg 주입
    bash sim/run_integration_one.sh "${LABEL}" "${A_P}" "${B_P}"

    # 3) compare gate
    BANKS=$(printf "../work/${LABEL}/hw_out/bank%d.mem " {0..31})
    (cd MXP_Tools && \
      python -m mxp_tools compare \
        --ref ../work/${LABEL}/sw_ref/C_sw_mxint${B_P}_mxint${A_P}.npz \
        --hw-banks ${BANKS} \
        --layout interleaved_row_major_32bank)
  done
done
```

---

## 5. `run_integration_one.sh` 의 xsim 호출

```bash
(cd "$BUILD" && \
    xvlog -sv \
        <HardFloat + MXP + SRAM + RMW + sram_1rw_banked_mp + GEMM + top + tb> && \
    xelab -L work gemm_sram_top_tb -snapshot gemm_sram_top_tb_snap && \
    cmd //c "xsim gemm_sram_top_tb_snap -runall \
              -testplusarg \"A_PREC=$A_PREC\" \
              -testplusarg \"B_PREC=$B_PREC\" \
              -testplusarg \"WORK_DIR=../../../$WORK\" \
              -testplusarg \"DUMP_DIR=../../../$DUMP\"")
```

**Git Bash on Windows gotcha**: `-testplusarg "K=V"` 의 `=` 가 cmd.exe quoting
으로 stripped 됨. `cmd //c "..."` 로 한 번 더 감싸야 정상 전달.

---

## 6. Mode 간 격리

mode 끼리 간섭하지 않는 이유:

| 격리 단위 | 경로 | 내용 |
|---|---|---|
| Vivado/XSim build | `sim/build/<LABEL>/` | xvlog/xelab 산출물 (xsim.dir, .pb, snapshot) |
| MXP_Tools 입력 | `work/<LABEL>/hw_input/` | mode 별 hex (`a_input_BS_mxint{8,4,2}.hex` 등) |
| Golden reference | `work/<LABEL>/sw_ref/` | `C_sw_mxint${B_P}_mxint${A_P}.npz` |
| HW dump | `work/<LABEL>/hw_out/` | 32 .mem files (이 sim 결과) |
| xsim process | per-invocation | 매번 새로 시작 — state 0 부터 |

`run_integration_sweep.sh` 가 순차로 9 번 invoke 하므로 동시 실행 충돌 없음.
병렬화하고 싶다면 `sim/run_integration_parallel.sh` (가이드만 출력, 9 subagent
디스패치 패턴) 참고.

---

## 7. PASS / FAIL 판정 (2 단계)

### TB in-sim (구조적)
종료 직전 4 개 invariant 확인:
1. capture 총 push 수 == EXPECTED_TOTAL (65536)
2. drain 총 pop 수 == EXPECTED_TOTAL
3. bank-col mismatch assert 가 sim 도중 한 번도 `$finish FATAL` 안 함
4. dump task 가 32 파일 모두 `$fopen` 성공

모두 OK → `[PASS] INTEGRATION TB A<P>_B<P> — structural checks OK` 배너.
하나라도 실패 → `[FAIL] ...` 배너 + count 진단.

### 외부 compare (비트 정확도)
sweep 의 마지막 단계가 `python -m mxp_tools compare`. HW dump 의 32 bank
.mem 을 `interleaved_row_major_32bank` 매핑으로 (M, N) 행렬로 재구성한 뒤
golden npz 의 `C_sw` 와 word-level 비교:

```
pair                   max          rmse      mean_abs      snr_dB    n_diff
hw_sw            0.000e+00     0.000e+00     0.000e+00        inf         0   ← bit-exact PASS
```

`n_diff == 0` 이어야 PASS. 한 워드라도 다르면 sweep 의 exit code 가 비-0
→ "FAILED: <LABEL>" 출력 + 다음 mode 로 continue.

### 최종 게이트
9 modes 의 compare 결과를 sweep 스크립트가 tally:
```
SWEEP RESULT: 9/9 PASS
PASSED: A2_B2 A2_B4 A2_B8 A4_B2 A4_B4 A4_B8 A8_B2 A8_B4 A8_B8
ALL 9 MODES PASSED
```

`ALL 9 MODES PASSED` 가 최종 회귀 게이트.

---

## 8. 단일 모드만 돌리기

A8_B8 만 빠르게 회귀하고 싶을 때:

```bash
# 1) 입력 생성
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8
cd ..

# 2) HW sim
bash sim/run_integration_one.sh A8_B8 8 8

# 3) compare
cd MXP_Tools
BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..31})
python -m mxp_tools compare \
    --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
    --hw-banks ${BANKS} \
    --layout interleaved_row_major_32bank
```

A8_B8 ~ 2분 (vs 전체 9-mode sweep ~11분).

---

## 9. 새 mode 추가 절차 (참고)

만약 미래에 A1 같은 새 precision 추가 시:

1. **MXP_Tools 측**: `mxp_tools/quant.py` 의 quant 함수에 1-bit 지원 추가.
2. **MXP RTL 측**: Station / Accumulator 의 precision 디코더에 A_INT1 케이스 추가 (`../MXP/`).
3. **TB CONFIG case**: `tb/gemm_sram_top_tb.v` 의 CONFIG 단계 case 에 새 mode 의 8 상수 추가.
4. **Capture lane 디코딩**: A_PREC=1 일 때 col 당 몇 lane 인지 → fifo push 분기 추가.
5. **EXPECTED_TOTAL**: 새 mode 도 32 col × N RMW/col 로 같은 totals 가 나오는지 확인. 다르면 mode 별 EXPECTED 로 분기.
6. **sweep loop**: `run_integration_sweep.sh` 의 `for A_P in 2 4 8` 에 `1` 추가.

TB 의 plusarg 진입점과 mode-aware 디스패치 구조 자체는 그대로.

---

## 10. 관련 파일

- `tb/gemm_sram_top_tb.v` — TB 본체 (plusarg 진입점 + CONFIG case + DRIVE/DRAIN/DUMP).
- `sim/run_integration_one.sh` — 단일 mode 실행 (xvlog/xelab/xsim).
- `sim/run_integration_sweep.sh` — 9-mode loop + compare gate.
- `sim/run_integration_parallel.sh` — 9-way 병렬 dispatch 가이드 (subagent 용 템플릿 출력).
- `MXP_Tools/mxp_tools/cli.py` — `gen / emit / ref / compare` 서브커맨드.
- `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_32bank` — 32-bank 역매핑.
- `precision_modes_protocol.md` — MXP 의 driving protocol 원본 spec (mode 별 chain delay / packing 정의).
- `docs/superpowers/notes/lane-to-c-mapping.md` — A4/A2 lane → C[m,n] 매핑 표.
