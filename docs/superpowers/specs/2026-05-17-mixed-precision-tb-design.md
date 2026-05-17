# Mixed-precision (per-block W_PREC) 검증 TB — Design

**Date**: 2026-05-17
**Author**: TaeJun (with Claude)
**Status**: Draft → User review

---

## 1. 검증 목적

기존 9-mode integration sweep (`bash sim/run_integration_sweep.sh`) 은 한 시뮬레이션이 **행렬 전체에 단일 (A_PREC, B_PREC)** 조합만 검증한다. 실제 MXP HW 는 control 신호 (`in_Wcontrol`, `in_Mode_oh`) 가 cycle 단위로 driving 가능하도록 설계되어 있어 **행렬 내부의 weight block 마다 다른 W_PREC** 도 지원해야 한다. 본 TB 는 이 능력의 **연산 정합성** 을 bit-exact 로 검증한다.

검증 통과 조건: random W_PREC 패턴 (블록별 {2,4,8}) 으로 driving 한 HW 의 FP32 누적 결과가 같은 패턴으로 계산한 Python 골든 FP32 와 **비트 단위 일치**.

---

## 2. 검증 대상의 mix 차원

| 차원 | 단위 | 개수 | mix 여부 |
|---|---|---|---|
| Activation precision | layer 전체 | 1 | **균일** (per-sim 한 값, plusarg) |
| Weight precision | (M-row, K-block) | 128 × 4 = **512** | **block 마다 random** |

- **block 정의**: MX 양자화의 native block = K-axis 32 element. weight matrix [M=128, K=128] 에서 (m row, k_t ∈ {0,1,2,3}) pair 마다 한 block.
- **random 분포**: {2, 4, 8} 균등 (1/3 each), seed=0 고정 재현.

### 2.1 왜 이 단위가 RTL 변경 없이 가능한가 (확정 사실)

SystolicArray.v 검증 결과:
- `in_a[r]` (1-bit per row) = **K-axis 의 K=r 의 weight bit** → row 차원 = K-axis.
- cycle 진행 = **M-axis** (한 M 처리에 W_PREC cycle).
- 한 cycle 에 32 K-element 가 같은 K-block 안 → 한 cycle 의 in_Wcontrol 한 값으로 그 K-block 의 W_PREC 표현 가능.
- M 진행마다 in_Wcontrol 갱신 → 그 (M, K-block) 의 W_PREC 적용.
- Accumulator 의 cnt 가 in_Wcontrol 따라 W cycle 후 fire → 다음 M.

**dataflow / RTL 변경 0**. driving 측만 변경.

---

## 3. 컴포넌트

```
┌────────────────────────────────────────────────────────────────┐
│ sim/gen_mixed.py  (Python: data + golden + visualize)          │
│   - seed=0 random W_PREC_MAP[M=128, K_T=4]                     │
│   - random FP weight + activation (M=K=N=128)                  │
│   - activation: MX quant @ A_PREC (mxp_tools.quant import)     │
│   - weight: (M, K-block) 마다 그 block 의 W_PREC 로 MX quant   │
│   - FP32 GEMM golden = Q(W) × Q(A) → C_sw_mixed.npz            │
│   - hex emit (TB 입력)                                          │
│   - precision_map.png + .txt (visualize)                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ tb/gemm_sram_top_mixed_tb.v  (신규)                            │
│   plusarg: A_PREC, WORK_DIR, DUMP_DIR                          │
│   기존 TB 의 INIT/LOAD/CONFIG/DRIVE/DRAIN/DUMP 구조 답습       │
│   LOAD 시 w_prec_per_block.hex 추가 로드                       │
│   DRIVE 의 inner-loop 만 mixed-aware 변경                      │
│   capture/decode/RMW/dump 는 기존 그대로 (A_PREC 만 의존)      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ sim/run_mixed_one.sh   <A_PREC>                                │
│ sim/run_mixed_sweep.sh  (A=2/4/8 직렬, "ALL 3 MIXED PASSED")   │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Hex / 파일 contract

| 파일 | 라인 수 | 한 라인 폭 | 의미 |
|---|---|---|---|
| `a_input_BS_mixed.hex` | 4096 | 32-bit | weight bit-serial. **W=8 padding 컨벤션** — W<8 인 (M, K-block) 의 상위 padding bit 는 0 (= contribution 0, 결과 동일) |
| `b_input_mixed_A{P}.hex` | 512 | 256-bit | activation (기존 `b_input_mxint{P}.hex` 와 동일 layout, A_PREC=P) |
| `a_scale_mixed.hex` | 512 | 8-bit | weight scale (E8M0) |
| `b_scale_mixed.hex` | 512 | 8-bit | activation scale |
| `w_prec_per_block.hex` | 128 | 8-bit | (M-row) 별 4 K-block 의 W_PREC code 팩킹. bit[2k+1:2k] = (m, k_t=k) 의 W_CTRL_CODE (W_INT8/4/2 = 11/10/01) |

TB 는 `$readmemh` 로 5 파일 로드, driving 시 `W_PREC_MAP[m][k_t]` lookup → 그 iteration 의 in_Wcontrol 결정.

---

## 5. 정답 (golden) 정의

```
For each (m, n) ∈ [0,128) × [0,128):
    C_sw[m, n] = FP32 sum over k_t ∈ [0,4) of (
        FP32 dot product of (
            Q_W[m, k_t * 32 + k] for k ∈ [0,32),   # W_PREC = W_PREC_MAP[m][k_t]
            Q_A[k_t * 32 + k, n]                    # A_PREC = layer 단일
        )
    )
```

- `Q_W`, `Q_A` 는 각각 (block, A_PREC) MX 양자화 후 FP 복원 값 (= scale × INT 값).
- 한 (m, n) 의 K-axis 32 element dot product 가 한 K-block 의 contribution → FP32 누적 → 4 K-block 합산 = 한 RMW step 4 회 누적과 일치.
- `C_sw_mixed.npz` 에 FP32 array 로 저장.

---

## 6. 비교 gate

| 시나리오 | A_PREC | seed | 검증 |
|---|---|---|---|
| `mixed_A8` | 8 | 0 | bit-exact vs `C_sw_mixed.npz` |
| `mixed_A4` | 4 | 0 | bit-exact |
| `mixed_A2` | 2 | 0 | bit-exact |

`sim/run_mixed_sweep.sh` 가 3 시나리오 직렬 실행, 마지막에 `ALL 3 MIXED SCENARIOS PASSED` 출력 (현 9-mode sweep 의 `ALL 9 MODES PASSED` 와 동일 양식). 비교는 기존 MXP_Tools `compare` subcommand 재사용 (32-bank dump → 골든 npz).

---

## 7. 검증 항목

| 레벨 | 항목 | 책임 |
|---|---|---|
| In-sim | capture 총 push 수 == drain 총 pop 수 == EXPECTED (65536) | TB |
| In-sim | bank-col 매핑 assert (모든 lane 의 flat&0x1F == col) | TB |
| In-sim | dump 파일 32개 모두 fopen 성공 | TB |
| Post-sim | FP32 32-bank dump = `C_sw_mixed.npz` bit-exact | MXP_Tools `compare` |

---

## 8. 비-목표 (non-goals)

- timing closure / 합성 — 본 TB 는 behavioral sim 한정.
- W_PREC chain 의 col-별 wave (= N-axis weight mix) 활용 — 현 spec 은 M-axis (cycle 진행) mix 만.
- mixed-precision 의 throughput / bandwidth 측정 — random pattern 의 sim time 은 평균 W ≈ 4.67 (균등 분포). 본 TB 의 1차 목표는 연산 정합성. dataflow / bandwidth 디테일은 implementation 단계에서 결정, spec 에는 잠그지 않음.
- MXP_Tools 의 gen/emit/ref 확장 — 본 TB 는 `sim/gen_mixed.py` 가 독립. MXP_Tools 는 utility (quant) 만 import.
- RTL 변경 — `../MXP/`, `../sram/`, 본 프로젝트의 `gemm_sram_top.v` 모두 변경 없음.

---

## 9. Visualize 산출물

- `work/mixed_A{P}/precision_map.png`: 128 row × 4 K-block heatmap (색 3종, 라벨 W=2/4/8) + 캡션에 `A_PREC=P, seed=0`.
- `work/mixed_A{P}/precision_map.txt`: ASCII 동등 도표 (matplotlib 없는 환경 대비).

---

## 10. 파일 layout (산출물)

```
sim/
  gen_mixed.py                          # 신규
  run_mixed_one.sh                      # 신규
  run_mixed_sweep.sh                    # 신규
tb/
  gemm_sram_top_mixed_tb.v              # 신규 (헤더 한글, 검증 목적/시나리오/동작 의도 명시)
work/mixed_A{2,4,8}/                    # gitignored, gen_mixed.py 산출
  hw_input/a_input_BS_mixed.hex
  hw_input/b_input_mixed_A{P}.hex
  hw_input/a_scale_mixed.hex
  hw_input/b_scale_mixed.hex
  hw_input/w_prec_per_block.hex
  sw_ref/C_sw_mixed.npz
  precision_map.png
  precision_map.txt
  hw_out/bank{0..31}.mem                # TB dump 산출
```

---

## 11. 검증 통과 = 무엇이 입증되는가

3 시나리오 (A=2,4,8) 모두 bit-exact PASS 이면:
- MXP HW 가 **K-tile / m_in iteration 단위로 in_Wcontrol 을 cycle-by-cycle 갱신** 했을 때 Accumulator cnt 가 정확히 그 W_PREC 의 fire 시점을 만들어내고, downstream chain (scale_weight, station selector) 도 mixed-pattern 에서 정합성 유지.
- random 512 block 패턴이 dataflow 깨지지 않고 정상 동작.
- RMW + SRAM 의 FP32 축적 결과가 골든과 ULP 0 일치.
