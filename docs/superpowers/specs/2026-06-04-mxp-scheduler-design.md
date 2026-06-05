# MXP_scheduler — precision-weighted GEMM mapping + variable-cost tile scheduling (cost model + optimizer)

작성일: 2026-06-04 (rev. prior-art 반영)
상태: 설계 (사용자 리뷰 대기)
관련:
- 통합 spec `2026-05-14-integration-design.md` §9 "Loop order", CLAUDE.md "다음 세션 후보 #4 Loop order explorer"
- prior-art 연구 보고서 `docs/research/2026-06-04-mapping-scheduler-prior-art.md` (반영: C1~C9)

---

## 1. 목표 (한 줄)

이 MXP bit-serial SA 가속기를 위한 **GEMM mapping cost model + optimizer**.
SRAM bank 크기/개수와, 타일링된 행렬의 **타일별 평균 weight 비트수**를 입력으로 받아,
(성능 제약 하) 가장 에너지가 낮은 **mapping(= outer 순열 + on-chip blocking)** 을 출력한다.

핵심 가치는 **두 개의 precision-구동 레버** (연구 보고서 §0):
1. **energy 레버** — 어느 피연산자를 reload 할지(= blocking partition + stationary 선택, 타일별 비트수로 가중). traffic/energy 의 1급 결정자.
2. **cycle 레버** — 타일별 compute time 이 달라(저비트=짧음) double-buffer 의 latency-hiding 예산이 타일마다 다름 → **타일 실행 순서가 실제 cycle(=compute+stall)을 결정**. precision-구동 latency-hiding ordering 은 prior art 미개척(가장 가까운 선례 sparsity load-imbalance).

**순수 outer-permutation 자체는 저레버리지/대부분 중복** — 절대 "loop order optimizer" 로 과대포장하지 않는다.

**Software-only. RTL 무수정. 두 개의 `.py`** — 표준본 + 주석 상세본(동일 결과) (§12). 기존 `sim/`, `MXP_Tools/` 와 독립.

---

## 2. 범위 / 비범위

### v1 (본 spec)
- 결정 공간 = **nested-blocked mapping**: 차원(M,K,N)별 outer×inner blocking + outer 루프 순열. partial psum spill 포함.
- precision-aware: 타일별 평균 weight 비트수 → cycle/byte 비용.
- evaluator(이벤트 카운터, 순수) → energy(정규화 상대 coeff) + **sequence-aware stall** + actual cycle.
- optimizer = **exhaustive** (순열 × blocking, 용량 필터) + **LPT-greedy baseline**(스케줄링 헤드룸 측정용) + 성능 제약.
- 제약 ON/OFF Pareto tradeoff.
- (있으면) TB 실측 cycle 로 calibration (§10).

### Phase 2 (비범위, 인터페이스만 열어둠)
- **자유 per-tile 스케줄링** (nested 구조 밖, T! 공간) — job-shop/LPT 정식화 + **CP-SAT/MILP** solver. cycle 레버의 완전 탐색.
- multi-level tiling (L2 on-chip 블록 + L1 inner), uneven/per-operand asymmetric blocking (ZigZag, 최대 33%).
- precision **공동 최적화** (블록별 비트수를 optimizer가 선택). v1 은 비트수를 *입력*으로만 받음.
- fill/drain·array-utilization 정밀화, 실측 energy 계수(FPGA/ASIC) 도입.

### 명시적 비목표
- RTL/TB 변경, 실제 하드웨어 합성, cycle-accurate 시뮬레이션.

---

## 3. I/O 계약

### 입력
| 항목 | 형태 | 비고 |
|---|---|---|
| 행렬 크기 `(M, K, N)` | int×3 | 32 의 배수 가정(아니면 ceil-pad, A1) |
| bank 크기, bank 개수 | int×2 | on-chip 용량 `C_buf` = bank_size × bank_count × word_bits |
| 타일별 평균 weight 비트수 | `M_T × K_T` float 배열 | tile = 32×32. 값 ∈ [2,8] |
| activation precision | int (2/4/8) | layer-uniform. 타일 내 uniform 보장 |
| DRAM BW | bits/cycle | **sequence-aware stall 의 핵심 파라미터** (A4) |
| energy 계수 | dict (내장 기본값) | 정규화 상대값 (§7). `--coeffs file.json` 로 교체 가능 |
| 성능 제약 | (선택) cycle budget 또는 "stall=0" | 제약 ON 용 (§8) |

### 출력
- **추천 mapping**: `(outer 순열, 차원별 blocking factor)`.
- breakdown: DRAM traffic(텐서별), on-chip access, MAC, RMW, energy(항목별), compute-work cycle, **stall cycle, actual cycle**.
- Pareto 집합: 모든 feasible mapping 의 (energy, actual_cycle) 점 + 제약 ON/OFF 추천 + LPT baseline 대비.

---

## 4. 하드웨어 모델 (grounding — RTL 정독 기반)

cost model 은 추측이 아니라 실제 RTL(`SystolicArray.v`, `PE_feeder.v`, `Accumulator*.v`, `RMW.v`, `sram_1rw_banked_mp.v`) 구조에 grounding.

### 4.1 Dataflow
- **activation-stationary + weight bit-serial.** `in_b`(activation) 가 PE station(Buf1/Buf2 ping-pong)에 적재·재사용. `in_a`(weight) 1비트씩 stream (`add_operand = in_a ? station : 0`).
- **K-reduction 은 spatial** — 32 row(K) tree-sync 로 합쳐져 col 출력.
- **M 은 temporal** — cycle 진행. bit-serial accumulator 는 `cnt = W_PREC−1` 에 fire (INT8=8, INT4=4, INT2=2 cycle).
- 한 SA pass(= cube-op) = **32(K) × 32(N) × 32(M) 큐브를 `32 × W_PREC` cycle** 에 처리.

### 4.2 텐서 인덱스 = 재사용 구조 (cost model 의 핵심)
| 텐서 | 인덱스 | 무관 축 → 재사용 |
|---|---|---|
| A (activation) | `[K, N]` | M 무관 → M 축 재사용 |
| W (weight) | `[M, K]` | N 무관 → N 축 재사용 |
| C (output) | `[M, N]` | K 무관 → K 축 **누적** (RMW running sum) |

### 4.3 비용 관련 상수
- **compute work/cube** = `32 × b` (b = 그 (m_t,k_t) 타일 평균 weight 비트). 실제 cycle 은 여기에 stall 가산(§8).
- **출력 원소당 RMW dispatch**: A8=1, A4=2, A2=4 (activation mode 별 상수 배수).
- **RMW**: latency 5 cycle, pipelined, 32 col-parallel (col j → bank j).
- **SRAM(현 HW)**: 32 bank × 1024 word × 32-bit. read latency 1 cycle. **v1 은 용량을 입력 파라미터로 일반화** (현 HW = 기본값).

### 4.4 "32×32" 의 의미 & DRAM 부재 → 가상 계층 (연구 C2)
- **"32×32" = 물리적 systolic array = spatial mapping = HW 고정.** 행렬 크기가 아님. Timeloop/CoSA/ZigZag 가 탐색하는 spatial-mapping/PE-array 축이 우리는 통째로 빠짐(탐색 안 함).
- 현 RTL 은 C psum 만 on-chip, A/W 는 array 직접 구동. v1 은 **Timeloop식 DRAM ↔ on-chip 버퍼 ↔ array** 3계층을 모델링(A/W/C 모두 on-chip 버퍼링 가능, future dataflow 탐색용). 재사용/비용은 §4.1–4.3 에 grounding.

---

## 5. 결정 공간 — nested-blocked mapping

### 5.1 타일링
`M_T = ⌈M/32⌉`, `K_T = ⌈K/32⌉`, `N_T = ⌈N/32⌉` 큐브. 총 cube-op `T = M_T·K_T·N_T` (각 정확히 1회).

### 5.2 mapping = (blocking factor) × (outer 순열)
각 차원을 **outer(DRAM-stream) × inner(on-chip resident)** 분할:
```
M_T = M_out · M_in,   K_T = K_out · K_in,   N_T = N_out · N_in
```
- inner block = on-chip 동시 resident. outer = 바깥 stream.
- **outer 루프 순열** = `{M_out, K_out, N_out}` 3 루프 순서 (≤ 6).
- inner 루프 순서 / m_in(32) cycle 진행은 array 고정.
- blocking factor = 각 차원 tile 수의 **약수**만 (균등 분할 가정, A2).
- **이 (순열 + blocking) 이 cube-op 의 실행 *시퀀스* 를 결정** → §8 sequence-aware stall 의 입력.

### 5.3 K-legality
cube-op `(m_t,k_t,n_t)` 는 정의상 `W[m_t,k_t]·A[k_t,n_t]` 로 K 자동 정합 → 어떤 순열/blocking 이든 legality 공짜. (K 정합은 *한 번의 곱* 조건, 전역 순서 제약 아님.)

### 5.4 partial psum spill
`K_out > 1` → (m_t,n_t) 출력의 K 기여가 outer-K 에 흩어짐 → C psum 을 outer-K 경계마다 DRAM spill/reload. `K_out = 1` → spill 없음(on-chip 누적). **이 spill 경우의 수가 공간에 포함됨**(사용자 요구).

### 5.5 순서 vs blocking — 정밀한 위치 (연구 C1, ★ 정정)
on-chip 이 LRU 캐시가 아니라 **explicit scratchpad** 라 두 층위로 갈린다:

- **(2a) 고정 partition 하 outer 순열은 총 DRAM 볼륨을 안 바꿈** (reload_factor 가 outer trip-count 의 곱 = commutative) **이고 다수 순열이 traffic 상 중복** (dMazeRunner: 5040 순서 → 고유 reuse ~15; Timeloop: innermost level 에서만 순서 무관). → **순수 순열은 저레버리지**.
- **(2b) 그러나 blocking partition + stationary 피연산자 선택은 traffic/energy 의 1급 레버** (SALSA: 순서 단독 energy 50%; Demystifying MSE: 14.4× EDP; CoSA: 1.7×). **"순서는 energy 무관" 은 거짓** — 절대 쓰지 않는다.

→ energy 주역 = **blocking partition + stationary 선택**(순수 순열 아님). cycle 주역 = **sequence-aware stall**(§8) — 여기서 순서가 실제로 강하게 작용. dMazeRunner 식 "고유 reuse factor 순열만 enumerate" pruning 채택(중복 제거).

---

## 6. Evaluator — 이벤트 카운터 (순수 함수, 가중치 없음)

`evaluate(mapping, hw, workload) → (EventCounts, compute_work, feasible)` — energy/stall 은 별도(§7,§8).

### 6.1 용량 제약 (feasible 판정)
on-chip 동시 resident footprint (bit):
```
foot(A) = K_in·N_in · (32·32) · act_bits
foot(W) = M_in·K_in · (32·32) · avg_W_bits(resident 타일들)
foot(C) = M_in·N_in · (32·32) · 32          # FP32 psum
footprint = foot(A)+foot(W)+foot(C)  ≤  C_buf
```
double-buffer prefetch 여유는 §8 에서 처리. footprint > C_buf → infeasible.

### 6.2 DRAM traffic (energy 주변동분)
reload_factor = outer 루프 중 해당 텐서를 인덱싱하지 *않는* 루프 trip-count 곱:
```
reload(A) = M_out ;  reload(W) = N_out ;  reload(C) = K_out   # = psum spill 횟수
DRAM_read(A)  = (K·N · act_bits)           · M_out
DRAM_read(W)  = (Σ_tiles 32·32·avg_W_bits) · N_out
DRAM_C_write  = (M·N · 32) · K_out                 # outer-K partial write
DRAM_C_read   = (M·N · 32) · (K_out − 1)           # 누적 reload (첫 touch zero-init)
```
`K_out = 1` → C_read=0, C_write = 최종 1회분.

### 6.3 on-chip access / MAC / RMW (연구 C3: ★ 상수 아님)
```
MAC               = M·K·N                                   # 논리 MAC (고정)
onchip_PE_read    = T·(32·32)·act_bits + N_T·Σ_tiles 32·32·avg_W_bits  # array 가 매 cube 피연산자 read (고정분)
onchip_refill_wr  = DRAM_read(A)+DRAM_read(W)+DRAM_C_read   # DRAM→on-chip 적재 (★ mapping 변동분)
RMW               = T·(32·32)·disp                          # disp = A8×1/A4×2/A2×4
```
**주의**: on-chip access 는 "상수" 가 아니다 — `onchip_refill_wr` 가 DRAM traffic 을 따라 변동(Timeloop: 같은 min-DRAM 인데 energy 11× 차이의 원인이 바로 on-chip access). 변동분을 반드시 카운트.

### 6.4 compute work (결정론적 ideal 하한 — actual cycle 아님, 연구 C3 ★)
```
compute_work = Σ_{cube} 32 · b(m_t,k_t) = N_T · Σ_{m_t,k_t} 32 · b(m_t,k_t)
```
순열/blocking **무관** — 이건 **stall 없는 ideal 하한**("ideal cycle"). **실제 cycle = compute_work + stall**, 그리고 stall 은 순서 의존(§8). "compute work 총합이 order-무관" 을 "실제 cycle 이 order-무관" 으로 확장하면 안 됨.

---

## 7. Energy 모델 (분리된 마지막 단계)

evaluator 는 **절대 energy 를 안 봄.** 별도:
```
energy = Σ_event  count(event) · coeff(event)
```
`coeff` = 정규화 상대값(Timeloop/Accelergy/ZigZag 식), 내장 기본 dict + `--coeffs` 교체. 출발 기본표(합의 대상 A3):

| event | 단위 | 기본 상대 coeff (예시) |
|---|---|---|
| DRAM access | per bit | 200 |
| on-chip access | per bit | 6 |
| MAC | per op | 1 |
| RMW (add+convert) | per op | TBD — A3 합의 |

핵심: **카운트는 선호-free.** "reuse 좋다/spill 나쁘다" 는 coeff(투명한 입력)에서만.

---

## 8. Cycle / stall 모델 — sequence-aware (연구 C2/C3 ★ roofline 폐기)

**roofline `max(Σcompute, Σmem)` 은 두 항 다 총합이라 순서에 눈멀어 cycle 레버를 통째로 놓침 → 폐기.** 대신 mapping 이 정한 cube-op **시퀀스를 따라 stall 을 누적**한다(작은 파이프라인 시뮬레이션):

```
시퀀스 = mapping(순열+blocking) 이 정한 cube-op 순서 [0..T-1]
c_i      = 32 · b(cube_i)                                   # 타일 compute time (precision 의존)
f_{i+1}  = (cube_{i+1} 의 non-resident 피연산자 bit) / DRAM_BW   # 다음 fetch time
stall_i  = max(0, f_{i+1} − c_i)                            # double-buffer 가 못 숨긴 분
actual_cycle = fill + Σ_i c_i + Σ_i stall_i
```
- **저비트 타일은 c_i 가 짧음 = hiding 예산 적음 → stall** (사용자 직관). 따라서 **순서가 actual cycle 을 좌우** = cycle 레버.
- `f_{i+1}` 은 resident 집합(blocking)과 다음 cube 가 공유하는 피연산자에 따라 달라짐 — 같은 inner 루프 내 연속 cube 는 fetch 작음, outer 경계는 큼.
- prefetch 여유: double-buffer 하려면 resident footprint 외 stream 타일 1개분 여유 필요. `C_buf − footprint < 1 stream tile` 이면 prefetch 못 함 → 해당 구간 fully exposed.
- ⚠ 이 stall 모델이 본 설계에서 **가장 손 검증/calibration 필요**(§10).

### 성능 제약 ON/OFF (step 4)
- **OFF**: 전역 min-energy mapping.
- **ON**: `actual_cycle ≤ budget` (또는 stall=0) 제약 하 min-energy.
- 출력: feasible mapping 을 (energy, actual_cycle) 평면 → **Pareto front** + ON/OFF 추천 + LPT baseline.

---

## 9. Optimizer (연구 C2/C9 ★ "MILP 불필요" 는 공간 한정)

### 9.1 v1 = exhaustive (순열 × blocking)
후보 = (outer 순열 ≤6, 중복 reuse pruning 후) × (M_out,K_out,N_out 약수 조합), 용량 feasible 필터. 각 mapping 에 §6→§7→§8 적용. exact + oracle.

**공간은 행렬 크기와 무관하게 작음** — 약수 개수 d 로만 증가:

| 행렬 (M=K=N) | M_T=K_T=N_T | d | (순열×blocking) ≈ |
|---|---|---|---|
| 128³ | 4 | 3 | 6·3³ = **162** |
| 512³ | 16 | 5 | 6·5³ = **750** |
| 1024³ | 32 | 6 | 6·6³ = **1296** |
| 4096³ | 128 | 8 | 6·8³ = **3072** |

→ 거대 행렬도 수천 → exhaustive 안전. (full-DNN mapspace `~10²⁴`(typical edge), Mind Mappings `~10²⁵` 대비 극소이므로 MILP/GA/gradient 불필요.)

### 9.2 LPT-greedy baseline (v1, 연구 C5)
선택된 mapping 의 시퀀스를 **타일을 compute time 내림차순으로 재배열**(LPT, O(T log T))해 stall 이 얼마나 줄 수 있는지 *headroom* 만 측정. nested 시퀀스 대비 개선폭 = 자유 스케줄링(Phase 2)의 잠재 이득 추정치. (LPT/job-shop 은 *독립 작업·makespan* 용이라 reuse 결합·energy 목적이 있는 우리엔 drop-in 아님 — baseline/headroom 으로만.)

### 9.3 Phase 2 = 자유 per-tile 스케줄링
nested 밖 자유 순서는 `T!`(128³→64!, 1024³→32768!)로 폭발 → **CP-SAT/job-shop(OR-Tools) 가 여기서 정당**. evaluator 가 pure function 이라 그대로 위에 얹음. v1 의존성 없음(stdlib).

---

## 10. 검증 전략

- **손 계산 단위 테스트** (`--selftest`): 작은 케이스(M_T=K_T=N_T=2, b 균일/비균일)에서 reload_factor, DRAM 볼륨, compute_work, **그리고 sequence stall**(직접 시퀀스 따라 손 계산) 고정.
- **경계 케이스**: K_out=1(spill 없음) vs K_out=K_T(매번 spill) C traffic 대조. 용량 1 타일(극소) vs ∞. BW→∞(stall=0) vs BW 작음(stall 지배).
- **단조성/sanity**: 용량↑ → DRAM traffic 단조 비증가. 8bit A reload vs 저비트 W reload 비용 역전(energy 레버). 저비트 타일을 큰 fetch 앞에 두면 stall↑(cycle 레버).
- **calibration (2-tier, 연구 C6)**: 우리는 **TB 실측 cycle** 자산이 있음 — 분석 stall 모델을 `tb/gemm_sram_top_mixed_tb.v` 의 `[CYC] STAGE` 출력에 맞춰 보정/검증(문헌 분석모델은 RTL 대비 MAESTRO 3.9%, ZigZag 5–7.5%; 우리는 자체 RTL 로 더 직접 검증 가능).
- **oracle 회귀**: exhaustive 결과를 Phase 2 solver 검증 기준으로 보존.

---

## 11. 명시적 가정 (리뷰 확인 요망)

- **A1 (패딩)**: 32 배수 아니면 ceil-pad. pad 비용 포함(보수적).
- **A2 (균등 blocking)**: blocking factor = tile 수 약수만. 불균등/uneven 은 Phase 2.
- **A3 (energy 계수)**: §7 기본표 = 출발점. 실제 상대값 합의 후 고정.
- **A4 (DRAM BW)**: **sequence stall 의 핵심 입력**(§8). 값 합의 필요. "compute=ideal 보장" 의도면 BW 충분히 크게(stall→0) 두거나 제약 ON 을 stall=0 으로.
- **A5 (on-chip 단일 레벨)**: A/W/C 단일 공유 버퍼(용량 = bank×개수). 분리 버퍼/멀티레벨 Phase 2.
- **A6 (byte 모델)**: DRAM 볼륨 = precision 비트수(W=avg, A=act, C=FP32). 저비트 W 가 버퍼·traffic 둘 다 적게 먹음(검증된 1급 효과, Stripes/Loom).
- **A7 (fill/drain·utilization, 연구 C4·신규)**: v1 은 fill 을 상수 근사. 작은/edge 타일의 pipeline 미충전 비효율(BISMO)은 Phase 2 정밀화 — 단조 결과를 왜곡할 수 있음을 명시.

---

## 12. 파일 구조 — 두 개의 `.py` (표준본 + 주석 상세본)

동일 알고리즘·**동일 결과**를 내는 두 파일. 둘 다 self-contained(상호 import 없이 위→아래로 읽힘), stdlib only.

```
MXP_scheduler/
    mxp_scheduler.py            # [표준본] 깔끔한 구현. 내부 섹션(주석 구분):
                                #  [config]   HWConfig, Workload, 내장 기본 energy coeff dict
                                #  [mapping]  순열×blocking(약수) 후보 생성 + 중복 reuse pruning
                                #  [evaluate] 순수 카운터 §6 (capacity/DRAM/on-chip/MAC/RMW/compute_work)
                                #  [timing]   §8 sequence-aware stall (시퀀스 따라 stall_i 누적)
                                #  [energy]   §7 events·coeff
                                #  [optimize] §9 exhaustive + LPT-greedy baseline
                                #  [report]   breakdown 표 + Pareto (텍스트; matplotlib 있으면 그림)
                                #  [selftest] §10 손 계산 케이스 / [cli] argparse main
    mxp_scheduler_annotated.py  # [주석 상세본] 같은 로직·같은 결과. 줄마다 한국어 주석 + 수식 유도
                                #  (reload_factor, sequence stall 등) + `--explain` 모드: 한 입력의 단계별
                                #  중간값(footprint/reload/DRAM 볼륨/ c_i·f_i·stall_i 시퀀스/energy 항목)을
                                #  한 줄씩 출력 → 사용자가 따라가며 이해
    (선택) configs/coeffs.json  # energy 계수 override (기본은 내장 dict, 파일 불필요)
```
- **동일 결과 보장**: 두 파일 모두 `--selftest` 에 *같은* golden 케이스(같은 기대 숫자) 내장 → 둘 다 통과하면 일치. 추가로 `mxp_scheduler.py --crosscheck` 가 두 파일을 같은 입력 sweep 으로 돌려 결과 **byte-동일을 assert**(drift 방지).
- **의존성: stdlib only** (math, itertools, argparse, json). matplotlib 은 guarded import(없으면 텍스트만).
- energy 계수는 내장 dict(파일 불필요), `--coeffs file.json` 로 override.
- **fallback**: Phase 2 CP-SAT 도입 등으로 과해지면 core 모듈 분리(두 표면은 유지) — 단 v1 규모는 파일당 충분.

CLI 예:
```
python mxp_scheduler.py --M 128 --K 128 --N 128 \
    --bank-size 1024 --banks 32 --bits-file wbits.json --act 8 --dram-bw 64
python mxp_scheduler.py --selftest
python mxp_scheduler.py --crosscheck                    # 표준본 ↔ 주석본 결과 동일 확인
python mxp_scheduler_annotated.py --explain --M 64 --K 64 --N 64 ...   # 단계별 따라가기
```

---

## 13. 빌드 순서

1. **evaluator (카운트만)** — §6. energy/stall/optimizer 없이 이벤트·compute_work. + `--selftest` 손 검증.
2. **손 검증** — §10 작은 케이스(특히 reload_factor, spill) 통과.
3. **sequence-aware stall** — §8. 시퀀스 시뮬레이션 + 손 검증(저비트→stall). roofline 아님 주의.
4. **energy + optimizer** — §7,§9. exhaustive min-energy + LPT baseline.
5. **제약 ON/OFF tradeoff + 리포트** — §8 Pareto. precision/spill/순서 효과 리포트.
6. (있으면) **TB calibration** — §10, `[CYC] STAGE` 대조.

※ 각 단계는 **표준본(`mxp_scheduler.py`)에 먼저 구현·검증** → 완료 후 **주석 상세본으로 정확히 mirror**(+`--explain`) → `--crosscheck`(두 파일 같은 입력 sweep 결과 byte-동일 assert)로 drift 방지. 두 파일이 1~6 전체에서 동일 결과를 유지해야 함.

---

## 14. 참조
- prior-art 보고서 `docs/research/2026-06-04-mapping-scheduler-prior-art.md` (C1~C9 근거, 출처 URL).
- Timeloop/Accelergy, CoSA, ZigZag, MAESTRO, Interstellar, Mind Mappings, dMazeRunner, Sparseloop, SALSA, Demystifying-MSE; bit-serial: Stripes/Loom, BISMO, Bit Fusion; scheduling: LPT/job-shop, Tile-Wise Sparsity/HYTE, Adyna.
- 본 repo: `2026-05-14-integration-design.md` §9, CLAUDE.md "MXP control surface" (SA 차원 매핑 row=K/col=N/cycle=M).

---

## 변경 이력
- 2026-06-04 rev: prior-art 연구(보고서) 반영 — C1(§5.5 순서/blocking 정밀화), C2(§4.4 "32×32"=spatial, §9 공간 스케일링), C3(§6.4 compute work=ideal 하한 / §8 sequence-aware stall, roofline 폐기), C5(§9.2 LPT baseline), C6(§10 TB calibration), C7(§1 두-레버 재프레이밍), §6.3 on-chip 변수화, A7 fill/drain 신규.
- 2026-06-04 rev2: §12 파일 구조 = **두 개의 `.py`** (표준본 `mxp_scheduler.py` + 주석 상세본 `mxp_scheduler_annotated.py`, 동일 결과). 주석본 `--explain` 단계별 출력, `--crosscheck` 로 두 파일 결과 byte-동일 보장. §13 에 mirror+crosscheck 절차 추가.
