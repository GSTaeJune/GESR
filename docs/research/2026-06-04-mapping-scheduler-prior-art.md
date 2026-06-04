# Prior-art research — mapping/scheduling cost models vs. our variable-per-tile-cost ordering project

작성일: 2026-06-04
출처: deep-research 워크플로 (103 agents, ~20 sources). 합성 단계 실패 → journal(104 claims, 67 adversarial verdicts: 58 upheld / 9 refuted, confidence 66 high)에서 복구·재합성.
대상 spec: `docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md`

---

## 0. 한 줄 결론

**"기존 툴(Timeloop 등)이 타일별 가변 비트→가변 cycle 을 못 다룬다 → 직접 만든다"는 논거는 강하게 검증됨.** 단, *왜 ordering 이 중요한가*의 메커니즘을 정확히 잡아야 한다 — 진짜 새로운 레버는 **두 개**다:

1. **energy 레버** — 어느 피연산자를 reload 할지(= blocking partition + stationary 피연산자 선택, 타일별 비트수로 가중). 이건 traffic/energy 의 1급 결정자다.
2. **cycle 레버** — variable per-tile 비용이 만드는 **latency-hiding stall ordering**. 타일별 compute time 이 다르면 double-buffer 가 다음 fetch 를 숨길 수 있는 예산이 타일마다 다르고, 따라서 **타일 실행 순서가 실제 cycle(=compute+stall)을 결정**한다. precision 으로 구동되는 이 ordering 은 prior art 에 미개척이다(가장 가까운 선례는 sparsity load-imbalance 뿐).

반대로 **순수 outer-loop permutation 자체는 저레버리지이고 대부분 중복**이다(고정 blocking partition 하에서 outer 순열은 DRAM 총량을 안 바꿈 — reload factor 가 곱이라 commutative). 따라서 프로젝트를 "loop-order optimizer" 가 아니라 **"precision-weighted mapping + variable-cost tile scheduling"** 으로 프레이밍해야 prior art 대비 정확히 새롭고 방어 가능하다. **"순서는 energy 와 무관"** 같은 일반 명제는 절대 쓰지 않는다(반증된 9개 주장이 전부 그 과장이었다).

---

## 1. Q1 — 기존 cost model 이 가변 per-tile 실행시간(bit-serial/mixed-precision)을 지원하나? → **아니오 (강하게 검증)**

| 툴 | precision 취급 | cycle 모델 | 출처 |
|---|---|---|---|
| **Timeloop** (ISPASS'19) | bit-width 는 *energy-per-MAC* 만 (DB, 곱셈 quadratic/덧셈 linear). cycle 은 `MAC수 / 곱셈기수` 균일. sparsity 의 *time* 절감은 명시적 future work | 균일 per-MAC | people.csail.mit.edu/anurag_m/papers/2019.timeloop.ispass.pdf |
| **CoSA** (ISCA'21) | 고정 8-bit dense, psum 24-bit. precision 은 architectural 상수 | 균일, MIP | arxiv 2105.01898 |
| **ZigZag** (TC'21) | precision = element→bit 변환용 고정 파라미터(메모리 cost only). 데이터 의존 per-tile cycle 없음 | 균일 | arxiv 2007.11360 |
| **Interstellar** (ASPLOS'20) | 고정 16-bit | 균일 | arxiv 1809.04070 |
| **MAESTRO** (MICRO'19) | ALU precision 고정 HW 입력. data-dependent 는 *균일분포 sparsity* 만 | 균일 | arxiv 1805.02566 |
| **Mind Mappings** (ASPLOS'21) | cost = f(accelerator, problem shape), **입력데이터 무관**. input-dependent 는 future work | 균일 | kartikhegde Mind_Mappings |
| **Sparseloop** (MICRO'22) | **유일하게 data-dependent cycle 모델** — 단 *sparsity* 를 **통계적 density 모델**로(exact per-tile 아님; per-fiber 분석은 너무 느림). zero-skip 기반, precision 아님 | 통계적 | arxiv 2205.05826 |

→ **off-the-shelf 로는 우리 케이스(타일별 정밀도→cycle)를 못 다룸.** 커스텀 빌드 정당. 게다가 우리 비용은 **exact(평균비트로 결정론적)** 라 Sparseloop 의 통계적 근사(2000× 빠르지만 0.1%–8% 오차)보다 오히려 정확함 — 이건 gap 이 아니라 우리 *강점*. (단 "exact compute work" 가 "exact actual cycle" 은 아님 — stall 은 순서 의존; §3-3, §6, §8 참조.)

---

## 2. Q2 — "DRAM traffic 은 blocking 에만 의존, 순서는 무관" → **고정 partition 하 outer 순열에 한해 참, 그러나 일반화는 반증됨 ⚠**

검증에서 **반증된 9개 주장이 전부 이 Q2 의 over-claim**("순서는 energy 와 무관")이었음. 정밀한 진실은 **두 층위로 분리**해야 한다.

### 2a. 순수 outer-permutation 은 저레버리지·대부분 중복 (참)

고정 inner/outer partition(blocking) 하에서, **outer 루프 순열은 총 DRAM 볼륨을 안 바꾼다** — reload factor 가 outer trip-count 의 곱이라 commutative. 추가로 다수 순열이 traffic 상 redundant:

- **Timeloop** pruning rule: permutation 은 **innermost tiling level 에서만** cost 무관(*innermost only*, 일반 명제 아님). [timeloop.pdf §V-E]
- **dMazeRunner**: 7-루프의 `7!=5040` 순서 중 **고유 reuse factor 는 ~15개뿐**, 나머지 중복 → brute-force 가 가능. [dl.acm 3358198]
- **Interstellar**: 고정 hierarchy 에서 **blocking 이 energy 지배**(blocking scheme 의 30% 만 min 의 1.25× 이내). [1809.04070 §6.1]
- **Timeloop Fig.1**: **6,582 mapping 이 같은 min DRAM 인데 energy 는 11× 차이** → DRAM 최소는 tiling 이 정하고, *나머지 energy* 는 on-chip access + 순서/stationary 가 좌우.

→ **순수 outer-permutation 만 보면 저레버리지 + 다수 중복.** 그래서 dMazeRunner 식 "고유 reuse factor 순서만 enumerate" pruning 채택 권장(우리 6 순열 중 중복 제거 가능).

### 2b. 그러나 blocking partition + stationary 선택은 고레버리지 (= "loop order" 의 느슨한 의미) — "순서는 energy 무관"은 거짓

어느 루프를 inner/outer 로 둘지(= blocking) **그리고** 어느 피연산자를 innermost/stationary 로 둘지는 traffic/energy 를 강하게 좌우한다. 문헌이 "loop order" 를 partition+stationary 까지 묶어 쓰는 느슨한 용법에서는 순서가 크게 중요:

- **SALSA** (arxiv 2304.12931): loop-order **단독**으로 energy 최대 **50%**, latency "orders of magnitude".
- **Demystifying MSE** (IISWC'22, 2210.03731): best-vs-worst order = **14.4× EDP**.
- **CoSA**: outer-vs-inner 루프 교체(PCK vs PKC)로 단일 layer **1.7× speedup**. "traffic is NOT independent of loop order."
- **ZigZag / MAESTRO**: tiling 과 reordering 이 **공동으로** access count 결정. 특히 **innermost(=stationary) 피연산자 선택**이 어떤 reuse 를 살릴지 정함 — 순서는 reuse 의 1급 결정자(단순 stall 아님).
- Interstellar 인용으로 "blocking ≫ order" 를 *일반 명제로* 주장한 claim 들은 **반증**됨: Interstellar 의 "loop blocking" 용어가 **reorder 를 포함**하기 때문(용어 오독).

### 종합

PURE outer-permutation = **저레버리지 + 다수 중복**. BLOCKING partition + STATIONARY-operand 선택 = **traffic/energy 고레버리지**. 우리 spec §5.5 의 "총 DRAM traffic 은 blocking partition 에만 의존, outer 순열에는 무관" 은 **고정 partition 하 outer 순열에 한해서만 참** — 그 문장 옆에 "단, partition+stationary 선택 자체는 traffic/energy 의 1급 레버" 를 반드시 병기해야 하고, "순서는 energy 무관" 같은 일반 명제로 확장하면 안 됨. (정정: spec-change C1.)

---

## 3. Q3 — 우리가 간과 중인 것

1. **Array-fill / pipeline 이용률** (BISMO): execute-stage 효율은 **행렬 폭 / array pipeline depth Dk** 비율에 달림 — 폭이 Dk 를 못 채우면 파이프라인 미충전. (BISMO 측정: 고정 8192-col 행렬에서도 깊은 파이프라인(wide-Dk) 인스턴스는 64%, 얕은(narrow-Dk) 인스턴스는 89% — 즉 Dk 가 클수록 fill 비효율 ↑.) 우리 "compute=결정론적 32×b" 모델은 **edge 타일/작은 타일의 fill-drain 비효율을 간과**. → fill/drain·utilization 항 추가.
2. **Uneven/per-operand asymmetric blocking** (ZigZag): 공유 메모리 레벨에서 피연산자별로 blocking 경계를 다르게 → **최대 33% 에너지 개선**. 우리 nested-blocked 는 차원당 동일 blocking 강제 → 이 자유도 누락(Phase 2 후보).
3. **double-buffer stall = max(comm, compute), 그러나 *총량 max 가 아니라 타일 순차 누적* 이어야 함** (MAESTRO + BISMO 2.2× 더블버퍼): MAESTRO 는 outstanding delay = max(통신, 연산) 로 명시. 단 우리 케이스에서는 이 max 를 **타일 *시퀀스* 를 따라 누적**해야 한다(roofline 식 전역 totals 의 max 는 ordering 효과를 통째로 놓침; §6, §8). → §8 stall 모델을 sequence-aware 로 명확화.
4. **precision 은 memory traffic 도 줄임** (Stripes/Loom): weight/act 비트가 precision 비례로 DRAM/on-chip 모두 감소((16−Pw)/16, (16−Pa)/16). 우리 A6 에 있으나 **검증된 1급 효과**로 강조.
5. **on-chip access 가 잔여 energy 를 좌우** (Timeloop 11× @ same DRAM): on-chip access 를 "상수" 로 본 우리 §6.3 가정은 부정확 — **mapping 마다 변동**. on-chip access 도 변수로 카운트.
6. **co-design 교훈** (Sparseloop): 직교한 축(dataflow vs sparsity)은 따로 최적화하면 안 됨 → 우리가 precision-opt 를 Phase 2 로 뺀 게 *언젠가* ordering 과 공동최적화 필요할 수 있다는 경고.

---

## 4. Q4 — 불필요/over-engineering (안 해도 되는 것)

1. **MILP/CP-SAT 가 v1 의 (순열 × blocking) 공간에는 불필요 (검증)**: Timeloop 도 small space 는 exhaustive linear search. 우리 단일-GEMM 의 (순열 × blocking) 공간은 수천 규모(아래 표)로, MILP/GA/gradient 를 부르는 full-DNN-layer mapspace(`~10²⁴` typical edge accel., Mind Mappings 의 `~10²⁵`; ZigZag 의 exhaustive 공간도 "수백만(millions)" 규모) 대비 극소. → **exhaustive v1 정당**.
   - **단 중요한 단서(정정 Correction 2):** 이 "불필요" 는 *오직* (순열 × blocking) 공간에 한함. 일단 **per-tile 스케줄링**(T = M_T·K_T·N_T 개 타일의 실행 순서)을 열면 공간이 factorial(T!)로 폭발하고, 큰 행렬에서 CP-SAT/job-shop solver 가 **정당해짐**(§5, Phase 2). 즉 "MILP 불필요" 는 무조건이 아니라 *공간 한정* 명제다.
2. **"32×32" 는 search axis 가 아니다 — multi-level tiling / spatial mapping 탐색 불필요**: 핵심 — **"32×32" 는 행렬 크기가 아니라 *물리적 systolic array = spatial mapping* 이고, 하드웨어 고정**이다. Timeloop/CoSA/ZigZag 가 큰 이유 중 하나가 spatial mapping(어느 루프를 공간 전개할지, PE array 차원)까지 탐색하기 때문 — 우리는 그게 HW-고정이라 그 축이 통째로 빠진다. (행렬 크기 (M,K,N) 는 *변수*; 아래 표 참조.) → spatial/PE-sizing/multi-level 탐색 skip.
3. **Accelergy plug-in energy table 불필요**: MAESTRO/ZigZag 모두 energy back-end 는 decoupled/pluggable. **정규화 상대계수로 충분**(우리 A3 접근 = 표준, ZigZag 도 동일). full Accelergy 안 만들어도 됨.
4. **통계적 density 모델(Sparseloop) 불필요**: 우리 비용은 exact. 통계 근사는 오히려 ordering 신호를 뭉갬.

### 4a. 검색 공간 스케일링 — 행렬 크기와 무관하게 exhaustive 가 버티는 이유 (Correction 2)

(순열 × blocking) 공간은 행렬 크기에 따라 **M_T=M/32, K_T=K/32, N_T=N/32 의 약수 개수**(d)로만 커진다 → outer 순열(≤6) × 약수 조합:

| 행렬 (M=K=N) | M_T(=K_T=N_T) | 약수 d | (순열×blocking) 크기 ≈ |
|---|---|---|---|
| 128³ | 4 | 3 | 6 · 3³ = **162** |
| 512³ | 16 | 5 | 6 · 5³ = **750** |
| 1024³ | 32 | 6 | 6 · 6³ = **1296** |
| 4096³ | 128 | 8 | 6 · 8³ = **3072** |

→ 거대 행렬에서도 **수천 규모** → exhaustive 는 **행렬 크기와 무관하게** 안전.

### 4b. 진짜 폭발하는 곳 = per-tile 스케줄링 (Correction 2)

반면 **T = M_T·K_T·N_T 개의 개별 타일을 순서 매기는** 문제는 factorial 로 폭발:

| 행렬 | 타일 수 T = M_T·K_T·N_T | 타일 순서 공간 |
|---|---|---|
| 128³ | 4³ = 64 | 64! (다루기 어려움) |
| 1024³ | 32³ = 32768 | 사실상 불가능 |

→ 이것이 heuristic(LPT)/CP-SAT(job-shop)의 **진짜 trigger**이고, **행렬 크기와 함께 커진다**(§5, Phase 2). 결론: (순열 × blocking) 에는 MILP 불필요, 그러나 per-tile 스케줄링을 열면 큰 행렬에서 Phase-2 solver 가 정당.

### 정확도 기대치 (calibration)

분석 모델의 실측 오차 — **MAESTRO 3.9%**, **ZigZag 5–7.5%** (vs RTL/silicon). 우리는 **자체 RTL/TB 의 실측 cycle** 이 있으니 거기에 calibrate 가능(문헌 대부분이 부러워할 자산). **Adyna/ZigZag 식 2-tier**(빠른 상대 cost model 로 탐색 + cycle-accurate 로 검증) 채택 권장.

---

## 5. Q5 — 가변-latency 스케줄링에 더 맞는 정식화

- **List scheduling / LPT (Graham)**: 이질적 *known* duration 작업의 makespan 최소화 = 우리 "타일을 비용순으로 정렬" 그대로. LPT = 내림차순 정렬 후 최소부하 머신에 배정, **O(n log n)**, 4/3 − 1/(3m) 근사(m=2 면 7/6). plain list scheduling 과의 유일한 차이가 *내림차순 presort* 라는 점이 곧 "비용순 ordering 이 schedule 을 개선하는 레버" 라는 증거. → **싸고 검증된 greedy baseline 이 존재**(exhaustive 와 대조용으로 구현 권장).
- **Job-shop / makespan + OR-Tools CP-SAT**: interval var + AddNoOverlap(자원 배타) + precedence. task duration 이 명시적으로 heterogeneous → variable per-tile cycle 에 자연스러운 정식화. 우리가 자원(연산/메모리) 공유 + precedence 를 넣을 때의 표준(§4b 의 per-tile 스케줄링 폭발 시 Phase 2 solver 후보).
- **Tile-Wise Sparsity / SparTen / HYTE**: 비균일 per-tile work → load imbalance 가 1급 효율 문제. 완화책 = **같은 비용 타일을 batch + concurrency**(MILP 아님). "타일을 data-dependent 비용으로 reorder" 의 직접 선례 — precision 이 아니라 sparsity 라는 점만 다름.
- **Adyna** (HPCA'25): data-dependent 워크로드에 **offline 통계 스케줄링이 online 을 이김**(online 은 critical path, sub-0.39ms 여야 역전). + Timeloop 식 분석 모델로 탐색 후 cycle-accurate(RTL-calibrated)로 검증하는 **2-tier**. → 우리 offline/exhaustive + 2-tier 계획 지지.

**주의 (정직한 한계)**: LPT/job-shop 은 *독립* 작업·*makespan* 용이다. 우리 타일은 **reuse 결합**(공유 피연산자) 이 있고 목적은 **energy(traffic) + cycle/makespan** 복합이다. 그래서 job-shop/LPT 는 좋은 *프레이밍·baseline* 이지 drop-in 이 아니다 — GEMM mapper 계열이 얹는 reuse/traffic 결합 + variable per-tile 비용을 latency-hiding 으로 보는 것이 핵심.

---

## 6. 사용자 핵심 thesis 평결

> "타일별 평균비트 → execution time 다름 → ordering 중요 → 그래서 Timeloop 안 쓰고 직접 탐색"

- **"Timeloop 류가 이걸(타일별 가변 정밀도→가변 cycle) 못 한다" → 참 (강함).** 빌드 정당. (Sparseloop 만 data-dependent cycle 을 다루나 통계적 sparsity 뿐, precision 아님.)
- **"가변 비트 → compute *work* 가변" → 참, 그리고 그 *총합*은 order-무관** (Σ 32·b(tile) = **ideal, stall-free 하한** = 사용자가 말하는 "ideal cycle"). **그러나 *실제 실행 cycle* = compute + stall 이고 stall 은 order-의존이다** — 따라서 "가변 비트 → cycle 가변 → 그러나 order-무관" 은 **틀린 진술이다**(정정 Correction 3).
  - 메커니즘: double-buffering 이 tile_{i+1} 의 피연산자 fetch 를 tile_i 의 compute 뒤에 숨긴다. `stall_i ≈ max(0, f_{i+1} − c_i)`, 여기서 `c_i = 32·b_i`(타일 i compute time), `f_{i+1}` = 다음 타일의 non-resident 피연산자 fetch time. **저비트 타일은 c_i 가 작아 latency-hiding 예산이 작고 → 다음 fetch 를 못 숨겨 stall**. 즉 variable per-tile 비트 = variable latency-hiding 예산 → **타일 순서가 stall = 실제 cycle 을 결정**. (사용자 표현: "cycle 이 너무 짧으면 stall 된다.")
- **"그래서 *ordering* 이 중요" → 참, 그리고 진짜로 새로운 두 레버는**:
  1. **energy 레버 — 어느 피연산자를 reload 할지 = blocking partition + stationary 선택**, 타일 비트수로 가중 — 우리 spec §6 에 이미 씨앗 있음.
  2. **cycle 레버 — precision-driven latency-hiding stall ordering** = job-shop/LPT(이질적 known-duration list scheduling)가 들어맞는 곳 — **여기가 진짜 새로움**, 선례는 sparsity-load-imbalance 문헌(Tile-Wise Sparsity/HYTE)뿐 precision 은 미개척.
- **순수 outer-loop permutation 자체는 저레버리지·대부분 중복**(§2a) — 과대평가 금지.

→ **프로젝트를 "loop-order optimizer" 가 아니라 "precision-weighted mapping + variable-cost(latency-hiding) tile scheduling" 으로 재프레이밍**하면 thesis 가 prior art 대비 정확히 새롭고 방어 가능.

---

## 7. spec 에 반영 제안 (검토 대상)

| # | 변경 | 근거 |
|---|---|---|
| C1 | **§5.5 정정** — "총 DRAM traffic 은 blocking partition 에만, outer 순열에는 무관" 은 *고정 partition 하 outer 순열에 한해* 참으로 한정. "순서는 energy 무관" 일반 명제는 삭제. "단 partition + stationary 선택은 traffic/energy 의 1급 레버" 병기 + dMazeRunner 식 unique-reuse pruning | Q2 (2a/2b) |
| C2 | **§8 stall 모델을 sequence-aware 로** — roofline `actual = max(Σcompute, Σmem)` 은 양변 다 order-무관 totals 라 **ordering 효과를 통째로 놓침**. 대신 **타일 순서를 walk 하며 `stall_i = max(0, f_{i+1} − c_i)` 누적**(작은 pipeline 시뮬). roofline 은 blocking 효과만, sequence 시뮬이 ordering 효과를 노출. (단일 타일 더블버퍼 형식은 MAESTRO max(comm,compute), BISMO 2.2×) | Q3-3, Correction 3 |
| C3 | **§6.3 on-chip access 를 상수→변수**로 카운트 | Q3-5, Timeloop 11× |
| C4 | **fill/drain·utilization 항 추가** (작은/edge 타일) | Q3-1, BISMO |
| C5 | **LPT/cost-순 greedy baseline 추가** (exhaustive 대조; per-tile 스케줄링용) | Q5 |
| C6 | **TB 실측 cycle 로 calibration** (2-tier 검증) | Q4, Adyna/MAESTRO/ZigZag |
| C7 | **재프레이밍**: "loop-order optimizer" → "precision-weighted mapping + variable-cost tile scheduling". 두 레버 = (a) reload-operand/stationary 선택(energy) + (b) latency-hiding stall ordering(cycle) | §6 평결 |
| C8 | (선택) uneven/per-operand blocking 을 Phase 2 명시 | Q3-2, ZigZag 33% |
| C9 | **per-tile 스케줄링은 별도 Phase** — (순열 × blocking) 은 exhaustive(수천, 행렬 크기 무관)로 충분하나, T 개 타일 순서(factorial)는 큰 행렬에서 LPT/CP-SAT 정당. spec §9 "MILP 불필요" 를 *(순열 × blocking) 한정* 으로 명시 | Q4 (4a/4b), Correction 2 |

---

## 부록 — 주요 출처 URL
- Timeloop ISPASS'19: people.csail.mit.edu/anurag_m/papers/2019.timeloop.ispass.pdf
- Sparseloop MICRO'22: arxiv.org/pdf/2205.05826
- CoSA ISCA'21: arxiv.org/abs/2105.01898
- ZigZag TC'21: arxiv.org/abs/2007.11360
- Interstellar ASPLOS'20: ar5iv.labs.arxiv.org/html/1809.04070
- MAESTRO MICRO'19: arxiv.org/pdf/1805.02566
- Mind Mappings ASPLOS'21: kartikhegde.net/media/Mind_Mappings_ASPLOS2021_CR.pdf
- dMazeRunner TECS'19: dl.acm.org/doi/fullHtml/10.1145/3358198
- Demystifying MSE IISWC'22: arxiv.org/abs/2210.03731
- SALSA: arxiv.org/abs/2304.12931
- Stripes/Loom (bit-serial precision-proportional): arxiv.org/pdf/1706.07853
- BISMO (bit-serial, validated cost model): arxiv.org/pdf/1806.08862
- Bit Fusion: arxiv.org/pdf/1712.01507
- Tile-Wise Sparsity: cs.sjtu.edu.cn/~leng-jw/resources/Files/guo20sc-twsparsity.pdf
- HYTE ISCA'25 / Adyna HPCA'25: people.iiis.tsinghua.edu.cn/~gaomy/pubs/
- LPT/list scheduling: en.wikipedia.org/wiki/Longest-processing-time-first_scheduling
- Job-shop / OR-Tools CP-SAT: developers.google.com/optimization/scheduling/job_shop

---

## 변경 이력 (이번 재작성)

이전 판의 내부 모순을 도메인 전문가 리뷰에 맞춰 정정했다. 적용한 3개 교정:

1. **Correction 1 — "loop order vs blocking vs energy" 입장 일관화.** 이전 판은 §0/§2 제목/§6 평결이 서로 어긋났다("순서는 energy 무관" 과 "순서가 중요" 가 공존). 단일 입장으로 통일: *고정 partition 하 **outer 순열**은 DRAM 총량 불변(commutative) + 다수 중복*(저레버리지) 이지만, **blocking partition + stationary 피연산자 선택**(문헌의 느슨한 "loop order")은 traffic/energy 의 1급 레버(SALSA 50%, MSE 14.4× EDP, CoSA 1.7×). "순서는 energy 무관" 일반 명제는 전 섹션에서 삭제. (§0, §2 를 2a/2b 로 분리, §6, §7 C1)
2. **Correction 2 — "32×32" 의미 + 검색 공간 스케일링 정정.** "32×32" 는 행렬 크기가 아니라 **물리적 systolic array = spatial mapping = 하드웨어 고정**(우리가 Timeloop/CoSA/ZigZag 보다 공간이 작은 진짜 이유 — 그들은 spatial mapping 도 탐색). (순열 × blocking) 공간은 M_T/K_T/N_T 의 약수 개수로만 커져 거대 행렬에서도 수천 규모(스케일링 표 추가) → exhaustive 는 행렬 크기 무관하게 안전. **진짜 폭발은 per-tile 스케줄링(T!)** 이고 이것이 Phase-2 solver 의 진짜 trigger. "MILP 불필요" 를 *(순열 × blocking) 한정* 으로 명시. (§4 Q4-1/Q4-2 정정, §4a/§4b 표 추가, §7 C9 신설)
3. **Correction 3 — compute cycle 은 *실제 cycle* 에 대해 order-무관 아님.** 이전 판 §6 의 "가변 비트 → compute cycle 가변 → 참이나 order-무관" 을 삭제. compute **work** 총합(Σ32·b)은 order-무관(= ideal, stall-free 하한)이나, **실제 cycle = compute + stall 이고 stall_i = max(0, f_{i+1} − c_i) 는 order-의존**(저비트 타일 = 작은 latency-hiding 예산). 이것이 프로젝트의 두 번째(cycle) 레버이자 진짜 새로움. 모델링 함의: roofline `max(Σcompute, Σmem)` 은 order-blind 라 이 효과를 놓침 → **sequence-aware pipeline 시뮬**필요. (§1 단서, §3-3, §6 평결, §8 관련 §7 C2)

### 독립 리뷰 정정 (2026-06-04, end-to-end 일관성 감사)

위 3개 교정이 전 섹션에 일관 적용됐는지 독립 검토. 두 곳에서 ground-truth claims 파일과의 사실 불일치를 발견·수정 (입장 자체는 일관됐음):

- **R1 — Q3-1 BISMO array-fill 수치 오귀속.** 이전 판은 "좁은 행렬은 … 효율 64% vs 89%" 로 64% 를 좁은 행렬에 귀속했으나, claims 파일은 *고정* 8192-col 행렬에서 **wide-Dk=64% / narrow-Dk=89%** (효율은 pipeline depth Dk 에 의존, Dk↑ → fill 비효율↑) 로 명시. 수치를 Dk 기준으로 재서술 (정성적 결론 "narrow matrix 가 파이프라인 미충전" 은 유지, 근거 grounding 만 정정).
- **R2 — Q4-1 검색공간 규모 오귀속.** 이전 판은 `10²¹~10²⁵` 를 "ZigZag/Mind Mappings" 에 묶었으나 claims 상 ZigZag 의 exhaustive 공간은 "millions" 이고 `10²⁴/10²¹` 은 full-DNN-layer/edge-accel 일반치, `10²⁵` 만 Mind Mappings. full-DNN mapspace(`~10²⁴`) + Mind Mappings(`~10²⁵`) + ZigZag("millions") 로 분리 귀속 (우리 수천 ≪ 그들 대공간 대비는 유지).

스케일링 표(§4a) 산술 재검증: 128³→6·3³=162, 512³→6·5³=750, 1024³→6·6³=1296, 4096³→6·8³=3072 — 전부 정확. per-tile 표(§4b): 128³→4³=64, 1024³→32³=32768 — 정확. verdict 카운트: 58+9=67 — 정확. Correction 1/2/3 은 §0·§1·§2·§3·§4·§6·§7 전체에서 모순 없이 일관.
