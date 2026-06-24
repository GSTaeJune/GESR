# MXP_scheduler — exact joint (order + eviction) scheduler with stall=0 + precision-adaptive residency (design)

- 날짜: 2026-06-23 (rev5 — 3차 병렬검토 반영: Belady 비최적 교정, joint order+eviction 탐색)
- 대상: `MXP_scheduler/mxp_scheduler.py` (+ `mxp_scheduler_annotated.py`, `hwconfig.py`)
- 선행: [`2026-06-04-mxp-scheduler-design.md`](2026-06-04-mxp-scheduler-design.md) — §5/§6/§8/§9 supersede.
- 관련 메모리: `feedback_optimize_everything_no_silent_drop`, `feedback_scheduler_belady_not_optimal`, `project_scheduler_static_footprint_no_precision_packing`, `project_scheduler_divisor_blocking_limitation`, `project_scheduler_dram_energy_coeffs`

---

## 0. 한 줄 요약

스케줄 = **cube-op 실행 순서 + eviction 결정**(둘 다 탐색 대상). 비용을 **순서+eviction 평가기(forward pass)**로 도출하고, **A*/branch-and-bound (직접 구현, 의존성 0)**로 joint 전역최적(또는 보고된 gap)을 구한다. 모든 레버(정밀도-적응 상주·ragged·C-window·ordering·eviction)를 joint exact로 포착. stall=0은 하드 feasibility 제약. cycle 모델 파라미터화. **타협 없음 — exact 못 닫으면 gap 명시.**

## 1. rev4 대비 교정 (3차 검토 반영)

1. **Belady ≠ 최적** (변크기 타일 = weighted caching, NP-hard). 실측 ~24% 초과. → "order가 master/residency 자동 파생" **철회**. **eviction을 탐색 차원으로** 승격, A*가 (order+eviction) joint 최적.
2. **목적함수 교정**: 재로드/spill은 DRAM뿐 아니라 **on-chip refill 에너지(c_onchip)**도 발생 (`onchip_bits` refill 항). g/h가 **(c_dram + c_onchip)/bit** 로 세고, h에 **불가피 final-C-write floor** 포함. mac/rmw는 상수라 제외.
3. **상태 Markov 보강**: stall=0 feasibility가 직전 cube compute에 의존 → 상태에 **last-cube-compute** 포함(없으면 dominance가 최적 잃거나 false-feasible).
4. **oracle 교정**: 기존 closed-form은 inner-blocked에서 C-spill을 all-or-nothing으로 **과대계상(상한)**, exact 아님. → **진짜 oracle = DP-exact(작은 T, order+eviction 전수)**. 기존 모델은 warm-start incumbent + 상한 sanity.
5. **각 cube 정확히 1회** 스케줄 불변식을 평가기에 명시.

## 2. HW 전제

- 저장 = 정밀도 비례(weight bit-serial). C psum = FP32, 정밀도 무관, footprint ≥67%(act=8) 지배. activation = layer-uniform. SA는 steady-state 중간 정지 불가(→stall=0 하드). 스케줄은 오프라인 정적(eviction까지 컴파일 타임 결정 → 탐색이 그대로 실현됨).
- **atom = 한 SA pass = cube `(mt,kt,nt)`** (32M×32K×32N). CLAUDE.md SA 매핑과 정합(한 tile 처리=32W cycle, 32 M-row). cube가 SA의 원자 단위라 재배치 가능; 더 잘게는 물리적으로 불가.

## 3. 문제 정의 (최적화)

> **min 에너지  s.t. (∀step 용량) ∧ (steady_stall=0),  변수 = (cube 실행 순서, eviction 결정).**

- 에너지 = `Σ(load+spill 비트)·(c_dram + c_onchip) + 상수(mac·rmw)`.
- 사용자 직관("순서 정하면 상주량 정해짐")은 **고정 eviction 정책에선 맞지만**, 변크기라 *최적* eviction은 정책으로 안 나옴 → eviction도 변수.

## 4. 결정사항

- **D1 변수 = (order + eviction)**, A*가 joint 탐색.
- **D2 탐색 = A*/branch-and-bound (stdlib).** T=64 증명-최적 *목표*(heuristic 의존); 못 닫으면 incumbent + 하한 = **gap 보고**.
- **D3 비용 = (order+eviction) 평가기 = 단일 진실.** Belady 같은 정책 가정 없음.
- **D4 stall=0 = 하드 feasibility.** steady-state stall ↔ fill/drain 분리.
- **D5 oracle = DP-exact(작은 T)** + warm-start = 기존 closed-form incumbent.
- **D6 M1 ≥ M0**: 보고값 = `min(A* 결과, 평가기로 채점한 구조적 incumbent)` → 구조적이 exact한 곳에서 A*가 gap 보고하는 퇴행 방지.
- **D7 cycle 파라미터화**; **D8 DRAM 계수 마무리**; **D9 결정성**: open-set + merge tie-break 명시.

## 5. 아키텍처

| 컴포넌트 | 역할 | 신규/재사용 |
|---|---|---|
| **(order+eviction) 평가기** `eval_sched` | 1패스로 reload/spill/stall/에너지 (eviction 주어짐). 단일 진실. | 신규 |
| **A* 최적기** | (order+eviction) joint 탐색 → exact or gap | 신규 |
| **DP-exact oracle** (작은 T) | order+eviction 전수 최적 → 평가기·A* 검증 | 신규 |
| **기존 closed-form** | warm-start incumbent + 상한 sanity | 재사용 |

## 6. 평가기 (eval_sched)

입력 = (cube 순서, 각 step의 eviction 집합). 1패스:
- 각 cube 실행 시 그 3타일 상주 필요. 비상주면 **load**(reload). eviction은 *입력*(A*가 정함) — 평가기는 정책 가정 안 함.
- **C 누적 카운터** per (mt,nt): evict 시 카운터<KT면 spill **write**(+나중 reload **read**), ==KT면 final **write**. (카운터로 bits 정확 — 어떤 kt인지 불필요.)
- **stall**: `steady_stall_i = max(0, fetch_i/eff_bw − compute_{i-1})`, fetch_i=비상주 타일 비트(load). fill=첫 cube load. drain=마지막 C write. **float 산술 → freq_ratio 비정수도 exact.**
- **에너지** = `(load+spill 비트)·(c_dram+c_onchip)` + 상수. (검토2: on-chip refill 반드시 포함.)
- **불변식**: 각 cube 정확히 1회(중복/누락 거부).
- **feasible** = (∀step 용량) ∧ (steady_stall=0).

## 7. A* 최적기

- **상태** = (스케줄된 cube 집합, resident set, C 누적 카운터, **last-cube-compute**). g = 누적 (load+spill)·(c_dram+c_onchip). 
- **expansion** = 다음 cube 선택 + (용량 초과 시) **eviction 선택 분기**. fetch가 직전 compute에 안 가려지면(stall>0) 또는 용량 불가면 가지치기.
- **admissible h** = 잔여 영역에 필요+비상주 각 타일 ≥1 load × (c_dram+c_onchip) + 미완 각 C타일 final write × (c_dram+c_onchip). (과대평가 안 함 → 최적성.)
- **warm-start** = 기존 구조적 최적을 incumbent (평가기로 재채점한 값). **D6 floor 보장.**
- **종료** = 닫히면 증명-최적; 한도 초과면 incumbent + min open f = **gap 보고**.
- **결정성**: open-set 키 `(f, g, 순서서명)`; **merge 동률 시 사전순 작은 prefix 유지** (D9).
- **규모(정직)**: resident set이 상태라 공간 큼. T=64 증명-최적은 **heuristic 품질 의존 — 닫으면 best, 보통은 gap 보고가 현실**. plan에서 상태 표현/병합/메모이즈 + heuristic 강화가 핵심 작업. 큰 T(예 T=512)는 gap.

## 8. 검증
- **DP-exact == A*** (작은 T≤~12, order+eviction 전수 vs A*).
- **평가기 == 기존 closed-form** (구조적 + inner=1 한정; inner-blocked은 closed-form이 상한이라 부등호 sanity만).
- **골든 재설정**: `test_stall_fill_g3`, `test_stall_includes_c_drain`, `test_stall_zero_when_bw_huge`, `selftest` G3.
- mixed 저비트 더 상주 / prime KT=5 / stall=0 저비트 binding / cycle 파라미터 / 결정성 재현 / 각-cube-1회 / D6 floor(A*≥구조적).
- 빈-영역: feasible 0 → 진단(min-steady-stall + 사유), `tradeoff`/CLI 에러. 조용한 fallback 금지.

## 9. cycle 파라미터화 / DRAM 계수
- `cycles_per_bit`(1), `sa_fill_cycles`/`sa_drain_cycles`(0). DRAM input-fill·DRAM C-drain(데이터의존)·SA fill/drain(상수) 분리, 중복계상 금지. `hwconfig.resolve` 주입.
- `dram_presets.json`(출처 명기) 검증/문서/테스트 마무리.

## 10. 기존 코드 영향
| 함수 | 변경 |
|---|---|
| `_stall_of_order`/`stall_fill`/`evaluate`/`report` | steady_stall ↔ fill/drain 분리 |
| `compute_work`/`_blocks`/`HW`/`hwconfig.resolve` | cycle 파라미터화 |
| 기존 `dram_bits`/`gen_mappings`/`optimize` | 유지 → warm-start incumbent (상한 sanity) |
| 신규 `eval_sched.py` | §6 평가기 |
| 신규 `astar.py` (+ DP-exact) | §7 A* / §8 oracle |
| `selftest`/`crosscheck`/annotated | 골든 재설정 + 평가기·A*↔DP 일치 |

## 11. 납품 단계 (nested, 결정 고정 없음)
- **M0** (저위험, 기존 모델 위): stall 분리 + cycle 파라미터화 + DRAM 계수 + §6 평가기 + (평가기==closed-form, inner=1) 일치. **그 자체로 쓸모 + warm-start 기반.**
- **M1** (joint 최적기): §7 A* + §8 DP oracle + stall=0 강제 + D6 floor. (order+eviction)라 모든 레버 joint exact(또는 gap).
- M0 incumbent ⊂ M1 탐색이라 결정 고정 없음, joint 최적 안 잃음.

## 12. 위험 / 미해결 (정직)
- **A* 규모**: resident set 상태 + eviction 분기 → 공간 큼. T=64 증명-최적은 heuristic 강도에 달림 (gap fallback). state 표현/dominance(last-cube-compute 포함)/메모이즈 = plan 핵심.
- **변크기 caching NP-hard**: eviction 분기로 흡수, 작은 T는 DP로 검증. 큰 T exact 보장 없음(gap).
- **prior-art "임의 ordering low-leverage"**: payoff 작을 수 있음 → §8에서 A* vs 구조적 gain **보고**(투명). 작아도 안 버림(exact로 구하고 gap 명시). D6로 구조적 floor 보장.
- **cube-stall vs block-stall**: prefetch 발행 시점 가정 명시 필요 — 본 설계는 cube-단위 hide(직전 cube로 가림) 채택, DP가 stall arbiter.

## 13. 3차 검토 반영 매핑
| 항목 | 반영 |
|---|---|
| Belady 비최적 (변크기) | §1.1/§3 eviction을 탐색 변수로, A* joint; "order=master" 철회 |
| g/h on-chip 누락 | §1.2/§6/§7 (c_dram+c_onchip)/bit + final-C floor |
| 상태 non-Markov (stall) | §1.3/§7 last-cube-compute 상태 포함 |
| closed-form oracle 아님 (inner-blocked) | §1.4/§8 DP-exact가 oracle, closed-form은 상한 sanity |
| 각 cube 1회 | §1.5/§6 불변식 |
| §6↔§8 eviction 소유 모순 | eviction=탐색차원 확정(§3/§7), 평가기는 주어진 eviction 채점 |
| M1<M0 퇴행 | §D6/§7 floor = min(A*, 구조적 incumbent) |
| T=64 exact 과장 | §7/§12 heuristic 의존, gap 현실 명시 |
| 결정성 merge tie-break | §D9/§7 |
| determinism/cycle/granularity/empty-region | §7/§9/§2/§8 (이전 rev 유지) |
