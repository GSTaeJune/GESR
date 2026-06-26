# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## 사용자 직접 정의 규칙
1. Address the user in Korean; use English for everything else. Use established terms for recurring concepts consistently — don't invent new or alternative names for something already named. When a genuinely new term is introduced, briefly define it on first use, then reuse it as-is.

2. After completing a logical chunk of work (one or several related tasks, e.g. tasks 1–3), run `/superpowers:requesting-code-review` and `/review`, iterating on their feedback until no blocking issues remain. Reviews don't need to run per-task — group related tasks into a coherent unit before reviewing. Minor or optional suggestions don't need to block convergence.

## Next session kickoff (2026-06-26, **band-serpentine (B,d) 휴리스틱 스케줄러 빌드·검증·main 병합 — 큰 T 스케일, Qwen Q/K/V 데모, honest gap. 미push**)

**진행 상태**: `MXP_scheduler/band_sched.py`(신규, stdlib-only) + `demo_qwen.py`(신규) + `measure_gap.py --backend band` **빌드·검증·main 로컬 ff-병합 완료**(브랜치 `feat/mxp-scheduler-band` 삭제). **`origin/main` 보다 앞섬 — 미push**. RTL/M1/CP-SAT/mxp_scheduler/hwconfig **무변경**. 검증: `cd MXP_scheduler && python -m pytest -q` → **168 passed, 1 skip(CACTI)**, `band_sched.py --selftest` OK, stdlib-clean(ortools 미import).

### 무엇 (spec rev2 converged + 6-task plan, subagent-driven, 2-리뷰어+수렴감사+deviation 리뷰)
spec `docs/superpowers/specs/2026-06-26-mxp-scheduler-band-bd-design.md` → plan `docs/.../plans/2026-06-26-mxp-scheduler-band.md`. **큰 T(10³~10⁶)용 휴리스틱 + honest 하한**: 지수적 exact(CP-SAT/A*/oracle)가 못 가는 영역. 설계 = **region(=m-row)별 두 노브 (B=열린 C tile 수, d=K-depth)** + **band-serpentine order**(k-outer/n-inner → W를 band의 B열에 amortize). **eviction = lazy capacity-aware**(타일 상주 유지, cap 압박 시 largest-outside-working-set 축출, 튜플 tie-break로 결정론적; region 간 reuse 포착 — deviation 리뷰서 SOUND 판정, 용량안전 증명 + 378 oracle 교차검증서 절대 optimum 밑돌지 않음). 비용은 **eval_sched(단일진실)로 채점**(모듈 자체 비용계산 없음). **honest 하한 = first-touch A/W floor**, gap=(E−floor)/floor. 결과 dict는 astar.optimize_exact 형. **W sizing은 변경 불필요**(wbits 가 이미 per-tile fractional; uniform/3-값 가정은 W 아닌 A=act_bits 에만 — A는 uniform 유지). C-spill fallback(spec §7.2)은 v1서 **unreachable**(B=1,d=1=no-spill floor)이라 미구현(용량/stall 진단만).

### Qwen 데모 (요청 산출물)
`demo_qwen.py`: Qwen2.5(0.5b/7b/72b) Q/K/V projection dim → `Work(M=seq,K=H,N=H_out)` + **임의 per-tile avg-bit**(블록32 mixed 모사, 결정론 LCG) → `optimize_band` → order+(B,d)+honest gap 출력, 전체 cube order 파일 기록. 예: `--model qwen2.5-0.5b --proj q --seq 128` → M128·K896·N896, **T=3136, honest gap ~115%**(cap 디폴트), order `work/<label>/band_order.txt`. measure_gap band(T=512 mixed): honest gap **~7%**, warmstart 동급/우위.

### 다음 후보
- band 휴리스틱 품질 개선: Belady victim(현 largest-first), cross-region A reuse 명시 모델(현 lazy로 일부 포착), tighter honest 하한(read-only A/W min-cost-flow / Lagrangian — first-touch floor 보다 tight).
- 실 워크로드 wbits 맵 연동(현 임의 assigner). CP-SAT first-touch cut + AddHint(별도, [[project_scheduler_cpsat_landed]]).

## Next session kickoff (2026-06-25, **CP-SAT joint 스케줄러 빌드·검증 완료 — 핵심 발견: T=32/64 prover 불가(honest gap만), main 로컬 병합·미push**)

**진행 상태**: `MXP_scheduler/cpsat_sched.py` (신규, 오프라인 OR-Tools 도구) **빌드·검증·main 로컬 ff-병합 완료** (브랜치 `feat/mxp-scheduler-cpsat` 삭제). **`origin/main` 보다 앞섬 — 미push** (M0+M1+CP-SAT 로컬만). RTL/M1 4모듈(eval_sched/warmstart/oracle/astar)/mxp_scheduler/hwconfig **무변경**. 검증: `cd MXP_scheduler && python -m pytest -q` → **156 passed, 1 skip(CACTI)**, `cpsat_sched.py --selftest`/`astar.py --selftest` OK, runtime-clean(런타임이 ortools/cpsat_sched 미import). 메모리: `project_scheduler_cpsat_landed`.

### 무엇을 만들었나 (spec rev2 + 10-task plan, subagent-driven)
spec `docs/superpowers/specs/2026-06-25-mxp-scheduler-cpsat-design.md` (rev2, 3-리뷰어 라운드 + 수렴감사) → plan `docs/.../plans/2026-06-25-mxp-scheduler-cpsat.md` → 3 unit 구현(각 spec+quality 리뷰) + 최종리뷰 + pre-merge 리뷰 **READY TO MERGE**. 딥리서치(103 에이전트, COSMA/CoSA/RCPSP-max LCG/Writeback-Aware Caching/Buffets)로 정초. **step-indexed CP-SAT**: COSMA식 C/P/S/R 잔류액션 + **demand-driven load**(`load[tau,t]<=sum x[c,t]` — vacuous-stall 익스플로잇 차단, 프로토타입 검증) + M1 capacity(post-load<=cap, prefetch-peak 미과금) + C-누적 reservoir + **stall=0 하드 선형제약** + **exact-rational(Fraction LCM) 스케일링**(round() 금지). 반환 dict는 astar.optimize_exact와 동형. **cpsat == oracle == astar bit-exact**(작은/중간 T, 독립 재현 0불일치) — 세 번째 독립 exact 엔진.

### 핵심 발견 (이번 세션의 실제 산출물 — open Q A 부정 답)
**CP-SAT 는 T=32/64 를 못 닫는다.** honest gap 으로 graceful degrade(OOM 아님, plan 대로)지만:
- **bound 가 first-touch A/W floor 보다도 낮음**(root LP 릴랙세이션 수준, "모든 입력은 최소 1회 로드" 전파조차 안 함) → honest gap **86x(T=32)/422x(T=64)**.
- **warmstart hint 없으면 30s incumbent 가 구조적 베이스라인보다 2.5~7x 나쁨**.
- 독립 리뷰어가 재현 확인. → CP-SAT 는 **small/medium T exact 레퍼런스로는 신뢰**되나, **큰-T optimizer/heuristic 으로는 부적합**.
- **open Q B(1.5×WS gap-① 임계가 큰 T서 유지되나)는 여전히 미측정** — T≥32 최적을 증명하는 엔진이 없음(A* OOM, CP-SAT bound 너무 느슨).

### 다음 스텝 (메모리 `project_scheduler_cpsat_landed` 에 상세)
1. **핵심은 tight honest 하한** — 자명한 first-touch-floor 하한조차 CP-SAT bound 보다 나음. 큰-T phase 는 tight LB(FOO-L/Lagrangian/first-touch floor) 없으면 gap-② 무의미.
2. **`AddHint(warmstart incumbent)`** — incumbent ≤ structural 보장(리뷰어 검증). 미구현(이번 scope 밖).
3. 큰 T(10³~10⁶): 구조적 heuristic(**Writeback-Aware Landlord**) + tight honest 하한. CP-SAT 는 exact 레퍼런스로 유지(큰-T 엔진 아님).

## Next session kickoff (2026-06-25, **스케줄러 큰-T 최적화: 6-에이전트 평가로 솔버 확정 — CP-SAT joint. min-cost-flow 분해 기각**)

코드 무변경. **6명 독립 전문가 에이전트 교차평가**로 솔버 확정 (6/6 수렴). **다음 세션: A* 확장·RL·flow 분해·loop-order 열거 재제안 금지.** gap 하니스 `MXP_scheduler/measure_gap.py`(untracked).

**문제**: 타겟 DeiT(S=197) + Qwen2.5(0.5B~72B). 스케줄러는 `Work(M,K,N,wbits,act_bits)` 만 봄 — 모델/prefill/decode 무관, **크기만 다른 GEMM**. `T=M·K·N/32³`: 10³~10⁶+. K·N 은 늘 32 배수(ragged 는 M 축뿐). 실 워크로드 **항상 footprint≫cap → eviction 불가피**. NP-hard = **변크기 캐싱 + C write-back + sequencing, joint** (변크기 원천: W mixed-precision + C psum=W의 4~16×).

**측정(`measure_gap.py`)**: gap=(구조적−최적)/최적, A* proven 낸 **T≤12 만**. **gap 은 cap≲1.5×큐브WS(49,152b)서만 15~44%**(저비트·T 클수록 큼); cap≥1.5×WS 면 구조적이 이미 최적. **128³@디폴트(cap 1.05M>footprint 786K)는 전부 fit→trivial**(일반화 아님). A* 는 T≥32 압박서 eviction 전수전개로 **open-heap 14.8GB OOM**.

**기각: "order 고정 → min-cost-flow eviction" 분해 (내가 냈다 평가서 BROKEN, oracle 검증 반례)**:
1. (fatal) **loop order 열거 ≠ 최적** — 최적 order 는 비-loop 순열(serpentine/per-row·col). 반례 mixed **+10.5%**, uniform serpentine **+25%**.
2. (fatal) **stall=0 이 flow 를 깸** — flow 는 traffic 만 봐 per-step rate 못 봄 → 물리적 stall 스케줄을 "최적"이라 반환.
3. (serious) **C write-back ∉ flow** — FOO 는 read-only·bound(i.i.d. 가정, 결정론 trace 위반). C spill = **Writeback-Aware Caching(Beckmann&Gibbons SPAA'19) = APX-hard**, Belady (ω+1)-경쟁뿐([[feedback_scheduler_belady_not_optimal]] 재확인). per-order eviction 자체가 APX-hard.
4. (serious) **"10⁶ 스케일" 거짓** — C 누적이 long-lived interval → short-lived FOO 영역 아님.
→ **기존 M1 A* 의 joint(order+eviction 동시, stall=0 하드프룬, C spill 정확과금, A*==oracle)이 옳았음. 분해는 regression.**

**확정 방향 (6/6)**:
- **작은/중간 T exact 엔진 = CP-SAT**: interval + `AddCumulative`(cap) + `AddReservoirConstraint`(C 누적) + stall=0 하드제약 + 대칭깨기. lazy clause gen → **A* 의 state-materialize OOM 회피** → T=64급 닫거나 honest gap 으로 graceful(OOM 아님). MILP(time-indexed)=anti-pattern(약LP·big-M·대칭), tiny-T 교차검증만. DP oracle 유지.
- **큰 T(10⁶) = exact 없음**: 구조 활용 — loop-order **클래스 해석적 스코어**(Timeloop/CoSA식; mixed-prec=closed-form W 비용) + windowed eviction + **Writeback-Aware Landlord**(plain GreedyDual 아님) + Lagrangian/FOO-L **honest 하한**. C 는 K-누적밴드 pin.
- **min-cost flow 강등**: read-only **A/W 하한 생성기**로만(write-back·stall 무시→exact 아님).
- **order/eviction 분리 금지** (stall+비-loop 최적 coupling). **joint 가 맞음.** RL 아님(오프라인+stdlib·결정론 깸).
- **stdlib 충돌**: OR-Tools 외부 의존 → 오프라인 도구, stdlib 런타임 미탑재.

**다음 스텝**: **CP-SAT joint 모델**(cumulative+reservoir+stall) 구축 → A* OOM 난 T=32/64 gap 실측·oracle 교차검증 → 큰 T 는 구조적 heuristic+honest 하한. refs: Writeback-Aware Caching SPAA'19, FOO POMACS'18(bound-not-exact), CoSA ISCA'21, OR-Tools CP-SAT.

## Next session kickoff (2026-06-24, **MXP_scheduler M1 — order+eviction joint exact 최적기 완료·main 병합**, RTL 무변, 미push)

**진행 상태**: M1 구현 완료, **`main` 에 로컬 fast-forward 병합** (feature 브랜치 `feat/mxp-scheduler-m1` 삭제). RTL/closed-form 모델/twin 무변경. **`origin/main` 보다 앞섬 — 아직 push 안 함** (M0+M1 로컬만; push 는 명시 요청 시). 검증: `cd MXP_scheduler && python -m pytest -q` → **134 passed, 1 skip(CACTI)**, `--selftest`/`--crosscheck`/`astar.py --selftest` OK. 메모리: `project_scheduler_m1_landed`.

### M1 = 정밀도-적응 잔류를 포함한 (실행순서 + eviction) joint **exact** 최적기
spec `docs/superpowers/specs/2026-06-23-mxp-scheduler-precision-adaptive-design.md` (rev5) → plan `docs/superpowers/plans/2026-06-24-mxp-scheduler-m1.md` (2 라운드 멀티에이전트 검토 + verbatim 구현·실행으로 수렴) → subagent-driven 7-task 실행 → 최종 코드리뷰 2회 READY TO MERGE (Critical/Important 0). 신규 stdlib 모듈 4개 (twin 아님 — single-source):

- `MXP_scheduler/eval_sched.py`: cube=(mt,kt,nt) 원자, tile 3종(A=(kt,nt)/W=(mt,kt)/C=(mt,nt)). **`apply_cube`** = 비용 단일 진실 (first-touch A/W 과금, zero-init C 무료(공간점유), counter>0 C 로드=reload, partial-C evict=spill, capacity/stall). **`eviction_choices`** = 전수 정확 move set(오라클·A* 공유). `eval_sched` = 한 스케줄 fold.
- `MXP_scheduler/warmstart.py`: 구조적 mapping→스케줄(D6 floor 인큐번트) + `min_structural_steady_stall`(빈영역 진단).
- `MXP_scheduler/oracle.py`: `dp_optimal(stall0=False)` 전순열×eviction DP — A* 와 **독립**(move set만 공유), 작은 T(≤6) 교차검증. `stall0=True` 로 유한BW 검증.
- `MXP_scheduler/astar.py`: `optimize_exact` best-first joint. 상태=(done,resident,last_compute), admissible-not-consistent h+`best_g_to_state` 재확장, **stall=0 하드 prune**, heap key `(f,g,sig,seq)`(SchedState 비교 크래시 방지), **budget-hit→proven_optimal=False**, 빈영역=진단(silent inf 금지, CLI exit 1). 큰 T 는 증명 못 닫으면 honest gap.

**핵심 의미값**: `energy = (read+spill)·(dram+onchip)` — **first-touch 로드 floor 포함**(all-resident energy ≠ 0). final C write 는 schedule-invariant 상수로 g/h 제외(argmin 무영향 exact 재정식화). A*==oracle 으로 정확성 검증(랜덤 sweep 0 불일치).

**다음 후보**: → **2026-06-25 섹션(맨 위)에서 방향 재확정** — 큰-T 최적화는 수리최적화 솔버(A* scale·RL 아님; 정확한 솔버 선택은 문제정의 재조사로 결정). ② 실 워크로드 wbits 맵 연동 ③ optimize 결과를 `mxp_scheduler` report/CLI 통합.

## Next session kickoff (2026-06-23, **DRAM 에너지 계수 full-system 확정** — MXP_scheduler, RTL/코드로직 무변, 미커밋)

**진행 상태**: `dram_presets.json` 의 `pj_per_bit`(=`coeffs.dram`)를 **full-system 경계로 통일** 확정 (사용자 결정: 주력 LPDDR5/5X, 경계 full-system = device core+I/O+SoC PHY/controller+refresh). 직전 시드값은 LPDDR=device-internal vs DDR=I/O포함으로 경계 혼재였음 → 교체.

- **확정값** (full-system pJ/b): LPDDR5-6400_x16=**9.0**, LPDDR5X-8533_x16=**7.5**, DDR4-3200_x64=20.0(secondary), DDR5-4800_x64=14.0(secondary).
- **주 앵커(도표 직접 판독)**: Ha 2018 Stanford PhD Fig 4.8 — LPDDR4 device-incl-I/O+refresh ~12-13 pJ/b. 체인: x0.75 LPDDR5 세대이득 + ~2-2.5 SoC PHY/controller. LPDDR=I/O가 전체 ~10%뿐(DDR/GDDR과 반대).
- **산출물**: `docs/dram-energy/README.md` (경계정의·도출체인·교차검증·DRAM선택 = source of truth) + `docs/dram-energy/refs/` (출처논문 6 PDF: Ha2018 앵커 + Fig4.8 크롭증거, OConnor2017/Chatterjee2017/Ghose2019/LPSpec2025). 기본 DRAM 권고 = **LPDDR5-6400_x16**.
- **검증**: `cd MXP_scheduler && python -m pytest -q` → **80 passed, 1 skip(CACTI)** · selftest OK · crosscheck OK. 테스트가 pj_per_bit golden 안 박음(>0, !=200만) → 값 교체 무회귀.
- **caveat**: 9.0/7.5는 die density·컨트롤러 포함분 따라 LPDDR5 ~8.5-12.5 중앙값. 합성/실측 pJ 나오면 config `coeffs.dram`로 덮어쓰기. 메모리: `project_scheduler_dram_energy_coeffs`.
- **주의**: 이 변경 + `MXP_scheduler/explore_*.ipynb` 2개는 **미커밋**(git untracked/modified). 커밋 원하면 다음 세션에.

## Next session kickoff (2026-06-10, **MXP_scheduler 리뷰픽스 + hwconfig 구현·머지** — PR #2, main `d21e9ba`)

**진행 상태**: 이번 세션은 전부 `MXP_scheduler/` (RTL 무변). ① 코드리뷰→수정 (`ec194ec`): coeffs 오타키 거부, `evaluate()` 블록 walk 6→1회 공유, CLI 친절 에러, report 헤더 freq_ratio/eff_bw, 테스트 55→58. ② **hwconfig (config 자동화)** spec→plan→subagent-driven 7-task 구현, PR #2 머지.

### hwconfig 요약 (spec: `docs/superpowers/specs/2026-06-10-mxp-scheduler-hwconfig-design.md`)
- `--config hw_config.json` (SRAM 뱅크 스펙 + DRAM 표준명 + chip_freq_mhz) → **CACTI 7** (onchip pJ/bit, SRAM max freq) + **`dram_presets.json`** (dram_bw/freq_ratio/dram pJ/bit, 문헌 출처 명기) 자동 도출. 우선순위: 명시 CLI 플래그 > config > 기본값, `--coeffs` 파일 최종 승리.
- 신규 파일: `MXP_scheduler/hwconfig.py` (단일 어댑터, 트윈 아님) · `dram_presets.json` · `hw_config.example.json` · `docs/cacti-setup.md`. CACTI는 `third_party/cacti` (gitignored, 재클론 시 setup 문서).
- **Gotcha**: CACTI 7은 full cache.cfg 키셋 필수 — 최소 cfg는 10분+ hang 또는 SIGFPE. `hwconfig._cacti_cfg` docstring 참조, 비활성 DRAM/NUCA 키 "정리" 금지.
- Ramulator는 v1 제외 (bandwidth 모델이라 불필요; effective-BW derate 캘리브레이션 필요 시 재고려) — 메모리 `project-mxp-scheduler-config-autoparam`.
- 검증: `cd MXP_scheduler && python -m pytest -q` → **81 passed** (실 CACTI 통합 1개 포함) · selftest OK · crosscheck OK (word_bits=16 parity 케이스 추가).

### 다음 세션 후보
1. (스케줄러) DRAM 프리셋 확장 / effective-BW derate 캘리브레이션 / 실 워크로드 wbits 맵 연동
2. (RTL) 보고-only 업스트림 항목 반영 — 아래 2026-06-04 섹션의 HIGH/MEDIUM 목록
3. production dataflow 최적화 · Vivado timing closure · 9-mode+mixed CI 자동화

## 직전 세션 (2026-06-04, **전체 코드리뷰 + 회귀 확인 완료** — commit `8a0e673`)

**진행 상태**: 멀티에이전트 파일단위 리뷰→수정 루프 1회 수렴 (88 agents). 21건 수정 / 15파일. **연산 로직 버그는 없었음** — 고친 건 전부 주변부(빌드·실행 스크립트, 테스트 게이트, 데드코드). 검증된 데이터패스(`int_to_fp32` 클램프, `fp32_adder` 파이프라인)는 PASS 깨질 위험으로 시니어 에이전트가 의도적으로 거절 → timing-closure 단계로 보류.

### 이번 세션 적용 (commit `8a0e673`)
- **Linux 실행 블로커**: `run_integration_one|smoke|mixed_one.sh` 의 `cmd //c xsim` (윈도우 전용) → `case $(uname -s)` 분기. repo 가 Linux 로 클론돼서 그대로면 죽었음.
- **빌드 누락**: `run_integration_smoke.sh` xvlog 리스트에 `sram_1rw_banked_mp.v` 추가.
- **가짜 green**: 단위 TB 5개 ($finish 로 항상 exit 0) → run_*.sh 에 pass-sentinel grep 게이트.
- **테스트/데드코드**: test_gen_mixed prec_b 인자, int_to_fp32_tb 음수 scale, sram_mp_tb idle-bank 검증, GEMM.v `sc_bcast` 데드넷 제거(동작 무변).

### 회귀 확인 (2026-06-04)
- `bash sim/run_integration_sweep.sh` → **ALL 9 MODES PASSED**
- `python sim/runner.py mixed-sweep` → **ALL 3 MIXED MODES PASSED**
- `sim/tests` 10 PASS · `MXP_Tools` 53 PASS. 회귀 0.

### 다음 세션 후보: 보고-only 항목 업스트림 반영 (수정 안 됨)
리뷰가 imports/vendored/MXP_Tools(fork)에서 발견했으나 정책상 미수정 (full 목록은 워크플로 결과 참조):
- **HIGH** `MXP_Tools/examples/01_smoke.py`: `cli_main()` 가 항상 SystemExit → 스모크가 1단계 후 죽는데 exit 0.
- **MEDIUM** `MXP_Tools/tests/test_gemm.py`: prec_B 고정 + K-block 내 prec 불변 → 64× A=2 버그 회귀 방어 안 됨.
- **MEDIUM** `viz.py` NaN vmin/vmax, `rmw_gen.py` MAX_N=128 오버플로, imported `PE_feeder/PE_naive` 8-bit 슬라이스 하드코딩(latent).

기타 후보(기존): production dataflow 최적화 · Vivado timing closure · 9-mode+mixed CI 자동화 · loop order explorer.

## 이전 세션 요약 (2026-05-18, **mixed-sweep BLOCKER 해소** — root cause = golden side prec_b 누락)

mixed-sweep A=2/4/8 전 모드 PASS. BLOCKER 는 RTL 이 아니라 **golden 측 bug** — `sim/gen_mixed.py` 가 `prec_b` 인자 누락 → A=2 에서 2^6=64× off. Fix: `mxint_gemm_golden` mixed-aware 확장 + `prec_B` 명시 전달 + 테스트 4건. RTL fix #4 (`impl_w` 파이프라인) 는 유효 확인. viz 3-row 단순화, TB 에 `[CYC]` stage cycle 출력 추가.

**진단 단서 (재사용 가치 — 다음에 빠르게 식별)**:
- `hw_fp32 snr > sw_fp32 snr` 시그니처 = "정상 RTL + 잘못된 golden" (A=2 mixed 에서 4.21 dB vs 0.12 dB).
- `(C_hw / C_old_golden).median()` 이 깔끔한 2^k 면 implicit_scale 인자 누락.

**핵심 RTL 매핑**: SA 의 row=K, col=N, cycle=M 진행 — 자세한 표는 `## MXP control surface` 첫 subsection. mixed-prec 디버그 시 row=M 으로 단정 금지.

### Fix chain (mixed-prec 진화 history, 참조용)

| # | Commit | 위치 | 내용 |
|---|---|---|---|
| 1 | `b0cb974` | TB | `in_scale_weight` cadence 일반화: `drive_cyc[k] = 18 + A_FIRE_DELAY + Σ_{j=0..k} W[j]` |
| 2 | `405f4ec` | TB | `in_Wcontrol` cum_W exclusive schedule: `drive_w_cyc[k] = 18 + Σ_{j=0..k-1} W[j]` |
| 3 | `d03771c` | TB | dynamic TOGGLE_VAL: K-tile 별 W lookup (`18 + A_FIRE_DELAY + W[k_t·M_T·TILE_SIZE]/2`) |
| 4 | `5beb24c` | **RTL** | `Accumulator_Col.v`: `impl_w` fire-chain 파이프라인 (`impl_w_q1/q2/q3`) — mode-별 effective impl_w |
| 5 | (본 세션) | **Python golden** | `mxint_gemm_golden` mixed-aware + `gen_mixed.py` prec_B 명시 전달 |

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

Three ways to run sim:

1. **Vivado GUI** — open `gemm_sram.xpr`. Synth top = `gemm_sram_top`; sim top = `gemm_sram_top_tb`. Run Behavioral Simulation.
2. **Python orchestrator (recommended for VSCode users)** — `sim/runner.py`:

   ```bash
   python sim/runner.py                                # default: mixed-one A=8 (random mixed) + viz
   python sim/runner.py mixed-one --A 8 --uniform 8    # uniform W isolation test
   python sim/runner.py mixed-one --A 8 --k-tile 8,4,2,8   # K-tile granular
   python sim/runner.py mixed-sweep                    # 3-mode: A in {2,4,8} x random W (~5 min)
   python sim/runner.py integration-one --A 8 --B 8    # single (A,B) uniform mode
   python sim/runner.py integration-sweep              # 9-mode uniform (A,B) in {2,4,8}^2 (~10 min)
   ```

   VSCode "Run Python File" 버튼 (no-args) → default mixed-one. 산출물: `work/<LABEL>/result.png` (4-row 종합 figure: 입력 W/A/prec_map · 출력 C_hw/C_sw/C_fp32 · diff log10 · PASS/FAIL + 통계), sweep 추가로 `work/sweep_summary.png` (9-mode rmse matrix). 내부적으로 기존 `sim/run_*.sh` 를 bash 로 호출 (xsim Python 직접 호출 시 좀비 issue 회피 — 메모리 `feedback_xsim_zombie_windows.md`).

3. **XSim batch (bash)** — 기존 `sim/run_*.sh` 직접 호출:

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
    runner.py                          # Python orchestrator (VSCode "Run" 호환); auto viz → work/<LABEL>/result.png
    gen_mixed.py                       # mixed-prec 입력/golden 생성 + inputs_mixed.npz (for viz Row 1)
    run_top_elab.sh                    # gemm_sram_top elab smoke
    run_integration_smoke.sh           # TB zero-prime + dump (no GEMM driving)
    run_integration_one.sh             # 1 mode end-to-end (<LABEL> <A_PREC> <B_PREC>)
    run_integration_sweep.sh           # 9-mode serial sweep + compare gate
    run_integration_parallel.sh        # 9-mode parallel dispatch guide (print template)
    run_mixed_one.sh                   # mixed-prec 단발 (random / uniform / K-tile via env vars)
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
