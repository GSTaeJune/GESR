# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## 사용자 직접 정의 규칙
1. Address the user in Korean; use English for everything else. Use established terms for recurring concepts consistently — don't invent new or alternative names for something already named. When a genuinely new term is introduced, briefly define it on first use, then reuse it as-is.

2. After completing a logical chunk of work (one or several related tasks, e.g. tasks 1–3), run `/superpowers:requesting-code-review` and `/review`, iterating on their feedback until no blocking issues remain. Reviews don't need to run per-task — group related tasks into a coherent unit before reviewing. Minor or optional suggestions don't need to block convergence.

## Next session kickoff (2026-07-13 오후, **BF16 native 재작성(A6) 랜딩 — 손코딩 bf16 덧셈기/변환기, HardFloat-free 데이터패스, 전 게이트 green, LUT -32%/최악 로직 2.28ns**)

**무엇/왜 (사용자 지시: "다시 BF16 계산기로 RTL 짜. 주석 잘 달아놔 — 내가 읽어볼 거니까")**: Phase 2c 정직 측정이 병목 = monolithic fp32 `AddRecFN`(24b significand 로 8b 일 하는 구조, est 12.4ns)임을 확정 → T3(생성물 내부 분할) 대신 **A6 승격**: `bf16_adder.v` + `int_to_bf16.v` 를 진짜 bf16 폭(내부 11~12b GRS 그리드)의 단일-라운딩 native 산술로 재작성. **주석이 1급 산출물** — 각 파일 헤더가 알고리즘(정렬/GRS/정규화 클램프/RNE/subnormal/0 부호 규칙)을 한글로 상술, 사용자 직독 대상. plan+outcome(리뷰 수렴 기록 포함): `docs/superpowers/plans/2026-07-13-bf16-native-datapath.md`. **main ff-병합 완료(8커밋), 브랜치 삭제, 2026-07-14 `origin/main` push 완료** (`fp32-rmw-final` 태그 포함. 주의: 아래 이력 섹션들에 남아 있는 "미push" 표기는 각 세션 당시의 기록 — 현재는 전부 push 됨). 리뷰 2라운드 수렴: 라운드1(종합+adversarial, 독립 퍼즈 5.5M/4.1M+ 0불일치 — Important 1건 = 주석의 full-INT32 golden 일치 과잉주장 → "라운딩 도메인 주의"로 축소, RTL 무변) → 라운드2(/review: testing/maintainability/adversarial/red-team — CRITICAL 0, INFORMATIONAL 전부 수정). Codex 크로스모델은 계정 모델 설정 문제로 불가(비차단).

- **계약 전부 불변**: 포트/파라미터 표면(L_IN/L_ADD/L_SUM/L_OUT = 절단점 깊이, L_CONV), RMW 5cy(내부 분배 S1..S5 블록당 1단 — 블록명만 native 로: S1 i2b[F] LZC32+8b RNE / S2 i2b[B] scale·denorm / S3 adder[A] 정렬·가감산 / S4 adder[B] 정규화 / S5 adder[C] RNE·패킹), out_RMW 레지스터 출력, TB 로직 무수정(헤더만 갱신).
- **게이트 전부 green(실측, 두 유닛 다 첫 실행에 통과)**: bf16_adder **200025**(oracle +20 directed: 유한 overflow→inf, min-normal−min-subnormal 경계, d=24 sticky+round-carry, x+(−x) 0 부호, tie, one-sided inf±finite; **트리플 DUT** = 기본/RMW/전조합) / int_to_bf16 **32360**(+48 full-range INT32) / rmw **113** / top elab / 통합 **ALL 9 MODES PASSED**(bit-exact) / mixed **ALL 3 PASSED** / preserved fp32 유닛 70012·PASS·PASS. 카운트 변경은 oracle 강화(약화 아님). **게이트 인프라 경화(/review 라운드)**: 3개 bf16 유닛 sentinel 이 정확-카운트 고정(빈/잘린 벡터파일의 "ALL 0 PASSED" false-green 차단, 벡터 추가 시 스크립트 숫자도 갱신), run_rmw.sh 가 rmw-gen 자동 수행(stale-vector 함정 제거), bf16_vectors.py errstate 에 invalid 추가(warnings-as-errors 환경서 mid-write 잘림 방지).
- **합성 실측(OOC K7, 4ns, pre-place)**: LUT-cell 744→**505**(-32%; 동일 metric 시리즈, Slice LUT 는 404), FF 148→139, WNS -8.79→**-1.64**. 최악 = S3 정렬·가감산 5.64ns = **logic 2.28**(13lvl) + route 추정 3.36 — 구 병목 로직(4.09ns) 절반 이하, 잔여 위반은 pre-place route 추정이 지배. 백엔드 = ASIC PNR(Vivado 는 상대 프록시)이라 **250MHz 가시권, T3 는 대상 블록 소멸로 폐기**.
- **HardFloat-free**: 전 bf16 컴파일 리스트(rmw/top/integration/smoke/mixed/synth)에서 번들·fp32_adder·fp32_to_bf16_rne 제거(elab 이 자립성 증명). 번들은 preserved fp32 유닛 TB + `fp32-rmw-final` 복구 전용. `fp32_to_bf16_rne.v` ACTIVE→PRESERVED(자체 70012 게이트 유지).
- **커버리지 유의**: rmw_tb 스트리밍이 여전히 유일한 skew/latency 게이트(약화 금지). 단위 oracle 이 subnormal/tie/overflow 의 유일한 영구 게이트라는 Phase 2b 원칙 그대로 — red 면 RTL 을 고칠 것(벡터 약화 금지, 추가만 허용).

## Next session kickoff (2026-07-13 오전, (부분 대체 — 같은 날 오후 native 재작성 섹션(맨 위) 참조: T3 는 폐기, HardFloat 스테이지명은 native 블록명으로 교체) **RMW pipeline rebalance Phase 2c 랜딩 — 5cy 유지 레지스터 재배치 + out_RMW 레지스터 출력화. 게이트 전부 green**)

**무엇/왜**: 9-에이전트 RTL 분석 + fable adversarial 선별(2026-07-12)의 T2 실행. 기존 5단 파이프라인은 레지스터 5개 중 3개가 wire-shift 자리(빈 스테이지)이고 두 reg-to-reg 구간에 FP 블록이 몰려 있었음. 재배치 후 = **주요 FP 블록당 정확히 1단**: S1 INToRecFN+exp-add / S2 FNFromRecFN+narrow / S3 RecFNFromFN / S4 AddRecFN / S5 FNFromRecFN+narrow, **out_RMW 레지스터 출력**(SRAM D 조합 꼬리 제거 — PNR 매크로 D-setup 에 유리). 외부 계약 불변(L_CONV+L_ADD=5cy, rmw_tb/top TB 무수정). plan+outcome: `docs/superpowers/plans/2026-07-13-rmw-pipeline-rebalance.md`. 브랜치 `feat/rmw-pipeline-rebalance`.

- **파라미터 표면**(전부 default=구동작 → 단위 TB 무변경 green): `fp32_adder #(L_ADD=3, L_SUM=0)` / `bf16_adder #(L_IN=0, L_ADD=3, L_SUM=0, L_OUT=0)`(L_ADD 의미가 "전체지연"->"피연산자단"으로 변경) / RMW 내부 분배 = i2b `L_CONV-1` + adder `L_IN1/L_ADD-2/L_SUM1/L_OUT1` (제약 L_CONV>=2, L_ADD>=3).
- **게이트 전부 green(실측)**: bf16_adder **200005**(dual-DUT: 기본 + RMW구성 양쪽) / int_to_bf16 **32312**(L_CONV 2·1 양쪽) / rmw **113** / fp32 유닛 PASS / 70012 / top elab / 통합 **ALL 9 MODES PASSED**(bit-exact) / mixed **ALL 3 PASSED**. **커버리지 핵심**: rmw_tb 스트리밍(1 vec/cy, 파이프라인 모델 없는 per-index oracle)이 **유일한 skew 게이트** — 통합 TB 는 10cy 정적 유지 입력이라 skew 못 잡음. rmw_tb 약화 금지.
- **합성 정직화 발견(중요)**: 종전 "RMW WNS -2.547(~153MHz)" 는 **반쪽 측정** — out_RMW 조합→포트 경로(AddRecFN 포함)가 no_output_delay 로 untimed(스크립트가 create_clock 만 설정). 재배치 후 전 경로 reg-to-reg → 정직 측정 **WNS -8.79 @4ns**, 병목 = **S4 AddRecFN 단독 est. 12.4ns**(logic 4.1/route 8.3, pre-place). LUT 768->744(SRL 34->0), FF 99->148. **250MHz 는 T3(AddRecFN 내부 3분할, [GENERATED] 사본) 없이 불가 — 다음 결정 후보**. synth 호출 gotcha: `-nojournal -nolog` 는 `-tclargs` **앞**(뒤에 두면 part 인자로 먹혀 fail + stale 리포트).
- SRAM/BRAM 계열 제안은 사용자 지시로 제외(SRAM RTL 은 sim 전용, 실물 = CACTI/파운드리 매크로). MXP core(imports) 항목(adder_lane 등)은 [UPSTREAM] — 이번 스코프 밖.

## Next session kickoff (2026-07-10, **RMW FP32->BF16 Phase 2b 랜딩 — RTL 32->16 하드교체 + 통합 재-bit-exact 완료. main 로컬 ff-병합 예정, 미push**)

**활성 라인**: RMW inter-tile accumulator 를 **FP32->BF16** 으로 (SRAM psum 저장 절반 + DRAM O-write 에너지(지배항) 절반). Phase 1(golden) + 2a(primitives) + **2b(RTL 하드교체+통합) 랜딩으로 이 라인 종결**. **spec**: `docs/superpowers/specs/2026-07-08-rmw-bf16-design.md` (§6.2 끝 "Phase 2b outcome"). **plan**: `docs/superpowers/plans/2026-07-10-rmw-bf16-phase2b.md`. 브랜치 `feat/rmw-bf16-phase2b` (Task 8 문서화 시점; Task 10 에서 main ff-병합 예정, 미push).

### 랜딩 완료 (feat/rmw-bf16-phase2b)
- **RTL 폭 32->16 하드교체**: `RMW.v` in_SRAM/out_RMW/sram_dly 16b (in_GEMM 은 INT32 유지) + int_to_fp32->int_to_bf16 v2 / fp32_adder->bf16_adder 스왑; `gemm_sram_top.v` Q/rmw_out_RMW/sram_D_w/WMASK `*32->*16` + `.DATA_WIDTH(16)` + zero-mux `16'h0000`; 양 통합 TB(`gemm_sram_top_tb.v`, `gemm_sram_top_mixed_tb.v`) dump `%08x->%04x`; `hwio.py` `read_writememh_bf16` (16b word -> `uint32(bits)<<16` fp32 exact upcast, gap sentinel 은 written-mask 로 분리).
- **int_to_bf16 v2 (계획의 clamp 를 넘어선 재설계)**: 리뷰 라운드1 이 `FNFromRecFN_bf16_wrapper` 의 denorm TRUNCATE (`HardFloatBundle_bf16.v:179-181`, round/sticky 없음)를 발견 — 어떤 flush 임계로도 subnormal 밴드가 golden(RNE) 대비 1 ULP 낮음 (143/32312 mismatch floor, 리뷰어 2인 독립 재현). v2 = `INToRecFN_i32_e8_s8`(int->8-sig RNE = golden r8) -> recoded fp32 exact widen `{sign,exp9,frac7,16'b0}` -> 지수 add(10b signed, [-127,415]) -> flush `new_exp10<122`(bf16 min subnormal 2^-133 의 절반 미만 -> 정확히 +-0) -> `FNFromRecFN_wrapper`(fp32; 남는 subnormal 밴드 recExp [122,129] 는 shift 0..7 로 padding zero 만 버려 EXACT) -> `fp32_to_bf16_rne`(단일 RNE). `FNFromRecFN_bf16_wrapper` 는 더 이상 인스턴스 안 함(번들 보존).
- **게이트 전부 GREEN (실측)**: `int_to_bf16_tb` ALL **32312** PASSED (음수 scale + subnormal boundary 확장 oracle: 4012 ints x 8 scales + 216 boundary); `rmw_tb` ALL **113** PASSED (underflow flush, subnormal round-up `(3,-8)->0x0001`, ties, inf saturation); `fp32_to_bf16_rne` 70012 / `bf16_adder` 200005; 통합 sweep **ALL 9 MODES PASSED**(bit-exact 16384x9 vs bf16 golden); mixed sweep **ALL 3 MIXED MODES PASSED**(bit-exact x3); MXP_Tools pytest **74 passed**(69 baseline + 5 hwio bf16). fp32 단위 TB(int_to_fp32, fp32_adder) 무회귀.
- **정정(중요)**: Phase 2a 의 "D6 클리어" 는 **normal 영역에서만** 유효했음 — scale in [0,254] oracle 은 lossy subnormal 에 구조적으로 도달 못 해(최소 도달값 2^-127, frac=0) truncation defect 이 안 보였음. Phase 2b 의 확장 oracle + v2 가 normal/subnormal/flush/inf-saturation 전 밴드에서 bit-exact 를 닫음.
- **D4 정보성 SNR (bf16 vs fp32-truth, 128^3 seed 0, 게이트 아님)**: A2_B2 2.21 / A2_B4 5.08 / A2_B8 5.36 / A4_B2 5.11 / A4_B4 14.57 / A4_B8 17.58 / A8_B2 5.41 / A8_B4 17.64 / A8_B8 38.21 dB; mixed_A2 4.21 / mixed_A4 9.45 / mixed_A8 10.22 dB. 전부 양수 (catastrophic 경고 없음).
- **핵심 caveat**: 통합 sweep 의 standard-normal 워크로드는 `comb_s < 0` 도 subnormal psum 도 절대 안 만듦 — **확장 단위 oracle(32312) 이 v2 의 subnormal/flush 동작을 검증하는 유일한 영구 게이트**. 단위 게이트가 red 면 RTL 을 고칠 것(oracle 벡터 약화 금지).

### 다음 후보
1. **SRAM 실이득 정량화**: psum word 32->16 = SRAM psum 저장 절반; buffer_sweep 의 O-buffer(binding constraint, best 1:1:6) 유효 용량 2x. cap 별 재-sweep 또는 CACTI 실측.
2. **Vivado timing closure**: (대체됨 — 2026-07-13 native 재작성으로 데이터패스는 HardFloat-free 손코딩이며 합성 실측은 맨 위 섹션 참조) `.xpr` 소스 목록이 Phase 2a 이전이라 stale — GUI flow 는 `.xpr` 재정비 전까지 미지원, bash `sim/run_*.sh` 가 authoritative.
3. **필요 시 fp32 fallback 경로**: bf16 정확도가 특정 워크로드에 부족하면 태그 `fp32-rmw-final` 로 복구(전 fp32 통합 상태 검증됨). golden 은 `accum_dtype={fp32,bf16}` 로 이미 양쪽 지원.

**재현/복구**: 복구 태그 `fp32-rmw-final`(local, 태그 시점 green 검증: pytest 69 + 전 단위 TB + 9-mode fp32 sweep + mixed fp32 sweep). `git checkout fp32-rmw-final && bash sim/run_integration_sweep.sh` -> `ALL 9 MODES PASSED`. bf16 재검증: `bash sim/run_int_to_bf16.sh`(32312) / rmw-gen 선행 후 `bash sim/run_rmw.sh`(113) / `bash sim/run_integration_sweep.sh`(9, --accum bf16 내장) / `python sim/runner.py mixed-sweep`(3). **주의**: `rmw_gen.py` 가 이제 bf16-semantics = MXP_Tools 업스트림 동기화 시 보존할 project-only 분기.

## Next session kickoff (2026-07-09, (완료 — 2026-07-10 섹션 참조) **RMW FP32->BF16 새 라인: Phase 1(golden) + Phase 2a(RTL bf16 primitives) 랜딩 완료. 다음 = Phase 2b(RTL 32->16 하드교체+통합). main 로컬 ff-병합, 미push**)

**활성 라인** (이전 "모든 라인 종료" 상태를 뒤집음): RMW inter-tile accumulator 를 **FP32->BF16** 으로. SRAM psum 저장 절반 + DRAM O-write 에너지(지배항) 절반. 자세한 배경/결정/발견은 메모리 `project_rmw_bf16_phase1_landed` 참조. **spec**: `docs/superpowers/specs/2026-07-08-rmw-bf16-design.md` (D1~D6 결정 + §6.1 Phase 2a 결과·이월).

### 랜딩 완료 (main, 미push)
- **Phase 1 (golden)** plan `docs/superpowers/plans/2026-07-08-rmw-bf16-phase1-golden.md`: `mxint_gemm_golden(accum_dtype in {fp32(기본),bf16})`. fp32 byte-identical(frozen 회귀), bf16 = ml_dtypes(optional `[bf16]` extra, lazy). 모델 = K-block당 int32 sum -> `int->bf16` RNE -> `np.ldexp` 단일 exp-shift -> native bf16 add. `ref --accum bf16`(파일명 분리 + 정보성 SNR). 검증 `cd MXP_Tools && python -m pytest -q` -> **69 passed**. 비-tautology 게이트(독립 bf16-RNE 레퍼런스 + fp64-truth + hand-derived + topology + subnormal).
- **Phase 2a (RTL primitives)** plan `docs/superpowers/plans/2026-07-09-rmw-bf16-phase2a.md`: **핵심 아키텍처 = bf16 add는 fp32 도메인에서** (스파이크가 HardFloat `AddRecFN(8,8)` elaborate 불가 발견 -> widen `{bf16,16'h0}` exact -> 기존 fp32 `AddRecFN` -> RNE narrow = golden 모델과 일치, provably bit-exact). 3 primitive 전부 ml_dtypes 와 bit-exact(각 cross-check TB, 게이트 실측): `fp32_to_bf16_rne.v`(70012) / `int_to_bf16.v`(24072, `INToRecFN_i32_e8_s8`+exp-add) / `bf16_adder.v`(200005). vendored `third_party/berkeley-hardfloat/HardFloatBundle_bf16.v`. oracle `sim/bf16_vectors.py`. 검증 `bash sim/run_{fp32_to_bf16_rne,int_to_bf16,bf16_adder}.sh` (각 `ALL ... TESTS PASSED`). fp32 데이터패스 byte-identical, fp32 unit TB 무회귀. **D6 클리어**.

### 다음 = Phase 2b (RTL 폭 32->16 하드교체 + 통합) — 각자 spec/plan/구현/리뷰 사이클
1. **[MUST FIX 먼저] `int_to_bf16` 깊은 언더플로 발산**: int_to_fp32 에서 물려받은 `new_exp=new_exp10[8:0]` 절단에 underflow clamp 없음. 음수 `comb_s`(작은 act/weight, E8M0<127 인 실워크로드)에서 wrap -> golden 의 ldexp-flush-to-zero 와 발산. 현재 oracle 이 scale in [0,254] 만 테스트해 **un-gated**. 조치: `sim/bf16_vectors.py` 에 음수 9-bit scale 벡터 추가 + `int_to_bf16.v`(신규 파일, 수정 가능)에 flush-to-zero clamp. int_to_fp32.v 는 보존(무변경).
2. **폭 전파 32->16** (spec §6.2 구체 사이트): `RMW.v` in_SRAM/out_RMW/sram_dly 16b(in_GEMM은 INT32 유지) + int_to_fp32->int_to_bf16 / fp32_adder->bf16_adder 스왑; `sram_1rw*` DATA_WIDTH 16; `gemm_sram_top.v:75,80,81` `*32->*16`+`16'h0`; TB dump `%08x->%04x`; `hwio.py` bf16 reader(16b word->fp32 upcast) + **gap sentinel 분리**(bf16 `0x0000` 이 unwritten-slot sentinel 과 충돌 -> written-mask 별도).
3. **fp32 RTL 하드교체 전 git 태그**(복구용), fp32 단위 RTL/TB는 HEAD 보존. 그 뒤 **통합 sweep(bf16 golden, `ref --accum bf16`) 재-bit-exact** = Phase 2b 게이트. (MUST-FIX 1 이 여기서 실워크로드로 드러남.)
4. inf-inf NaN 부호 차이(HardFloat +qNaN vs ml_dtypes -qNaN, IEEE 미규정) — 실질 무해(bounded GEMM엔 inf/NaN 없음), bf16_adder_tb 는 NaN sign-agnostic 비교.

**재vendor 재현**: bf16 emit 은 `C:/Users/ptj72/hf_src`(HardFloat clone, Chisel 3.5.6) 의 `hardfloat/src/main/scala/EmitBf16.scala` + Java 17 은 `JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED ..."`(= form) 로 `sbt "runMain hardfloat.EmitBf16 ./generated_bf16"`. AddRecFN(8,8) 는 elaborate 안 됨(의도적으로 미사용).

## Next session kickoff (2026-07-07, **buffer_sweep + timeloop-sweep 완료 — 이 라인 종료(사용자 결정). scheduler 라인도 함께 종료. main 로컬 ff-병합, 미push**)

**사용자 결정 (2026-07-07)**: "buffer_sweep이나 스케쥴러는 이걸로 다 끝내자" — **buffer/timeloop 매핑 라인과 MXP_scheduler(joint order+eviction: M1/cpsat/band) 라인 둘 다 종료.** 두 라인 모두 코드 무변경 보존. 새 방향 나오기 전까지 이쪽 개선 제안 금지.

**진행 상태**: `buffer_sweep/timeloop_sweep.py` (신규 standalone 드라이버) + `buffer_sweep/timeloop/` (arch/problem/mapper YAML 템플릿 + `run_jobs.sh` + `NOTES.md`) **빌드·리뷰 round1 수렴·main 로컬 ff-병합 완료**(브랜치 `feat/timeloop-sweep` 삭제, 5커밋). RTL/MXP_scheduler/buffer_sweep.py(v1) **무변경**. **`origin/main` 보다 앞섬 — 미push**. 검증: `python buffer_sweep/buffer_sweep.py --selftest`(4,866) + `python buffer_sweep/timeloop_sweep.py --selftest`(25) OK. eval_nest math core는 리뷰어 2명 독립 fuzz 828케이스 0불일치 확인.

### 무엇 (spec `docs/superpowers/specs/2026-07-06-timeloop-sweep-design.md`, survey `docs/superpowers/notes/2026-07-06-mapper-survey.md`)
- **buffer_sweep v1 정정**: "dataflow 고정 = O-stationary"는 **intra-tile(SA bit-serial row=K/col=N/cycle=M, RTL 고정)만** 의미. **inter-tile 스케줄(loop order+tiling)은 search space** — Timeloop 같은 매퍼가 탐색하는 영역. 그래서 tile 스케줄은 하드코딩 안 함.
- **역할 분담**: Timeloop(WSL, timeloopfe v4)이 **proposer**(파티션·32배수 HW조건을 arch로 받고 (m,k,n)+perm 제안), 우리 `eval_nest`가 **scorer**(bit-serial×8, 핑퐁 stall, psum spill 정확과금, LPDDR5 9pJ/b). **best = min(timeloop, v1 O-stationary floor)** per metric.
- **sweep 축**: cap을 (W,A,O) 버퍼로 분할(21 파티션/cap, 8-slice). (m,k,n)·loop order는 Timeloop 출력. 워크로드=buffer_sweep과 동일(deit_tiny/small + qwen0.5b/1.5b, attention BMM 포함).
- **왜 CoSA/최신매퍼 아님**: 작은 single-level dense-GEMM mapspace에선 exhaustive가 최적. CoSA/GAMMA/DOSA는 거대 multi-level mapspace의 근사해(proxy objective)용 — 우리 regime엔 과잉. 재검토 트리거 = multi-level SRAM 계층/fusion.

### 핵심 결과 (실측)
- **Timeloop이 v1에 100% 패배**: 64KB 84개 파티션(21×4모델) 전부 best_src=v1. tl_E가 v1보다 **3.5~8배 나쁨**(qwen0.5b 8×). 원인 = Timeloop이 K-outer(psum spill) 순열을 다수 제안(details 438행 중 321행이 K@DRAM=spill), 우리 scorer가 32b psum spill을 정확과금. → **min()=v1**. Timeloop은 "명백한 손해 없음"의 독립 cross-check로만 유효. 128/256은 사용자 결정으로 **v1-only 산출**(tl 컬럼 blank, best=v1).
- **cap↑ → best energy 단조감소** (모든 모델 -33~-42%, 64→256KB): deit_tiny 5.32→4.30→3.57 / deit_small 16.07→12.46→10.63 / qwen0.5b 113.8→86.6→68.4 / qwen1.5b 386.5→288.6→223.0 mJ. (deit_small 16.07mJ = buffer_sweep v1과 일치 → 상호검증.)
- **best 파티션 = 항상 O-heavy 1:1:6** (W/A/O = 8/8/48, 16/16/96, 32/32/192 KB). DRAM 분해로 필연: O write=`M·N·32b` 고정(psum 완성 1회write), A read=`(M/T_m)·K·N·8`, W read=`(N/T_n)·M·K·8` → 줄일 수 있는 입력 refetch가 T_m·T_n(=O tile=O버퍼)에 반비례. **이긴 타일이 매번 T_m·T_n로 O버퍼를 꽉 채움**(예 256KB 128/·/192 → 96KB=O/2), **T_k는 작게**(32~64). cycles는 cap 무관 ~flat(compute-bound). "psum 용량 최대 + K 안쪽 완성"이 실측 최적 규칙.

### 산출물 / 다음 후보
- `buffer_sweep/results/timeloop/` (gitignored): 모델별 `*.csv`(파티션×cap best) + `*_details.csv`(shape별 nest/E/cyc) + cap별 energy/cycles heatmap 24 + summary 2. `--score-only`로 캐시 재채점(수분). 캐시는 `results/tl_cache/cfg_<md5(mapper_sweep.yaml)>`.
- 다음(재개 시): CACTI로 cap별 C_ONCHIP 실측 / seq·정밀도 sweep / RTL 연동 검증 / (tl이 이기게 하려면) Z-corrected DRAM arch(현 8b-O undercount bias, NOTES.md 참조).

## Next session kickoff (2026-07-03, **방향 전환: joint 스케줄러 폐기 예정 → buffer_sweep(SRAM 버퍼분할 sweep) 빌드·리뷰·main 병합. 미push**)

**방향 전환 (사용자 결정)**: MXP_scheduler 의 joint order+eviction 라인(M1/cpsat/band)은 **폐기 예정** — 스케줄러 개선 제안 금지. 새 방향 = **SRAM 구성·dataflow 를 먼저 고정하고 매핑**. 기존 M0/M1/cpsat/band 코드는 무변경 보존.

**진행 상태**: `buffer_sweep/buffer_sweep.py` (신규 top-level 디렉토리, **단일 파일 standalone** — MXP_scheduler 무의존, 사용자가 직접 읽고 수정하는 파일) 빌드·리뷰 2라운드 수렴·**main 로컬 ff-병합 완료**(브랜치 `feat/buffer-sweep` 삭제). **`origin/main` 보다 28커밋 앞섬 — 미push**. spec: `docs/superpowers/specs/2026-07-03-buffer-sweep-design.md`. 검증: `python buffer_sweep/buffer_sweep.py --selftest` → **4,866케이스 OK** (walk==grouped 이중평가기 상호일치; 리뷰어 독립 fuzz 20k케이스 추가 무결).

### 무엇 (spec + 리뷰 확정 의미론)
- **SRAM 고정 구성**: W(m·k·8b) / A(k·n·8b) / O(m·n·32b) 세 공간, **각각 핑퐁 ×2**. budget `2(8mk+8kn+32mn) ≤ cap`, cap ∈ {64,128,256}KB, m/k/n 32배수, 정밀도 8/8/32 고정.
- **dataflow 고정 = O-stationary**: 출력타일 (i,j)마다 kk innermost, O 버퍼 누적, DRAM write 1회(psum spill 없음). fetch 규칙 = "직전 step 과 tile id 다르면 fetch"(Kt==1 이면 W(i)가 j sweep 내내 상주).
- **사용자 확정 3건**: ① 핑퐁 shadow = prefetch/drain 전용(residency 확장 안 함 — Kt==2 W traffic ≤2× 과대는 의도된 보수성) ② **O writeback 은 다음 타일 Kt step 전체에 균등분산**(한-step 창은 가짜 stall) ③ C_ONCHIP=0.27 pJ/b (M0 비율을 DRAM 9.0 pJ/b 앵커로 재스케일; argmin 은 onchip 계수 무관).
- **워크로드**: deit_tiny/small(S=197→224 패딩) + qwen2.5-0.5b/1.5b(prefill 128, GQA, SwiGLU), **attention BMM 포함**(count=Q헤드).
- **산출물**: `buffer_sweep/results/` (gitignored) — 모델별 CSV + cap별 energy/cycles heatmap(별=best) + summary 2장. 콘솔 top-10(energy 순, cycles 병기) ASCII-only.
- 결과 요지: cap↑ → best energy 단조감소(예: deit_small 64→256KB 에서 16.1→9.9 mJ), cycles 최적은 stall=0 작은 타일. m/n 은 모델 최대 패딩치수에서 클리핑(dominated 구성 제거, 콘솔 명기).

### 다음 후보
- CACTI 로 cap별 C_ONCHIP 실측 갱신(현 0.27 고정) / effective-BW derate / seq-length sweep(`--seq`) / 정밀도 sweep((W,A)∈{2,4,8}²) / k>KT residency(핑퐁 양쪽 카피 reuse 인정 모델) / RTL 연동 검증.

## Next session kickoff (2026-06-26, **band-serpentine (B,d) 휴리스틱 스케줄러 빌드·검증·main 병합 — 큰 T 스케일, Qwen Q/K/V 데모, honest gap. 미push** — 참고: 2026-07-03 방향 전환으로 이 라인은 폐기 예정)

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
GEMM (INT psum + 9-bit scale) ──► RMW (INT→BF16, BF16 add) ──► sram_1rw_banked (BF16, 16-bit words)
                                       ▲                              │
                                       └────── BF16 prior psum ◄──────┘
```

Verified end-to-end via `bash sim/run_integration_sweep.sh` (9/9 modes bit-exact PASS vs MXP_Tools golden, 128×128 = 16384 elements per mode).

### RMW contract (from `RMW.v` header — committed decisions)

```verilog
module RMW (
    input  wire        clk, rst,
    input  wire [15:0] in_SRAM,   // BF16 prior partial sum read from SRAM
    input  wire [31:0] in_GEMM,   // INT32 from GEMM accumulator (one lane's worth)
    input  wire [8:0]  scale,     // combined 9-bit scale (matches one sub-word
                                  // of Accumulator_Col `out_scale`, = scale_len+1)
    output wire [15:0] out_RMW    // BF16 sum to write back to SRAM
);
```

This pins down **two** of the previously-open questions:

1. **Bit-width reconciliation**: dequantize INT→**BF16 at the RMW boundary** (Phase 2b, 2026-07-10). Both SRAM storage and the accumulator live in bfloat16 — **16-bit words** (`DATA_WIDTH=16`), no INT-domain accumulation in SRAM. The bf16 add itself is a **hand-written native single-round RNE adder** (2026-07-13 A6 rewrite: `bf16_adder.v` v3 — align/GRS → add/sub → normalize → round, 11~12b internal, HardFloat-free; the earlier fp32-domain detour is git history / tag `fp32-rmw-final`). This lines up with `MXP_Tools/hwio.py::read_writememh_bf16`, which reads 16-bit bf16 words from the `$writememh` dump and upcasts exactly (`fp32 = uint32(bf16_bits) << 16`).
2. **Scale handling**: the 9-bit `scale` is the per-sub-word combined `(act_scale + weight_scale − 127)` that `Accumulator_Col` produces (note: 9-bit signed, `{1'b0, act_scale} + {1'b0, weight_scale} - 9'sd127`). RMW consumes it during the INT→BF16 conversion; it is **not** stored alongside the psum.

The RMW controller (whether inside RMW or wrapping it) must still:
1. On `out_fire[col]` rising, latch the lane's INT32 psum and the corresponding 9-bit scale sub-word.
2. Issue a READ to the SRAM address holding that lane's prior BF16 psum.
3. Wait for the read return (1 cycle if leaf `PIPELINE=0`, 2 cycles if `PIPELINE=1`).
4. Convert INT→BF16 (using `scale`), add to `in_SRAM`, write back.

### Resolved (during integration) — committed values

- **Granularity**: **32 RMW instances** (col-parallel; each col j has its own RMW + dedicated bank j port). Lane decode (A8 = 1, A4 = 2, A2 = 4 dispatches per col fire) lives in `tb/gemm_sram_top_tb.v` per-col FIFO + drain state machine. Re-visit trigger: timing closure 시 합성 코스트 측정.
- **Latency budget**: `L_CONV = 2`, `L_ADD = 3`, total = **5 cycles**. **2026-07-13 native 재작성(A6)**: 두 유닛 모두 손코딩 bf16 폭, HardFloat-free. 내부 분배는 Phase 2c 계약 유지 — int_to_bf16 1단 + bf16_adder 4단(L_IN/L_ADD/L_SUM/L_OUT), 블록당 1단(S1 i2b[F] LZC32+8b RNE / S2 i2b[B] scale·denorm·인코딩 / S3 adder[A] 정렬·가감산 / S4 adder[B] 정규화 / S5 adder[C] RNE·패킹), out_RMW 레지스터 출력. "int_to_bf16 안에 2단"으로 읽지 말 것.
- **bf16 arithmetic implementation**: **native 손코딩** (`bf16_adder.v`/`int_to_bf16.v` v3 — 단일 라운딩 RNE, 내부 11~12b GRS 그리드; 알고리즘은 각 파일 헤더가 상술). Not Vivado FP IP. HardFloat 는 preserved fp32 유닛(`fp32_adder.v`/`int_to_fp32.v`) TB 와 fp32 복구 라인용으로만 vendored 유지.
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
   **Caveat (2026-07-10)**: the `.xpr` source list is stale — it pre-dates Phase 2a (lists `RMW.v` but not `fp32_adder.v`/`int_to_fp32.v`/`int_to_bf16.v`/`bf16_adder.v`/`fp32_to_bf16_rne.v`/the HardFloat bundles), so **GUI synth/sim is unsupported until the `.xpr` is refreshed**. The bash `sim/run_*.sh` flow (way #3) is authoritative.
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
   bash sim/run_int_to_fp32.sh       # int_to_fp32 unit TB (fp32, preserved at HEAD)
   bash sim/run_fp32_adder.sh        # fp32_adder unit TB (fp32, preserved at HEAD)
   bash sim/run_fp32_to_bf16_rne.sh  # fp32->bf16 RNE narrow (70012 vectors; preserved unit)
   bash sim/run_int_to_bf16.sh       # int_to_bf16 v3 native (32360 vectors: neg scales +
                                     #   subnormal boundary + full-range INT32)
   bash sim/run_bf16_adder.sh        # bf16 adder v3 native (200025 vectors vs ml_dtypes,
                                     #   triple-DUT: default/RMW/all-comb)
   bash sim/run_rmw.sh               # full RMW vector TB (113 cases, bf16)
   bash sim/run_top_elab.sh          # gemm_sram_top elab-only smoke
   bash sim/run_integration_smoke.sh # TB zero-prime + dump (no GEMM driving)

   # MXP_Tools Python 단위 검증 (compare / hwio / quant / gemm)
   cd MXP_Tools && python -m pytest tests/ -q   # 74 cases, ~0.3 s

   # Integration (end-to-end vs MXP_Tools golden)
   bash sim/run_integration_one.sh <LABEL> <A_PREC> <B_PREC>   # 1 mode (e.g. "A8_B8" 8 8)
   bash sim/run_integration_sweep.sh                            # all 9 modes serial
   bash sim/run_integration_parallel.sh                         # print parallel dispatch guide
   ```

   **Full RMW vector test** — `run_rmw.sh` 가 rmw-gen 을 자동 수행하므로 수동 선행
   불필요 (2026-07-13 /review 라운드에서 stale-vector 함정 제거):

   ```bash
   bash sim/run_rmw.sh             # expect: rmw_tb: ALL 113 TESTS PASSED
   # (내부적으로: cd MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0)
   ```

   **Integration sweep** is self-contained — invokes `gen / emit / ref` internally then `run_integration_one.sh` + `compare`. Last line on success: `ALL 9 MODES PASSED`. Runtime **633s ≈ 10.5 min** for the full 9-mode sweep (32-RMW col-parallel; 이전 1-RMW sweep ~20–25 min 대비 약 2× 단축).

   To author a new TB, copy the pattern from any existing `sim/run_*.sh`. Required directive that **must** appear in both RTL and TB or XSim errors out: `` `timescale 1ns/1ps ``.

   **Gotcha — `-testplusarg "K=V"` under Git Bash**: the `=` is stripped before xsim sees it. Wrap with `cmd //c "xsim ... -testplusarg \"K=V\""` — the integration scripts already do this.

When parameter overrides are needed at sim time, **do not use `xelab -generic_top "P=V"`** — there's a cmd.exe quoting bug under Git Bash on Windows that strips the `=`. Use `` `ifdef `` in the source + `xvlog -d FLAG` instead. The SRAM TBs already use this pattern:

- Leaf TB (`sram_1rw_tb.v`): `USE_PIPELINE`, `RUN_INIT_TEST`.
- Banked TB (`sram_1rw_banked_tb.v`): `USE_PIPELINE`, `USE_SEQUENTIAL` (selects `BANK_STRATEGY="SEQUENTIAL"` vs default `"INTERLEAVED"`).

**TB sizing convention.** Use a small config for fast sim (banked TB uses 4 banks × 64 depth × 32-bit), full-size parameters only via the elab sweep (`../sram/sim/run_wrapper_elab.sh`). When writing this project's TBs, follow the same pattern — do not run a 2 MB sim by default.

## Verification helper: `MXP_Tools/` (Python, local subdir)

**Provenance**: fork of upstream `~/Desktop/Desktop/MXP_Tools`. 사용자가 업스트림에서 버그 픽스를 하면 프로젝트 사본으로 옮겨야 함 (절차는 메모리 `reference_mxp_tools_upstream.md` 참고). 프로젝트 전용 추가분 — `mxp_tools/rmw_gen.py` (**bf16-semantics rewrite**, Phase 2b), cli 의 `rmw-gen` 서브커맨드, hwio 의 `interleaved_row_major_16bank` + `interleaved_row_major_32bank` mapping + `read_writememh_bf16` (16b word → fp32 exact upcast), `tests/test_hwio_interleaved.py` + `tests/test_hwio_bf16.py` — 는 머지 시 반드시 보존.

**pytest 슈트** (`tests/`, 74 cases) 는 단위 동작 (NaN/Inf 처리, `@addr` 파싱, gather_banks duplicate detection, emit shape validation, MX quant edge cases, bf16 reader) 을 0.3 초 만에 검증. RTL sim 전에 Python tool 변경 검증용으로 우선 돌릴 것.

Generates HW inputs and SW golden GEMM. Typical invocation (the sweep script already wraps these):

```bash
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8                        # emits all 3 precs
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8 --accum bf16   # SW golden (bf16)
# run HW → produces work/A8_B8/hw_out/bank{0..31}.mem ($writememh)
BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..31})
python -m mxp_tools compare --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8_bf16.npz \
                            --hw-banks ${BANKS} \
                            --layout interleaved_row_major_32bank
```

**Naming gotcha (was a Task 8 silent-bug source)**: MXP_Tools' `--prec-a` = WEIGHT precision = our plusarg `B_PREC` (weight is the bit-serial first operand in `mxint_gemm_golden(A, ..., prec_A=weight)`). `--prec-b` = ACTIVATION precision = our `A_PREC`. The resulting `.npz` filename slot order is `C_sw_mxint{prec_a}_mxint{prec_b}.npz` = `C_sw_mxint{B_PREC}_mxint{A_PREC}.npz`. Symmetric modes (A2_B2, A4_B4, A8_B8) hide the bug; asymmetric modes catch it as all-zero HW dumps. The sweep + per-mode scripts already use the corrected convention.

HW dump = one `$writememh` file per SRAM bank (32 files). Mapping callable: `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_32bank`. Numerical contract: HW output words are interpreted as **16-bit bf16** bit patterns (RMW dequantizes INT→BF16 before storing); `read_writememh_bf16` upcasts each exactly to fp32 via `uint32(bf16_bits) << 16`.

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
| RMW (L_CONV, L_ADD) | (2, 3), total 5 cyc — 내부 분배 S1..S5 (블록당 1단, 위 Latency budget 참조; 2026-07-13 부터 블록 = native 손코딩) | timing closure phase |
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
        RMW.v                          # bf16 RMW unit (int_to_bf16 + delay + bf16_adder), 16-bit psum
        int_to_bf16.v                  # INT32 + 9-bit scale -> bf16 (v3 native: LZC32 + 8b RNE r8
                                       #   -> exp+scale -> subnormal GRS rounder; HardFloat-free)
        bf16_adder.v                   # bf16 add (v3 native: align/GRS -> add/sub -> normalize -> RNE,
                                       #   11~12b internal; HardFloat-free)
        fp32_to_bf16_rne.v             # IEEE fp32 -> bf16 RNE narrow -- PRESERVED (v2-era, own 70012 TB)
        int_to_fp32.v                  # fp32 unit RTL, preserved at HEAD (not instantiated by bf16 top)
        fp32_adder.v                   # IEEE-754 FP32 adder (HardFloat) -- PRESERVED (own TB + fp32 recovery)
    imports/Desktop/MXP/...            # MXP compute RTL (Accumulator_Col.v has IMPLICIT_total patch)
    imports/Desktop/sram/rtl/...       # SRAM RTL (sram_1rw + sram_1rw_banked)
tb/
    gemm_sram_top_tb.v                 # Integration TB (9-mode plusarg-driven)
    rmw_tb.v (bf16), int_to_bf16_tb.v,
    bf16_adder_tb.v, fp32_to_bf16_rne_tb.v,   # bf16 unit testbenches
    int_to_fp32_tb.v, fp32_adder_tb.v,        # fp32 unit testbenches (preserved)
    rmw_smoke_tb.v, accumulator_col_elab.v    # Unit testbenches
sim/
    runner.py                          # Python orchestrator (VSCode "Run" 호환); auto viz → work/<LABEL>/result.png
    gen_mixed.py                       # mixed-prec 입력/golden 생성 + inputs_mixed.npz (for viz Row 1)
    run_top_elab.sh                    # gemm_sram_top elab smoke
    run_integration_smoke.sh           # TB zero-prime + dump (no GEMM driving)
    run_integration_one.sh             # 1 mode end-to-end (<LABEL> <A_PREC> <B_PREC>)
    run_integration_sweep.sh           # 9-mode serial sweep + compare gate
    run_integration_parallel.sh        # 9-mode parallel dispatch guide (print template)
    run_mixed_one.sh                   # mixed-prec 단발 (random / uniform / K-tile via env vars)
    run_rmw*.sh, run_int_to_bf16.sh, run_bf16_adder.sh, run_fp32_to_bf16_rne.sh,
    run_int_to_fp32.sh, run_fp32_adder.sh, clean.sh
    bf16_vectors.py                    # oracle for the bf16 unit TBs (ml_dtypes cross-check)
third_party/berkeley-hardfloat/        # Vendored HardFloat (HardFloatBundle.v [fp32] +
                                       #   HardFloatBundle_bf16.v + VENDORING.md). 2026-07-13 native
                                       #   재작성 이후 bf16 데이터패스 미사용 — preserved fp32 유닛 TB 전용
MXP_Tools/                             # Python verification toolkit (fork of ~/Desktop/Desktop/MXP_Tools)
    pyproject.toml                     # mxp-tools entry point + pytest config
    mxp_tools/                         # gen / emit / ref / compare / viz / rmw-gen subcommands
    tests/                             # 74 pytest cases (compare/gemm/hwio/quant + interleaved 16/32 + bf16)
        test_compare.py                # NaN/Inf-aware diff_3way (upstream)
        test_gemm.py                   # mxint_gemm_golden (upstream)
        test_hwio.py                   # @addr writememh, dup-write, emit shape, LF-only (upstream)
        test_quant.py                  # MX quant edge cases + NaN/Inf rejection (upstream)
        test_hwio_interleaved.py       # interleaved_row_major_{16,32}bank round-trip (project-only)
        test_hwio_bf16.py              # read_writememh_bf16 16b->fp32 upcast (project-only, Phase 2b)
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
