# 다음 세션 kickoff — 통합 후 단계

작성: 2026-05-15, GEMM ↔ RMW ↔ SRAM 통합 완료 직후.

---

## 지금 어디까지 왔나

**완성 — phase 1 (RMW unit, 2026-05-14)**
- HardFloat vendoring (`third_party/berkeley-hardfloat/`)
- `int_to_fp32.v`, `fp32_adder.v`, `RMW.v` 단위 산술기 (latency 5 cyc)
- `Accumulator_Col.v` IMPLICIT_total 패치
- `MXP_Tools/mxp_tools/rmw_gen.py` 골든 벡터 생성
- 검증: `bash sim/run_rmw.sh` → 71/71 PASS

**완성 — phase 2 (시스템 통합, 2026-05-15)**
- `gemm_sram_top.v` (pure structural wrapper, GEMM + RMW + sram_1rw_banked + 1-bit zero-prime mux)
- `tb/gemm_sram_top_tb.v` (단일 9-mode plusarg-driven TB, 870+ lines)
- `sim/run_integration_one.sh` / `sweep.sh` / `parallel.sh`
- `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_16bank` 매핑 + `compare --layout` 분기
- `int_to_fp32.v` zero-passthrough 버그 fix (in_int=0 + scale≠127 케이스)
- TOGGLE_VAL 확정값 (A8W4=22, A4W8=23 — `precision_modes_protocol.md` §3 의 TBD 해결)
- 검증: `bash sim/run_integration_sweep.sh` → **9/9 PASS** (A,B ∈ {2,4,8}, 각 128×128 bit-exact)

**완성 — 부수 작업 (TB 한글 주석, 2026-05-15)**
- TB 6개 (`tb/*.v`) 헤더와 본문 주석 한글화 — 파일별 "검증 목적 / 검증 내용 / 동작 의도" 명시.
  코드 로직 무변경 (comment-only), 단위 + 통합 회귀 PASS 후 commit. HEAD = `92d6b1e`.
- 다음 세션 진입 시 임의의 TB 를 열어서 헤더만 봐도 그 TB 가 뭘 검증하는지 즉시 파악 가능.

**잠정 결정값** (재검토 트리거는 `CLAUDE.md` "Settled — with re-visit triggers" 표 참고)
- RMW instance: **1개** (TB-side mode-aware dispatch)
- Loop order: **K-outermost** (`n_t → k_t → m_t → o`)
- 워크로드: **M=K=N=128**
- Bank strategy: **INTERLEAVED** (16 banks × 1024 word 사용 = 16384 elements)
- SRAM PIPELINE: **0** (read latency 1 cyc)

---

## 다음 단계 후보 (사용자 선택)

### 1. Throughput 확장 — RMW 다중 instance

현재 RMW 1개로 65536 fire 를 직렬 dispatch. K-tile / m-tile 별 fire 가 burst 로 몰리는 구간에서 SRAM bank-level 병목 발생 가능. 후보 변경:

- RMW 2개로 늘려서 A2 mode 의 4-lane × 32-col fire 를 두 갈래로 분산
- RMW 4개 + per-col 가벼운 arbiter
- 각 옵션의 sim 시간 / 합성 후 cycle count 비교

(영향 파일: `gemm_sram_top.v` 구조 변경, `tb/gemm_sram_top_tb.v` dispatcher 분기, 그리고 spec § 9 "RMW instance 수" 항목 갱신)

### 2. Timing closure — Vivado 합성

- target: 250 MHz @ xc7vx485 (MXP standalone 의 closure 결과 답습)
- 합성 진입 전 readiness check: `gemm_sram.xpr` open → "Run Synthesis" 무에러
- 잠재적 critical path: HardFloat `addRecFN`, GEMM 의 station chain, SRAM 의 1RW write
- 새 spec/plan 필요 (timing closure 는 별도 workflow)

### 3. Future scope — SRAM 을 weight (B input) 저장소로

현재 `tb/gemm_sram_top_tb.v` 는 `$readmemh` 로 `in_b` 를 직접 driving. HW deployment 시 B 도 SRAM 에서 읽도록 변경하려면:

- SRAM 한 bank 를 weight 영역으로 reserve (또는 별도 SRAM instance)
- TB 의 LOAD 단계에서 weight 를 SRAM 에 prime (zero-prime mux 와 같은 구조 활용)
- DRIVE 단계는 매 cycle SRAM read → `in_b` 라우팅

별도 spec 작성 필요. 통합 spec § 1 "Future scope" 에 명시.

### 4. 9-mode 자동 회귀 CI

`sim/run_integration_sweep.sh` 를 CI (GitHub Actions / 로컬 cron) 로 묶음. 변경 사항이 RTL / TB / MXP_Tools 어디든 닿을 때마다 9/9 PASS 자동 회귀.

- 20-25 min sweep — nightly 가 자연스러움
- 또는 modes={A8_B8, A2_B2} 만 빠르게 PR-time 에 돌리고, 나머지 7 modes 는 nightly

---

## 새 spec/plan 작성 흐름

위 4가지 중 하나로 진행 결정 시:

1. `superpowers:brainstorming` → 결정 사항 정리 → `docs/superpowers/specs/2026-MM-DD-<topic>.md`
2. `superpowers:writing-plans` → `docs/superpowers/plans/2026-MM-DD-<topic>.md`
3. `superpowers:executing-plans` (혹은 `superpowers:subagent-driven-development`) 로 진행

이전 두 phase (RMW unit, 통합) 가 같은 흐름으로 굴러갔음. 답습하면 됨.
