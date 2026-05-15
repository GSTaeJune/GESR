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

**완성 — MXP_Tools 업스트림 동기화 (2026-05-15)**
- 업스트림 `~/Desktop/Desktop/MXP_Tools` 에서 들어온 버그 픽스 머지:
  - `compare.py` — NaN/Inf-aware 통계 (이제 fully-NaN HW 출력이 "0 mismatches" 로 안 보임)
  - `hwio.py` — `@addr` writememh 파서 (sparse dump 정렬), gather_banks duplicate-write detection,
    `_require_block_multiple` enforcement, LF-only newline 고정
  - `quant.py` — NaN/Inf 입력 거부 (이전엔 downstream OverflowError 로 묻힘)
  - `cli.py` — `cmd_emit` 의 ab_fp32.npz shape validation
  - `viz.py` — inf 라벨 표시 개선
  - `gemm.py` — bit-exactness contract docstring 추가
- 프로젝트 전용 추가분 보존: `rmw_gen.py`, cli `rmw-gen`, hwio `interleaved_row_major_16bank`, `test_hwio_interleaved.py`
- pytest 슈트 신규 도입 — `tests/test_{compare,gemm,hwio,quant}.py` + 기존 interleaved 테스트 = **43/43 PASS**
- `pyproject.toml` 추가 — `pip install -e .` 가능, pytest config 도 여기서 잡힘
- 재검증: pytest 43/43 PASS → `bash sim/run_rmw.sh` 71/71 PASS → `bash sim/run_integration_sweep.sh` 9/9 PASS
- 다음 세션에서 업스트림 변경분이 또 들어오면 메모리 `reference_mxp_tools_upstream.md` 참고해서 동일 절차로 동기화.

**완성 — phase 3 (RMW 32× col-parallel, 2026-05-15)**
- `sram_1rw_banked_mp.v` (per-bank port-exposed, NB=32, depth=1024).
- `gemm_sram_top.v` 32-RMW + 32-bank generate.
- `tb/gemm_sram_top_tb.v` 32-stream capture/drain (per-col FIFO + parallel state-machine drain).
- MXP_Tools `interleaved_row_major_32bank` layout (32-bank compare).
- 검증: 9/9 PASS (`bash sim/run_integration_sweep.sh`), wall-clock **633s ≈ 10.5 min** (이전 1-RMW sweep ~20-25 min 대비 약 2× 단축).
- 산출 commit: f102756 (wrapper) → c4ae7fc (MXP_Tools) → 3d21544 (top) → 20f60e6 (TB) → 76bb597 (cleanup) → 326ce1f (sweep).

**잠정 결정값** (재검토 트리거는 `CLAUDE.md` "Settled — with re-visit triggers" 표 참고)
- RMW instance: **32개** (col-parallel, per-bank SRAM port)
- Loop order: **K-outermost** (`n_t → k_t → m_t → o`)
- 워크로드: **M=K=N=128**
- Bank strategy: **per-bank port 노출** (col j → bank j, 충돌 0; NUM_BANKS=32, BANK_DEPTH=1024)
- SRAM PIPELINE: **0** (read latency 1 cyc)

---

## 다음 단계 후보 (사용자 선택)

### ~~1. Throughput 확장 — RMW 다중 instance~~ ✓ 완료 (phase 3)

RMW 32 col-parallel + 32-bank per-bank port 로 완성. sweep wall-clock 633s ≈ 10.5 min (이전 ~20-25 min 대비 2×). 이 항목은 닫힘.

---

### 1. Real-time drain — DRIVE 와 DRAIN 동시 진행

현재 capture/replay 모델 (DRIVE 완료 후 DRAIN 직렬). fire 즉시 per-col RMW → SRAM write 가 overlap 하면 추가 sim time 단축 가능. 필요 변경:

- TB drain loop 를 DRIVE clock 과 interleave (per-col FIFO 는 이미 있음)
- back-pressure 처리 (FIFO full 시 DRIVE stall)

### 2. SRAM multi-port 진짜 활용

현재 `sram_1rw_banked_mp.v` 는 1RW × 32 bank. leaf 를 1R1W / 2RW 로 교체하면 read + write 동시 → latency overlap → throughput 추가 향상 가능.

- `sram_1rw.v` 를 1R1W 로 파생 (upstream sram repo 에서 spec 작성 필요)
- RMW 의 READ 와 다음 WRITE 사이 latency 를 hiding

### 3. Timing closure — Vivado 합성

- target: 250 MHz @ xc7vx485 (MXP standalone 의 closure 결과 답습)
- 합성 진입 전 readiness check: `gemm_sram.xpr` open → "Run Synthesis" 무에러
- 잠재적 critical path: HardFloat `addRecFN`, GEMM 의 station chain, SRAM 의 1RW write
- 새 spec/plan 필요 (timing closure 는 별도 workflow)

### 4. 9-mode 자동 회귀 CI

`sim/run_integration_sweep.sh` 를 CI (GitHub Actions / 로컬 cron) 로 묶음. 변경 사항이 RTL / TB / MXP_Tools 어디든 닿을 때마다 9/9 PASS 자동 회귀.

- 633s sweep (≈ 10.5 min) — nightly 가 자연스러움
- 또는 modes={A8_B8, A2_B2} 만 빠르게 PR-time 에 돌리고, 나머지 7 modes 는 nightly

### 5. Future scope — SRAM 을 weight (B input) 저장소로

현재 `tb/gemm_sram_top_tb.v` 는 `$readmemh` 로 `in_b` 를 직접 driving. HW deployment 시 B 도 SRAM 에서 읽도록 변경하려면:

- SRAM 한 bank 를 weight 영역으로 reserve (또는 별도 SRAM instance)
- TB 의 LOAD 단계에서 weight 를 SRAM 에 prime (zero-prime mux 와 같은 구조 활용)
- DRIVE 단계는 매 cycle SRAM read → `in_b` 라우팅

별도 spec 작성 필요. 통합 spec § 1 "Future scope" 에 명시.

---

## 새 spec/plan 작성 흐름

위 5가지 중 하나로 진행 결정 시:

1. `superpowers:brainstorming` → 결정 사항 정리 → `docs/superpowers/specs/2026-MM-DD-<topic>.md`
2. `superpowers:writing-plans` → `docs/superpowers/plans/2026-MM-DD-<topic>.md`
3. `superpowers:executing-plans` (혹은 `superpowers:subagent-driven-development`) 로 진행

이전 세 phase (RMW unit, 통합, RMW 32× col-parallel) 가 같은 흐름으로 굴러갔음. 답습하면 됨.
