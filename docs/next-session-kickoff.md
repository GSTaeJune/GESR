# 다음 세션 kickoff — 시스템 통합 (GEMM ↔ RMW ↔ SRAM)

작성: 2026-05-14, RMW unit 완성 직후.
용도: 다음 세션이 brainstorming 부터 바로 들어갈 수 있게 컨텍스트 미리 정리.

---

## 지금 어디까지 왔나

**완성 (이전 세션):**
- HardFloat vendoring (`third_party/berkeley-hardfloat/`)
- `int_to_fp32.v`, `fp32_adder.v`, `RMW.v` (FP32 단위 연산기, L_CONV+L_ADD=5 cyc latency)
- `Accumulator_Col.v` IMPLICIT_total 보정 패치
- `MXP_Tools/mxp_tools/rmw_gen.py` (RMW unit 검증용 골든 벡터 생성)
- 전부 standalone 검증 끝 — `bash sim/run_rmw.sh` → 71/71 PASS

**아직 안 한 것 (이번 세션 목표):**
- `RMW.v`는 unit으로만 존재. `GEMM.v` 내부에 instantiate 되지 않았고, SRAM (`sram_1rw_banked.v`)도 instantiate 되지 않음.
- 이걸 하나의 datapath로 연결하는 게 다음 단계.

---

## 통합에서 결정해야 할 4가지 열린 질문

CLAUDE.md "## What's NOT settled" 섹션의 항목 2, 3, 4, 6. 여기 다시 정리:

### Q1. RMW granularity — column당 1개(32) vs lane당 1개(128)?

배경:
- `in_GEMM[31:0]`는 한 lane의 INT32 부분합. 하지만 column 1개의 `out_accumulate`는 60비트인데 모드별로 다르게 패킹됨:
  - **A8**: 1개 21-bit signed (1 lane만 유효)
  - **A4**: 2개 18-bit signed (2 lane)
  - **A2**: 4개 15-bit (4 lane 다 유효)
- `out_scale`도 같은 식으로 4개의 9-bit sub-word로 쪼개져 있음 (`Accumulator_Col.v` 참고).

선택지:
- **per-lane (128 instance)**: 자연스럽게 A2까지 처리됨. A8 모드에선 3/4 instance 놀음. 면적/전력 큼.
- **per-column (32 instance)**: 각 instance가 모드별로 lane을 시간 분할(time-mux)해서 처리. 면적은 1/4지만 컨트롤러 복잡도+throughput 감소.
- 하이브리드 등.

### Q2. Bank ↔ column 매핑

배경:
- `sram_1rw_banked` 기본값: `NUM_BANKS=16`, `BANK_DEPTH=32768` (총 2 MB).
- 32 column을 16 bank에 어떻게 매핑할지. 한 cycle에 한 bank 1 RW만 가능하니 충돌 회피 필수.
- `BANK_STRATEGY="INTERLEAVED"` (LSB로 bank select) vs `"SEQUENTIAL"` (MSB) — `../sram/docs/banking-wrapper-decisions.md` 참고.

### Q3. Fire-timing 스케줄링

배경:
- `out_fire`는 column별로 시점이 다름 (station/accumulator 체인의 chain delay 때문).
- RMW unit latency = 5 cyc, SRAM read 1~2 cyc. 같은 bank로 두 column이 충돌하면 안 됨.
- 옵션: per-column FIFO, 32 parallel RMW + arbiter, serialized scheduler 등.
- Q1 (granularity)이 결정되면 자연스럽게 좁혀짐.

### Q4. First-tile 초기화

배경:
- 어떤 출력 주소에 대한 첫 RMW는 SRAM에 있는 쓰레기값(initialization 안 된 NaN 등)을 읽으면 안 됨.
- 옵션:
  - sim 시작 시 zero-prime (banked wrapper는 INIT_FILE 없음 — `do_write` 루프로 초기화)
  - 첫 read를 게이트해서 dequant 결과를 그대로 write
  - K-tile 카운터로 첫 tile만 특수 처리

---

## 브레인스토밍 진입 시 먼저 읽을 파일

순서대로:

1. `CLAUDE.md` — 특히:
   - "## Top-level wiring (current state vs target)" — 현재 상태와 타깃 그림
   - "## MXP control surface" — out_fire가 column별로 시점 다른 이유, A8/A4/A2 패킹
   - "## SRAM control surface" — read latency, 활성화 polarity, banked wrapper 동작
   - "## What's NOT settled" — 위 4개 질문의 원래 위치
2. `gemm_sram.srcs/sources_1/new/RMW.v` — 헤더 주석에 데이터패스 다이어그램 + 호출자 책임 명시.
3. `gemm_sram.srcs/sources_1/new/GEMM.v` — 현재 unmodified MXP TOP. 어디에 RMW를 끼워넣을지 보려면 필수.
4. `gemm_sram.srcs/sources_1/imports/Desktop/sram/rtl/sram_1rw_banked.v` — banked wrapper 인터페이스.
5. `gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v` — `out_accumulate`/`out_scale` 패킹 + 새로 추가된 IMPLICIT_total 보정.

---

## 브레인스토밍 산출물

`superpowers:brainstorming` 끝나면 다음을 결정해서 spec으로 저장:

| 결정사항 | 어디에 영향 |
|---|---|
| RMW granularity (per-column / per-lane / 하이브리드) | RMW instance 수, 컨트롤러 복잡도 |
| 모드별 lane → RMW 매핑 (특히 A4/A2의 sub-word 디스패치) | RMW 입력 mux + scale sub-word 선택 |
| Bank-column 매핑 + bank strategy | SRAM 주소 생성 로직 |
| 충돌 회피 메커니즘 (FIFO / arbiter / scheduler) | RMW 컨트롤러 FSM |
| First-tile 처리 방식 | 컨트롤러 + 시뮬레이션 init |
| 컨트롤러 위치 (`GEMM.v` 내부 vs 별도 모듈) | 파일 구조 |
| 검증 전략 (unit TB 추가 / 기존 MXP_Tools `compare`로 end-to-end / 둘 다) | TB + run script 추가 분량 |

저장 위치: `docs/superpowers/specs/2026-MM-DD-integration-design.md`
(같은 패턴으로 plan은 `docs/superpowers/plans/2026-MM-DD-integration-implementation.md`)

---

## 진행 흐름 (이전 단계 답습)

1. `superpowers:brainstorming` → spec 작성
2. `superpowers:writing-plans` → plan 작성 (작업을 N개 task로 쪼갬)
3. plan 내부에 "resume protocol" 섹션 명시 (이전엔 CLAUDE.md "Build state"가 그 역할)
4. 구현 시작 — 독립 task는 병렬 dispatch, 종속 task는 sequential
5. 각 task 끝날 때마다 `superpowers:requesting-code-review`로 리뷰
6. 마지막에 CLAUDE.md 정리 + kickoff 문서 다시 갱신 (또는 삭제)

이번 RMW unit이 위 패턴 그대로 했고 잘 굴러갔음. 같은 식으로 가면 됨.
