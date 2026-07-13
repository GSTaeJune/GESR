# `sources_1/new/` — 프로젝트-로컬 RTL

## 개요

이 폴더는 **프로젝트-로컬 RTL** 이다 — `gemm_sram` 이 직접 소유하고 자유롭게
편집하는 소스. RMW BF16 데이터패스의 글루와 bf16 primitive 가 여기 산다.

옆의 두 디렉토리는 성격이 다르니 주의:

- `../imports/Desktop/{MXP,sram}/...` = **업스트림 사본** (Vivado import). MXP compute
  엔진과 SRAM leaf 는 sister repo 가 source of truth. **여기서 in-place 수정 금지** —
  알고리즘/마이크로아키텍처를 바꿀 일이 있으면 업스트림(`../MXP/`, `../sram/`)에서
  고치고 재-import 한다.
- `../../../../third_party/berkeley-hardfloat/` = **vendored HardFloat**. 아래 "HardFloat
  번들" 참고. 역시 손대지 않는다 (재-vendor 절차는 `docs/hardfloat-setup.md`).

## 데이터패스

```
                              per-col (j = 0..31), col-parallel
                              ┌─────────────────────────────────────────────┐
  GEMM (32×32 bit-serial SA)  │  RMW[j]                                      │
  ─────────────────────────  │                                              │
   in_GEMM  ─ INT32 (32b) ───┼─► int_to_bf16 ─ bf16 ─┐                       │
   scale    ─ s9 (9b)    ────┤   (S1, 1reg)          ├─► bf16_adder ─ bf16 ─┐│
                              │                       │   ┌──────────────┐  ││
   Q[j] (SRAM read) ─ bf16 ──┼─► sram_dly[1] ─────────┘   │ fp32_adder + │  ││
                    (16b)     │   (in_SRAM 정렬)           │ fp32→bf16 RNE│  ││
                              │                            └──────────────┘  ││
                              │                            (S2..S5)          ││
                              └───────────────────────────────────────────┬─┘│
                                                                           │  │
   out_RMW ─ bf16 (16b) ───────────────────────────────────────────────┐  │  │
                                                                        ▼  ▼  ▼
                              sram_1rw_banked_mp  (32 bank × 1024 × 16b, bf16 psum)
                                 col j ─► bank j  (충돌 0)

  폭 표기:  in_GEMM = INT32 32b  ·  scale = 9b signed  ·  나머지 psum 경로 = bf16 16b
  지연:     RMW 전체 = L_CONV(2) + L_ADD(3) = 5 cycle — 외부 계약 불변 (Phase 2c 재배치)
  단 배치:  S1 INToRecFN+exp-add / S2 FNFromRecFN+narrow / S3 RecFNFromFN(widen)
            / S4 AddRecFN / S5 FNFromRecFN+narrow → out_RMW 는 레지스터 출력
            (내부 분배: int_to_bf16 1단 + bf16_adder 4단(L_IN/피연산자/L_SUM/L_OUT);
             주요 FP 블록당 정확히 1단 — SRAM D 까지의 조합 꼬리 제거)
```

첫 K-tile 은 `sram_D_use_zero=1` 로 SRAM 을 0x0000 으로 zero-prime 한 뒤,
이후 스텝은 `=0` 으로 RMW 결과를 write-back 한다 (누적 running sum).

## 모듈 맵

| 파일 | 역할 (한 줄) | 인스턴스하는 것 | 검증 스크립트 → 기대 출력 | 상태 |
|---|---|---|---|---|
| `gemm_sram_top.v` | GEMM+RMW[32]+SRAM 통합 최상위 (pure structural, FSM 없음) | GEMM, RMW×32, sram_1rw_banked_mp | `run_top_elab.sh` → elab OK · `run_integration_sweep.sh` → `ALL 9 MODES PASSED` | active |
| `GEMM.v` | MXP 비트-시리얼 SA 의 얇은 TOP 래퍼 | SystolicArray, station×32, Accumulator_Col×32 | (통합 sweep 로만) | active |
| `RMW.v` | Read-Modify-Write 단위 (변환→덧셈), bf16 | int_to_bf16, bf16_adder | `run_rmw.sh` → `ALL 113 TESTS PASSED` | active |
| `int_to_bf16.v` | INT32 + s9 scale → bf16 (v2) | INToRecFN_i32_e8_s8, FNFromRecFN_wrapper, fp32_to_bf16_rne | `run_int_to_bf16.sh` → `ALL 32312 TESTS PASSED` | active |
| `bf16_adder.v` | bf16 덧셈 (fp32 도메인 계산 후 RNE narrow) | fp32_adder, fp32_to_bf16_rne | `run_bf16_adder.sh` → `ALL 200005 TESTS PASSED` | active |
| `fp32_to_bf16_rne.v` | fp32 → bf16 RNE (순수 조합, subnormal 포함) | — (leaf) | `run_fp32_to_bf16_rne.sh` → `ALL 70012 TESTS PASSED` | active |
| `fp32_adder.v` | IEEE-754 fp32 덧셈 (HardFloat AddRecFN) | RecFNFromFN_wrapper×2, AddRecFN, FNFromRecFN_wrapper | `run_fp32_adder.sh` → 단위 TB PASS | active (bf16_adder 경유) |
| `int_to_fp32.v` | INT32 + s9 scale → fp32 (fp32 시절 변환단) | INToRecFN_i32_e8_s24, FNFromRecFN_wrapper | `run_int_to_fp32.sh` → 단위 TB PASS | **preserved** |
| `sram_1rw_banked_mp.v` | per-bank 포트 노출 1RW SRAM (32 bank, bf16 16b) | sram_1rw (leaf) ×NUM_BANKS | (통합 sweep 로만) | active |

`int_to_fp32.v` 는 어디에도 인스턴스되지 않는다 — Phase 2b 에서 RMW 변환단이
`int_to_bf16` 로 교체되며 데이터패스에서 빠졌다. HEAD 에 남겨둔 이유는 fp32 복구
태그 `fp32-rmw-final` 의 앵커 + 단위 TB 회귀 유지. **무변경 보존.**

## HardFloat 번들

Berkeley HardFloat 를 두 파일로 vendored (`third_party/berkeley-hardfloat/`):

- `HardFloatBundle.v` — fp32 계열: `AddRecFN`, `FNFromRecFN_wrapper`,
  `RecFNFromFN_wrapper`, `INToRecFN_i32_e8_s24`, ...
- `HardFloatBundle_bf16.v` — bf16 계열: `INToRecFN_i32_e8_s8`,
  `FNFromRecFN_bf16_wrapper`.

두 번들은 모듈명이 서로소(name-disjoint)라 함께 컴파일된다. **`AddRecFN(8,8)`
(bf16 폭 가산기) 는 elaborate 되지 않아** bf16 덧셈은 fp32 도메인 우회를 쓴다
(`bf16_adder` 참고). `FNFromRecFN_bf16_wrapper` 는 v1 의 int→bf16 환원용이었으나,
그 denormalization 이 RNE 없이 TRUNCATE 라 subnormal 이 1 ULP 낮게 나오는 문제로
**현재 미사용** (번들에는 보존, `int_to_bf16.v` 헤더에 폐기 사유 상술).

## fp32 데이터패스 복구

bf16 이전의 fp32 RMW 데이터패스로 되돌리려면 복구 태그를 checkout:

```bash
git checkout fp32-rmw-final -- gemm_sram.srcs/sources_1/new/
```

(RMW 가 `int_to_fp32` + `fp32_adder` 를 직접 인스턴스하고, SRAM/psum 폭이 32b 이던
상태. 현재 HEAD 의 `int_to_fp32.v`/`fp32_adder.v` 는 이 복구를 위해 보존돼 있다.)

## 합성 실측 (per-unit OOC)

Kintex-7 160T-1, 250MHz 제약, 2026-07-12 측정:

| 유닛 | LUT | FF | Fmax | 비고 |
|---|---|---|---|---|
| RMW (bf16) | 768 | 99 | ~153 MHz | fp32 시절 1183 / 193 |
| RMW (fp32, 참고) | 1183 | 193 | ~157 MHz | — |
| psum SRAM bank 16b | 293 | — | — | LUTRAM. 32b 시절 581 |

**주의:** 위 수치는 Kintex-7 160T-1 기준 per-unit OOC 합성값이다. bf16/fp32 RMW 둘 다
250MHz 제약에 못 미친다 (~153 / ~157 MHz) — **timing closure 는 알려진 향후 과제**.
프로젝트 타겟 디바이스(`xc7vx485`)나 배치·배선 후 값과는 다를 수 있다.

## 검증 전체 게이트

빠른 단위 배터리 (elaborate + 각 primitive, foreground):

```bash
bash sim/run_top_elab.sh && bash sim/run_rmw.sh && \
bash sim/run_int_to_bf16.sh && bash sim/run_bf16_adder.sh && \
bash sim/run_fp32_to_bf16_rne.sh && bash sim/run_int_to_fp32.sh && \
bash sim/run_fp32_adder.sh
```

종단(end-to-end) 게이트 — MXP_Tools bf16 golden 과 9-mode bit-exact:

```bash
bash sim/run_integration_sweep.sh     # 마지막 줄: ALL 9 MODES PASSED
```
