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
   scale    ─ s9 (9b)    ────┤   (S1·S2, 1reg)       ├─► bf16_adder ─ bf16 ─┐│
                              │                       │   (native 3-블록:    ││
   Q[j] (SRAM read) ─ bf16 ──┼─► sram_dly[0:0] ───────┘    정렬/가감산 →     ││
                    (16b)     │   (in_SRAM 정렬)            정규화 → RNE)    ││
                              │                            (S3..S5)          ││
                              └───────────────────────────────────────────┬─┘│
                                                                           │  │
   out_RMW ─ bf16 (16b) ───────────────────────────────────────────────┐  │  │
                                                                        ▼  ▼  ▼
                              sram_1rw_banked_mp  (32 bank × 1024 × 16b, bf16 psum)
                                 col j ─► bank j  (충돌 0)

  폭 표기:  in_GEMM = INT32 32b  ·  scale = 9b signed  ·  나머지 psum 경로 = bf16 16b
  지연:     RMW 전체 = L_CONV(2) + L_ADD(3) = 5 cycle — 외부 계약 불변 (Phase 2c 재배치)
  단 배치:  S1 i2b[F] |int|→LZC32→8b RNE / S2 i2b[B] scale·denorm·인코딩 /
            S3 가산기[A] 정렬·가감산 / S4 가산기[B] 정규화 / S5 가산기[C] RNE·패킹
            → out_RMW 는 레지스터 출력 (SRAM D 까지의 조합 꼬리 제거)
            (내부 분배: int_to_bf16 1단 + bf16_adder 4단(L_IN/L_ADD/L_SUM/L_OUT);
             2026-07-13 native 재작성 — 두 유닛 모두 손코딩 bf16 폭, HardFloat-free.
             내부 데이터패스 11~12b 그리드, 알고리즘 설명은 각 파일 헤더 참조)
```

첫 K-tile 은 `sram_D_use_zero=1` 로 SRAM 을 0x0000 으로 zero-prime 한 뒤,
이후 스텝은 `=0` 으로 RMW 결과를 write-back 한다 (누적 running sum).

## 모듈 맵

| 파일 | 역할 (한 줄) | 인스턴스하는 것 | 검증 스크립트 → 기대 출력 | 상태 |
|---|---|---|---|---|
| `gemm_sram_top.v` | GEMM+RMW[32]+SRAM 통합 최상위 (pure structural, FSM 없음) | GEMM, RMW×32, sram_1rw_banked_mp | `run_top_elab.sh` → elab OK · `run_integration_sweep.sh` → `ALL 9 MODES PASSED` | active |
| `GEMM.v` | MXP 비트-시리얼 SA 의 얇은 TOP 래퍼 | SystolicArray, station×32, Accumulator_Col×32 | (통합 sweep 로만) | active |
| `RMW.v` | Read-Modify-Write 단위 (변환→덧셈), bf16 | int_to_bf16, bf16_adder | `run_rmw.sh` → `ALL 113 TESTS PASSED` | active |
| `int_to_bf16.v` | INT32 + s9 scale → bf16 (v3 native: LZC32 + 8b RNE ×2) | — (leaf) | `run_int_to_bf16.sh` → `ALL 32360 TESTS PASSED` | active |
| `bf16_adder.v` | bf16 덧셈 (v3 native: 정렬→가감산→정규화→RNE, 11~12b) | — (leaf) | `run_bf16_adder.sh` → `ALL 200025 TESTS PASSED` | active |
| `fp32_to_bf16_rne.v` | fp32 → bf16 RNE (순수 조합, subnormal 포함) | — (leaf) | `run_fp32_to_bf16_rne.sh` → `ALL 70012 TESTS PASSED` | **preserved** |
| `fp32_adder.v` | IEEE-754 fp32 덧셈 (HardFloat AddRecFN) | RecFNFromFN_wrapper×2, AddRecFN, FNFromRecFN_wrapper | `run_fp32_adder.sh` → 단위 TB PASS | **preserved** |
| `int_to_fp32.v` | INT32 + s9 scale → fp32 (fp32 시절 변환단) | INToRecFN_i32_e8_s24, FNFromRecFN_wrapper | `run_int_to_fp32.sh` → 단위 TB PASS | **preserved** |
| `sram_1rw_banked_mp.v` | per-bank 포트 노출 1RW SRAM (32 bank, bf16 16b) | sram_1rw (leaf) ×NUM_BANKS | (통합 sweep 로만) | active |

`int_to_fp32.v`/`fp32_adder.v`/`fp32_to_bf16_rne.v` 는 어디에도 인스턴스되지 않는다 —
2026-07-13 native 재작성으로 bf16 데이터패스가 HardFloat/fp32 유닛 없이 자립하면서
데이터패스에서 빠졌다. HEAD 에 남겨둔 이유는 fp32 복구 태그 `fp32-rmw-final` 의
앵커 + 각자의 단위 TB 회귀 유지. **무변경 보존.**

## HardFloat 번들

Berkeley HardFloat 를 두 파일로 vendored (`third_party/berkeley-hardfloat/`):

- `HardFloatBundle.v` — fp32 계열: `AddRecFN`, `FNFromRecFN_wrapper`,
  `RecFNFromFN_wrapper`, `INToRecFN_i32_e8_s24`, ...
- `HardFloatBundle_bf16.v` — bf16 계열: `INToRecFN_i32_e8_s8`,
  `FNFromRecFN_bf16_wrapper`.

두 번들은 모듈명이 서로소(name-disjoint)라 함께 컴파일된다. **2026-07-13 native
재작성 이후 bf16 데이터패스는 HardFloat 를 전혀 쓰지 않는다** — 번들이 남아 있는
이유는 preserved fp32 유닛(`fp32_adder`/`int_to_fp32`)의 단위 TB 와 fp32 복구 라인.
히스토리 참고: `AddRecFN(8,8)` (bf16 폭 가산기) 가 elaborate 되지 않아 Phase 2a~2c
의 bf16 덧셈은 fp32 도메인 우회를 썼고, v1 의 `FNFromRecFN_bf16_wrapper` 는
denormalization 이 RNE 없이 TRUNCATE 라 subnormal 1 ULP 결함으로 폐기됐다
(스펙 §6.1~6.2). native v3 는 두 문제 모두 원천적으로 벗어난다.

## fp32 데이터패스 복구

bf16 이전의 fp32 RMW 데이터패스로 되돌리려면 복구 태그를 checkout:

```bash
git checkout fp32-rmw-final -- gemm_sram.srcs/sources_1/new/
```

(RMW 가 `int_to_fp32` + `fp32_adder` 를 직접 인스턴스하고, SRAM/psum 폭이 32b 이던
상태. 현재 HEAD 의 `int_to_fp32.v`/`fp32_adder.v` 는 이 복구를 위해 보존돼 있다.)

## 합성 실측 (per-unit OOC)

Kintex-7 160T-1, 250MHz(4ns) 제약, 2026-07-13 native v3 측정 (상세 표는 `rtl/README.md`):

| 유닛 | LUT cell (동일 metric 시리즈) | FF | WNS (pre-place) | 비고 |
|---|---|---|---|---|
| RMW (bf16 native v3) | **505** (Slice LUT 404) | 139 | **-1.64** | 최악 S3 = logic 2.28 + route(추정) 3.36 |
| RMW (bf16 HardFloat, 2c) | 744 | 148 | -8.79 | 병목이던 AddRecFN logic 4.1ns — native 로 소멸 |
| RMW (fp32 시절, 참고) | 1183 | 193 | (반쪽 측정) | — |
| psum SRAM bank 16b | 293 | — | — | LUTRAM. 32b 시절 581. SRAM RTL 은 sim 전용 |

**주의:** 위 수치는 Kintex-7 160T-1 기준 per-unit OOC **pre-place** 합성값 — 백엔드는
ASIC PNR 이고 Vivado 수치는 상대 프록시다. native v3 의 남은 위반은 route 추정분이
지배해 (logic 2.28ns < 4ns) 배치 후 250MHz 는 가시권으로 판단.

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
