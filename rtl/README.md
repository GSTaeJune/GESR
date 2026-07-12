# rtl/ — 읽기용 RTL 정리 사본 (SNAPSHOT COPY)

> **이 폴더는 브라우징 편의를 위한 사본이다. 원본이 진실(source of truth)이며,
> 어떤 수정도 반드시 원본에서 할 것.** 원본 수정 후 `bash rtl/refresh.sh` 로 재복사.
>
> | 원본 위치 | 내용 |
> |---|---|
> | `gemm_sram.srcs/sources_1/new/` | 프로젝트 로컬 RTL (자유 수정) — 상세 문서는 그 폴더의 README.md |
> | `gemm_sram.srcs/sources_1/imports/.../MXP/...` | MXP compute (수정은 `../MXP/` 업스트림에서) |
> | `gemm_sram.srcs/sources_1/imports/.../sram/rtl/` | SRAM (수정은 `../sram/` 업스트림에서) |
>
> 시뮬/합성 스크립트는 전부 **원본 경로**를 컴파일한다 — 이 폴더는 어떤 빌드에도 안 물림.

## 읽는 순서 (폴더 번호 = 권장 순서)

```
1_top/         gemm_sram_top.v   ── 전체 묶음 (pure structural, FSM 없음 — TB 가 컨트롤러)
               GEMM.v            ── MXP TOP 래퍼 (32x32 bit-serial systolic; in_a=weight!)
2_rmw_bf16/    RMW.v             ── psum 누산 유닛 (x32 col-parallel), L_CONV+L_ADD=5cy
               int_to_bf16.v     ── INT32+scale -> bf16 (v2: fp32 도메인 shift + 단일 RNE + flush)
               bf16_adder.v      ── bf16+bf16 (fp32 도메인 가산 -> RNE narrow)
               fp32_to_bf16_rne.v── fp32 -> bf16 RNE narrower (조합, subnormal 포함)
3_fp32_units/  fp32_adder.v      ── ACTIVE (bf16_adder 내부에서 사용)
               int_to_fp32.v     ── PRESERVED (데이터패스 미사용; fp32 복구 태그 + 단위 TB 용)
4_sram/        sram_1rw_banked_mp.v ── 32-bank per-bank 포트 래퍼 (현 사용, DATA_WIDTH=16)
               sram_1rw.v           ── leaf 1RW (CEB/WEB active-low, WMASK active-high)
               sram_1rw_banked.v    ── 16-bank mux 형 래퍼 (다른 caller 용, 본 top 미사용)
5_gemm_core/   SystolicArray / PE_feeder / PE_naive / station / adder_lane /
               Accumulator / Accumulator_Col  ── MXP compute (SA row=K, col=N, cycle=M!)
```

## 데이터패스 (한 col 기준; 실제로는 x32 병렬, col j -> bank j)

```
GEMM ──INT32 psum──► RMW ──┬─ int_to_bf16 ──fp_a(bf16)──┐
     ──9b scale────►       │                             ├─ bf16_adder ──► out_RMW(bf16 16b)
                           └─ in_SRAM(bf16) ─sram_dly────┘        │
                                    ▲                              ▼
                                    └────── sram_1rw_banked_mp (1024 x 16b/bank) ◄──
```

- HardFloat 번들 2개 co-compile: `third_party/berkeley-hardfloat/HardFloatBundle.v`(fp32) +
  `HardFloatBundle_bf16.v`(INToRecFN_i32_e8_s8). 생성물이라 읽기 비권장 — 여기 복사 안 함.
  `FNFromRecFN_bf16_wrapper` 는 denorm TRUNCATE 결함으로 미사용(번들에만 보존).
- fp32 시절 전체 복구: `git checkout fp32-rmw-final`

## 검증 게이트 (원본 기준 실행)

| 대상 | 명령 | 기대 |
|---|---|---|
| 전체 (end-to-end) | `bash sim/run_integration_sweep.sh` | `ALL 9 MODES PASSED` (bf16 golden bit-exact) |
| RMW 유닛 | `bash sim/run_rmw.sh` | `ALL 113 TESTS PASSED` |
| int_to_bf16 | `bash sim/run_int_to_bf16.sh` | `ALL 32312 TESTS PASSED` |
| bf16_adder / narrower | `run_bf16_adder.sh` / `run_fp32_to_bf16_rne.sh` | 200005 / 70012 |
| top elab | `bash sim/run_top_elab.sh` | elab clean |

## 합성 실측 (OOC, Kintex-7 160T-1, 250MHz 제약 — V7 라이선스 부재로 대체 측정)

| 유닛 | LUT | FF | Fmax(추정) |
|---|---|---|---|
| RMW (bf16) | 768 | 99 | ~153 MHz |
| RMW (fp32 시절) | 1183 | 193 | ~157 MHz |
| psum SRAM bank 16b | 293 (LUTRAM) | 16 | 250MHz 여유 |
| psum SRAM bank 32b 시절 | 581 | 32 | 동일 |

x32 합계 RMW+SRAM 기준 **-40% LUT**. 두 RMW 모두 250MHz 는 미달 — timing closure 는 향후 과제.
스크립트: `work/synth_rmw/synth.tcl`
