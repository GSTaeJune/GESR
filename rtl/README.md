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
               int_to_bf16.v     ── INT32+scale -> bf16 (v3 native: LZC32 + 8b RNE x2, leaf)
               bf16_adder.v      ── bf16+bf16 (v3 native: 정렬->가감산->정규화->RNE, 11~12b, leaf)
3_fp32_units/  fp32_adder.v      ── PRESERVED (native 전환으로 데이터패스 미사용)
               int_to_fp32.v     ── PRESERVED (데이터패스 미사용; fp32 복구 태그 + 단위 TB 용)
               fp32_to_bf16_rne.v── PRESERVED (v2 시절 공용 narrow 단; 자체 TB 70012 보존)
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

- **bf16 데이터패스는 HardFloat-free** (2026-07-13 native 재작성): RMW/int_to_bf16/
  bf16_adder 는 leaf 손코딩 Verilog 만으로 자립. HardFloat 번들 2개
  (`third_party/berkeley-hardfloat/HardFloatBundle*.v`)는 preserved fp32 유닛의
  단위 TB 와 fp32 복구 라인용으로만 남아 있다 (생성물이라 읽기 비권장 — 여기 복사 안 함).
- fp32 시절 전체 복구: `git checkout fp32-rmw-final`

## 검증 게이트 (원본 기준 실행)

| 대상 | 명령 | 기대 |
|---|---|---|
| 전체 (end-to-end) | `bash sim/run_integration_sweep.sh` | `ALL 9 MODES PASSED` (bf16 golden bit-exact) |
| RMW 유닛 | `bash sim/run_rmw.sh` (rmw-gen 자동 수행) | `ALL 113 TESTS PASSED` |
| int_to_bf16 | `bash sim/run_int_to_bf16.sh` | `ALL 32360 TESTS PASSED` |
| bf16_adder / narrower | `run_bf16_adder.sh` / `run_fp32_to_bf16_rne.sh` | 200025 / 70012 |
| top elab | `bash sim/run_top_elab.sh` | elab clean |

## 합성 실측 (OOC, Kintex-7 160T-1, 250MHz 제약 — V7 라이선스 부재로 대체 측정)

| 유닛 | LUT cell¹ | FF | 타이밍 (4ns 제약, pre-place 추정) |
|---|---|---|---|
| RMW (bf16, **native v3, 2026-07-13**) | **505** (SRL 0; Slice LUT 404) | 139 | WNS **-1.64** — 최악 = S3(정렬·가감산) 5.64ns = **logic 2.28**(13 lvl) + pre-place route 3.36 |
| RMW (bf16, Phase 2c 재배치, HardFloat) | 744 (SRL 0) | 148 | WNS -8.79 — S4(AddRecFN 단독) est. 12.4ns (logic 4.1 + route 8.3) |
| RMW (bf16, 재배치 전) | 768 (SRL 34) | 99 | "-2.55 (~153MHz)" 는 **반쪽 측정** — AddRecFN 경로가 no_output_delay 로 untimed. 직접 비교 금지 |
| RMW (fp32 시절) | 1183 | 193 | ~157 MHz (동일하게 반쪽 측정) |
| psum SRAM bank 16b | 293 (LUTRAM) | 16 | 250MHz 여유 (참고: SRAM RTL 은 sim 전용, 실물은 CACTI/파운드리 매크로) |
| psum SRAM bank 32b 시절 | 581 | 32 | 동일 |

¹ LUT cell = SYNTH_SUMMARY 의 LUT primitive 카운트 — 표의 종전 수치(768/744/1183)와
동일 metric 이라 상호 비교 가능. util.rpt 헤드라인 "Slice LUTs"(combining 후)는 404.

native 재작성(A6)으로 구 병목(AddRecFN, logic 4.1ns)이 소멸 — 최악 스테이지 로직이
2.28ns 로 절반 이하, 나머지는 pre-place route 추정(보수적). Vivado 수치는 ASIC PNR 의
상대 프록시일 뿐이므로 (배치 후 route 단축 감안) **250MHz 는 가시권**; 구 T3(AddRecFN
내부 분할) 후보는 대상 블록이 사라져 폐기. latency 5cy 불변 + bit-exact 게이트 전부
green + out_RMW 레지스터 출력 유지. 스크립트: `work/synth_rmw/synth.tcl` (호출:
`vivado -mode batch -nojournal -nolog -source work/synth_rmw/synth.tcl -tclargs bf16`
— 플래그는 반드시 -tclargs **앞**에).
