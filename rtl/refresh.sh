#!/bin/bash
# rtl/refresh.sh — 읽기용 RTL 사본 폴더를 원본에서 재복사한다.
#
# rtl/ 은 브라우징 편의를 위한 SNAPSHOT COPY 다. 원본이 진실(source of truth):
#   - 프로젝트 로컬 RTL : gemm_sram.srcs/sources_1/new/
#   - MXP compute (import): gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/
#   - SRAM (import)       : gemm_sram.srcs/sources_1/imports/Desktop/sram/rtl/
# 수정은 반드시 원본에서 하고, 이 스크립트로 사본을 갱신할 것.
#   bash rtl/refresh.sh
set -e
cd "$(dirname "$0")/.."

NEW=gemm_sram.srcs/sources_1/new
MXP=gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new
SRAM=gemm_sram.srcs/sources_1/imports/Desktop/sram/rtl

mkdir -p rtl/1_top rtl/2_rmw_bf16 rtl/3_fp32_units rtl/4_sram rtl/5_gemm_core

# 1) 최상위 (구조적 래퍼 + MXP TOP 래퍼)
cp "$NEW"/gemm_sram_top.v        rtl/1_top/
cp "$NEW"/GEMM.v                 rtl/1_top/

# 2) RMW bf16 데이터패스 (active)
cp "$NEW"/RMW.v                  rtl/2_rmw_bf16/
cp "$NEW"/int_to_bf16.v          rtl/2_rmw_bf16/
cp "$NEW"/bf16_adder.v           rtl/2_rmw_bf16/
cp "$NEW"/fp32_to_bf16_rne.v     rtl/2_rmw_bf16/

# 3) fp32 유닛 (fp32_adder = active(bf16_adder 내부) / int_to_fp32 = preserved)
cp "$NEW"/fp32_adder.v           rtl/3_fp32_units/
cp "$NEW"/int_to_fp32.v          rtl/3_fp32_units/

# 4) SRAM (per-bank 포트 래퍼는 프로젝트 로컬, leaf/16-bank 래퍼는 import)
cp "$NEW"/sram_1rw_banked_mp.v   rtl/4_sram/
cp "$SRAM"/sram_1rw.v            rtl/4_sram/
cp "$SRAM"/sram_1rw_banked.v     rtl/4_sram/

# 5) MXP GEMM compute core (import — 수정은 ../MXP/ 업스트림에서)
cp "$MXP"/SystolicArray.v        rtl/5_gemm_core/
cp "$MXP"/PE_feeder.v            rtl/5_gemm_core/
cp "$MXP"/PE_naive.v             rtl/5_gemm_core/
cp "$MXP"/station.v              rtl/5_gemm_core/
cp "$MXP"/adder_lane.v           rtl/5_gemm_core/
cp "$MXP"/Accumulator.v          rtl/5_gemm_core/
cp "$MXP"/Accumulator_Col.v      rtl/5_gemm_core/

echo "rtl/ refreshed from originals ($(git rev-parse --short HEAD))"
