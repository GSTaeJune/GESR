#!/bin/bash
# sim/run_top_elab.sh — gemm_sram_top elaboration smoke (no TB).
set -e
cd "$(dirname "$0")/.."
BUILD=sim/build/top_elab
mkdir -p "$BUILD"
cd "$BUILD"

SRC_ROOT="../../../gemm_sram.srcs/sources_1"

# MXP + SRAM + native bf16 RMW + 통합 top (HardFloat-free, 2026-07-13)
xvlog -sv \
    $SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
    $SRC_ROOT/imports/Desktop/sram/rtl/*.v \
    $SRC_ROOT/new/int_to_bf16.v \
    $SRC_ROOT/new/bf16_adder.v \
    $SRC_ROOT/new/RMW.v \
    $SRC_ROOT/new/sram_1rw_banked_mp.v \
    $SRC_ROOT/new/GEMM.v \
    $SRC_ROOT/new/gemm_sram_top.v

xelab -L work gemm_sram_top -snapshot gemm_sram_top_snap

echo "PASS: gemm_sram_top elaborates cleanly"
