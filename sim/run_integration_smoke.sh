#!/bin/bash
# sim/run_integration_smoke.sh — Task 2 smoke: SRAM zero priming + dump.
set -e
cd "$(dirname "$0")/.."

LABEL="smoke"
WORK="work/${LABEL}"
DUMP="${WORK}/hw_out"
BUILD="sim/build/${LABEL}"

mkdir -p "$DUMP" "$BUILD"

SRC_ROOT="gemm_sram.srcs/sources_1"
HF_ROOT="third_party/berkeley-hardfloat"

(cd "$BUILD" && \
    xvlog -sv \
        ../../../$HF_ROOT/HardFloatBundle.v \
        ../../../$HF_ROOT/HardFloatBundle_bf16.v \
        ../../../$SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
        ../../../$SRC_ROOT/imports/Desktop/sram/rtl/*.v \
        ../../../$SRC_ROOT/new/fp32_adder.v \
        ../../../$SRC_ROOT/new/fp32_to_bf16_rne.v \
        ../../../$SRC_ROOT/new/int_to_bf16.v \
        ../../../$SRC_ROOT/new/bf16_adder.v \
        ../../../$SRC_ROOT/new/RMW.v \
        ../../../$SRC_ROOT/new/sram_1rw_banked_mp.v \
        ../../../$SRC_ROOT/new/GEMM.v \
        ../../../$SRC_ROOT/new/gemm_sram_top.v \
        ../../../tb/gemm_sram_top_tb.v && \
    xelab -L work gemm_sram_top_tb -snapshot gemm_sram_top_tb_snap && \
    { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cmd //c "xsim gemm_sram_top_tb_snap -runall -testplusarg \"DUMP_DIR=../../../$DUMP\"";; *) xsim gemm_sram_top_tb_snap -runall -testplusarg "DUMP_DIR=../../../$DUMP";; esac; })

echo "Smoke sim done. Check $DUMP/bank{0..31}.mem"
