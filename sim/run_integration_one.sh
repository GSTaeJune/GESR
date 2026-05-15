#!/bin/bash
# sim/run_integration_one.sh — Task 7+: parameterized single-mode integration sim.
#
# Usage:   bash sim/run_integration_one.sh <LABEL> <A_PREC> <B_PREC>
# Example: bash sim/run_integration_one.sh A8_B8 8 8
#
# Requires MXP_Tools to have generated the inputs first:
#   cd MXP_Tools && python -m mxp_tools gen   --out ../work/<LABEL> -M 128 -K 128 -N 128 --seed 0
#                   python -m mxp_tools emit  --out ../work/<LABEL> --prec <A_PREC>
#                   python -m mxp_tools ref   --out ../work/<LABEL> --prec-a <A_PREC> --prec-b <B_PREC>
set -e
cd "$(dirname "$0")/.."

LABEL="${1:-A8_B8}"
A_PREC="${2:-8}"
B_PREC="${3:-8}"

WORK="work/${LABEL}"
DUMP="${WORK}/hw_out"
BUILD="sim/build/${LABEL}"

mkdir -p "$DUMP" "$BUILD"

SRC_ROOT="gemm_sram.srcs/sources_1"
HF_ROOT="third_party/berkeley-hardfloat"

# Note: GEMM.v MUST appear before gemm_sram_top.v in xvlog list (task-7 plan fix).
# Note: snapshot name is bare, not work.<snap>.
# Note: -testplusarg "K=V" requires cmd //c wrap on Windows Git Bash (= is stripped otherwise).
(cd "$BUILD" && \
    xvlog -sv \
        ../../../$HF_ROOT/HardFloatBundle.v \
        ../../../$SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
        ../../../$SRC_ROOT/imports/Desktop/sram/rtl/*.v \
        ../../../$SRC_ROOT/new/int_to_fp32.v \
        ../../../$SRC_ROOT/new/fp32_adder.v \
        ../../../$SRC_ROOT/new/RMW.v \
        ../../../$SRC_ROOT/new/sram_1rw_banked_mp.v \
        ../../../$SRC_ROOT/new/GEMM.v \
        ../../../$SRC_ROOT/new/gemm_sram_top.v \
        ../../../tb/gemm_sram_top_tb.v && \
    xelab -L work gemm_sram_top_tb -snapshot gemm_sram_top_tb_snap && \
    cmd //c "xsim gemm_sram_top_tb_snap -runall -testplusarg \"A_PREC=$A_PREC\" -testplusarg \"B_PREC=$B_PREC\" -testplusarg \"WORK_DIR=../../../$WORK\" -testplusarg \"DUMP_DIR=../../../$DUMP\"")

echo "Integration sim ${LABEL} done. Check $DUMP/bank{0..31}.mem"
