#!/bin/bash
# sim/run_mixed_one.sh — mixed-precision 단일 시나리오 (1 A_PREC).
#
# Usage:   bash sim/run_mixed_one.sh <A_PREC>
# Example: bash sim/run_mixed_one.sh 8
#
# 절차: gen_mixed.py → xsim → MXP_Tools compare (vs C_sw_mixed.npz).
set -e
cd "$(dirname "$0")/.."

A_PREC="${1:-8}"
LABEL="mixed_A${A_PREC}"
WORK="work/${LABEL}"
DUMP="${WORK}/hw_out"
BUILD="sim/build/${LABEL}"

mkdir -p "$DUMP" "$BUILD"

# 1) Python: hex + golden + viz.
python sim/gen_mixed.py --A "$A_PREC" --seed 0 --out "$WORK"

# 2) HW sim.
SRC_ROOT="gemm_sram.srcs/sources_1"
(cd "$BUILD" && \
    xvlog -sv \
        ../../../$SRC_ROOT/imports/Desktop/MXP/MXP.srcs/sources_1/new/*.v \
        ../../../$SRC_ROOT/imports/Desktop/sram/rtl/*.v \
        ../../../$SRC_ROOT/new/int_to_bf16.v \
        ../../../$SRC_ROOT/new/bf16_adder.v \
        ../../../$SRC_ROOT/new/RMW.v \
        ../../../$SRC_ROOT/new/sram_1rw_banked_mp.v \
        ../../../$SRC_ROOT/new/GEMM.v \
        ../../../$SRC_ROOT/new/gemm_sram_top.v \
        ../../../tb/gemm_sram_top_mixed_tb.v && \
    xelab -L work gemm_sram_top_mixed_tb -snapshot mixed_snap && \
    { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cmd //c "xsim mixed_snap -runall -testplusarg \"A_PREC=$A_PREC\" -testplusarg \"WORK_DIR=../../../$WORK\" -testplusarg \"DUMP_DIR=../../../$DUMP\"";; *) xsim mixed_snap -runall -testplusarg "A_PREC=$A_PREC" -testplusarg "WORK_DIR=../../../$WORK" -testplusarg "DUMP_DIR=../../../$DUMP";; esac; })

# 3) bit-exact compare.
BANKS=$(printf "../work/${LABEL}/hw_out/bank%d.mem " {0..31})
(cd MXP_Tools && \
    python -m mxp_tools compare \
        --ref ../work/${LABEL}/sw_ref/C_sw_mixed.npz \
        --hw-banks ${BANKS} \
        --layout interleaved_row_major_32bank)

echo "mixed_A${A_PREC}: PASS"
