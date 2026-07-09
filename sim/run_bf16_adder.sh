#!/usr/bin/env bash
# sim/run_bf16_adder.sh — bf16_adder cross-check TB (vs ml_dtypes).
#
# 1) generate ml_dtypes vectors  2) xvlog fp32 HardFloat + fp32_adder +
# fp32_to_bf16_rne + bf16_adder + TB  3) xelab  4) xsim  5) grep sentinel.
# Runs from sim/ so all XSim scratch (xsim.dir/, *.log, sim/work/) stays
# gitignored; vectors land in sim/work/bf16_vec/, which the TB opens as
# work/bf16_vec/... (CWD=sim/).
set -e
cd "$(dirname "$0")"
bash clean.sh

SRC=../gemm_sram.srcs/sources_1/new
HF=../third_party/berkeley-hardfloat

# 1) ml_dtypes oracle -> sim/work/bf16_vec/bf16_add.mem (and the other two)
python bf16_vectors.py --out work/bf16_vec

# 2) compile fp32 HardFloat bundle (needed by fp32_adder) + DUTs + TB
xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    "$SRC"/fp32_adder.v \
    "$SRC"/fp32_to_bf16_rne.v \
    "$SRC"/bf16_adder.v \
    ../tb/bf16_adder_tb.v

# 3) elaborate
xelab -nolog -debug typical bf16_adder_tb -s bf16_adder_tb_sim

# 4) simulate (match the repo's case $(uname -s) xsim-invocation pattern)
out=$( { case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) cmd //c "xsim bf16_adder_tb_sim -nolog -R";;
    *)                    xsim bf16_adder_tb_sim -nolog -R;;
esac; } 2>&1 )
echo "$out"

# 5) pass-sentinel gate — no false green
echo "$out" | grep -qE 'bf16_adder_tb: ALL .* TESTS PASSED' \
    || { echo 'run_bf16_adder.sh: bf16_adder TB FAILED' >&2; exit 1; }
