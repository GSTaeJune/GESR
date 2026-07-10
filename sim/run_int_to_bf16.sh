#!/usr/bin/env bash
# sim/run_int_to_bf16.sh — int_to_bf16 cross-check TB (vs ml_dtypes).
#
# 1) generate ml_dtypes vectors  2) xvlog bf16 HardFloat + DUT + TB  3) xelab
# 4) xsim  5) grep sentinel. Runs from sim/ so all XSim scratch (xsim.dir/, *.log,
# sim/work/) stays gitignored; vectors land in sim/work/bf16_vec/, which the TB
# opens as work/bf16_vec/... (CWD=sim/).
set -e
cd "$(dirname "$0")"
bash clean.sh

SRC=../gemm_sram.srcs/sources_1/new
HF=../third_party/berkeley-hardfloat

# 1) ml_dtypes oracle -> sim/work/bf16_vec/int_to_bf16.mem (and the other two)
python bf16_vectors.py --out work/bf16_vec

# 2) compile bf16 HardFloat bundle + DUT + TB
xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    "$HF"/HardFloatBundle_bf16.v \
    "$SRC"/fp32_to_bf16_rne.v \
    "$SRC"/int_to_bf16.v \
    ../tb/int_to_bf16_tb.v

# 3) elaborate
xelab -nolog -debug typical int_to_bf16_tb -s int_to_bf16_tb_sim

# 4) simulate (match the repo's case $(uname -s) xsim-invocation pattern)
out=$( { case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) cmd //c "xsim int_to_bf16_tb_sim -nolog -R";;
    *)                    xsim int_to_bf16_tb_sim -nolog -R;;
esac; } 2>&1 )
echo "$out"

# 5) pass-sentinel gate — no false green
echo "$out" | grep -qE 'int_to_bf16_tb: ALL .* TESTS PASSED' \
    || { echo 'run_int_to_bf16.sh: int_to_bf16 TB FAILED' >&2; exit 1; }
