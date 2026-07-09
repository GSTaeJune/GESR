#!/usr/bin/env bash
# sim/run_fp32_to_bf16_rne.sh — fp32_to_bf16_rne cross-check TB (vs ml_dtypes).
#
# 1) generate ml_dtypes vectors  2) xvlog DUT+TB  3) xelab  4) xsim  5) grep sentinel.
# Runs from sim/ so all XSim scratch (xsim.dir/, *.log, sim/work/) stays gitignored;
# vectors land in sim/work/bf16_vec/, which the TB opens as work/bf16_vec/... (CWD=sim/).
set -e
cd "$(dirname "$0")"
bash clean.sh

SRC=../gemm_sram.srcs/sources_1/new

# 1) ml_dtypes oracle -> sim/work/bf16_vec/fp32_to_bf16.mem (and the other two)
python bf16_vectors.py --out work/bf16_vec

# 2) compile DUT + TB
xvlog -nolog \
    "$SRC"/fp32_to_bf16_rne.v \
    ../tb/fp32_to_bf16_rne_tb.v

# 3) elaborate
xelab -nolog -debug typical fp32_to_bf16_rne_tb -s fp32_to_bf16_rne_tb_sim

# 4) simulate (match the repo's case $(uname -s) xsim-invocation pattern)
out=$( { case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) cmd //c "xsim fp32_to_bf16_rne_tb_sim -nolog -R";;
    *)                    xsim fp32_to_bf16_rne_tb_sim -nolog -R;;
esac; } 2>&1 )
echo "$out"

# 5) pass-sentinel gate — no false green
echo "$out" | grep -qE 'fp32_to_bf16_rne_tb: ALL .* TESTS PASSED' \
    || { echo 'run_fp32_to_bf16_rne.sh: fp32_to_bf16_rne TB FAILED' >&2; exit 1; }
