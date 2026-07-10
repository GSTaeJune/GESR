#!/usr/bin/env bash
# sim/run_rmw.sh — RMW top wrapper unit TB.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF=../third_party/berkeley-hardfloat
SRC=../gemm_sram.srcs/sources_1/new

xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    "$HF"/HardFloatBundle_bf16.v \
    "$SRC"/fp32_adder.v \
    "$SRC"/fp32_to_bf16_rne.v \
    "$SRC"/int_to_bf16.v \
    "$SRC"/bf16_adder.v \
    "$SRC"/RMW.v \
    ../tb/rmw_tb.v

xelab -nolog -debug typical rmw_tb -s rmw_tb_sim
out=$(xsim rmw_tb_sim -nolog -R 2>&1); echo "$out"; echo "$out" | grep -q 'rmw_tb: ALL .* TESTS PASSED' || { echo 'run_rmw.sh: RMW vector test FAILED' >&2; exit 1; }
