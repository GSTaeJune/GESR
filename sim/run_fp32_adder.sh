#!/usr/bin/env bash
# sim/run_fp32_adder.sh — fp32_adder unit TB.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF=../third_party/berkeley-hardfloat
SRC=../gemm_sram.srcs/sources_1/new

xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    "$SRC"/fp32_adder.v \
    ../tb/fp32_adder_tb.v

xelab -nolog -debug typical fp32_adder_tb -s fp32_adder_tb_sim
out=$(xsim fp32_adder_tb_sim -nolog -R 2>&1); echo "$out"; echo "$out" | grep -q 'fp32_adder_tb: ALL TESTS PASSED' || { echo 'run_fp32_adder.sh: fp32_adder TB FAILED' >&2; exit 1; }
