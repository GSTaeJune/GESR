#!/usr/bin/env bash
# sim/run_int_to_fp32.sh — int_to_fp32 unit TB.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF=../third_party/berkeley-hardfloat
SRC=../gemm_sram.srcs/sources_1/new

xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    "$SRC"/int_to_fp32.v \
    ../tb/int_to_fp32_tb.v

xelab -nolog -debug typical int_to_fp32_tb -s int_to_fp32_tb_sim
xsim int_to_fp32_tb_sim -nolog -R
