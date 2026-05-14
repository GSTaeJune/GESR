#!/usr/bin/env bash
# sim/run_rmw.sh — RMW top wrapper unit TB.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF=../third_party/berkeley-hardfloat
SRC=../gemm_sram.srcs/sources_1/new

xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    "$SRC"/int_to_fp32.v \
    "$SRC"/fp32_adder.v \
    "$SRC"/RMW.v \
    ../tb/rmw_tb.v

xelab -nolog -debug typical rmw_tb -s rmw_tb_sim
xsim rmw_tb_sim -nolog -R
