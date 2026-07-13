#!/usr/bin/env bash
# sim/run_rmw.sh — RMW top wrapper unit TB.
set -e
cd "$(dirname "$0")"
bash clean.sh

SRC=../gemm_sram.srcs/sources_1/new

# native bf16 데이터패스 (2026-07-13) — HardFloat/fp32 유닛 불필요
xvlog -nolog \
    "$SRC"/int_to_bf16.v \
    "$SRC"/bf16_adder.v \
    "$SRC"/RMW.v \
    ../tb/rmw_tb.v

xelab -nolog -debug typical rmw_tb -s rmw_tb_sim
out=$(xsim rmw_tb_sim -nolog -R 2>&1); echo "$out"; echo "$out" | grep -q 'rmw_tb: ALL .* TESTS PASSED' || { echo 'run_rmw.sh: RMW vector test FAILED' >&2; exit 1; }
