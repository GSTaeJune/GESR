#!/usr/bin/env bash
# sim/run_rmw_smoke.sh — HardFloat round-trip smoke test (INT→recFN→FP32).
set -e
cd "$(dirname "$0")"
bash clean.sh

HF=../third_party/berkeley-hardfloat

xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    ../tb/rmw_smoke_tb.v

xelab -nolog -debug typical rmw_smoke_tb -s rmw_smoke_tb_sim
out=$(xsim rmw_smoke_tb_sim -nolog -R 2>&1); echo "$out"; echo "$out" | grep -q 'rmw_smoke_tb: ALL TESTS PASSED' || { echo 'run_rmw_smoke.sh: rmw_smoke FAILED' >&2; exit 1; }
