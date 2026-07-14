#!/usr/bin/env bash
# sim/run_rmw_smoke.sh — HardFloat round-trip smoke test (INT→recFN→FP32).
#
# 주의 (2026-07-13 native 재작성 이후): 이 스모크가 검증하는 HardFloat 경로는
# 더 이상 RMW 데이터패스에 없다. 지금 역할 = third_party/berkeley-hardfloat
# 번들의 vendoring 무결성 체크 (preserved fp32 유닛 TB 들의 전제) — 이 green 을
# "RMW 데이터패스 검증"으로 읽지 말 것.
set -e
cd "$(dirname "$0")"
bash clean.sh

HF=../third_party/berkeley-hardfloat

xvlog -nolog \
    "$HF"/HardFloatBundle.v \
    ../tb/rmw_smoke_tb.v

xelab -nolog -debug typical rmw_smoke_tb -s rmw_smoke_tb_sim
out=$(xsim rmw_smoke_tb_sim -nolog -R 2>&1); echo "$out"; echo "$out" | grep -q 'rmw_smoke_tb: ALL TESTS PASSED' || { echo 'run_rmw_smoke.sh: rmw_smoke FAILED' >&2; exit 1; }
