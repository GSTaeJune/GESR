#!/usr/bin/env bash
# sim/run_rmw.sh — RMW top wrapper unit TB.
set -e
cd "$(dirname "$0")"
bash clean.sh

SRC=../gemm_sram.srcs/sources_1/new

# 0) 벡터를 매번 결정론적으로 재생성 (stale-file 함정 방지 — 종전에는 수동
#    rmw-gen 선행이 필요해 오래된 expected 와 비교할 수 있었다).
#    --n 64 -> 113 벡터 (지수적 케이스 확장), rmw_tb 의 MAX_N=128 제약과 짝.
#    --n 을 바꾸면 아래 sentinel 의 113 과 rmw_tb MAX_N 도 함께 갱신할 것.
(cd ../MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0)

# native bf16 데이터패스 (2026-07-13) — HardFloat/fp32 유닛 불필요
xvlog -nolog \
    "$SRC"/int_to_bf16.v \
    "$SRC"/bf16_adder.v \
    "$SRC"/RMW.v \
    ../tb/rmw_tb.v

xelab -nolog -debug typical rmw_tb -s rmw_tb_sim
# sentinel 은 카운트까지 고정 (truncated 벡터로 "ALL 0 PASSED" green 차단)
out=$(xsim rmw_tb_sim -nolog -R 2>&1); echo "$out"; echo "$out" | grep -qF 'rmw_tb: ALL 113 TESTS PASSED' || { echo 'run_rmw.sh: RMW vector test FAILED (or vector count changed)' >&2; exit 1; }
