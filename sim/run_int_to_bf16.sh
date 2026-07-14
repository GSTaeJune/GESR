#!/usr/bin/env bash
# sim/run_int_to_bf16.sh — int_to_bf16 cross-check TB (vs ml_dtypes).
#
# 1) generate ml_dtypes vectors  2) xvlog int_to_bf16 (native, leaf — HardFloat
# 미사용) + TB  3) xelab  4) xsim  5) grep sentinel. Runs from sim/ so all XSim
# scratch (xsim.dir/, *.log, sim/work/) stays gitignored; vectors land in
# sim/work/bf16_vec/, which the TB opens as work/bf16_vec/... (CWD=sim/).
set -e
cd "$(dirname "$0")"
bash clean.sh

SRC=../gemm_sram.srcs/sources_1/new

# 1) ml_dtypes oracle -> sim/work/bf16_vec/int_to_bf16.mem (and the other two)
python bf16_vectors.py --out work/bf16_vec

# 2) compile DUT + TB (native converter is self-contained)
xvlog -nolog \
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

# 5) pass-sentinel gate — no false green. 카운트까지 고정: 이 유닛 oracle 이
#    subnormal/flush 의 유일한 영구 게이트라 벡터 수 감소는 커버리지 손실이다
#    (벡터 추가 시 이 숫자도 함께 갱신할 것).
echo "$out" | grep -qF 'int_to_bf16_tb: ALL 32360 TESTS PASSED' \
    || { echo 'run_int_to_bf16.sh: int_to_bf16 TB FAILED (or vector count changed)' >&2; exit 1; }
