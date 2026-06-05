#!/bin/bash
# sim/run_sram_mp.sh — sram_1rw_banked_mp 단위 회귀.
set -e
cd "$(dirname "$0")/.."
BUILD=sim/build/sram_mp
mkdir -p "$BUILD"
cd "$BUILD"

SRC_ROOT="../../../gemm_sram.srcs/sources_1"

xvlog -sv \
    $SRC_ROOT/imports/Desktop/sram/rtl/sram_1rw.v \
    $SRC_ROOT/new/sram_1rw_banked_mp.v \
    ../../../tb/sram_1rw_banked_mp_tb.v

xelab -L work sram_1rw_banked_mp_tb -snapshot sram_mp_snap
out=$(xsim sram_mp_snap -runall 2>&1); echo "$out"; echo "$out" | grep -q 'sram_1rw_banked_mp_tb: ALL .* TESTS PASSED' || { echo 'run_sram_mp.sh: sram_mp TB FAILED' >&2; exit 1; }
