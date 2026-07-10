#!/bin/bash
# sim/run_integration_sweep.sh — 9-mode serial sweep + compare gate.
#
# Loops over (A_P, B_P) in {2,4,8}^2, runs:
#   1) MXP_Tools gen  -> ../work/<LABEL>
#   2) MXP_Tools emit -> ../work/<LABEL>   (emits all 3 precs; no --prec flag)
#   3) MXP_Tools ref  -> ../work/<LABEL>   (NOTE arg swap: --prec-a = WEIGHT = B_P,
#                                                          --prec-b = ACT    = A_P)
#      (--accum bf16 -> C_sw_*_bf16.npz; compare auto-selects the 16-bit dump
#       reader from the npz accum field)
#   4) sim/run_integration_one.sh <LABEL> <A_P> <B_P>
#   5) MXP_Tools compare against C_sw_mxint${B_P}_mxint${A_P}_bf16.npz
#      (npz filename slot 1 = WEIGHT prec = B_P, slot 2 = ACT prec = A_P)
#
# Exits 1 on any mismatch. Final success line: "ALL 9 MODES PASSED".
#
# Note: this script intentionally does NOT use `set -e` because the compare
# step is allowed to fail (it's caught by the if/then below for tallying);
# `set -e` would abort the loop on the first FAIL and skip remaining modes.
#
# 동작 원리 / TB 와의 인터페이스 / mode 격리 / PASS 게이트 2단계 등 전체
# 메커니즘 설명은 docs/sweep-architecture.md 참고.
cd "$(dirname "$0")/.."

PASSED=()
FAILED=()

for A_P in 2 4 8; do
  for B_P in 2 4 8; do
    LABEL="A${A_P}_B${B_P}"
    echo ""
    echo "==================== ${LABEL} ===================="

    # 1) gen + emit + ref (note --prec-a / --prec-b swap vs our plusarg convention)
    (cd MXP_Tools && \
      python -m mxp_tools gen   --out ../work/${LABEL} -M 128 -K 128 -N 128 --seed 0 && \
      python -m mxp_tools emit  --out ../work/${LABEL} && \
      python -m mxp_tools ref   --out ../work/${LABEL} --prec-a ${B_P} --prec-b ${A_P} --accum bf16) || {
        echo "${LABEL}: gen/emit/ref FAILED"
        FAILED+=("${LABEL}")
        continue
      }

    # 2) HW sim — produces work/<LABEL>/hw_out/bank{0..31}.mem (32-RMW phase)
    bash sim/run_integration_one.sh "${LABEL}" "${A_P}" "${B_P}" || {
        echo "${LABEL}: HW sim FAILED"
        FAILED+=("${LABEL}")
        continue
      }

    # 3) compare gate (32-bank layout). npz filename slot order = B_P then A_P
    BANKS=$(printf "../work/${LABEL}/hw_out/bank%d.mem " {0..31})
    if (cd MXP_Tools && \
        python -m mxp_tools compare \
            --ref ../work/${LABEL}/sw_ref/C_sw_mxint${B_P}_mxint${A_P}_bf16.npz \
            --hw-banks ${BANKS} \
            --layout interleaved_row_major_32bank); then
      PASSED+=("${LABEL}")
      echo "${LABEL}: PASS"
    else
      FAILED+=("${LABEL}")
      echo "${LABEL}: FAIL"
    fi
  done
done

echo ""
echo "===================================="
echo "SWEEP RESULT: ${#PASSED[@]}/9 PASS"
echo "PASSED: ${PASSED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "ALL 9 MODES PASSED"
