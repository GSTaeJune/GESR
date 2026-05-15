#!/bin/bash
# sim/run_integration_parallel.sh — Task 10: 9-way parallel dispatch guidance.
#
# This script does NOT itself fork 9 sims (xsim's xsim.dir lock would collide).
# Instead it prints the per-subagent task template for an AI agent that uses
# `superpowers:dispatching-parallel-agents` to dispatch 9 independent subagents
# at once.
#
# For straight serial verification (no AI agent), use run_integration_sweep.sh.
#
# Verified end-state from Task 9: serial sweep already shows 9/9 PASS bit-exact
# vs MXP_Tools golden across all (A_PREC, B_PREC) in {2,4,8}^2.
cat <<'EOF'
============================ PARALLEL DISPATCH GUIDE =============================

Inputs to each subagent:
  A_PREC ∈ {2, 4, 8}
  B_PREC ∈ {2, 4, 8}
  WORKING_DIR = repo root (read-only RTL + isolated work/A${A_PREC}_B${B_PREC}/)
  Dispatch 9 subagents in parallel — one per (A_PREC, B_PREC) combination.

Per-subagent steps:
  1. cd MXP_Tools && python -m mxp_tools gen \
       --out ../work/A${A_PREC}_B${B_PREC} -M 128 -K 128 -N 128 --seed 0
  2. python -m mxp_tools emit --out ../work/A${A_PREC}_B${B_PREC}
     (no --prec flag emits all 3 precisions; the same input dir serves any mode)
  3. python -m mxp_tools ref --out ../work/A${A_PREC}_B${B_PREC} \
       --prec-a ${B_PREC} --prec-b ${A_PREC}
     (NOTE arg swap: MXP_Tools --prec-a = WEIGHT precision = our B_PREC;
                     MXP_Tools --prec-b = ACTIVATION  precision = our A_PREC)
  4. cd .. && bash sim/run_integration_one.sh \
       "A${A_PREC}_B${B_PREC}" ${A_PREC} ${B_PREC}
  5. cd MXP_Tools && \
       BANKS=$(printf "../work/A${A_PREC}_B${B_PREC}/hw_out/bank%d.mem " {0..31}) && \
       python -m mxp_tools compare \
         --ref ../work/A${A_PREC}_B${B_PREC}/sw_ref/C_sw_mxint${B_PREC}_mxint${A_PREC}.npz \
         --hw-banks ${BANKS} \
         --layout interleaved_row_major_32bank
     (compare exit code 0 = PASS; non-zero = FAIL.
      .npz filename slot 1 = WEIGHT prec = B_PREC; slot 2 = ACT prec = A_PREC.)

Isolation between subagents:
  - Each subagent writes only to its own work/A${A_PREC}_B${B_PREC}/ tree.
  - Each subagent's xsim build dir is sim/build/A${A_PREC}_B${B_PREC}/, also
    mode-disjoint (run_integration_one.sh uses LABEL as the build subdir).
  - RTL sources (gemm_sram.srcs/, tb/, third_party/) are read-only — all
    subagents share without conflict.

Aggregator (controlling agent):
  - Collect (LABEL, exit_code) tuples from all 9 subagents.
  - Report "9/9 PASS" or list failing LABELs with first-mismatch diagnostic.

Resource note:
  - One xsim invocation peaks ~1.5 GB RSS; 9 parallel ≈ 14 GB. Adequate on
    a 16 GB machine. If memory-constrained, fall back to a 3-wave x 3-mode
    rolling dispatch (e.g., 3 parallel agents at a time).
==================================================================================
EOF
