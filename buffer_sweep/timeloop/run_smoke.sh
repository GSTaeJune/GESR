#!/usr/bin/env bash
# Timeloop v4 smoke run for the gemm_sram accelerator model.
#
# Run FROM WINDOWS (Git Bash / cmd):
#   wsl bash -lc '/mnt/c/Users/ptj72/Desktop/Desktop/00project/gemm_sram/buffer_sweep/timeloop/run_smoke.sh'
# or from inside WSL directly:
#   bash /mnt/c/Users/ptj72/Desktop/Desktop/00project/gemm_sram/buffer_sweep/timeloop/run_smoke.sh
#
# NOTE: the run executes in a WSL-native scratch dir (~/tl-gemm-smoke) because
# running the mapper directly on /mnt/c (9P filesystem) is slow and flaky for
# the many small output-file writes. Final outputs are copied back next to
# this script under ./output/.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH=~/tl-gemm-smoke

# 1. Conda env with accelergy + timeloopfe (timeloop-mapper itself is /usr/local/bin)
source ~/miniconda3/bin/activate timeloop

# 2. Stage inputs into WSL-native scratch
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
cp -r "$SRC_DIR"/{arch.yaml,problem.yaml,mapper.yaml,run_smoke.py,components} "$SCRATCH/"

# 3. Run the mapper via the timeloopfe front-end (writes ./output/ in scratch)
cd "$SCRATCH"
python -u run_smoke.py

# 4. Copy results back into the repo
rm -rf "$SRC_DIR/output"
cp -r "$SCRATCH/output" "$SRC_DIR/output"
echo "Results copied to $SRC_DIR/output"
