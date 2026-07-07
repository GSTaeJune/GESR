#!/usr/bin/env bash
# One-off: re-run a cached job, optionally with an alternate mapper yaml.
#   wsl bash <this file> <job_key> [mapper_yaml]
set -euo pipefail
JOB="${1:?usage: debug_job.sh <job_key> [mapper_yaml]}"
MAPPER="${2:-}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# jobs live under a per-mapper-config subdir (cfg_<hash>); default to the
# most recently modified one, override with CACHE_DIR env (review I4)
CACHE="${CACHE_DIR:-$(ls -dt "$SRC_DIR/../results/tl_cache"/cfg_* 2>/dev/null | head -1)}"
[ -n "$CACHE" ] || { echo "no cfg_* cache dir found" >&2; exit 1; }
source ~/miniconda3/bin/activate timeloop
rm -rf ~/tl-debug && mkdir -p ~/tl-debug
cp "$CACHE/$JOB/arch.yaml" "$CACHE/$JOB/problem.yaml" ~/tl-debug/
cp -r "$SRC_DIR/components" "$SRC_DIR/run_job.py" ~/tl-debug/
if [ -n "$MAPPER" ]; then
  sed 's/\r$//' "$MAPPER" > ~/tl-debug/mapper_sweep.yaml
else
  sed 's/diagnostics: False/diagnostics: True/' "$SRC_DIR/mapper_sweep.yaml" > ~/tl-debug/mapper_sweep.yaml
fi
cd ~/tl-debug
timeout 550 python run_job.py 2>&1 | tail -40
