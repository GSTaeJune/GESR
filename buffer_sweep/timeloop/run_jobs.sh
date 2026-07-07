#!/usr/bin/env bash
# Batch-run pending timeloop-mapper jobs for the partition sweep.
#
#   run_jobs.sh <cache_dir_wsl> [parallel_jobs]
#
# <cache_dir_wsl> = /mnt/c/.../buffer_sweep/results/tl_cache -- one subdir per
# job holding arch.yaml + problem.yaml (written by timeloop_sweep.py). A job is
# DONE when map.yaml exists, FAILED-NO-MAPPING when INVALID exists; everything
# else is pending. Jobs execute in WSL-native scratch (~/tl-sweep) because the
# mapper's many small writes are slow/flaky on /mnt/c (NOTES.md pitfall 6),
# then only map.yaml + stats.txt are copied back.
set -euo pipefail

CACHE="$1"
JOBS="${2:-3}"
DEADLINE="${3:-240}"   # per-job wall-clock (s). The mapper's own termination
                       # knobs count VALID mappings, which are sparse on
                       # factor-rich shapes (K=768 ran 3h+ unbounded). At the
                       # deadline we SIGINT the mapper binary: timeloop dumps
                       # its best-so-far mapping gracefully and the job
                       # completes as a normal 'done'.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$HOME/tl-sweep"

source ~/miniconda3/bin/activate timeloop

mkdir -p "$SCRATCH/_shared"
cp -r "$SRC_DIR/components" "$SRC_DIR/mapper_sweep.yaml" "$SRC_DIR/run_job.py" "$SCRATCH/_shared/"

pending=$(for d in "$CACHE"/*/; do
  n=$(basename "$d")
  if [ ! -f "$d/map.yaml" ] && [ ! -f "$d/INVALID" ]; then echo "$n"; fi
done)
total=$(printf '%s\n' "$pending" | grep -c . || true)
echo "run_jobs: $total pending, parallel=$JOBS"
[ "$total" -eq 0 ] && exit 0

export CACHE SCRATCH DEADLINE
printf '%s\n' "$pending" | xargs -r -P "$JOBS" -I{} bash -c '
  set -e
  n="{}"
  rm -rf "$SCRATCH/$n"; mkdir -p "$SCRATCH/$n"
  cp "$CACHE/$n/arch.yaml" "$CACHE/$n/problem.yaml" "$SCRATCH/$n/"
  cp -r "$SCRATCH/_shared/components" "$SCRATCH/_shared/mapper_sweep.yaml" \
        "$SCRATCH/_shared/run_job.py" "$SCRATCH/$n/"
  cd "$SCRATCH/$n"
  python run_job.py > run.log 2>&1 &
  jpid=$!
  waited=0
  while kill -0 $jpid 2>/dev/null; do
    sleep 10; waited=$((waited+10))
    if [ "$waited" -eq "$DEADLINE" ]; then
      # deadline: SIGINT only the timeloop-mapper BINARY of this job (exe
      # check avoids the /bin/sh wrapper; the pattern cannot match this
      # scripts own cmdline because there $n is still unexpanded text).
      for p in $(pgrep -f "$n/output/parsed-processed-input.yaml"); do
        if [ "$(readlink /proc/$p/exe 2>/dev/null)" = "/usr/local/bin/timeloop-mapper" ]; then
          kill -INT "$p" 2>/dev/null || true
        fi
      done
      echo "note:    $n hit ${DEADLINE}s deadline, SIGINT sent (best-so-far)"
    elif [ "$waited" -ge $((DEADLINE + 180)) ]; then
      kill -9 $jpid 2>/dev/null || true      # dump/parse did not finish: give up
      break
    fi
  done
  wait $jpid 2>/dev/null || true
  if [ -f output/timeloop-mapper.map.yaml ] && [ -s output/timeloop-mapper.stats.txt ]; then
    cp output/timeloop-mapper.map.yaml "$CACHE/$n/map.yaml"
    cp output/timeloop-mapper.stats.txt "$CACHE/$n/stats.txt"
    echo "done:    $n"
  else
    tail -5 run.log > "$CACHE/$n/INVALID" 2>/dev/null || echo no-valid-mapping > "$CACHE/$n/INVALID"
    echo "invalid: $n"
  fi
  rm -rf "$SCRATCH/$n"
'
echo "run_jobs: all finished"
