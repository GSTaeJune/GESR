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
  n="$1"
  rm -rf "$SCRATCH/$n"; mkdir -p "$SCRATCH/$n"
  cp "$CACHE/$n/arch.yaml" "$CACHE/$n/problem.yaml" "$SCRATCH/$n/"
  cp -r "$SCRATCH/_shared/components" "$SCRATCH/_shared/mapper_sweep.yaml" \
        "$SCRATCH/_shared/run_job.py" "$SCRATCH/$n/"
  cd "$SCRATCH/$n"
  python run_job.py > run.log 2>&1 &
  jpid=$!
  waited=0
  int_sent=0
  while kill -0 $jpid 2>/dev/null; do
    sleep 10; waited=$((waited+10))
    if [ "$int_sent" -eq 0 ] && [ "$waited" -ge "$DEADLINE" ]; then
      int_sent=1
      # deadline: SIGINT only the timeloop-mapper BINARY of this job. Match by
      # process NAME (pgrep -x cannot self-match this script) and scope to this
      # job via its cmdline. NOTE: do NOT match /proc/pid/exe against
      # /usr/local/bin/timeloop-mapper -- that path is a symlink and exe
      # resolves to the real binary, so the compare never fired (2026-07-07:
      # orphan mappers piled up because no INT was ever delivered).
      for p in $(pgrep -x timeloop-mapper); do
        if grep -q "$n" "/proc/$p/cmdline" 2>/dev/null; then
          kill -INT "$p" 2>/dev/null || true
        fi
      done
      echo "note:    $n hit ${DEADLINE}s deadline, SIGINT sent (best-so-far)"
    elif [ "$waited" -ge $((DEADLINE + 120)) ]; then
      # dump/parse did not finish: hard-kill python AND this jobs mapper
      # (no apostrophes in this quoted block -- it would end the xargs quote)
      for p in $(pgrep -x timeloop-mapper); do
        grep -q "$n" "/proc/$p/cmdline" 2>/dev/null && kill -9 "$p" 2>/dev/null || true
      done
      kill -9 $jpid 2>/dev/null || true      # dump/parse did not finish: give up
      break
    fi
  done
  wait $jpid 2>/dev/null || true
  if [ -f output/timeloop-mapper.map.yaml ] && [ -s output/timeloop-mapper.stats.txt ] \
     && grep -q "Summary Stats" output/timeloop-mapper.stats.txt; then
    # publish via rename: a crash mid-copy must not leave truncated files that
    # look done (review F3)
    cp output/timeloop-mapper.map.yaml "$CACHE/$n/.map.tmp"
    cp output/timeloop-mapper.stats.txt "$CACHE/$n/.stats.tmp"
    mv "$CACHE/$n/.stats.tmp" "$CACHE/$n/stats.txt"
    mv "$CACHE/$n/.map.tmp" "$CACHE/$n/map.yaml"
    rm -f "$CACHE/$n/TIMEOUT"
    echo "done:    $n"
  elif grep -q "Utilization =" run.log 2>/dev/null; then
    # mapper HAD valid mappings but died before dumping: leave the job
    # retryable, do not poison the cache with a false INVALID (review C2)
    tail -3 run.log > "$CACHE/$n/TIMEOUT" 2>/dev/null || true
    echo "timeout: $n (retryable; valid mappings were seen)"
  else
    tail -5 run.log > "$CACHE/$n/INVALID" 2>/dev/null || echo no-valid-mapping > "$CACHE/$n/INVALID"
    echo "invalid: $n"
  fi
  rm -rf "$SCRATCH/$n"
' _ {}
echo "run_jobs: all finished"
