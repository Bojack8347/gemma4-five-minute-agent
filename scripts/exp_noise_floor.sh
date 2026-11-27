#!/usr/bin/env bash
# Noise floor for the iE behaviour vector: the same agent, same 24 tasks, same
# budget, run REPEATS times. The between-run spread of each behaviour metric is
# the covariance iE whitens with and sets the gain scales (memo section 10.8, step 2).
# Each run: dev24, concurrency 8, 34-minute cap (= 5 Kaggle minutes at c8), v0_long.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REPEATS="${1:-2}"
SUB="${2:-submissions/v0_long}"
TAG="${3:-noise_v0long_dev24_c8}"

scripts/wait_server.sh || exit 1
for i in $(seq 1 "$REPEATS"); do
  name="${TAG}_r${i}"
  echo "=== $(date '+%F %T') start $name"
  .venv/bin/python scripts/run_guarded.py --name "$name" -- \
    --submission-dir "$SUB" --results-dir "runs/$name" \
    --task-ids-file data/dev24.txt \
    --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 \
    --concurrency 8 2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  rc=${PIPESTATUS[0]}
  echo "=== $(date '+%F %T') end $name rc=$rc"
  if [ "$rc" -eq 3 ]; then echo "watchdog killed $name; stopping chain"; exit 3; fi
done
