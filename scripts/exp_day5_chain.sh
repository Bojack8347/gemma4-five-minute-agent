#!/usr/bin/env bash
# Day-5 chain: the pilot's end point (temperature 1.0, thinking 2048, read window 150) is a
# candidate, not a result, until measured on a fixed task set (IMPLEMENTATION.md 6.15).
# Run it on dev24 twice, like the noise floor, after the day-4 chain finishes.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day4_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-4 chain finished"
scripts/wait_server.sh || exit 1
.venv/bin/python scripts/ie_control.py configs/ie_pilot.yaml build 2>&1 | grep -v -E "UserWarning|check_feature"
for i in 1 2; do
  name="pilot_end_dev24_c8_r$i"
  echo "=== $(date '+%F %T') start $name"
  .venv/bin/python scripts/run_guarded.py --name "$name" -- \
    --submission-dir submissions/ie/pilot/current --results-dir "runs/$name" \
    --task-ids-file data/dev24.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
done
.venv/bin/python scripts/behaviour_vector.py runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 runs/pilot_end_dev24_c8_r1 runs/pilot_end_dev24_c8_r2 runs/v0long_format1_dev24_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') day-5 chain done"
