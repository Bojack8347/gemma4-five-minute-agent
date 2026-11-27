#!/usr/bin/env bash
# Day-7 chain: iE over the two trained directions (Usage 2), after the day-6 chain. Runs only
# if the decisiveness adapter exists (the day-6 sanity gate may have refused it).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STEPS="${1:-3}"
while pgrep -f "^bash scripts/exp_day6b?_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-6 chain finished"
if [ ! -f adapters_out/decisive_r16/adapter_model.safetensors ]; then
  echo "no decisiveness adapter; Usage 2 not started"; exit 2
fi
echo "=== $(date '+%F %T') iE over directions: sigma + probe"
.venv/bin/python scripts/ie_control.py configs/ie_directions.yaml probe \
  --noise-runs runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 \
  2>&1 | grep -v -E "UserWarning|check_feature"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') probe rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
echo "=== $(date '+%F %T') iE over directions: $STEPS RM steps"
.venv/bin/python scripts/ie_control.py configs/ie_directions.yaml iterate --steps "$STEPS" 2>&1 | grep -v -E "UserWarning|check_feature"
echo "=== $(date '+%F %T') day-7 chain done rc=${PIPESTATUS[0]}"
