#!/usr/bin/env bash
# Day-1 GPU chain after the noise-floor runs (memo section 10.8, steps 2-4):
#   1. wait for scripts/exp_noise_floor.sh to finish;
#   2. mine on-policy format pairs from the thinking-off runs (server otherwise idle,
#      so its load does not bias any timing metric);
#   3. iE pilot: sigma from the two noise-floor runs, Jacobian probe (5 batches),
#      then RM steps until the night is over.
# Everything is detached-safe and resumable: the miner skips mined contexts, the
# controller resumes from runs/ie_pilot/state.json.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STEPS="${1:-4}"

while pgrep -f "^bash scripts/exp_noise_floor.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') noise floor finished"
scripts/wait_server.sh || exit 1

echo "=== $(date '+%F %T') mining format pairs"
.venv/bin/python scripts/mine_format_pairs.py \
  runs/v0_think0_t06_dev24_c2x2 runs/v0_think0_dev24_c2x2 runs/v0_think0_t10_dev24_c2x2 \
  --out data/pairs/format_pairs.jsonl --samples 8 --max-tokens 1024 --limit 50 \
  2>&1 | grep -v -E "UserWarning|check_feature"
echo "=== $(date '+%F %T') mining done"

echo "=== $(date '+%F %T') iE pilot: sigma + probe"
.venv/bin/python scripts/ie_control.py configs/ie_pilot.yaml probe \
  --noise-runs runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 \
  2>&1 | grep -v -E "UserWarning|check_feature"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') probe rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== $(date '+%F %T') iE pilot: $STEPS RM steps"
.venv/bin/python scripts/ie_control.py configs/ie_pilot.yaml iterate --steps "$STEPS" \
  2>&1 | grep -v -E "UserWarning|check_feature"
echo "=== $(date '+%F %T') chain done rc=${PIPESTATUS[0]}"
