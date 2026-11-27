#!/usr/bin/env bash
# Day-12 chain: round 3 of the gated iE over the directions on fixed dev24 (configs/ie_directions_gated3.yaml),
# four RM steps from the format-only point, Jacobian reused from round 1. After the day-10 chain.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STEPS="${1:-4}"
while pgrep -f "^bash scripts/exp_day11_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-10 chain finished"
scripts/wait_server.sh || exit 1
echo "=== $(date '+%F %T') gated iE over directions: $STEPS RM steps on dev24"
.venv/bin/python scripts/ie_control.py configs/ie_directions_gated3.yaml iterate --steps "$STEPS" 2>&1 | grep -v -E "UserWarning|check_feature"
echo "=== $(date '+%F %T') day-12 chain done rc=${PIPESTATUS[0]}"
