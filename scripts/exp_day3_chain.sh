#!/usr/bin/env bash
# Day-3 chain (2026-10-12), after the day-1 probe was lost to a sandbox OOM and the first
# trainer attempt failed on the vision tower's modules (both fixed):
#   1. exp_day2_chain.sh: stop server, smoke-test + train the format direction, restart the
#      server with v0_long routed through it, evaluate on dev24;
#   2. iE pilot: finish the probe (one batch left; resumes from state.json), then RM steps.
# Note: from today every sandbox command runs under a 4 GiB / 2-core cgroup (run_eval.py),
# so timing coordinates are not strictly comparable with the unlimited noise-floor runs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STEPS="${1:-4}"

scripts/exp_day2_chain.sh
echo "=== $(date '+%F %T') day-2 chain rc=$? (server should be up)"
scripts/wait_server.sh || exit 1

echo "=== $(date '+%F %T') iE pilot: finish probe"
.venv/bin/python scripts/ie_control.py configs/ie_pilot.yaml probe 2>&1 | grep -v -E "UserWarning|check_feature"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') probe rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== $(date '+%F %T') iE pilot: $STEPS RM steps"
.venv/bin/python scripts/ie_control.py configs/ie_pilot.yaml iterate --steps "$STEPS" 2>&1 | grep -v -E "UserWarning|check_feature"
echo "=== $(date '+%F %T') chain done rc=${PIPESTATUS[0]}"
