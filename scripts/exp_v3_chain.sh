#!/usr/bin/env bash
# Variant 3 (v3_skills_complete: code-tools skill + guidance for title-only statements: read the
# existing tests first, change every parallel place, export new names) on the 76 title-only hard
# tasks, after the v2 chain (which restores the base server at its end).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_v2_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') v2 chain finished"
scripts/wait_server.sh || exit 1
name=heldout_hard_terse_v3_c8
echo "=== $(date '+%F %T') start $name"
PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
  --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
  --submission-dir submissions/v3_skills_complete --results-dir "runs/$name" --data-dir data/heldout_hard_terse \
  --task-ids-file data/heldout_hard_terse/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_skills_c8 runs/heldout_hard_terse_v2_c8 runs/heldout_hard_terse_v3_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') v3 chain done"
