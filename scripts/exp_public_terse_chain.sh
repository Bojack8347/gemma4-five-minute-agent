#!/usr/bin/env bash
# The base agent (v0_long) title-only on the 100 sound public tasks (data/public_terse), same runner and
# settings as base_sound100_c8/_r2. Replicates the title-only effect on a second task set and provides the
# student states for statement distillation (IMPLEMENTATION.md 6.31).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
scripts/wait_server.sh || exit 1
name=public_terse_base_c8
echo "=== $(date '+%F %T') start $name"
.venv/bin/python scripts/run_guarded.py --name "$name" -- \
  --submission-dir submissions/v0_long --results-dir "runs/$name" --data-dir data/public_terse \
  --task-ids-file data/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/base_sound100_c8 runs/base_sound100_c8_r2 runs/public_terse_base_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') public terse chain done"
