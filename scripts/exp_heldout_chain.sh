#!/usr/bin/env bash
# Held-out evaluation: the base agent (v0_long) and the submitted format bundle on the sound
# held-out tasks (data/heldout/sound_tasks.txt), under the scorer's harness (0.2.11) with the
# held-out environment, at the calibrated cap. The server serves the unzipped format bundle,
# whose base-model route serves v0_long.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
scripts/wait_server.sh || exit 1
for point in "base:submissions/v0_long" "format1:submissions/_bundle_format1"; do
  tag=${point%%:*}; SUB=${point##*:}
  name="heldout_${tag}_c8"
  echo "=== $(date '+%F %T') start $name"
  PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
    --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
    --submission-dir "$SUB" --results-dir "runs/$name" --data-dir data/heldout \
    --task-ids-file data/heldout/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
done
.venv/bin/python scripts/behaviour_vector.py runs/heldout_base_c8 runs/heldout_format1_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') held-out chain done"
