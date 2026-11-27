#!/usr/bin/env bash
# The budget frame as a lever (IMPLEMENTATION.md 6.42). The most-run public configuration ("Budget-Fit" by
# verracodeguacas, Apache 2.0; submissions/public_budget_fit, byte-for-byte from the public notebook) differs
# from ours in the frame, not in weights: 8 minutes, 28 tool calls, 80 turns, thinking 4096, 240 s commands,
# and a call-plan prompt. Ours: 5 minutes, 100 calls, 250 turns, thinking 512. Every earlier comparison of
# ours held the 5-minute frame fixed. On the 76 sound title-only hard tasks at concurrency 8:
#   1. the public bundle at its own frame (8 Kaggle minutes = 54.4 local minutes at c8: 34 x 8/5)
#   2. our base (v0_long) with only the time cap raised to 8 minutes, the control that separates time from the rest
#   3. the public bundle again, the replicate
# Base at the 5-minute frame: runs/heldout_hard_terse_base_c8 and runs/proxy_f000_base_c8 (13 and 13 of 76).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
DATA=data/heldout_hard_terse
scripts/wait_server.sh || exit 1

run() {  # run NAME SUBMISSION MINUTES CALLS TURNS TIMEOUT
  local name="$1" sub="$2" minutes="$3" calls="$4" turns="$5" tmo="$6"
  if [ -f "runs/$name/summary.json" ]; then echo "=== $name already done"; return; fi
  echo "=== $(date '+%F %T') start $name ($sub, $minutes min, $calls calls, $turns turns, ${tmo}s commands)"
  PYTHONPATH="$ROOT/data/harness_v2_site" $PY scripts/run_guarded.py --name "$name" \
    --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
    --submission-dir "$sub" --results-dir "runs/$name" --data-dir "$DATA" \
    --task-ids-file "$DATA/sound_tasks.txt" --max-time-minutes "$minutes" --max-tool-calls "$calls" \
    --max-turns "$turns" --timeout-seconds "$tmo" --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
  $PY -c "import json; s=json.load(open('runs/$name/summary.json')); print('RESULT $name', s['resolved'], '/', s['total'], 'wall_h', round(s['wall_seconds']/3600, 2))"
}

run public_bf_terse_c8_r1 submissions/public_budget_fit 54.4 28 80 240
run v0_long_8min_terse_c8 submissions/v0_long 54.4 100 250 120
run public_bf_terse_c8_r2 submissions/public_budget_fit 54.4 28 80 240
echo "=== $(date '+%F %T') budget frame chain done"
