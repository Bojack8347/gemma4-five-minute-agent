#!/usr/bin/env bash
# Variant 2 (v2_think0_format): thinking off + the format direction (LoRA) on v0_long's prompt, on the
# 76 hard tasks with title-only statements, same settings as heldout_hard_terse_base_c8. Runs after the skills chain; restarts the
# server with the variant's adapter mounted, and restores the base server at the end.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_skills_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') skills chain finished"
restart_with() {
  pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
  pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
  mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
  setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$1" > logs/vllm_server.log 2>&1 < /dev/null &
  scripts/wait_server.sh
}
restart_with submissions/v2_think0_format || exit 1
name=heldout_hard_terse_v2_c8
echo "=== $(date '+%F %T') start $name"
PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
  --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
  --submission-dir submissions/v2_think0_format --results-dir "runs/$name" --data-dir data/heldout_hard_terse \
  --task-ids-file data/heldout_hard_terse/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_skills_c8 runs/heldout_hard_terse_v2_c8 2>&1 | grep -v -i warning
restart_with submissions/v0_long
echo "=== $(date '+%F %T') v2 chain done"
