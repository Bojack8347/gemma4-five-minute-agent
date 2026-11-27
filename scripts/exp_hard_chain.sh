#!/usr/bin/env bash
# Hard held-out set (IMPLEMENTATION.md 6.30): the base agent (v0_long, 31B) on the 76 sound hard tasks,
# with full commit-message statements and then with PR-title-only statements. 0.2.11 harness,
# held-out environment, calibrated cap.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup .venv/bin/python scripts/serve.py --submission-dir submissions/v0_long > logs/vllm_server.log 2>&1 < /dev/null &
scripts/wait_server.sh || exit 1
for variant in heldout_hard heldout_hard_terse; do
  name="${variant}_base_c8"
  echo "=== $(date '+%F %T') start $name"
  PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
    --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
    --submission-dir submissions/v0_long --results-dir "runs/$name" --data-dir "data/$variant" \
    --task-ids-file "data/$variant/sound_tasks.txt" --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
done
.venv/bin/python scripts/behaviour_vector.py runs/heldout_base_c8 runs/heldout_hard_base_c8 runs/heldout_hard_terse_base_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') hard chain done"
