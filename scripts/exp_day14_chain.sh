#!/usr/bin/env bash
# Day-14 chain: the score question at the scale that can see it. The base (v0_long) and the
# format point (v0_long_format1) on all 100 sound tasks at the calibrated cap (34 min, c8),
# guarded, with the per-command sandbox limits; about 7 hours each.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
restart_with() {
  pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
  pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
  mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
  setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$1" > logs/vllm_server.log 2>&1 < /dev/null &
  scripts/wait_server.sh
}
for point in "base:submissions/v0_long" "format1:submissions/v0_long_format1"; do
  tag=${point%%:*}; SUB=${point##*:}
  echo "=== $(date '+%F %T') $tag: restarting server with $SUB"
  restart_with "$SUB" || exit 1
  name="${tag}_sound100_c8"
  echo "=== $(date '+%F %T') start $name"
  .venv/bin/python scripts/run_guarded.py --name "$name" -- \
    --submission-dir "$SUB" --results-dir "runs/$name" \
    --task-ids-file data/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
done
.venv/bin/python scripts/behaviour_vector.py runs/base_sound100_c8 runs/format1_sound100_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') day-14 chain done"
