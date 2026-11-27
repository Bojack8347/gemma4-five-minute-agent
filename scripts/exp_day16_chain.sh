#!/usr/bin/env bash
# Day-16 chain: a second draw of the format point on the 100 sound tasks, to tighten the
# +7 estimate of 6.25 (about 13 tasks flip between draws of the same agent).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day15_chain.sh" >/dev/null; do sleep 60; done
SUB=submissions/v0_long_format1
pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$SUB" > logs/vllm_server.log 2>&1 < /dev/null &
scripts/wait_server.sh || exit 1
name="format1_sound100_c8_r2"
echo "=== $(date '+%F %T') start $name"
.venv/bin/python scripts/run_guarded.py --name "$name" -- \
  --submission-dir "$SUB" --results-dir "runs/$name" \
  --task-ids-file data/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/base_sound100_c8 runs/format1_sound100_c8 runs/format1_sound100_c8_r2 runs/gated3_end_sound100_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') day-16 chain done"
