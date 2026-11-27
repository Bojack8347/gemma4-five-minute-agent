#!/usr/bin/env bash
# Day-15 chain: the third candidate (round 3's gated end point, zero garbled calls on dev24)
# on the 100 sound tasks, so all three bundles are measured at the scale that can see a
# two- or three-point score difference. After the day-14 chain.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day14_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-14 chain finished"
SUB=submissions/ie/directions_gated3/current
pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$SUB" > logs/vllm_server.log 2>&1 < /dev/null &
scripts/wait_server.sh || exit 1
name="gated3_end_sound100_c8"
echo "=== $(date '+%F %T') start $name"
.venv/bin/python scripts/run_guarded.py --name "$name" -- \
  --submission-dir "$SUB" --results-dir "runs/$name" \
  --task-ids-file data/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/base_sound100_c8 runs/format1_sound100_c8 runs/gated3_end_sound100_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') day-15 chain done"
