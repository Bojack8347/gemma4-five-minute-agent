#!/usr/bin/env bash
# Day-13 chain: fixed-set check of round 3 (gated) end point (as 6.21 did for round 1 and 6.17 for the
# knob pilot). After the day-12 chain: build the end point, restart the server with its merged
# adapter, run dev24 twice, and compare with the noise floor and the format-only point.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day12_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-12 chain finished"
.venv/bin/python scripts/ie_control.py configs/ie_directions_gated3.yaml build 2>&1 | grep -v -E "UserWarning|check_feature"
SUB=submissions/ie/directions/current
pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$SUB" > logs/vllm_server.log 2>&1 < /dev/null &
scripts/wait_server.sh || exit 1
for i in 1 2; do
  name="gated3_end_dev24_c8_r$i"
  echo "=== $(date '+%F %T') start $name"
  .venv/bin/python scripts/run_guarded.py --name "$name" -- \
    --submission-dir "$SUB" --results-dir "runs/$name" \
    --task-ids-file data/dev24.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
done
.venv/bin/python scripts/behaviour_vector.py runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 runs/v0long_format1_dev24_c8 runs/ie_directions_gated3/step_001 runs/gated3_end_dev24_c8_r1 runs/gated3_end_dev24_c8_r2 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') day-13 chain done"
