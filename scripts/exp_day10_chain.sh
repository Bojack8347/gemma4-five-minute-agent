#!/usr/bin/env bash
# Day-10 chain: isolate the decisiveness weight on dev24 at fixed temperature (0.2), twice per
# point, against the format-only run on the same tasks (6.16) and the noise floor:
#   A: w_format 1.0, w_decisive +1.0      B: w_format 1.0, w_decisive -0.5
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day9_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-9 chain finished"
restart_with() {
  pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
  pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
  mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
  setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$1" > logs/vllm_server.log 2>&1 < /dev/null &
  scripts/wait_server.sh
}
for point in "A:1.0" "B:-0.5"; do
  tag=${point%%:*}; wd=${point##*:}
  SUB=submissions/decisive_${tag}
  rm -rf "$SUB"; cp -r submissions/v0_long "$SUB"; mkdir -p "$SUB/adapters"
  .venv-vllm/bin/python scripts/merge_directions.py --out "$SUB/adapters/merged" \
    --direction format=adapters_out/format_r16_v2:1.0 --direction decisive=adapters_out/decisive_r16:$wd >/dev/null
  rm -f "$SUB/adapters/merged/merge.json"
  sed -i 's/^model: gemma-4-31b-it-qat-w4a16-ct$/model: gemma-4-31b-it-qat-w4a16-ct\nadapter: merged/' "$SUB/agent.yaml"
  echo "=== $(date '+%F %T') point $tag (w_decisive $wd): restarting server"
  restart_with "$SUB" || exit 1
  for i in 1 2; do
    name="decisive_${tag}_dev24_c8_r$i"
    echo "=== $(date '+%F %T') start $name"
    .venv/bin/python scripts/run_guarded.py --name "$name" -- \
      --submission-dir "$SUB" --results-dir "runs/$name" \
      --task-ids-file data/dev24.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
      2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
    echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
  done
done
.venv/bin/python scripts/behaviour_vector.py runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 runs/v0long_format1_dev24_c8 \
  runs/decisive_A_dev24_c8_r1 runs/decisive_A_dev24_c8_r2 runs/decisive_B_dev24_c8_r1 runs/decisive_B_dev24_c8_r2 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') day-10 chain done"
