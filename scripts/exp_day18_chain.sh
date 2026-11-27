#!/usr/bin/env bash
# Day-18 chain: pre-submission smoke test of bundles/format1.zip under the scorer's current
# harness (swegemma 0.2.11, adk_submission 0.2.13; installed in data/harness_v2_site), after the
# day-17 chain. Serves the unzipped bundle, runs 12 dev24 tasks at the calibrated cap, and checks
# that requests reach the adapter route.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day17_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-17 chain finished"
SUB=submissions/_bundle_format1
rm -rf "$SUB"; mkdir -p "$SUB"; (cd "$SUB" && unzip -q "$ROOT/bundles/format1.zip")
pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$SUB" > logs/vllm_server.log 2>&1 < /dev/null &
scripts/wait_server.sh || exit 1
name="bundle_format1_v2harness_dev12"
echo "=== $(date '+%F %T') start $name"
( sleep 600; echo "--- mid-run LoRA gauge"; curl -s http://127.0.0.1:8000/metrics | grep '^vllm:lora_requests_info' ) &
PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" --runner run_eval_v2.py -- \
  --submission-dir "$SUB" --results-dir "runs/$name" \
  --task-ids-file data/dev24_first12.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
echo "--- vLLM metrics (LoRA routing and request totals)"
curl -s http://127.0.0.1:8000/metrics | grep -E '^vllm:(lora_requests_info|request_success_total)' | head -8
.venv/bin/python scripts/behaviour_vector.py runs/$name runs/format1_sound100_c8 runs/format1_sound100_c8_r2 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') day-18 chain done"
