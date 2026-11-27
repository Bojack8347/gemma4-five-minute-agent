#!/usr/bin/env bash
# Day-8 chain: train the decisiveness direction on hindsight pairs (IMPLEMENTATION.md 6.18),
# gated by the trainer's min-steps and sanity checks, then run iE over the two directions
# (Usage 2: probe + RM steps). Restores the base server on any failure.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STEPS="${1:-3}"
PAIRS=data/pairs/decisive_hindsight_pairs.jsonl
stop_server() {
  pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
  pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 30); do [ "$(free -g | awk 'NR==2{print $3}')" -lt 15 ] && break; sleep 2; done
  rm -f logs/vllm_server.pid
}
start_server() {
  mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null; rm -f logs/vllm_server.pid
  setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$1" > logs/vllm_server.log 2>&1 < /dev/null &
  scripts/wait_server.sh
}
echo "=== $(date '+%F %T') training decisiveness direction on hindsight pairs"
stop_server
.venv-vllm/bin/python scripts/train_direction_dpo.py --pairs "$PAIRS" --out adapters_out/decisive_r16 \
  --epochs 1 --max-prompt-tokens 4096 --max-completion-tokens 768 --chunk 96 --max-pairs-per-context 3 --grad-accum 2 \
  > logs/train_decisive_hindsight.log 2>&1
rc=$?
echo "=== $(date '+%F %T') training rc=$rc"; grep -E "early stop|TOO FEW|\[sanity\]|SANITY|saved adapter" logs/train_decisive_hindsight.log | cut -c1-160
start_server submissions/v0_long || exit 4
if [ "$rc" -ne 0 ] || [ ! -f adapters_out/decisive_r16/adapter_model.safetensors ]; then
  echo "no usable decisiveness adapter; Usage 2 not started"; exit 2
fi
echo "=== $(date '+%F %T') iE over directions: sigma + probe"
.venv/bin/python scripts/ie_control.py configs/ie_directions.yaml probe \
  --noise-runs runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 2>&1 | grep -v -E "UserWarning|check_feature"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') probe rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
echo "=== $(date '+%F %T') iE over directions: $STEPS RM steps"
.venv/bin/python scripts/ie_control.py configs/ie_directions.yaml iterate --steps "$STEPS" 2>&1 | grep -v -E "UserWarning|check_feature"
echo "=== $(date '+%F %T') day-8 chain done rc=${PIPESTATUS[0]}"
