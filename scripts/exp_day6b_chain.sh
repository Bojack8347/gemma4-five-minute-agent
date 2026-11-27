#!/usr/bin/env bash
# Day-6b: fallback for the decisiveness direction. If day 6 produced no adapter (too few
# pre-edit pairs or the sanity gate refused it), mine the earliest informed states instead
# (87 contexts offline, median 9 exploration steps before the edit) and train on those.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day6_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-6 chain finished"
if [ -f adapters_out/decisive_r16/adapter_model.safetensors ]; then
  echo "decisiveness adapter exists; nothing to do"; exit 0
fi
scripts/wait_server.sh || exit 1
echo "=== $(date '+%F %T') mining earliest-state pairs"
.venv/bin/python scripts/mine_decisive_pairs.py --state earliest \
  runs/v0_long_sound100_c8 runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 \
  runs/ie_pilot/probe_base runs/ie_pilot/step_002 runs/ie_pilot/step_004 \
  runs/v0long_format1_dev24_c8 runs/pilot_end_dev24_c8_r1 runs/pilot_end_dev24_c8_r2 \
  --submission-dir submissions/v0_long --out data/pairs/decisive_earliest_pairs.jsonl --samples 8 2>&1 | grep -v -E "UserWarning|check_feature"
# train on the union of the pre-edit pairs (42, day 6) and the earliest-state pairs
cat data/pairs/decisive_pairs.jsonl data/pairs/decisive_earliest_pairs.jsonl > data/pairs/decisive_all_pairs.jsonl
n_pairs=$(.venv/bin/python -c "import json; print(sum(len(json.loads(l).get('pairs',[])) for l in open('data/pairs/decisive_all_pairs.jsonl') if l.strip()))" 2>/dev/null || echo 0)
echo "=== $(date '+%F %T') mining done: $n_pairs pairs in the union"
if [ "$n_pairs" -lt 60 ]; then echo "too few pairs; not training"; exit 2; fi
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
echo "=== $(date '+%F %T') training decisiveness direction (earliest states)"
stop_server
.venv-vllm/bin/python scripts/train_direction_dpo.py --pairs data/pairs/decisive_all_pairs.jsonl --out adapters_out/decisive_r16 \
  --epochs 1 --max-prompt-tokens 4096 --max-completion-tokens 768 --chunk 96 --max-pairs-per-context 4 --grad-accum 2 \
  > logs/train_decisive_earliest.log 2>&1
echo "=== $(date '+%F %T') training rc=$?"; grep -E "early stop|\[sanity\]|SANITY|saved adapter" logs/train_decisive_earliest.log | cut -c1-160
start_server submissions/v0_long
echo "=== $(date '+%F %T') day-6b chain done"
