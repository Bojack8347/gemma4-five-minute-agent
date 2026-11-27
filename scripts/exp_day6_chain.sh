#!/usr/bin/env bash
# Day-6 chain: second direction (decisiveness, memo 10.5 item 2). After the day-5 chain:
#   1. mine decisiveness pairs at the pre-edit states of resolved trajectories (server idle);
#   2. stop the server, train the direction with the anchored objective and the sanity gate;
#   3. restart the server with the base submission.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day5_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-5 chain finished"
scripts/wait_server.sh || exit 1

echo "=== $(date '+%F %T') mining decisiveness pairs"
.venv/bin/python scripts/mine_decisive_pairs.py \
  runs/v0_long_sound100_c8 runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 \
  runs/ie_pilot/probe_base runs/ie_pilot/step_002 runs/ie_pilot/step_004 \
  --submission-dir submissions/v0_long --out data/pairs/decisive_pairs.jsonl --samples 8 2>&1 | grep -v -E "UserWarning|check_feature"
n_pairs=$(.venv/bin/python -c "import json; print(sum(len(json.loads(l).get('pairs',[])) for l in open('data/pairs/decisive_pairs.jsonl') if l.strip()))" 2>/dev/null || echo 0)
echo "=== $(date '+%F %T') mining done: $n_pairs pairs"
if [ "$n_pairs" -lt 20 ]; then echo "too few pairs; not training"; exit 2; fi

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
echo "=== $(date '+%F %T') training decisiveness direction"
stop_server
.venv-vllm/bin/python scripts/train_direction_dpo.py --pairs data/pairs/decisive_pairs.jsonl --out adapters_out/decisive_r16 \
  --epochs 1 --max-prompt-tokens 4096 --max-completion-tokens 768 --chunk 96 --max-pairs-per-context 2 \
  > logs/train_decisive.log 2>&1
echo "=== $(date '+%F %T') training rc=$?"; grep -E "early stop|\[sanity\]|SANITY|saved adapter" logs/train_decisive.log | cut -c1-160
start_server submissions/v0_long
echo "=== $(date '+%F %T') day-6 chain done"
