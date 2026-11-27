#!/usr/bin/env bash
# Day-2 GPU chain (memo section 10.8, step 4 -> first direction):
#   1. wait for exp_day1_chain.sh to finish;
#   2. stop the server (training needs the memory), smoke-test the DPO trainer on a few
#      contexts, then train the format direction on all mined pairs;
#   3. restart the server with a submission that routes v0_long through the adapter and
#      evaluate it on dev24 (guarded, same budget as the noise floor) for a first
#      before/after on the behaviour vector.
# The server is restarted at the end whatever happens, so the machine is never left idle.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PAIRS="${PAIRS:-data/pairs/format_pairs.jsonl}"
OUT="${OUT:-adapters_out/format_r16}"
EPOCHS="${EPOCHS:-1}"

while pgrep -f "^bash scripts/exp_day1_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-1 chain finished"

n_pairs=$(.venv/bin/python -c "import json,sys; print(sum(len(json.loads(l).get('pairs',[])) for l in open('$PAIRS') if l.strip()))" 2>/dev/null || echo 0)
echo "=== pairs available: $n_pairs"
if [ "$n_pairs" -lt 20 ]; then echo "too few pairs; not training"; exit 2; fi

stop_server() {
  # The API server can exit while the engine core (which holds the ~67 GB) lingers, and the
  # reverse; stop both and wait until the memory is actually back before training.
  pid=$(cat logs/vllm_server.pid 2>/dev/null)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
  pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 30); do [ "$(free -g | awk 'NR==2{print $3}')" -lt 15 ] && break; sleep 2; done
  rm -f logs/vllm_server.pid
  echo "server stopped; memory used $(free -g | awk 'NR==2{print $3}') GB"
}
start_server() {
  mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
  rm -f logs/vllm_server.pid   # wait_server.sh must not read the previous server's pid
  setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$1" > logs/vllm_server.log 2>&1 < /dev/null &
  scripts/wait_server.sh
}

echo "=== $(date '+%F %T') stopping server for training"
stop_server
mkdir -p adapters_out logs

echo "=== $(date '+%F %T') trainer smoke test (4 pairs, 1 epoch)"
.venv-vllm/bin/python scripts/train_direction_dpo.py --pairs "$PAIRS" --out "${OUT}_smoke" \
  --limit 4 --epochs 1 --max-prompt-tokens 4096 --max-completion-tokens 768 --chunk 96 --sanity-contexts 0 \
  > logs/train_format_smoke.log 2>&1
rc=$?
echo "=== $(date '+%F %T') smoke rc=$rc"; tail -5 logs/train_format_smoke.log
if [ "$rc" -ne 0 ]; then
  echo "smoke test failed; restarting server and stopping"
  start_server submissions/v0_long
  exit 3
fi

echo "=== $(date '+%F %T') training format direction"
.venv-vllm/bin/python scripts/train_direction_dpo.py --pairs "$PAIRS" --out "$OUT" \
  --epochs "$EPOCHS" --max-prompt-tokens "${PROMPT_TOKENS:-4096}" --max-completion-tokens 768 --chunk 96 --max-pairs-per-context "${PAIRS_PER_CTX:-2}" \
  > logs/train_format.log 2>&1
rc=$?
echo "=== $(date '+%F %T') training rc=$rc"; tail -3 logs/train_format.log

if [ "$rc" -eq 0 ] && [ -f "$OUT/adapter_model.safetensors" ]; then  # rc 5 = sanity gate refused the adapter
  # v0_long routed through the format direction (weight 1), as a plain submission
  SUB=submissions/v0_long_format1
  rm -rf "$SUB"; cp -r submissions/v0_long "$SUB"; mkdir -p "$SUB/adapters"
  .venv-vllm/bin/python scripts/merge_directions.py --out "$SUB/adapters/merged" --direction format="$OUT":1.0 >/dev/null
  rm -f "$SUB/adapters/merged/merge.json"
  sed -i 's/^model: gemma-4-31b-it-qat-w4a16-ct$/model: gemma-4-31b-it-qat-w4a16-ct\nadapter: merged/' "$SUB/agent.yaml"
  echo "=== $(date '+%F %T') starting server with $SUB"
  start_server "$SUB" || exit 4
  echo "=== $(date '+%F %T') evaluating $SUB on dev24"
  .venv/bin/python scripts/run_guarded.py --name format1_dev24 -- \
    --submission-dir "$SUB" --results-dir runs/v0long_format1_dev24_c8 \
    --task-ids-file data/dev24.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature"
  .venv/bin/python scripts/behaviour_vector.py runs/noise_v0long_dev24_c8_r1 runs/noise_v0long_dev24_c8_r2 runs/v0long_format1_dev24_c8
else
  start_server submissions/v0_long
fi
echo "=== $(date '+%F %T') day-2 chain done"
