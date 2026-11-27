#!/usr/bin/env bash
# Retraining for exp_distill_chain2.sh (IMPLEMENTATION.md 6.31). The first training run early-stopped at
# 80 of 156 contexts (margin 1.81) and was refused by the sanity gate: one greedy generation, a coherent
# thought followed by a reproduction script, ran past the 256-token cap. The gate cap is now 1024 for
# these thinking-on contexts, and a refused adapter is kept in <out>_quarantine. Same pairs, same steps after.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PAIRS=data/pairs/statement_pairs.jsonl
while pgrep -f "^bash scripts/exp_distill_chain2.sh" >/dev/null; do sleep 10; done
restart_with() {
  pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
  pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 30); do [ "$(free -g | awk 'NR==2{print $3}')" -lt 15 ] && break; sleep 2; done
  rm -f logs/vllm_server.pid
  if [ -n "${1:-}" ]; then
    mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
    setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$1" > logs/vllm_server.log 2>&1 < /dev/null &
    scripts/wait_server.sh
  fi
}
echo "=== $(date '+%F %T') training"
restart_with ""
.venv-vllm/bin/python scripts/train_direction_dpo.py --pairs "$PAIRS" --out adapters_out/statement_r16 \
  --epochs 1 --max-prompt-tokens 8192 --max-completion-tokens 768 --chunk 96 --max-pairs-per-context 2 --grad-accum 2 --sanity-tokens 1024 \
  > logs/train_statement.log 2>&1
rc=$?
echo "=== $(date '+%F %T') training rc=$rc"; grep -E "early stop|TOO FEW|\[sanity\]|SANITY|saved adapter" logs/train_statement.log | cut -c1-170
if [ "$rc" -ne 0 ] || [ ! -f adapters_out/statement_r16/adapter_model.safetensors ]; then
  restart_with submissions/v0_long; echo "no usable adapter; stopping"; exit 3
fi
SUB=submissions/v4_distill
rm -rf "$SUB"; cp -r submissions/v0_long "$SUB"; mkdir -p "$SUB/adapters"
.venv-vllm/bin/python scripts/merge_directions.py --out "$SUB/adapters/merged" --direction statement=adapters_out/statement_r16:1.0 >/dev/null
rm -f "$SUB/adapters/merged/merge.json"
sed -i 's/^model: gemma-4-31b-it-qat-w4a16-ct$/model: gemma-4-31b-it-qat-w4a16-ct\nadapter: merged/' "$SUB/agent.yaml"
restart_with "$SUB" || exit 4
name=heldout_hard_terse_v4_c8
echo "=== $(date '+%F %T') start $name"
PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
  --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
  --submission-dir "$SUB" --results-dir "runs/$name" --data-dir data/heldout_hard_terse \
  --task-ids-file data/heldout_hard_terse/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_v4_c8 2>&1 | grep -v -i warning
restart_with submissions/v0_long
echo "=== $(date '+%F %T') distill chain done"
