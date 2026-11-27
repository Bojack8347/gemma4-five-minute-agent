#!/usr/bin/env bash
# Statement distillation (IMPLEMENTATION.md 6.31), after the public title-only run:
#   1. mine teacher/student pairs at the student's early states (public fastapi/rich tasks only);
#   2. stop the server, train the direction (anchored DPO, sanity and min-steps gates);
#   3. package v4_distill = v0_long + the direction (weight 1), serve it, evaluate on the 76 title-only
#      hard held-out tasks from other repositories; 4. restore the base server.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_public_terse_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') public title-only run finished"
scripts/wait_server.sh || exit 1
PAIRS=data/pairs/statement_pairs.jsonl
echo "=== $(date '+%F %T') mining"
.venv/bin/python scripts/mine_statement_pairs.py --student runs/public_terse_base_c8 \
  --teacher-runs runs/base_sound100_c8 runs/base_sound100_c8_r2 --out "$PAIRS" --steps-per-task 8 --limit 320 --workers 4 \
  2>&1 | grep -v -E "UserWarning|check_feature" | tee logs/mine_statement.log | tail -3
n_pairs=$(.venv/bin/python -c "import json; print(sum(len(json.loads(l).get('pairs',[])) for l in open('$PAIRS') if l.strip()))" 2>/dev/null || echo 0)
echo "=== $(date '+%F %T') mining done: $n_pairs pairs"
[ "$n_pairs" -ge 60 ] || { echo "too few pairs; stopping"; exit 2; }
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
  --epochs 1 --max-prompt-tokens 4096 --max-completion-tokens 768 --chunk 96 --max-pairs-per-context 2 --grad-accum 2 \
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
