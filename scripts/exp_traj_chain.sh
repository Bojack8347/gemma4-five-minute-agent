#!/usr/bin/env bash
# Trajectory-level distillation (IMPLEMENTATION.md 6.34):
#   1. stop the server; train the LoRA on the informed agent's resolved trajectories rendered title-only;
#   2. package v6_traj = v0_long + the adapter; serve it;
#   3. per-call check at the 224 hard title-only states (agreement with the full-statement teacher at 84 of them),
#      which is the kind of objective the fixed-state check is valid for (6.33);
#   4. the deciding run: the 76 title-only hard tasks (base 13, v4_distill 13); 5. restore the base server.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
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
# 2026-10-23: cap 8192 with truncated over-cap steps and a memory guard; the 12k run of 10-21 thrashed at 108 GiB
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-vllm/bin/python scripts/train_traj_sft.py \
  --data data/traj_sft/trajectories.jsonl --out adapters_out/traj_r16 \
  --max-prompt-tokens 8192 --over-cap-sample 0.2 --max-completion-tokens 1024 --chunk 64 --chunk-long 48 --grad-accum 4 --lr 5e-5 \
  --mem-soft-gib 100 --sanity-tokens 1024 > logs/train_traj.log 2>&1
rc=$?
echo "=== $(date '+%F %T') training rc=$rc"; grep -E "trajectories,|training done|SANITY|\[sanity\]|saved adapter|Error|error" logs/train_traj.log | cut -c1-200 | tail -8
if [ "$rc" -ne 0 ] || [ ! -f adapters_out/traj_r16/adapter_model.safetensors ]; then
  restart_with submissions/v0_long; echo "no usable adapter; stopping"; exit 3
fi
SUB=submissions/v6_traj
rm -rf "$SUB"; cp -r submissions/v0_long "$SUB"; mkdir -p "$SUB/adapters"
.venv-vllm/bin/python scripts/merge_directions.py --out "$SUB/adapters/merged" --direction traj=adapters_out/traj_r16:1.0 >/dev/null
rm -f "$SUB/adapters/merged/merge.json"
sed -i 's/^model: gemma-4-31b-it-qat-w4a16-ct$/model: gemma-4-31b-it-qat-w4a16-ct\nadapter: merged/' "$SUB/agent.yaml"
restart_with "$SUB" || exit 4
echo "=== $(date '+%F %T') per-call check at fixed states"
.venv/bin/python scripts/reach_check.py draw --set hard_terse --config merged 2>&1 | grep -v -i warning | tail -1
cp runs/reach/hard_terse/merged.jsonl runs/reach/hard_terse/v6_traj.jsonl
.venv/bin/python scripts/reach_check.py report --set hard_terse --ref base --configs stmt1 v6_traj 2>&1 | grep -v -i warning
name=heldout_hard_terse_v6_c8
echo "=== $(date '+%F %T') start $name"
PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
  --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
  --submission-dir "$SUB" --results-dir "runs/$name" --data-dir data/heldout_hard_terse \
  --task-ids-file data/heldout_hard_terse/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\(|^Task |client_session|Unclosed"
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_v4_c8 "runs/$name" 2>&1 | grep -v -i warning
restart_with submissions/v0_long
echo "=== $(date '+%F %T') traj chain done"
