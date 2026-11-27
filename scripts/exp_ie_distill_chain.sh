#!/usr/bin/env bash
# iE calibration of the statement direction (IMPLEMENTATION.md 6.32), then the agent at the calibrated weight:
#   1. after the teacher samples exist: probe (w = 0, w = 1) and 6 RM steps (scripts/ie_distill.py iterate);
#   2. w* = mean of the last three iterates (Polyak averaging); if it differs from 1 by at least 0.25 and its
#      draw is not malformed, package v5_ie = v0_long + w* x statement, run the 76 title-only hard tasks;
#   3. restore the base server.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^.venv/bin/python scripts/ie_distill.py teacher" >/dev/null; do sleep 30; done
[ -s data/ie_distill/teacher.jsonl ] || { echo "no teacher samples; stopping"; exit 1; }
echo "=== $(date '+%F %T') iE iterate"
.venv/bin/python scripts/ie_distill.py iterate --steps 6 2>&1 | grep -v -i -E "warning|check_feature"
.venv/bin/python scripts/ie_distill.py status 2>&1 | grep -v -i warning
W=$(.venv/bin/python - <<'EOF'
import json
st = json.load(open('runs/ie_distill/state.json'))
steps = [r for r in st['runs'] if r['tag'].startswith('step_') and 'guard' not in r]
if len(steps) < 3:
    print('none')
else:
    w = sum(r['next_w'] for r in steps[-3:]) / 3
    print(f'{w:.3f}' if abs(w - 1.0) >= 0.25 and abs(w) >= 0.25 else 'none')  # not the old direction, not the base
EOF
)
echo "=== $(date '+%F %T') calibrated weight: $W"
restart_base() {
  pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
  pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
  mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
  setsid nohup .venv/bin/python scripts/serve.py --submission-dir "$1" > logs/vllm_server.log 2>&1 < /dev/null &
  scripts/wait_server.sh
}
if [ "$W" != "none" ]; then
  SUB=submissions/v5_ie
  rm -rf "$SUB"; cp -r submissions/v0_long "$SUB"; mkdir -p "$SUB/adapters"
  .venv-vllm/bin/python scripts/merge_directions.py --out "$SUB/adapters/merged" --direction "statement=adapters_out/statement_r16:$W" >/dev/null
  rm -f "$SUB/adapters/merged/merge.json"
  sed -i 's/^model: gemma-4-31b-it-qat-w4a16-ct$/model: gemma-4-31b-it-qat-w4a16-ct\nadapter: merged/' "$SUB/agent.yaml"
  restart_base "$SUB" || exit 4
  name=heldout_hard_terse_v5ie_c8
  echo "=== $(date '+%F %T') start $name (w = $W)"
  PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
    --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
    --submission-dir "$SUB" --results-dir "runs/$name" --data-dir data/heldout_hard_terse \
    --task-ids-file data/heldout_hard_terse/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\(|^Task |client_session|Unclosed"
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
  .venv/bin/python scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_v4_c8 "runs/$name" 2>&1 | grep -v -i warning
fi
restart_base submissions/v0_long
echo "=== $(date '+%F %T') ie distill chain done"
