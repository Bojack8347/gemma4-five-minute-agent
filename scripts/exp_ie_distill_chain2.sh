#!/usr/bin/env bash
# Continuation of exp_ie_distill_chain.sh (IMPLEMENTATION.md 6.32) after its iterate: the agent run is gated on
# the statistic. The probes showed M(-1) ~ M(0) > M(1), so the signed gain drives w towards the lower bound; an
# agent run at the final weight is worth its 4.5 hours only if the draws there beat the base's agreement by more
# than two paired standard errors.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^.venv/bin/python scripts/ie_distill.py iterate" >/dev/null; do sleep 30; done
.venv/bin/python scripts/ie_distill.py status 2>&1 | grep -v -i warning
W=$(.venv/bin/python - <<'EOF'
import json, sys
st = json.load(open('runs/ie_distill/state.json'))
steps = [r for r in st['runs'] if r['tag'].startswith('step_') and 'guard' not in r]
def per(tag):
    return {(r['instance_id'], r['step_id']): r['agree'] for r in map(json.loads, open(f'runs/ie_distill/draw_{tag}.jsonl'))}
if len(steps) < 3:
    print('none'); raise SystemExit
w = round(sum(r['next_w'] for r in steps[-3:]) / 3, 3)
at = [r['tag'] for r in st['runs'] if abs(r['w'] - w) < 0.05]   # every draw taken at the final weight
base = per('probe_w0')
if not at or abs(w) < 0.25 or abs(w - 1.0) < 0.25:
    print('none'); raise SystemExit
draws = [per(t) for t in at]
d = [sum(x[k] for x in draws) / len(draws) - base[k] for k in base]
m = sum(d) / len(d); se = (sum((v - m) ** 2 for v in d) / (len(d) - 1)) ** .5 / len(d) ** .5
print(f'final w {w}: {len(at)} draws, M - M(0) = {m:+.4f} (paired se {se:.4f})', file=sys.stderr)
print(f'{w:.3f}' if m > 2 * se else 'none')
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
