#!/usr/bin/env bash
# 12B comparison (IMPLEMENTATION.md 6.28), after the held-out chain:
#   1. serve gemma-4-12b-it-qat-w4a16-ct and measure local speed (31B local: 10.6 tok/s single-stream);
#   2. v0_12b (v0_long with the 12B declared) on the 95 held-out tasks (0.2.11 harness, as heldout_base_c8);
#   3. v0_12b on the 100 public sound tasks (0.2.7 runner, as base_sound100_c8 and _r2).
# The cap stays at 34 minutes at concurrency 8, which assumes the 12B's local-to-Kaggle speed ratio
# equals the 31B's (unmeasured).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_heldout_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') held-out chain finished"
until [ ! -f models/gemma-4-12b-it-qat-w4a16-ct/gemma-4.tar.gz.kaggle-partial ] && ls models/gemma-4-12b-it-qat-w4a16-ct/*.safetensors >/dev/null 2>&1; do sleep 30; done
pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup .venv/bin/python scripts/serve.py --model 12b --submission-dir submissions/v0_12b > logs/vllm_server.log 2>&1 < /dev/null &
scripts/wait_server.sh || exit 1
echo "=== $(date '+%F %T') speed probe (12B, local)"
.venv/bin/python scripts/speed_probe.py http://127.0.0.1:8000/v1 gemma-4-12b-it-qat-w4a16-ct > runs/speed_probe_local_12b.json 2>&1; cat runs/speed_probe_local_12b.json | head -40
name=heldout_12b_c8
echo "=== $(date '+%F %T') start $name"
PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$name" \
  --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
  --submission-dir submissions/v0_12b --results-dir "runs/$name" --data-dir data/heldout \
  --task-ids-file data/heldout/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
name=v012b_sound100_c8
echo "=== $(date '+%F %T') start $name"
.venv/bin/python scripts/run_guarded.py --name "$name" -- \
  --submission-dir submissions/v0_12b --results-dir "runs/$name" \
  --task-ids-file data/sound_tasks.txt --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
  2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
.venv/bin/python scripts/behaviour_vector.py runs/heldout_base_c8 runs/heldout_12b_c8 runs/base_sound100_c8 runs/base_sound100_c8_r2 runs/v012b_sound100_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') 12b chain done"
