#!/usr/bin/env bash
# The informed agent as the target (IMPLEMENTATION.md 6.41): waits for the proxy calibration chain (6.40) to
# release the GPU, sets the whitening from the two title-only base draws, probes the decisiveness and
# thinking-budget columns on the 76 title-only hard tasks (the base and the trajectory and statement columns
# come from existing runs), takes three iE steps toward the informed agent's measured behaviour vector, and
# restores the base server. Each draw is about 4.5 hours.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
CFG=configs/ie_informed.yaml
while pgrep -f "scripts/exp_ie_proxy_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') proxy chain finished; informed chain starts"
scripts/wait_server.sh || exit 1
NOISE="runs/heldout_hard_terse_base_c8"
if [ -f runs/proxy_f000_base_c8/summary.json ]; then NOISE="$NOISE runs/proxy_f000_base_c8"; else
  echo "WARNING: no title-only replicate (runs/proxy_f000_base_c8); dropping it from the probe base"
  sed -i '/proxy_f000_base_c8/d' "$CFG"
fi
echo "=== $(date '+%F %T') probe"
$PY scripts/ie_control.py "$CFG" probe --noise-runs $NOISE 2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\(|^\s*warnings.warn"
echo "=== $(date '+%F %T') probe done rc=${PIPESTATUS[0]}"
echo "=== $(date '+%F %T') iterate"
$PY scripts/ie_control.py "$CFG" iterate --steps 3 2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\(|^\s*warnings.warn"
echo "=== $(date '+%F %T') iterate done rc=${PIPESTATUS[0]}"
$PY scripts/ie_control.py "$CFG" status 2>&1 | grep -v -i warning
# restore the base server
pid=$(cat logs/vllm_server.pid 2>/dev/null); [ -n "$pid" ] && kill "$pid" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "vllm.entrypoints|VLLM::EngineCore" >/dev/null || break; sleep 2; done
pkill -9 -f "vllm.entrypoints|VLLM::EngineCore" 2>/dev/null; sleep 5; rm -f logs/vllm_server.pid
mv logs/vllm_server.log "logs/vllm_server_$(date +%F_%H%M).log" 2>/dev/null
setsid nohup $PY scripts/serve.py --submission-dir submissions/v0_long > logs/vllm_server.log 2>&1 < /dev/null &
scripts/wait_server.sh
echo "=== $(date '+%F %T') ie informed chain done"
