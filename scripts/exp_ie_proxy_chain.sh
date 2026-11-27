#!/usr/bin/env bash
# iE calibration of the proxy's informativeness knob (IMPLEMENTATION.md 6.40): the base agent (v0_long, 31B)
# on the 76 sound hard tasks with a fraction x of each statement body kept. First draw at x = 0.5, then
# iE steps toward M(x) = 0.13 (scripts/ie_proxy.py); when the iteration converges at a bound, one more draw
# at x = 0.25 for the shape of the curve. Same runner and settings as runs/heldout_hard_terse_base_c8.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
scripts/wait_server.sh || exit 1

draw() {  # draw X
  local x="$1"
  local tag
  tag=$(printf 'f%03d' "$($PY -c "print(int(round($x*100)))")")
  local data="data/heldout_hard_$tag"
  [ -f "$data/tasks.jsonl" ] || $PY scripts/heldout/build_frac_tasks.py "$x"
  local name="proxy_${tag}_base_c8"
  if [ -f "runs/$name/summary.json" ]; then name="${name}_r$(ls -d runs/proxy_${tag}_base_c8* | wc -l)"; fi
  echo "=== $(date '+%F %T') start $name (x=$x)"
  PYTHONPATH="$ROOT/data/harness_v2_site" $PY scripts/run_guarded.py --name "$name" \
    --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
    --submission-dir submissions/v0_long --results-dir "runs/$name" --data-dir "$data" \
    --task-ids-file "$data/sound_tasks.txt" --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency 8 \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
  echo "=== $(date '+%F %T') end $name rc=${PIPESTATUS[0]}"
  $PY scripts/ie_proxy.py record --x "$x" --run "runs/$name"
}

out=$($PY scripts/ie_proxy.py step); echo "$out"
x=$(echo "$out" | awk '/^NEXT/{print $2}')
draws=0
while [ -n "$x" ] && [ "$draws" -lt 3 ]; do
  draw "$x"; draws=$((draws+1))
  out=$($PY scripts/ie_proxy.py step); echo "$out"
  x=$(echo "$out" | awk '/^NEXT/{print $2}')
done
if [ "$draws" -lt 3 ]; then
  echo "=== iteration converged after $draws draw(s); drawing the curve point x=0.25"
  draw 0.25
fi
$PY scripts/ie_proxy.py report
echo "=== $(date '+%F %T') ie proxy chain done"
