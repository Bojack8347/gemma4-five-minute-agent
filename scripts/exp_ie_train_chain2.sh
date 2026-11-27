#!/usr/bin/env bash
# Continuation of exp_ie_train_chain.sh after the probe (IMPLEMENTATION.md 6.36): the targets are now
# ceilings (configs/ie_train.yaml `side: max`), the state was reset to the accepted point, and the four RM
# steps run from the probe's Jacobian. Decision and final run as in the first chain, with the one-sided
# residual; the base server is restored at the end.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG=configs/ie_train.yaml
PY=.venv/bin/python
echo "=== $(date '+%F %T') iterate (ceiling targets)"
$PY scripts/ie_train_loop.py $CFG iterate --steps 4 2>&1 | grep --line-buffered -v -i -E "warning|check_feature"
echo "=== $(date '+%F %T') status"
$PY scripts/ie_train_loop.py $CFG status 2>&1 | grep -v -i warning
DECIDE=$($PY - <<'PYEOF'
import json, sys
sys.path.insert(0, 'scripts')
import numpy as np, yaml
from ie_train_loop import residual_of
cfg = yaml.safe_load(open('configs/ie_train.yaml'))
st = json.load(open('runs/ie_train_mix/state.json'))
w = 1 / np.array(st['sigma'])
def res(r): return float(np.linalg.norm(w * residual_of(cfg, r['y'])))
runs = [r for r in st['runs'] if not r.get('infeasible')]
base = next(r for r in runs if r['label'] == 'probe_base')
acc = [r for r in runs if r.get('gate') == 'accepted' or r['label'] == 'probe_base']
best = min(acc, key=res)
print('final' if best['label'] != 'probe_base' and res(best) < res(base) and st.get('last_accepted_x') == best['x'] else 'none', file=sys.stdout)
print(f"best accepted {best['label']} residual {res(best):.2f} vs base {res(base):.2f}", file=sys.stderr)
PYEOF
)
echo "=== $(date '+%F %T') decision: $DECIDE"
if [ "$DECIDE" = "final" ]; then
  $PY scripts/ie_train_loop.py $CFG final 2>&1 | grep --line-buffered -v -i -E "warning|check_feature"
  last=$(ls -d runs/ie_train_mix/final_n* 2>/dev/null | tail -1)
  [ -n "$last" ] && $PY scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_v6_c8 "$last" 2>&1 | grep -v -i warning
fi
$PY - <<'PYEOF'
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path
from ie_control import restart_server
restart_server(Path('submissions/v0_long').resolve())
PYEOF
echo "=== $(date '+%F %T') ie train chain2 done"
