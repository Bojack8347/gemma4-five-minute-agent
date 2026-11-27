#!/usr/bin/env bash
# iE as the outer loop of DPO (IMPLEMENTATION.md 6.36): probe (base + one retrain per knob), four RM steps,
# then the accepted end point on all 76 title-only hard tasks if its whitened residual on the batch is below
# the probe base's; finally the base server is restored. Every draw stops the server, retrains, serves and
# evaluates on the fixed 24-task batch (about 3-4 h).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG=configs/ie_train.yaml
PY=.venv/bin/python
echo "=== $(date '+%F %T') probe"
$PY scripts/ie_train_loop.py $CFG probe 2>&1 | grep -v -i -E "warning|check_feature"
echo "=== $(date '+%F %T') iterate"
$PY scripts/ie_train_loop.py $CFG iterate --steps 4 2>&1 | grep -v -i -E "warning|check_feature"
echo "=== $(date '+%F %T') status"
$PY scripts/ie_train_loop.py $CFG status 2>&1 | grep -v -i warning
DECIDE=$($PY - <<'EOF'
import json, sys
sys.path.insert(0, 'scripts')
import numpy as np, yaml
from ie_control import z_of, z_target
cfg = yaml.safe_load(open('configs/ie_train.yaml'))
st = json.load(open('runs/ie_train_mix/state.json'))
w = 1 / np.array(st['sigma'])
def res(r): return float(np.linalg.norm(w * np.nan_to_num(z_target(cfg) - z_of(cfg, r['y']), nan=0.0)))
runs = [r for r in st['runs'] if not r.get('infeasible')]
base = next(r for r in runs if r['label'] == 'probe_base')
acc = [r for r in runs if r.get('gate') == 'accepted' or r['label'] == 'probe_base']
best = min(acc, key=res)
print('final' if best['label'] != 'probe_base' and res(best) < res(base) and st.get('last_accepted_x') == best['x'] else 'none', file=sys.stdout)
print(f"best accepted {best['label']} residual {res(best):.2f} vs base {res(base):.2f}", file=sys.stderr)
EOF
)
echo "=== $(date '+%F %T') decision: $DECIDE"
if [ "$DECIDE" = "final" ]; then
  $PY scripts/ie_train_loop.py $CFG final 2>&1 | grep -v -i -E "warning|check_feature"
  last=$(ls -d runs/ie_train_mix/final_n* 2>/dev/null | tail -1)
  [ -n "$last" ] && $PY scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_v6_c8 "$last" 2>&1 | grep -v -i warning
fi
$PY - <<'EOF'
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path
from ie_control import restart_server
restart_server(Path('submissions/v0_long').resolve())
EOF
echo "=== $(date '+%F %T') ie train chain done"
