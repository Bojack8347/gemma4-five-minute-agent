#!/usr/bin/env bash
# Validation of the fixed-state reachability check (IMPLEMENTATION.md 6.33): predict, at fixed agent states
# and with the agent's own sampling, the behaviour changes that agent runs already measured for the three
# trained directions, then show the gate verdicts. One server mounts every configuration.
#   public set (300 states, base agent on the 100 sound public tasks): base, format 1, format 2,
#     format 1 + decisiveness 1, format 1 + decisiveness -0.5
#   hard title-only set (224 states, 84 with a full-statement teacher): base, statement 1, statement 2
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=".venv/bin/python scripts/reach_check.py"
F=adapters_out/format_r16_v2; D=adapters_out/decisive_r16; S=adapters_out/statement_r16
echo "=== $(date '+%F %T') serving every configuration"
$PY serve --configs "fmt1=$F:1" "fmt2=$F:2" "fmt1_dec1=$F:1,$D:1" "fmt1_decm05=$F:1,$D:-0.5" "stmt1=$S:1" "stmt2=$S:2" \
  2>&1 | grep -v -i -E "warning|EngineCore|INFO" | tail -4
curl -sf localhost:8000/v1/models | grep -q stmt2 || { echo "server did not mount the configurations"; exit 1; }
for c in base fmt1 fmt1_dec1 fmt1_decm05 fmt2; do
  $PY draw --set public --config "$c" 2>&1 | grep --line-buffered -v -i warning
done
$PY teacher --set hard_terse 2>&1 | grep --line-buffered -v -i warning
for c in base stmt1 stmt2; do
  $PY draw --set hard_terse --config "$c" 2>&1 | grep --line-buffered -v -i warning
done
echo "=== $(date '+%F %T') reports"
$PY report --set public --ref base --configs fmt1 fmt2 2>&1 | grep -v -i warning
$PY report --set public --ref fmt1 --configs fmt1_dec1 fmt1_decm05 2>&1 | grep -v -i warning
$PY report --set hard_terse --ref base --configs stmt1 stmt2 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') gates"
$PY gate --set public --ref base --probes fmt1:1 fmt2:2 --target malformed=0.005 2>&1 | grep -v -i warning
$PY gate --set hard_terse --ref base --probes stmt1:1 stmt2:2 --target agree=$(.venv/bin/python -c "
import json, itertools, sys; sys.path.insert(0, 'scripts')
from reach_check import agree
rs = [json.loads(l) for l in open('data/reach/hard_terse/teacher.jsonl')]
per = [sum(agree(a, b) for a, b in itertools.combinations(r['labels'], 2)) / 6 for r in rs]
print(round(sum(per) / len(per), 4))" 2>/dev/null) 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') restoring the base server"
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path
from ie_control import restart_server
restart_server(Path('submissions/v0_long').resolve())
EOF
echo "=== $(date '+%F %T') reach chain done"
