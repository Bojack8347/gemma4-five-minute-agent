#!/usr/bin/env bash
# Skills variant (v1_skills: code-tools skill with locate.py and run_tests.py), after the hard chain:
#   1. smoke on 4 hard tasks; stop unless run_skill_script calls succeed in the sandbox;
#   2. the 76 hard tasks with title-only statements, same settings as heldout_hard_terse_base_c8.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_hard_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') hard chain finished"
scripts/wait_server.sh || exit 1
run() {  # name, task file, concurrency
  PYTHONPATH="$ROOT/data/harness_v2_site" .venv/bin/python scripts/run_guarded.py --name "$1" \
    --runner run_eval_v2.py --python "$ROOT/.venv-heldout/bin/python" -- \
    --submission-dir submissions/v1_skills --results-dir "runs/$1" --data-dir data/heldout_hard_terse \
    --task-ids-file "$2" --max-time-minutes 34 --max-tool-calls 100 --max-turns 250 --concurrency "$3" \
    2>&1 | grep -v -E "UserWarning|check_feature|EventsCompactionConfig\("
}
echo "=== $(date '+%F %T') smoke"
run heldout_hard_terse_skills_smoke data/heldout_hard/smoke4.txt 4
.venv/bin/python - <<'PY'
import json, glob, sys, collections
calls = collections.Counter(); ok = 0; bad = []
for f in glob.glob('runs/heldout_hard_terse_skills_smoke/traces/trace_*.json'):
    for s in json.load(open(f))['steps']:
        for c in s.get('tool_calls') or []:
            calls[c['function_name']] += 1
            if c['function_name'] == 'run_skill_script':
                obs = str((s.get('observation') or {}).get('content') or '')
                if '"status": "error"' in obs[:60] or 'Traceback' in obs or 'not found' in obs[:200].lower():
                    bad.append(obs[:300])
                else:
                    ok += 1
print('tool calls:', dict(calls))
print('run_skill_script ok:', ok, '| errors:', len(bad))
for b in bad[:3]: print('  ERROR:', b)
sys.exit(0 if ok > 0 and ok >= len(bad) else 3)
PY
rc=$?
if [ "$rc" -ne 0 ]; then echo "skill calls failing in the sandbox; stopping"; exit 3; fi
echo "=== $(date '+%F %T') full hard set"
run heldout_hard_terse_skills_c8 data/heldout_hard_terse/sound_tasks.txt 8
.venv/bin/python scripts/behaviour_vector.py runs/heldout_hard_terse_base_c8 runs/heldout_hard_terse_skills_c8 2>&1 | grep -v -i warning
echo "=== $(date '+%F %T') skills chain done"
