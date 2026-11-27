"""Behaviour vector y of one evaluation run, for the iE controller (memo section 10.3).

Each coordinate is a per-task or per-call quantity with far lower noise than the
resolution rate. Rates are reported raw and on the logit scale, times and counts on
the log scale, so that the forward map x -> y is closer to linear.

Usage:
  .venv/bin/python scripts/behaviour_vector.py runs/A [runs/B ...]   # table
  .venv/bin/python scripts/behaviour_vector.py --json runs/A         # one JSON object

Importable: `behaviour_vector(run_dir) -> dict[str, float]`.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_run import agent_turns, has_source_edit, load_results  # noqa: E402

# Local-to-Kaggle wall-clock ratio measured in IMPLEMENTATION.md 7.1 (one agent turn).
LOCAL_TO_KAGGLE = 1 / 3.2

# Coordinates the controller may target (name -> transform). Everything else is tracked only.
COORDS = {
    'garbled_call_rate': 'logit',
    'source_edit_rate': 'logit',
    'first_edit_s_kaggle': 'log',
    'turns_per_task': 'log',
    'tokens_per_turn': 'log',
    'repeat_loop_rate': 'logit',
    'scratch_write_rate': 'logit',
    'compactions_per_task': 'identity',
    'empty_patch_rate': 'logit',
    'resolution_rate': 'logit',
    'wrong_edit_rate': 'logit',
}


def _is_garbled(call: dict, observation: str) -> bool:
    """A tool call whose arguments did not survive the native call format.

    Signatures seen in traces (IMPLEMENTATION.md 6.6): argument keys that are not
    identifiers (`end_line"`, or a code fragment that swallowed the next argument),
    a path that absorbed `,start_line:...`, or the harness rejecting the call for a
    missing mandatory parameter.
    """
    args = call.get('arguments') or {}
    if any(not str(k).isidentifier() for k in args):
        return True
    fp = args.get('filepath')
    if isinstance(fp, str) and ('`' in fp or ',start_line' in fp or ',end_line' in fp):
        return True
    head = observation[:400].lower()
    return 'mandatory parameter' in head or 'missing required' in head


def _clip(p: float, n: int) -> float:
    # Laplace-style clipping keeps the logit finite at 0 and 1.
    lo, hi = 0.5 / max(n, 1), 1 - 0.5 / max(n, 1)
    return min(max(p, lo), hi)


def transform(name: str, value: float, n: int = 100) -> float:
    kind = COORDS.get(name, 'identity')
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float('nan')
    if kind == 'logit':
        p = _clip(value, n)
        return math.log(p / (1 - p))
    if kind == 'log':
        return math.log(max(value, 1e-9))
    return value


def behaviour_vector(run: Path) -> dict[str, float]:
    run = Path(run)
    results = load_results(run)
    n_tasks = len(results)
    calls = garbled = 0
    repeat_tasks = scratch_tasks = 0
    turns_all, tok_per_turn, first_edit, compactions_total = [], [], [], 0
    for iid in results:
        path = run / 'traces' / f'trace_{iid}.json'
        if not path.exists():
            continue
        trace = json.loads(path.read_text())
        n_turns = 0
        prev_prompt = 0
        prev_sig = None
        streak = max_streak = 0
        edit_at = None
        scratch = False
        for step in trace.get('steps', []):
            if step.get('source') != 'agent':
                continue
            obs = (step.get('observation') or {}).get('content') or ''
            for c in step.get('tool_calls') or []:
                calls += 1
                if _is_garbled(c, obs):
                    garbled += 1
                sig = (c.get('function_name'), json.dumps(c.get('arguments'), sort_keys=True))
                streak = streak + 1 if sig == prev_sig else 1
                max_streak = max(max_streak, streak)
                prev_sig = sig
                if c.get('function_name') == 'write_file':
                    scratch = True
        for seconds, prompt, completion, name, elapsed in agent_turns(trace):
            n_turns += 1
            if completion:
                tok_per_turn.append(completion)
            if n_turns > 1 and prompt - prev_prompt < -2000 and seconds:
                compactions_total += 1
            if name == 'edit_file' and edit_at is None:
                edit_at = elapsed
            prev_prompt = prompt
        turns_all.append(n_turns)
        first_edit.append(edit_at)
        repeat_tasks += max_streak >= 4
        scratch_tasks += scratch

    resolved = sum(r['resolved'] for r in results.values())
    source_edits = {iid: has_source_edit(run, iid) for iid in results}
    n_source = sum(source_edits.values())
    wrong = sum(1 for iid, r in results.items() if source_edits[iid] and not r['resolved'])
    empty = sum(1 for r in results.values() if not r.get('agent_patch_size'))
    edits = [e for e in first_edit if e is not None]
    y = {
        'n_tasks': n_tasks,
        'n_calls': calls,
        'resolution_rate': resolved / n_tasks if n_tasks else float('nan'),
        'garbled_call_rate': garbled / calls if calls else float('nan'),
        'source_edit_rate': n_source / n_tasks if n_tasks else float('nan'),
        'wrong_edit_rate': wrong / n_tasks if n_tasks else float('nan'),
        'empty_patch_rate': empty / n_tasks if n_tasks else float('nan'),
        'edit_attempt_rate': len(edits) / n_tasks if n_tasks else float('nan'),
        # Local seconds scaled to Kaggle wall clock; tasks that never edit are censored at the cap.
        'first_edit_s_kaggle': st.median(edits) * LOCAL_TO_KAGGLE if edits else float('nan'),
        'turns_per_task': st.median(turns_all) if turns_all else float('nan'),
        'tokens_per_turn': st.median(tok_per_turn) if tok_per_turn else float('nan'),
        'repeat_loop_rate': repeat_tasks / n_tasks if n_tasks else float('nan'),
        'scratch_write_rate': scratch_tasks / n_tasks if n_tasks else float('nan'),
        'compactions_per_task': compactions_total / n_tasks if n_tasks else float('nan'),
    }
    for name in COORDS:
        n = calls if name == 'garbled_call_rate' else n_tasks
        y[f'z_{name}'] = transform(name, y[name], n)
    return y


def main() -> int:
    argv = sys.argv[1:]
    as_json = '--json' in argv
    runs = [Path(a) for a in argv if a != '--json']
    if not runs:
        print(__doc__)
        return 2
    vectors = {r.name: behaviour_vector(r) for r in runs}
    if as_json:
        print(json.dumps(vectors if len(runs) > 1 else next(iter(vectors.values())), indent=1))
        return 0
    names = ['n_tasks', 'n_calls'] + list(COORDS) + ['edit_attempt_rate']
    width = max(len(n) for n in names)
    print(f'{"":{width}}  ' + '  '.join(f'{r.name[:18]:>18}' for r in runs))
    for name in names:
        row = []
        for r in runs:
            v = vectors[r.name][name]
            row.append(f'{v:18.3f}' if isinstance(v, float) else f'{v:18d}')
        print(f'{name:{width}}  ' + '  '.join(row))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
