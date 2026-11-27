"""Summarise one evaluation run, or compare two, from task_results.jsonl and traces.

Usage:
  .venv/bin/python scripts/analyze_run.py runs/A            # one run
  .venv/bin/python scripts/analyze_run.py runs/A runs/B     # per-task comparison
"""

from __future__ import annotations

import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

# write_file is deliberately excluded: in practice it is mostly scratch/reproduction
# scripts, which show up in the patch but are not a fix.
EDIT_TOOLS = ('edit_file',)


def load_results(run: Path) -> dict[str, dict]:
    return {
        json.loads(line)['instance_id']: json.loads(line)
        for line in (run / 'task_results.jsonl').read_text().splitlines()
        if line.strip()
    }


SCRATCH_NAME = re.compile(r'(repro|reproduce|debug|scratch|tmp|check_|verify|^test_|_test\.py$)', re.I)


def has_source_edit(run: Path, iid: str) -> bool:
    """True if the captured patch touches a file that is not a scratch script or a test."""
    path = run / 'patches' / f'{iid}.patch'
    if not path.exists():
        return False
    files = re.findall(r'^diff --git a/(\S+)', path.read_text(errors='replace'), flags=re.M)
    return any(not SCRATCH_NAME.search(f.rsplit('/', 1)[-1]) for f in files)


def outcome(r: dict, source_edit: bool) -> str:
    if r['resolved']:
        return 'resolved'
    err = r.get('error') or ''
    if 'timeout' in err:
        stopped = 'timeout'
    elif 'budget' in err:
        stopped = 'out of calls/turns'
    else:
        stopped = 'finished'
    return f'{stopped}, source edit was wrong' if source_edit else f'{stopped}, no source edit'


def agent_turns(trace: dict):
    """Yield (seconds, prompt_tokens, completion_tokens, tool_name, elapsed) per model turn."""
    prev_elapsed = None
    for step in trace.get('steps', []):
        extra = step.get('extra') if isinstance(step.get('extra'), dict) else {}
        elapsed = extra.get('elapsed_s')
        if step.get('source') != 'agent':
            if elapsed is not None:
                prev_elapsed = elapsed
            continue
        metrics = step.get('metrics') or {}
        calls = step.get('tool_calls') or []
        name = calls[0].get('function_name') if calls else '(text)'
        seconds = (elapsed - prev_elapsed) if (elapsed is not None and prev_elapsed is not None) else None
        yield seconds, metrics.get('prompt_tokens') or 0, metrics.get('completion_tokens') or 0, name, elapsed
        if elapsed is not None:
            prev_elapsed = elapsed


def summarise(run: Path) -> dict:
    results = load_results(run)
    by_tool: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    turns_per_task, tokens_per_task, first_edit, compactions = [], [], [], []
    for iid in results:
        path = run / 'traces' / f'trace_{iid}.json'
        if not path.exists():
            continue
        trace = json.loads(path.read_text())
        n_turns = n_tokens = 0
        prev_prompt = 0
        edit_at = None
        for seconds, prompt, completion, name, elapsed in agent_turns(trace):
            n_turns += 1
            n_tokens += completion
            by_tool[name].append((completion, seconds or 0.0))
            # A large drop in prompt size means ADK compacted the history before this turn.
            if n_turns > 1 and prompt - prev_prompt < -2000 and seconds:
                compactions.append(seconds)
            if name in EDIT_TOOLS and edit_at is None:
                edit_at = elapsed
            prev_prompt = prompt
        turns_per_task.append(n_turns)
        tokens_per_task.append(n_tokens)
        first_edit.append(edit_at)
    return {
        'results': results,
        'outcomes': collections.Counter(
            outcome(r, has_source_edit(run, iid)) for iid, r in results.items()
        ),
        'source_edits': sum(has_source_edit(run, iid) for iid in results),
        'by_tool': by_tool,
        'turns_per_task': turns_per_task,
        'tokens_per_task': tokens_per_task,
        'first_edit': first_edit,
        'compactions': compactions,
    }


def print_summary(run: Path) -> dict:
    s = summarise(run)
    results = s['results']
    n = len(results)
    resolved = sum(r['resolved'] for r in results.values())
    print(f'== {run} ==')
    print(f'resolved {resolved}/{n} ({resolved / n:.1%})' if n else 'no tasks')
    for name, count in s['outcomes'].most_common():
        print(f'  {count:3d}  {name}')
    if s['turns_per_task']:
        durations = [r['duration_seconds'] for r in results.values()]
        edits = [e for e in s['first_edit'] if e is not None]
        print(
            f'per task: median {st.median(s["turns_per_task"]):.0f} turns, '
            f'{st.median(s["tokens_per_task"]):.0f} generated tokens, {st.median(durations):.0f} s'
        )
        print(
            f'tasks whose patch touches real source: {s["source_edits"]}/{n}; '
            f'used edit_file: {len(edits)}/{n}'
            + (f', first at median {st.median(edits):.0f} s' if edits else '')
        )
        if s['compactions']:
            print(
                f'context compactions: {len(s["compactions"])}, '
                f'median {st.median(s["compactions"]):.0f} s each'
            )
        print(f'  {"tool":22s} {"turns":>5} {"tokens":>7} {"tok/turn":>8} {"seconds":>8} {"s/turn":>7}')
        for name, rows in sorted(s['by_tool'].items(), key=lambda kv: -sum(sec for _, sec in kv[1])):
            tokens = sum(c for c, _ in rows)
            seconds = sum(sec for _, sec in rows)
            print(f'  {name:22s} {len(rows):5d} {tokens:7d} {tokens / len(rows):8.0f} {seconds:8.0f} {seconds / len(rows):7.1f}')
    return s


def compare(a: Path, b: Path) -> None:
    sa, sb = print_summary(a), print_summary(b)
    ra, rb = sa['results'], sb['results']
    shared = sorted(set(ra) & set(rb))
    both = [i for i in shared if ra[i]['resolved'] and rb[i]['resolved']]
    only_a = [i for i in shared if ra[i]['resolved'] and not rb[i]['resolved']]
    only_b = [i for i in shared if rb[i]['resolved'] and not ra[i]['resolved']]
    print(f'== comparison on {len(shared)} shared tasks ==')
    print(f'both resolved: {len(both)} | only {a.name}: {len(only_a)} {only_a} | only {b.name}: {len(only_b)} {only_b}')


def main() -> int:
    runs = [Path(p) for p in sys.argv[1:]]
    if len(runs) == 1:
        print_summary(runs[0])
    elif len(runs) == 2:
        compare(runs[0], runs[1])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
