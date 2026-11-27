"""Mine on-policy pairs for the *decisiveness* direction (memo 10.5, item 2).

At the state just before a resolved trajectory's first successful `edit_file`, the model
demonstrably knew enough to edit. Rebuild that exact context, sample N raw completions, and
pair: chosen = a well-formed `edit_file` at the same file and location as the trajectory's
successful edit (correctness by agreement with a known-good edit); rejected = a well-formed
non-edit call (more reading or searching). Same record format as `mine_format_pairs.py`, so
`train_direction_dpo.py` consumes it unchanged.

Usage:
  .venv/bin/python scripts/mine_decisive_pairs.py runs/noise_v0long_dev24_c8_r1 runs/v0_long_sound100_c8 \
      --out data/pairs/decisive_pairs.jsonl --samples 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_format_pairs import (MODEL, Snapshot, classify, compacted_before, guess_submission,  # noqa: E402
                               post, steps_to_messages, tool_specs)

ROOT = Path(__file__).resolve().parent.parent


def first_good_edit(trace: dict):
    """(step, call) of the first edit_file whose observation reports success."""
    for s in trace['steps']:
        if s['source'] != 'agent':
            continue
        obs = (s.get('observation') or {}).get('content') or ''
        for c in s.get('tool_calls') or []:
            if c['function_name'] == 'edit_file' and obs[:40].startswith('{"status": "ok"'):
                return s, c
    return None, None


def earliest_informed_step(trace: dict, edit_step: dict, good_args: dict):
    """The first agent step after the model had already seen the text it would later edit.

    Scans observations in order; once the good edit's first non-empty line of `old_string`
    has appeared in a tool result, the next agent step is a state where editing was possible
    but the trajectory kept exploring. Returns None if that step is the edit itself.
    """
    core = ''
    for ln in str(good_args.get('old_string', '')).splitlines():
        if ln.strip():
            core = ln.strip()
            break
    if len(core) < 8:
        return None
    seen = False
    for s in trace['steps']:
        if s['step_id'] >= edit_step['step_id']:
            return None
        if s['source'] != 'agent':
            continue
        if seen:
            return s
        obs = (s.get('observation') or {}).get('content') or ''
        if core in obs or core.replace('"', '\\"') in obs:
            seen = True
    return None


def same_location(sample_args: dict, good_args: dict) -> bool:
    if sample_args.get('filepath') != good_args.get('filepath'):
        return False
    a, b = str(sample_args.get('old_string', '')), str(good_args.get('old_string', ''))
    if not a or not b:
        return False
    # the sampled edit targets the same region if either old_string contains the other's core
    core = lambda t: t.strip().splitlines()[0].strip() if t.strip() else ''
    return core(a) in b or core(b) in a


def mine_run(run: Path, out, args, seen: set) -> dict:
    sub_dir = Path(args.submission_dir).resolve() if args.submission_dir else guess_submission(run)
    sampling = yaml.safe_load((sub_dir / 'configs' / 'sampling.yaml').read_text())
    agent_name, tools = tool_specs(sub_dir)
    results = {json.loads(l)['instance_id']: json.loads(l) for l in (run / 'task_results.jsonl').read_text().splitlines() if l.strip()}
    stats = {'contexts': 0, 'mismatch': 0, 'compacted': 0, 'pairs': 0, 'edit_ok': 0, 'explore': 0, 'other': 0}
    for iid, r in results.items():
        if not r.get('resolved'):
            continue
        key = (run.name, iid)
        if key in seen or stats['contexts'] >= args.limit:
            continue
        trace_path = run / 'traces' / f'trace_{iid}.json'
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text())
        step, good = first_good_edit(trace)
        if step is None:
            continue
        if args.state == 'earliest':
            step = earliest_informed_step(trace, step, good.get('arguments') or {})
            if step is None:
                continue
        seen.add(key)
        if compacted_before(trace, step['step_id']):
            stats['compacted'] += 1
            continue
        # Thinking-off rendering: the completions endpoint carries no thinking budget, and the
        # format direction trained this way transferred to the thinking-on agent.
        messages = steps_to_messages(trace, step['step_id'], agent_name)
        kw = {'enable_thinking': False}
        tok = post('/tokenize', {'model': MODEL, 'messages': messages, 'tools': tools,
                                 'chat_template_kwargs': kw, 'add_generation_prompt': True})
        stats['contexts'] += 1
        body = {'model': MODEL, 'prompt': tok['tokens'], 'n': args.samples,
                'temperature': args.temperature, 'top_p': 0.95, 'max_tokens': args.max_tokens, 'skip_special_tokens': False}
        t0 = time.time()
        res = post('/v1/completions', body)
        snap = Snapshot(iid)
        # replay the trajectory's earlier successful edits so old_string checks see the right file
        for s in trace['steps']:
            if s['step_id'] >= step['step_id'] or s['source'] != 'agent':
                continue
            obs = (s.get('observation') or {}).get('content') or ''
            for c in s.get('tool_calls') or []:
                a = c.get('arguments') or {}
                if c['function_name'] == 'edit_file' and obs[:40].startswith('{"status": "ok"') and all(isinstance(a.get(k), str) for k in ('filepath', 'old_string', 'new_string')):
                    snap.apply_edit(a['filepath'], a['old_string'], a['new_string'])
        samples, edits, explores = [], [], []
        good_args = good.get('arguments') or {}
        for ch in res['choices']:
            lab = {'status': 'truncated'} if ch.get('finish_reason') == 'length' else classify(ch['text'], snap)
            rec_s = {'text': ch['text'], **{k: v for k, v in lab.items() if k != 'args'}}
            if lab['status'] == 'ok' and lab.get('tool') == 'edit_file' and same_location(lab.get('args', {}), good_args):
                rec_s['role'] = 'edit_at_location'
                edits.append(rec_s)
                stats['edit_ok'] += 1
            elif lab['status'] == 'ok' and lab.get('tool') in ('read_file', 'run_command', 'search_similar_code', 'get_code_neighbors', 'get_code_subgraph'):
                rec_s['role'] = 'explore'
                explores.append(rec_s)
                stats['explore'] += 1
            else:
                stats['other'] += 1
            samples.append(rec_s)
        pairs = [{'chosen': e['text'], 'rejected': x['text'], 'chosen_tool': 'edit_file', 'rejected_tool': x['tool'], 'why': 'explore_instead_of_edit'}
                 for e in edits for x in explores][:args.max_pairs]
        stats['pairs'] += len(pairs)
        rec = {'run': run.name, 'instance_id': iid, 'step_id': step['step_id'], 'prompt_tokens': tok['count'],
               'enable_thinking': False, 'original_tool': 'edit_file', 'messages': messages, 'tools': tools,
               'sampling': {'temperature': args.temperature, 'top_p': 0.95}, 'samples': samples, 'pairs': pairs,
               'good_edit': {'filepath': good_args.get('filepath'), 'old_string': str(good_args.get('old_string', ''))[:200]}}
        out.write(json.dumps(rec) + '\n')
        out.flush()
        print(f'  {iid} step {step["step_id"]:3d} ({tok["count"]:5d} tok): {len(edits)} edits at location / {len(explores)} explore / '
              f'{len(samples) - len(edits) - len(explores)} other, {len(pairs)} pairs, {time.time() - t0:.0f}s', flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--submission-dir', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'pairs' / 'decisive_pairs.jsonl')
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.6, help='sampling temperature for the alternatives')
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--max-pairs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=10 ** 9)
    ap.add_argument('--state', choices=['pre_edit', 'earliest'], default='pre_edit',
                    help='pre_edit: the state right before the successful edit; earliest: the first state after the edited text had been seen')
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen.add((r['run'], r['instance_id']))
    with args.out.open('a') as out:
        for run in args.runs:
            print(f'== {run}')
            print(f'   {mine_run(run, out, args, seen)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
