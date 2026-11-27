"""Trajectory-level distillation data (IMPLEMENTATION.md 6.34).

From the informed agent's resolved trajectories (full problem statement), build title-only imitation
targets: at every agent step before the first context compaction, the prompt is the history rendered
with the statement cut to its title, and the target is what the informed agent emitted there, in the
model's own generation form:

    <|channel>thought\n{thought}<channel|><|tool_call>call:{name}{args}<tool_call|>

with arguments serialised exactly as the chat template serialises them (sorted keys, <|"|> strings).
A thought that quotes the full statement (a 5-gram of the statement that the title lacks) is kept in
the target for conditioning but masked out of the loss, so the student never learns to recite text it
cannot see. Steps whose edit the harness rejected are skipped (they stay in the history).

Run once per harness version (tool declarations differ by 75 tokens between 0.2.7 and 0.2.11):
  .venv/bin/python scripts/build_traj_sft.py --runs base_sound100_c8 base_sound100_c8_r2 v0_long_sound100_c8 \
      --tasks data/competition/tasks.jsonl
  PYTHONPATH=data/harness_v2_site .venv/bin/python scripts/build_traj_sft.py --runs heldout_base_c8 \
      --tasks data/heldout/tasks.jsonl --exclude data/heldout_hard_terse/sound_tasks.txt --append
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_format_pairs import compacted_before, tool_specs  # noqa: E402
from mine_statement_pairs import build_messages, detect_thought_mode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'traj_sft' / 'trajectories.jsonl'
STR = '<|"|>'


def fmt_arg(v) -> str:
    """The chat template's format_argument with escape_keys=False."""
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, str):
        return f'{STR}{v}{STR}'
    if isinstance(v, dict):
        return '{' + ','.join(f'{k}:{fmt_arg(x)}' for k, x in sorted(v.items())) + '}'
    if isinstance(v, (list, tuple)):
        return '[' + ','.join(fmt_arg(x) for x in v) + ']'
    if v is None:
        raise ValueError('null argument')
    return str(v)


def native_call(name: str, args: dict) -> str:
    return f'<|tool_call>call:{name}{{' + ','.join(f'{k}:{fmt_arg(v)}' for k, v in sorted(args.items())) + '}<tool_call|>'


def title_of(statement: str) -> str:
    return re.sub(r'\s*\(#\d+\)', '', statement.split('\n')[0]).strip()


def grams(text: str, n: int = 5) -> set:
    w = re.findall(r'\w+', text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--tasks', required=True, help='tasks.jsonl with the full statements the runs saw')
    ap.add_argument('--exclude', default=None, help='instance ids never to train on (the evaluation set)')
    ap.add_argument('--submission-dir', default='submissions/v0_long')
    ap.add_argument('--append', action='store_true')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()

    full = {json.loads(l)['instance_id']: json.loads(l)['problem_statement'] for l in open(ROOT / args.tasks) if l.strip()}
    exclude = set()
    if args.exclude:
        exclude = {l.strip() for l in open(ROOT / args.exclude) if l.strip()}
    agent_name, tools = tool_specs((ROOT / args.submission_dir).resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.open('a' if args.append else 'w')
    tot = {'traj': 0, 'steps': 0, 'kept': 0, 'masked': 0, 'failed_edit': 0, 'no_call': 0, 'no_sub': 0}
    for run in args.runs:
        rdir = ROOT / 'runs' / run
        res = [json.loads(l) for l in (rdir / 'task_results.jsonl').read_text().splitlines() if l.strip()]
        mode = None
        for r in res:
            iid = r['instance_id']
            if not r['resolved'] or iid in exclude or iid not in full:
                continue
            tp = rdir / 'traces' / f'trace_{iid}.json'
            if not tp.exists():
                continue
            trace = json.loads(tp.read_text())
            if mode is None:
                mode = detect_thought_mode(trace, agent_name, tools)
                if mode is None:
                    raise SystemExit(f'{run}: could not match the harness rendering; use the other harness on the path')
            agent = [s for s in trace['steps'] if s['source'] == 'agent']
            pre = [s for s in agent if not compacted_before(trace, s['step_id'])]
            if not pre:
                continue
            msgs = build_messages(trace, pre[-1]['step_id'] + 1, agent_name, mode)
            ui = next(i for i, m in enumerate(msgs) if m['role'] == 'user')
            stmt, title = full[iid], title_of(full[iid])
            content = msgs[ui]['content']
            new = None
            for cand in (stmt, stmt.strip(), stmt.replace('\r\n', '\n'), stmt.replace('\r\n', '\n').strip()):
                if cand and cand in content:
                    new = content.replace(cand, title, 1)
                    break
            if new is None:
                tot['no_sub'] += 1
                continue
            msgs[ui]['content'] = new
            sg = grams(stmt) - grams(title)
            steps = []
            for s in pre:
                n_msgs = len(build_messages(trace, s['step_id'], agent_name, mode))
                calls = s.get('tool_calls') or []
                rec = {'step_id': s['step_id'], 'n_msgs': n_msgs}
                tot['steps'] += 1
                if len(calls) != 1:
                    tot['no_call'] += 1
                    rec['skip'] = 'no_single_call'
                    steps.append(rec)
                    continue
                c = calls[0]
                obs = s.get('observation')
                obs_text = (obs.get('content') if isinstance(obs, dict) else obs) or ''
                if c['function_name'] in ('edit_file', 'write_file') and not str(obs_text)[:40].startswith('{"status": "ok"'):
                    tot['failed_edit'] += 1
                    rec['skip'] = 'failed_edit'
                    steps.append(rec)
                    continue
                try:
                    call = native_call(c['function_name'], c.get('arguments') or {})
                except ValueError:
                    rec['skip'] = 'null_arg'
                    steps.append(rec)
                    continue
                thought = (s.get('message') or '').strip()
                head = f'<|channel>thought\n{thought}<channel|>' if thought else ''
                rec.update({'thought': head, 'call': call, 'tool': c['function_name'],
                            'mask_thought': bool(thought) and bool(grams(thought) & sg)})
                tot['kept'] += 1
                tot['masked'] += rec['mask_thought']
                steps.append(rec)
            out.write(json.dumps({'run': run, 'instance_id': iid, 'title': title, 'tools': tools, 'enable_thinking': True,
                                  'messages': msgs, 'steps': steps}) + '\n')
            tot['traj'] += 1
        print(f'{run}: done ({mode})', flush=True)
    out.close()
    print(tot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
