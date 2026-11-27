"""Statement distillation pairs: what the agent does with the full statement vs with only the title.

Privileged-teacher setup (IMPLEMENTATION.md 6.31). The student is the base agent run with title-only
statements; the teacher is the same model, at the same student state (same history), with the full
statement in the user message. At each early student state of a task the informed agent can solve:

  * sample the student's next completion (thought + call) from the title-only context;
  * sample the teacher's next completion from the full-statement context;
  * if the teacher's call differs from the student's (another tool, or another file / command),
    emit a pair whose two sides share the student's own thought:
        chosen   = student thought + teacher call
        rejected = student thought + student call
    The shared thought cancels in the DPO margin, so the gradient falls on the call alone, and the
    teacher's thought (which may quote the full statement) never enters the training text.

Contexts are rebuilt as the harness rendered them; the prior-thought rendering is detected per run by
matching the trace's recorded prompt lengths. Records use mine_format_pairs.py's format, so
train_direction_dpo.py consumes them unchanged.

Usage: .venv/bin/python scripts/mine_statement_pairs.py --student runs/public_terse_base_c8 \
           --teacher-runs runs/base_sound100_c8 runs/base_sound100_c8_r2 --out data/pairs/statement_pairs.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_format_pairs import MODEL, Snapshot, classify, compacted_before, post, tool_specs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_results(run: Path) -> dict:
    return {json.loads(l)['instance_id']: json.loads(l) for l in (run / 'task_results.jsonl').read_text().splitlines() if l.strip()}


def build_messages(trace: dict, upto: int, agent_name: str, thought_mode: str) -> list[dict]:
    msgs = []
    for s in trace['steps']:
        if s['step_id'] >= upto:
            break
        if s['source'] == 'system':
            identity = f'You are an agent. Your internal name is "{agent_name}".'
            msgs.append({'role': 'system', 'content': f'{s["message"]}\n\n{identity}'})
        elif s['source'] == 'user':
            msgs.append({'role': 'user', 'content': s['message']})
        elif s['source'] == 'agent':
            calls = s.get('tool_calls') or []
            thought = s.get('message') or ''
            m = {'role': 'assistant', 'content': None}
            if thought_mode == 'content':
                m['content'] = thought or None
            elif thought_mode == 'reasoning' and thought:
                m['reasoning_content'] = thought
            if calls:
                m['tool_calls'] = [{'id': c['tool_call_id'], 'type': 'function',
                                    'function': {'name': c['function_name'], 'arguments': json.dumps(c.get('arguments') or {})}}
                                   for c in calls]
            msgs.append(m)
            obs = s.get('observation')
            for c in calls:
                content = obs.get('content') if isinstance(obs, dict) else obs
                msgs.append({'role': 'tool', 'tool_call_id': c['tool_call_id'], 'content': content or ''})
    return msgs


def tokenize(messages, tools) -> dict:
    return post('/tokenize', {'model': MODEL, 'messages': messages, 'tools': tools,
                              'chat_template_kwargs': {'enable_thinking': True}, 'add_generation_prompt': True})


def detect_thought_mode(trace: dict, agent_name: str, tools) -> str | None:
    """Pick the prior-thought rendering whose length matches the trace's recorded prompt lengths."""
    agent_steps = [s for s in trace['steps'] if s['source'] == 'agent']
    probe = next((s for s in agent_steps[2:6] if (s.get('metrics') or {}).get('prompt_tokens')
                  and not compacted_before(trace, s['step_id'])), None)
    if probe is None:
        return None
    want = probe['metrics']['prompt_tokens']
    best = None
    for mode in ('reasoning', 'none', 'content'):
        n = tokenize(build_messages(trace, probe['step_id'], agent_name, mode), tools)['count']
        if best is None or abs(n - want) < best[1]:
            best = (mode, abs(n - want), n)
    print(f'thought rendering: {best[0]} (rendered {best[2]} vs recorded {want})', flush=True)
    return best[0] if best[1] <= 16 else None


def key_of(lab: dict) -> tuple:
    args = lab.get('args') or {}
    tool = lab.get('tool')
    if tool in ('read_file', 'edit_file', 'write_file'):
        return (tool, str(args.get('filepath', '')).lstrip('./'))
    if tool == 'run_command':
        return (tool, re.sub(r'\s+', ' ', str(args.get('command', ''))).strip()[:80])
    return (tool,)


STOP = {'grep', 'head', 'tail', 'python', 'python3', 'pytest', 'include', 'tests', 'docs', 'echo', 'find', 'name', 'cat',
        'sed', 'awk', 'true', 'null', 'type', 'print', 'import', 'from', 'self', 'none', 'return'}


def targets(lab: dict) -> set[str]:
    """What an action is about: the file for file tools; quoted search terms and paths for commands."""
    args = lab.get('args') or {}
    if lab.get('tool') in ('read_file', 'edit_file', 'write_file'):
        return {str(args.get('filepath', '')).lstrip('./')}
    cmd = str(args.get('command', ''))
    quoted = re.findall(r'["\']([^"\']{3,60})["\']', cmd)
    paths = re.findall(r'[\w./-]+\.py\b', cmd)
    words = {w for w in re.findall(r'[A-Za-z_][\w]{3,}', ' '.join(quoted)) if w.lower() not in STOP}
    return {p.lstrip('./') for p in paths} | words


def materially_different(a: dict, b: dict) -> bool:
    """Different decision about editing, or about where to look (Jaccard of targets below 0.5)."""
    if (a.get('tool') == 'edit_file') != (b.get('tool') == 'edit_file'):
        return True
    ta, tb = targets(a), targets(b)
    if not ta or not tb:
        return a.get('tool') != b.get('tool')
    return len(ta & tb) / len(ta | tb) < 0.5


def split(text: str) -> tuple[str, str] | None:
    """(thought prefix, call) of a raw completion with exactly one tool call."""
    i = text.find('<|tool_call>')
    j = text.find('<tool_call|>', i)
    if i < 0 or j < 0:
        return None
    return text[:i], text[i:j + len('<tool_call|>')]


def sample(prompt_tokens: list[int], n: int, args) -> list[str]:
    body = {'model': MODEL, 'prompt': prompt_tokens, 'n': n, 'temperature': args.temperature, 'top_p': 0.95,
            'max_tokens': args.max_tokens, 'skip_special_tokens': False}
    res = post('/v1/completions', body)
    return [c['text'] for c in res['choices'] if c.get('finish_reason') != 'length']


def replay(snap: Snapshot, step: dict) -> None:
    """Apply a step's successful edits so later classify() calls see the file state the agent saw."""
    obs = (step.get('observation') or {}).get('content') or ''
    for c in step.get('tool_calls') or []:
        a = c.get('arguments') or {}
        if c['function_name'] == 'edit_file' and obs[:40].startswith('{"status": "ok"') and \
                all(isinstance(a.get(k), str) for k in ('filepath', 'old_string', 'new_string')):
            snap.apply_edit(a['filepath'], a['old_string'], a['new_string'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--student', type=Path, required=True)
    ap.add_argument('--teacher-runs', type=Path, nargs='+', required=True, help='full-statement runs; a task qualifies if any resolved it')
    ap.add_argument('--full-tasks', type=Path, default=ROOT / 'data' / 'competition' / 'tasks.jsonl')
    ap.add_argument('--submission-dir', type=Path, default=ROOT / 'submissions' / 'v0_long')
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'pairs' / 'statement_pairs.jsonl')
    ap.add_argument('--steps-per-task', type=int, default=8)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--max-tokens', type=int, default=1200)
    ap.add_argument('--limit', type=int, default=10 ** 9, help='contexts in total')
    ap.add_argument('--workers', type=int, default=4, help='tasks mined in parallel')
    args = ap.parse_args()

    full = {json.loads(l)['instance_id']: json.loads(l)['problem_statement'] for l in args.full_tasks.read_text().splitlines() if l.strip()}
    competent = set()
    for run in args.teacher_runs:
        competent |= {i for i, r in load_results(run).items() if r['resolved']}
    student = load_results(args.student)
    tasks = sorted(i for i in student if i in competent)
    print(f'{len(tasks)} tasks resolved by the informed agent and run title-only', flush=True)
    agent_name, tools = tool_specs(args.submission_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {(json.loads(l)['instance_id'], json.loads(l)['step_id']) for l in args.out.read_text().splitlines() if l.strip()}
    stats = {'contexts': 0, 'pairs': 0, 'same': 0, 'unusable': 0}
    lock = threading.Lock()
    mode = None
    for iid in tasks:  # the prior-thought rendering is a property of the run; detect it once
        tp = args.student / 'traces' / f'trace_{iid}.json'
        if tp.exists():
            mode = detect_thought_mode(json.loads(tp.read_text()), agent_name, tools)
            break
    if mode is None:
        print('could not match the harness rendering; stopping')
        return 2

    def mine_task(iid: str, out) -> None:
        trace_path = args.student / 'traces' / f'trace_{iid}.json'
        if not trace_path.exists():
            return
        trace = json.loads(trace_path.read_text())
        snap = Snapshot(iid)
        steps = [s for s in trace['steps'] if s['source'] == 'agent']
        used = 0
        for s in steps:
            if used >= args.steps_per_task or compacted_before(trace, s['step_id']):
                break
            # keep the file state at this step current (successful edits before it)
            if (iid, s['step_id']) in done:
                used += 1
                replay(snap, s)
                continue
            with lock:
                if stats['contexts'] >= args.limit:
                    return
                stats['contexts'] += 1
            msgs = build_messages(trace, s['step_id'], agent_name, mode)
            user_i = next(i for i, m in enumerate(msgs) if m['role'] == 'user')
            teacher_msgs = copy.deepcopy(msgs)
            stmt_title = re.search(r'Problem Statement:\n(.*?)\n', msgs[user_i]['content'] + '\n', re.S)
            if not stmt_title:
                with lock:
                    stats['unusable'] += 1
                return
            teacher_msgs[user_i]['content'] = msgs[user_i]['content'].replace(stmt_title.group(1), full[iid], 1)
            st_tok, te_tok = tokenize(msgs, tools), tokenize(teacher_msgs, tools)
            t0 = time.time()
            st_samples = [x for x in sample(st_tok['tokens'], args.samples, args) if split(x)]
            te_samples = [x for x in sample(te_tok['tokens'], args.samples, args) if split(x)]
            used += 1
            pairs, same = [], 0
            for st in st_samples:
                st_lab = classify(st, snap)
                if st_lab['status'] != 'ok':
                    continue
                for te in te_samples:
                    te_lab = classify(te, snap)
                    if te_lab['status'] != 'ok':
                        continue
                    if not materially_different(te_lab, st_lab):
                        same += 1
                        continue
                    thought, st_call = split(st)
                    _, te_call = split(te)
                    pairs.append({'chosen': thought + te_call, 'rejected': st, 'chosen_tool': te_lab['tool'],
                                  'rejected_tool': st_lab['tool'], 'why': f'{key_of(st_lab)} -> {key_of(te_lab)}'[:160]})
            rec = {'run': args.student.name, 'instance_id': iid, 'step_id': s['step_id'], 'prompt_tokens': st_tok['count'],
                   'enable_thinking': True, 'original_tool': (s.get('tool_calls') or [{}])[0].get('function_name'),
                   'messages': msgs, 'tools': tools, 'pairs': pairs[:2],
                   'samples': [{'text': x} for x in st_samples], 'teacher_samples': [{'text': x} for x in te_samples]}
            with lock:
                out.write(json.dumps(rec) + '\n')
                out.flush()
                stats['pairs'] += len(rec['pairs'])
                stats['same'] += same
            print(f'  {iid} step {s["step_id"]:3d} ({st_tok["count"]:5d} tok): {len(rec["pairs"])} pairs, {time.time() - t0:.0f}s', flush=True)
            replay(snap, s)

    with args.out.open('a') as out, ThreadPoolExecutor(args.workers) as pool:
        for f in [pool.submit(mine_task, iid, out) for iid in tasks]:
            try:
                f.result()
            except Exception as e:  # one task's failure should not lose the others
                print(f'task failed: {type(e).__name__}: {e}', flush=True)
    print(stats)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
