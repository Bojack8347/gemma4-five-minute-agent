"""Fixed-state reachability check for adapter directions (IMPLEMENTATION.md 6.33).

Scope (validated 2026-10-20): it measures per-call statistics, such as agreement with a teacher, and
should gate only objectives that are per-call. It did NOT predict the agent-level effects of the format and
decisiveness directions, which act through the trajectory; it would have rejected the format direction.

Before an agent run (hours), measure what a direction does to the agent's next call at fixed agent
states (minutes), and whether a target is within its reach. This is the moment-equation form of iE used
in 6.32, generalised: a statistic M(config) is simulated at fixed states, compared with the reference
configuration state by state (paired), and with a target alpha.

Sampling is the agent's own: the chat endpoint (which enforces the thinking budget; /v1/completions
ignores `thinking_token_budget`), the submission's temperature, top_p, output cap and thinking budget,
and the harness's tool-call parser. A sample is classified from the parsed tool call the harness would
receive.

Per-state statistics (each the mean over n samples):
  malformed  no call, truncated, unparseable or missing required arguments   ~ garbled_call_rate
  repeat     well-formed and identical (tool and arguments) to the previous call ~ repeat_loop_rate
  edit       well-formed edit_file or write_file                             ~ edit_attempt_rate, empty patches
  tokens     completion tokens                                               ~ tokens_per_turn
  agree      agreement with a teacher sample (states with teacher samples)   (statement distillation)

Modes
  states   --set S --run RUN [--informed-run R --full-tasks F]   K states per task over the pre-compaction steps
  serve    --configs NAME=dir:w[,dir:w] ...                       one server with every config mounted as a LoRA
  teacher  --set S                                                 base model with the full statement
  draw     --set S --config NAME                                   (NAME 'base' = the base model)
  report   --set S --ref NAME --configs NAME ...                   paired differences against the reference
  gate     --set S --ref NAME --probes NAME:w ... --target COORD=VALUE   reachability verdict along one direction

Run `states` for public traces with the default environment and for held-out traces with
PYTHONPATH=data/harness_v2_site (the tool specs must match the harness that produced the traces).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_format_pairs import MODEL, REQUIRED, compacted_before, post, tool_specs  # noqa: E402
from mine_statement_pairs import build_messages, detect_thought_mode, materially_different  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'reach'
RUNS = ROOT / 'runs' / 'reach'
EDIT_TOOLS = ('edit_file', 'write_file')


# ----------------------------------------------------------------------------- sampling
def agent_sampling(submission: Path) -> dict:
    cfg = yaml.safe_load((submission / 'configs' / 'sampling.yaml').read_text()) or {}
    tc = cfg.get('thinking_config') or {}
    out = {'temperature': cfg.get('temperature', 0.2), 'top_p': cfg.get('top_p', 0.95),
           'max_tokens': cfg.get('max_output_tokens', 2048),
           'chat_template_kwargs': {'enable_thinking': bool(tc) and tc.get('thinking_budget', 1) != 0}}
    if tc.get('thinking_budget'):
        out['thinking_token_budget'] = int(tc['thinking_budget'])
    return out


def label_choice(choice: dict) -> dict:
    if choice.get('finish_reason') == 'length':
        return {'status': 'truncated'}
    calls = (choice.get('message') or {}).get('tool_calls') or []
    if not calls:
        return {'status': 'no_call'}
    fn = calls[0].get('function') or {}
    name = fn.get('name')
    try:
        args = json.loads(fn.get('arguments') or '{}')
    except json.JSONDecodeError:
        return {'status': 'garbled', 'tool': name}
    if not isinstance(args, dict):
        return {'status': 'garbled', 'tool': name}
    if name in REQUIRED and not REQUIRED[name] <= set(args):
        return {'status': 'garbled', 'tool': name}
    return {'status': 'ok', 'tool': name, 'args': args}


def same_call(lab: dict, prev: dict | None) -> bool:
    return bool(prev) and lab['status'] == 'ok' and lab['tool'] == prev['tool'] and \
        json.dumps(lab['args'], sort_keys=True) == json.dumps(prev['args'], sort_keys=True)


def agree(a: dict, b: dict) -> bool:
    return a['status'] == 'ok' and b['status'] == 'ok' and not materially_different(a, b)


def chat(messages, tools, model: str, n: int, sampling: dict) -> tuple[list[dict], int]:
    body = {'model': model, 'messages': messages, 'tools': tools, 'n': n, **sampling}
    res = post('/v1/chat/completions', body)
    return [label_choice(c) for c in res['choices']], res['usage']['completion_tokens']


def parallel(fn, items, workers: int) -> list:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


# ----------------------------------------------------------------------------- states
def build_states(args) -> None:
    run = ROOT / args.run
    agent_name, tools = tool_specs((ROOT / args.base_submission).resolve())
    full, solved = {}, None
    if args.full_tasks:
        full = {json.loads(l)['instance_id']: json.loads(l)['problem_statement']
                for l in (ROOT / args.full_tasks).read_text().splitlines() if l.strip()}
    if args.informed_run:
        solved = {json.loads(l)['instance_id'] for l in (ROOT / args.informed_run / 'task_results.jsonl').read_text().splitlines()
                  if l.strip() and json.loads(l)['resolved']}
    ids = None
    if args.task_ids:
        ids = {l.strip() for l in (ROOT / args.task_ids).read_text().splitlines() if l.strip()}
    mode, states = None, []
    for tp in sorted((run / 'traces').glob('trace_*.json')):
        iid = tp.stem[len('trace_'):]
        if ids is not None and iid not in ids:
            continue
        trace = json.loads(tp.read_text())
        if mode is None:
            mode = detect_thought_mode(trace, agent_name, tools)
            if mode is None:
                raise SystemExit('could not match the harness rendering (wrong harness on the path?)')
        agent = [s for s in trace['steps'] if s['source'] == 'agent']
        pre = [s for s in agent if not compacted_before(trace, s['step_id'])]
        if len(pre) < 2:
            continue
        picks = sorted({max(1, round((j + 1) * len(pre) / (args.per_task + 1))) for j in range(args.per_task)})
        for i in picks:
            if i >= len(pre):
                continue
            s = pre[i]
            msgs = build_messages(trace, s['step_id'], agent_name, mode)
            pc = (pre[i - 1].get('tool_calls') or [None])[0]
            prev = {'tool': pc['function_name'], 'args': pc.get('arguments') or {}} if pc else None
            rec = {'instance_id': iid, 'step_id': s['step_id'], 'index': i, 'n_pre': len(pre), 'messages': msgs, 'prev': prev}
            if full and (solved is None or iid in solved) and iid in full:
                ui = next(k for k, m in enumerate(msgs) if m['role'] == 'user')
                t = re.search(r'Problem Statement:\n(.*?)\n', msgs[ui]['content'] + '\n', re.S)
                if t:
                    tm = copy.deepcopy(msgs)
                    tm[ui]['content'] = msgs[ui]['content'].replace(t.group(1), full[iid], 1)
                    rec['teacher_messages'] = tm
            states.append(rec)
    d = DATA / args.set
    d.mkdir(parents=True, exist_ok=True)
    (d / 'states.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in states))
    (d / 'tools.json').write_text(json.dumps(tools))
    (d / 'meta.json').write_text(json.dumps({'run': args.run, 'base_submission': args.base_submission, 'per_task': args.per_task,
                                             'thought_mode': mode}))
    print(f'{args.set}: {len(states)} states from {len({s["instance_id"] for s in states})} tasks '
          f'({sum("teacher_messages" in s for s in states)} with a teacher context; thought rendering {mode})')


def load_set(name: str) -> tuple[list[dict], list, dict]:
    d = DATA / name
    states = [json.loads(l) for l in (d / 'states.jsonl').read_text().splitlines() if l.strip()]
    return states, json.loads((d / 'tools.json').read_text()), json.loads((d / 'meta.json').read_text())


# ----------------------------------------------------------------------------- server
def serve(args) -> None:
    """One server, every configuration mounted as its own merged LoRA (the base stays served by name)."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from ie_control import restart_server
    sub = ROOT / 'submissions' / 'reach' / args.server_name
    if sub.exists():
        shutil.rmtree(sub)
    shutil.copytree(ROOT / args.base_submission, sub)
    shutil.rmtree(sub / 'adapters', ignore_errors=True)
    (sub / 'adapters').mkdir()
    for spec in args.configs:
        name, parts = spec.split('=', 1)
        cmd = [str(ROOT / '.venv-vllm' / 'bin' / 'python'), str(ROOT / 'scripts' / 'merge_directions.py'), '--out', str(sub / 'adapters' / name)]
        for k, part in enumerate(parts.split(',')):
            path, w = part.rsplit(':', 1)
            cmd += ['--direction', f'd{k}={ROOT / path}:{float(w)}']
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        (sub / 'adapters' / name / 'merge.json').unlink(missing_ok=True)
    (sub / 'reach_configs.json').write_text(json.dumps(args.configs, indent=1))
    restart_server(sub)
    import urllib.request
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=30) as r:
        print('served:', [m['id'] for m in json.loads(r.read())['data']])


# ----------------------------------------------------------------------------- draws
def teacher(args) -> None:
    states, tools, meta = load_set(args.set)
    sampling = agent_sampling(ROOT / meta['base_submission'])
    todo = [s for s in states if 'teacher_messages' in s]

    def one(s):
        labs, ntok = chat(s['teacher_messages'], tools, MODEL, args.teacher_samples, sampling)
        return {'instance_id': s['instance_id'], 'step_id': s['step_id'], 'labels': labs}

    t0 = time.time()
    recs = parallel(one, todo, args.workers)
    (DATA / args.set / 'teacher.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recs))
    per = [sum(agree(a, b) for a, b in itertools.combinations(r['labels'], 2)) / max(1, len(r['labels']) * (len(r['labels']) - 1) // 2)
           for r in recs]
    print(f'{args.set}: teacher at {len(recs)} states in {time.time() - t0:.0f}s; alpha (self-agreement) = {sum(per) / len(per):.4f}')


def draw(args) -> None:
    states, tools, meta = load_set(args.set)
    sampling = agent_sampling(ROOT / meta['base_submission'])
    tpath = DATA / args.set / 'teacher.jsonl'
    teach = {}
    if tpath.exists():
        teach = {(r['instance_id'], r['step_id']): r['labels'] for r in map(json.loads, tpath.read_text().splitlines())}
    model = MODEL if args.config == 'base' else args.config
    done = [0]
    lock = threading.Lock()

    def one(s):
        labs, ntok = chat(s['messages'], tools, model, args.samples, sampling)
        n = len(labs)
        rec = {'instance_id': s['instance_id'], 'step_id': s['step_id'],
               'malformed': sum(l['status'] != 'ok' for l in labs) / n,
               'repeat': sum(same_call(l, s['prev']) for l in labs) / n,
               'edit': sum(l['status'] == 'ok' and l['tool'] in EDIT_TOOLS for l in labs) / n,
               'tokens': ntok / n, 'tools': [l.get('tool') for l in labs]}
        tl = teach.get((s['instance_id'], s['step_id']))
        if tl:
            rec['agree'] = sum(agree(a, b) for a in labs for b in tl) / (n * len(tl))
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f'  {args.config}: {done[0]}/{len(states)}', flush=True)
        return rec

    t0 = time.time()
    recs = parallel(one, states, args.workers)
    out = RUNS / args.set
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.config}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recs))
    summ = summarize(recs)
    print(f'[{time.strftime("%F %T")}] {args.set}/{args.config}: ' + ', '.join(f'{k} {v:.4f}' for k, v in summ.items())
          + f' ({time.time() - t0:.0f}s)', flush=True)


COORDS = ('malformed', 'repeat', 'edit', 'tokens', 'agree')


def summarize(recs: list[dict]) -> dict:
    out = {}
    for c in COORDS:
        v = [r[c] for r in recs if c in r]
        if v:
            out[c] = sum(v) / len(v)
    return out


def load_draw(set_name: str, config: str) -> dict:
    p = RUNS / set_name / f'{config}.jsonl'
    return {(r['instance_id'], r['step_id']): r for r in map(json.loads, p.read_text().splitlines())}


def paired(a: dict, b: dict, coord: str) -> tuple[float, float, int]:
    d = [b[k][coord] - a[k][coord] for k in a if k in b and coord in a[k] and coord in b[k]]
    if len(d) < 2:
        return float('nan'), float('nan'), len(d)
    m = sum(d) / len(d)
    sd = (sum((v - m) ** 2 for v in d) / (len(d) - 1)) ** 0.5
    return m, sd / len(d) ** 0.5, len(d)


def report(args) -> None:
    ref = load_draw(args.set, args.ref)
    rs = summarize(list(ref.values()))
    print(f'{args.set}: reference {args.ref} ' + ', '.join(f'{k} {v:.4f}' for k, v in rs.items()))
    for cfg in args.configs:
        cur = load_draw(args.set, cfg)
        cells = []
        for c in COORDS:
            m, se, n = paired(ref, cur, c)
            if n >= 2:
                flag = '*' if abs(m) > 2 * se else ' '
                cells.append(f'{c} {m:+.4f} (se {se:.4f}){flag}')
        print(f'  {cfg:16} ' + ' | '.join(cells))


def gate(args) -> None:
    """Reachability of COORD = VALUE along one direction from probes at several weights (paired against ref)."""
    coord, alpha = args.target.split('=')
    alpha = float(alpha)
    ref = load_draw(args.set, args.ref)
    pts = [(0.0, summarize(list(ref.values()))[coord], 0.0, summarize(list(ref.values()))['malformed'])]
    for spec in args.probes:
        name, w = spec.rsplit(':', 1)
        cur = load_draw(args.set, name)
        m, se, _ = paired(ref, cur, coord)
        pts.append((float(w), pts[0][1] + m, se, summarize(list(cur.values()))['malformed']))
    pts.sort()
    feasible = [p for p in pts if p[3] <= args.max_malformed]
    print(f'target {coord} = {alpha}; probes (w, M, paired se, malformed):')
    for p in pts:
        print(f'  w {p[0]:+.2f}  M {p[1]:.4f}  se {p[2]:.4f}  malformed {p[3]:.3f}' + ('' if p in feasible else '  [infeasible]'))
    ys = [p[1] for p in feasible]
    lo, hi = min(ys), max(ys)
    if lo <= alpha <= hi:
        # bracketed: linear interpolation between the bracketing probes gives the iE starting point
        for a, b in zip(feasible, feasible[1:]):
            if min(a[1], b[1]) <= alpha <= max(a[1], b[1]) and a[1] != b[1]:
                w = a[0] + (alpha - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                print(f'REACHABLE: target bracketed by feasible probes; root near w = {w:+.2f} (refine with iE draws there)')
                return
    # not bracketed: extrapolate the outermost significant slope, but only within the bounds
    best = max(feasible, key=lambda p: -abs(p[1] - alpha))
    moving = [p for p in feasible if p[0] != 0.0 and abs(p[1] - pts[0][1]) > 2 * p[2]]
    if not moving:
        # Not evidence of no effect: the validation (6.33) found that effects acting through the trajectory
        # (the format direction's 5.0% -> 0.4% malformed calls, 11.5% -> 1.5% loops) do not show at fixed states.
        print(f'NO EFFECT DETECTED on {coord} at these states (no probe beyond twice its paired se). This rules out a '
              f'per-call effect of this size, not a trajectory-level one; do not use it to reject a direction whose '
              f'target acts through the trajectory (loops, malformed calls, edit timing)')
        return
    p = max(moving, key=lambda q: abs(q[0]))
    g = (p[1] - pts[0][1]) / p[0]
    w = (alpha - pts[0][1]) / g
    if all((q[0] > 0) != (w > 0) for q in moving):
        print(f'NOT REACHABLE ALONG THE PROBED SIGN: the slope {g:+.4f} points away from the target; the root would need '
              f'w = {w:+.2f}, on the side no probe has measured (6.32 found M flat there); probe it before believing it')
        return
    verdict = 'EXTRAPOLATED' if args.lo <= w <= args.hi else 'NOT REACHABLE'
    print(f'{verdict}: closest feasible probe w {best[0]:+.2f} (M {best[1]:.4f}); slope {g:+.4f} per unit weight '
          f'puts the root at w = {w:+.2f} (bounds [{args.lo}, {args.hi}])')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['states', 'serve', 'teacher', 'draw', 'report', 'gate'])
    ap.add_argument('--set')
    ap.add_argument('--run')
    ap.add_argument('--task-ids')
    ap.add_argument('--informed-run')
    ap.add_argument('--full-tasks')
    ap.add_argument('--per-task', type=int, default=3)
    ap.add_argument('--base-submission', default='submissions/v0_long')
    ap.add_argument('--server-name', default='multi')
    ap.add_argument('--configs', nargs='*', default=[])
    ap.add_argument('--config')
    ap.add_argument('--ref', default='base')
    ap.add_argument('--probes', nargs='*', default=[])
    ap.add_argument('--target')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--teacher-samples', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-malformed', type=float, default=0.10)
    ap.add_argument('--lo', type=float, default=-2.0)
    ap.add_argument('--hi', type=float, default=3.0)
    args = ap.parse_args()
    {'states': build_states, 'serve': serve, 'teacher': teacher, 'draw': draw, 'report': report, 'gate': gate}[args.mode](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
