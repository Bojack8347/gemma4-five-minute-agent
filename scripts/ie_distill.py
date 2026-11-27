"""iE calibration of the statement direction's weight (IMPLEMENTATION.md 6.32; dissertation 3.2-3.3).

The statement direction (6.31) was trained on pairs of which only about a third carry what the full
statement adds, so the trained adapter is a shrunk estimate of the teacher's shift. iE corrects that by
solving the moment equation

    M(w) = alpha,   M(w) = E[ agreement of a title-only student sample (adapter weight w) with a teacher sample ]
                    alpha = E[ agreement of two independent teacher samples ]

on a fixed set of title-only agent states from the held-out repositories. The teacher is the base model
with the full statement in place of the title, sampled once (4 samples per state). If the student's next
call had the teacher's distribution, both expectations would be equal; the per-state U-statistics give
unbiased draws of each side. A malformed or truncated sample counts as disagreement on both sides, so the
equation stays exact. Agreement means not materially different (mine_statement_pairs.materially_different).

The iteration is the accelerated iE of the dissertation with one noisy forward draw per step,

    w_{n+1} = w_n + a_n (alpha - y_n),   a_n = P / (n^beta * g),

where g is the slope from a two-point probe (the scalar case of the matrix gain, 4.4), with a trust
region and a guard that rejects a weight whose student samples are mostly malformed.

Modes (the server must be up; `iterate` restarts it with each merged adapter):
  states   build data/ie_distill/states.jsonl from title-only traces of tasks the informed agent solved
  teacher  sample the teacher at every state (base model, full statement)
  draw     sample the student at the current server's adapter (or the base model with --base) and score it
  iterate  probe and RM steps; state in runs/ie_distill/state.json
  status   print the state
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_format_pairs import API, MODEL, classify, compacted_before, post, tool_specs  # noqa: E402
from mine_statement_pairs import build_messages, detect_thought_mode, materially_different, split, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'ie_distill'
RUN = ROOT / 'runs' / 'ie_distill'


def load_results(run: Path) -> dict:
    return {json.loads(l)['instance_id']: json.loads(l) for l in (run / 'task_results.jsonl').read_text().splitlines() if l.strip()}


def label(text: str | None) -> dict:
    if text is None:
        return {'status': 'truncated'}
    lab = classify(text, None)
    return {k: lab.get(k) for k in ('status', 'tool', 'args')}


def agree(a: dict, b: dict) -> bool:
    return a['status'] == 'ok' and b['status'] == 'ok' and not materially_different(a, b)


def sample(tokens: list[int], n: int, model: str, temperature: float, max_tokens: int) -> list[str | None]:
    body = {'model': model, 'prompt': tokens, 'n': n, 'temperature': temperature, 'top_p': 0.95,
            'max_tokens': max_tokens, 'skip_special_tokens': False}
    res = post('/v1/completions', body)
    return [c['text'] if c.get('finish_reason') != 'length' and split(c['text']) else None for c in res['choices']]


def served_models() -> list[str]:
    with urllib.request.urlopen(f'{API}/v1/models', timeout=30) as r:
        return [m['id'] for m in json.loads(r.read())['data']]


# ----------------------------------------------------------------------------- states
def build_states(args) -> None:
    full = {json.loads(l)['instance_id']: json.loads(l)['problem_statement']
            for l in (ROOT / args.full_tasks).read_text().splitlines() if l.strip()}
    informed = load_results(ROOT / args.informed_run)
    student = ROOT / args.student_run
    tasks = sorted(i for i, r in informed.items() if r['resolved'])
    agent_name, tools = tool_specs((ROOT / 'submissions' / 'v0_long').resolve())
    mode = None
    states = []
    for iid in tasks:
        tp = student / 'traces' / f'trace_{iid}.json'
        if not tp.exists():
            continue
        trace = json.loads(tp.read_text())
        if mode is None:
            mode = detect_thought_mode(trace, agent_name, tools)
            if mode is None:
                raise SystemExit('could not match the harness rendering')
        used = 0
        for s in [s for s in trace['steps'] if s['source'] == 'agent']:
            if used >= args.steps_per_task or compacted_before(trace, s['step_id']):
                break
            msgs = build_messages(trace, s['step_id'], agent_name, mode)
            ui = next(i for i, m in enumerate(msgs) if m['role'] == 'user')
            t = re.search(r'Problem Statement:\n(.*?)\n', msgs[ui]['content'] + '\n', re.S)
            if not t:
                break
            tmsgs = copy.deepcopy(msgs)
            tmsgs[ui]['content'] = msgs[ui]['content'].replace(t.group(1), full[iid], 1)
            states.append({'instance_id': iid, 'step_id': s['step_id'], 'repo': iid.rsplit('_', 1)[0],
                           'student_tokens': tokenize(msgs, tools)['tokens'], 'teacher_tokens': tokenize(tmsgs, tools)['tokens']})
            used += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'states.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in states))
    lens = sorted(len(s['student_tokens']) for s in states)
    print(f'{len(states)} states from {len({s["instance_id"] for s in states})} tasks (thought rendering {mode}); '
          f'student prompt tokens median {lens[len(lens) // 2]}, max {lens[-1]}')


def load_states() -> list[dict]:
    return [json.loads(l) for l in (OUT / 'states.jsonl').read_text().splitlines() if l.strip()]


def parallel(fn, items, workers: int) -> list:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


def teacher(args) -> None:
    states = load_states()
    done = [0]
    lock = threading.Lock()

    def one(s):
        texts = sample(s['teacher_tokens'], args.teacher_samples, MODEL, args.temperature, args.max_tokens)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f'  teacher {done[0]}/{len(states)}', flush=True)
        return {'instance_id': s['instance_id'], 'step_id': s['step_id'], 'labels': [label(t) for t in texts]}

    t0 = time.time()
    recs = parallel(one, states, args.workers)
    (OUT / 'teacher.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recs))
    a = alpha_of(recs)
    print(f'teacher sampled at {len(recs)} states in {time.time() - t0:.0f}s; alpha = {a["alpha"]:.4f} '
          f'(se {a["se"]:.4f}), teacher malformed {a["malformed"]:.3f}')


def alpha_of(recs: list[dict]) -> dict:
    per = []
    bad = n = 0
    for r in recs:
        labs = r['labels']
        pairs = list(itertools.combinations(labs, 2))
        per.append(sum(agree(a, b) for a, b in pairs) / len(pairs))
        bad += sum(l['status'] != 'ok' for l in labs)
        n += len(labs)
    m = sum(per) / len(per)
    sd = (sum((p - m) ** 2 for p in per) / (len(per) - 1)) ** 0.5
    return {'alpha': m, 'se': sd / len(per) ** 0.5, 'malformed': bad / n, 'per_state': per}


def draw(args, model: str, tag: str) -> dict:
    """One noisy forward draw y = M(w) + noise: student samples at every state, scored against the teacher."""
    states = load_states()
    teach = {(r['instance_id'], r['step_id']): r['labels'] for r in
             (json.loads(l) for l in (OUT / 'teacher.jsonl').read_text().splitlines() if l.strip())}

    def one(s):
        texts = sample(s['student_tokens'], args.student_samples, model, args.temperature, args.max_tokens)
        labs = [label(t) for t in texts]
        tl = teach[(s['instance_id'], s['step_id'])]
        return {'instance_id': s['instance_id'], 'step_id': s['step_id'], 'repo': s['repo'],
                'agree': sum(agree(a, b) for a in labs for b in tl) / (len(labs) * len(tl)),
                'malformed': sum(l['status'] != 'ok' for l in labs) / len(labs),
                'tools': [l.get('tool') for l in labs], 'labels': labs}

    t0 = time.time()
    per = parallel(one, states, args.workers)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / f'draw_{tag}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in per))
    ys = [p['agree'] for p in per]
    y = sum(ys) / len(ys)
    sd = (sum((v - y) ** 2 for v in ys) / (len(ys) - 1)) ** 0.5
    out = {'y': y, 'se': sd / len(ys) ** 0.5, 'malformed': sum(p['malformed'] for p in per) / len(per),
           'n_states': len(per), 'wall_s': round(time.time() - t0)}
    print(f'[{time.strftime("%F %T")}] draw {tag} ({model}): M = {y:.4f} (se {out["se"]:.4f}), '
          f'student malformed {out["malformed"]:.3f}, {out["wall_s"]}s', flush=True)
    return out


# ----------------------------------------------------------------------------- iE
CFG = {'base_submission': 'submissions/v0_long',
       'directions': [{'name': 'statement', 'path': 'adapters_out/statement_r16'}],
       'knobs': [{'name': 'w_statement', 'kind': 'adapter_weight', 'direction': 'statement'}]}


def at_weight(args, w: float, tag: str) -> dict:
    """Serve v0_long + w * statement direction and take one draw (w = 0 uses the base model, no restart)."""
    from ie_control import apply_knobs, restart_server
    if w == 0.0:
        if any(m != MODEL for m in served_models()):
            restart_server(ROOT / 'submissions' / 'v0_long')  # v0_long has no adapters: plain base server
        return draw(args, MODEL, tag)
    sub = ROOT / 'submissions' / 'ie' / 'distill' / tag
    apply_knobs(CFG, {'w_statement': w}, sub)
    restart_server(sub)
    lora = [m for m in served_models() if m != MODEL]
    if len(lora) != 1:
        raise RuntimeError(f'expected one adapter on the server, got {lora}')
    return draw(args, lora[0], tag)


def iterate(args) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    sp = RUN / 'state.json'
    st = json.loads(sp.read_text()) if sp.exists() else {'runs': [], 'n': 0}
    teach = [json.loads(l) for l in (OUT / 'teacher.jsonl').read_text().splitlines() if l.strip()]
    a = alpha_of(teach)
    st['alpha'], st['alpha_se'] = a['alpha'], a['se']

    def save():
        sp.write_text(json.dumps(st, indent=1))

    def record(tag, w, d, **extra):
        st['runs'].append({'tag': tag, 'w': w, **d, **extra})
        save()

    done = {r['tag']: r for r in st['runs']}
    # probe: three points (w = -1, 0, 1) give the slope g of the scalar gain by least squares and show
    # whether M is monotone around the base; the slope's noise is judged on the same states (paired)
    probes = [('probe_w0', 0.0), ('probe_w1', 1.0), ('probe_wm1', -1.0)]
    for tag, w in probes:
        if tag not in done:
            record(tag, w, at_weight(args, w, tag))
            done = {r['tag']: r for r in st['runs']}
    per = {tag: {(r['instance_id'], r['step_id']): r['agree'] for r in
                 (json.loads(l) for l in (RUN / f'draw_{tag}.jsonl').read_text().splitlines() if l.strip())}
           for tag, _ in probes}
    keys = list(per['probe_w0'])
    # per-state least-squares slope over the three weights (x = -1, 0, 1: slope = (y1 - y_-1) / 2)
    d = [(per['probe_w1'][k] - per['probe_wm1'][k]) / 2 for k in keys]
    g = sum(d) / len(d)
    g_se = (sum((v - g) ** 2 for v in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5
    st['g'], st['g_se'] = g, g_se
    print(f'alpha = {st["alpha"]:.4f}; M(-1) = {done["probe_wm1"]["y"]:.4f}, M(0) = {done["probe_w0"]["y"]:.4f}, '
          f'M(1) = {done["probe_w1"]["y"]:.4f}; slope g = {g:+.4f} (paired se {g_se:.4f})', flush=True)
    if abs(g) <= 2 * g_se:
        print('slope not distinguishable from noise: the direction does not move the statistic; iE has no usable gain')
        save()
        return
    if 'w' not in st:
        # first iterate: a Newton step from the base, inside the trust region
        w0 = (st['alpha'] - done['probe_w0']['y']) / g
        st['w'] = round(max(args.lo, min(args.hi, max(-args.trust, min(args.trust, w0)))), 3)
        print(f'first iterate: Newton step from the base gives w = {w0:+.3f}; starting at {st["w"]}', flush=True)
    w = st['w']
    trust = st.get('trust', args.trust)
    for _ in range(args.steps):
        st['n'] += 1
        n = st['n']
        tag = f'step_{n:02d}'
        d = at_weight(args, w, tag)
        if d['malformed'] > args.max_malformed:
            # guard: a weight that breaks the call format is outside the feasible set; go halfway back to the
            # last accepted weight (2026-10-20: the first version stepped by -trust, which assumed the iteration
            # moved upwards, so at the lower bound it re-drew w = -3 four times)
            trust *= 0.5
            last_ok = st.get('last_ok', 0.0)
            w_new = round(last_ok + 0.5 * (w - last_ok), 3)
            record(tag, w, d, guard='rejected', next_w=w_new)
            print(f'  {tag}: malformed {d["malformed"]:.3f} > {args.max_malformed}; step back to w = {w_new}', flush=True)
            w = w_new
            st['w'], st['trust'] = w, trust
            save()
            continue
        step = args.P / (n ** args.beta * g) * (st['alpha'] - d['y'])
        step = max(-trust, min(trust, step))
        w_new = round(max(args.lo, min(args.hi, w + step)), 3)
        st['last_ok'] = w
        record(tag, w, d, residual=st['alpha'] - d['y'], next_w=w_new)
        print(f'  {tag}: w = {w}, alpha - y = {st["alpha"] - d["y"]:+.4f}, next w = {w_new}', flush=True)
        w = w_new
        st['w'], st['trust'] = w, trust
        save()


def status(args) -> None:
    st = json.loads((RUN / 'state.json').read_text())
    print(f'alpha = {st["alpha"]:.4f} (se {st["alpha_se"]:.4f}); g = {st.get("g")}; current w = {st.get("w")}')
    for r in st['runs']:
        print(f'  {r["tag"]:10} w = {r["w"]:6.3f}  M = {r["y"]:.4f} (se {r["se"]:.4f})  malformed {r["malformed"]:.3f}'
              + (f'  -> {r["next_w"]}' if 'next_w' in r else '') + (f'  [{r["guard"]}]' if 'guard' in r else ''))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['states', 'teacher', 'draw', 'iterate', 'status'])
    ap.add_argument('--student-run', default='runs/heldout_hard_terse_base_c8')
    ap.add_argument('--informed-run', default='runs/heldout_hard_base_c8')
    ap.add_argument('--full-tasks', default='data/heldout_hard/tasks.jsonl')
    ap.add_argument('--steps-per-task', type=int, default=6)
    ap.add_argument('--teacher-samples', type=int, default=4)
    ap.add_argument('--student-samples', type=int, default=2)
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--max-tokens', type=int, default=1200)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--base', action='store_true', help='draw: sample the base model instead of the served adapter')
    ap.add_argument('--tag', default='manual')
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--P', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.6)
    ap.add_argument('--trust', type=float, default=1.0, help='largest weight change per step')
    ap.add_argument('--lo', type=float, default=-3.0)
    ap.add_argument('--hi', type=float, default=3.0)
    ap.add_argument('--max-malformed', type=float, default=0.25)
    args = ap.parse_args()
    if args.mode == 'states':
        build_states(args)
    elif args.mode == 'teacher':
        teacher(args)
    elif args.mode == 'draw':
        model = MODEL if args.base else next(m for m in served_models() if m != MODEL)
        draw(args, model, args.tag)
    elif args.mode == 'iterate':
        iterate(args)
    else:
        status(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
