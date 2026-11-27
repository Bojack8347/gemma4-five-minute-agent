"""Mine on-policy preference pairs for the tool-call *format* direction (memo 10.5, item 1).

Traces keep only parsed tool arguments, so a garbled call's raw text cannot be
recovered from them. Instead, at every turn where a run produced a garbled call,
rebuild the exact context the model saw, sample N raw completions from the server
(`/v1/completions`, special tokens kept), and let a strict parser label each one as
well-formed or garbled. Pairs from the same prefix then have exact strings and free
labels: chosen = well-formed call, rejected = garbled call.

The context is rendered exactly as the harness does: ADK's litellm conversion (system
prompt, user task, assistant `tool_calls` with JSON arguments, `tool` messages with the
JSON observation), the submission's tool declarations, and the chat template kwargs
that `adk_submission` bridges from the sampling config. Each rendering is checked
against the trace's recorded prompt token count before sampling.

Usage:
  .venv/bin/python scripts/mine_format_pairs.py runs/v0_think0_t06_dev24_c2x2 [more runs] \
      --out data/pairs/format_pairs.jsonl --samples 8 --limit 20
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
API = 'http://127.0.0.1:8000'
REQUIRED = {
    'read_file': {'filepath'}, 'edit_file': {'filepath', 'old_string', 'new_string'},
    'write_file': {'filepath', 'content'}, 'run_command': {'command'},
    'submit_patch': set(), 'get_status': set(),
}
CALL_RE = re.compile(r'<\|tool_call>call:([A-Za-z_][A-Za-z0-9_]*)\{(.*?)\}<tool_call\|>', re.S)
STR = '<|"|>'


# ----------------------------------------------------------------------------- server
def post(path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(f'{API}{path}', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ----------------------------------------------------------------------------- tool schemas
def tool_specs(submission_dir: Path) -> tuple[str, list[dict]]:
    """Agent name and OpenAI tool specs in agent.yaml order, derived exactly as ADK derives them."""
    from adk_submission.yaml_loader import load_yaml
    from google.adk.models.lite_llm import _function_declaration_to_tool_param
    from google.adk.tools.function_tool import FunctionTool
    from swegemma.tools import create_tools

    agent = load_yaml(submission_dir / 'agent.yaml', submission_dir)
    names = [t for t in agent['tools'] if isinstance(t, str)]
    fns = create_tools(None)
    specs = []
    for name in names:
        decl = FunctionTool(fns[name])._get_declaration()
        specs.append(_function_declaration_to_tool_param(decl))
    return agent['name'], specs


# ----------------------------------------------------------------------------- context
def steps_to_messages(trace: dict, upto_step_id: int, agent_name: str) -> list[dict]:
    """Messages for everything before step `upto_step_id` (the target agent turn)."""
    msgs = []
    for s in trace['steps']:
        if s['step_id'] >= upto_step_id:
            break
        src = s['source']
        if src == 'system':
            # ADK appends its identity line after the instruction (flows: instructions, identity),
            # joining with a blank line; the trace records only the instruction.
            identity = f'You are an agent. Your internal name is "{agent_name}".'
            msgs.append({'role': 'system', 'content': f'{s["message"]}\n\n{identity}'})
        elif src == 'user':
            msgs.append({'role': 'user', 'content': s['message']})
        elif src == 'agent':
            calls = s.get('tool_calls') or []
            m = {'role': 'assistant', 'content': s.get('message') or None}
            if calls:
                m['tool_calls'] = [{
                    'id': c['tool_call_id'], 'type': 'function',
                    'function': {'name': c['function_name'],
                                 'arguments': json.dumps(c.get('arguments') or {})},
                } for c in calls]
            msgs.append(m)
            obs = s.get('observation')
            for c in calls:
                content = obs.get('content') if isinstance(obs, dict) else obs
                msgs.append({'role': 'tool', 'tool_call_id': c['tool_call_id'], 'content': content or ''})
    return msgs


def compacted_before(trace: dict, step_id: int) -> bool:
    """True if ADK compacted the history before this step (prompt size dropped a lot)."""
    prev = None
    for s in trace['steps']:
        if s['step_id'] >= step_id:
            break
        if s['source'] != 'agent':
            continue
        p = (s.get('metrics') or {}).get('prompt_tokens') or 0
        if prev is not None and p - prev < -2000:
            return True
        prev = p
    return False


def garbled_args(call: dict, observation: str) -> bool:
    args = call.get('arguments') or {}
    if any(not str(k).isidentifier() for k in args):
        return True
    fp = args.get('filepath')
    if isinstance(fp, str) and ('`' in fp or ',start_line' in fp or ',end_line' in fp):
        return True
    return 'mandatory input parameters are not present' in observation[:400]


# ----------------------------------------------------------------------------- file state
class Snapshot:
    """Read files from a task's base snapshot, replaying the session's successful edits."""

    def __init__(self, instance_id: str):
        self.tar = tarfile.open(ROOT / 'data' / 'competition' / 'snapshots' / f'{instance_id}.tgz')
        self.names = self.tar.getnames()
        self.cache: dict[str, str] = {}

    def read(self, path: str) -> str | None:
        path = path.lstrip('./')
        if path in self.cache:
            return self.cache[path]
        for cand in (path, f'workspace/{path}', f'repo/{path}'):
            if cand in self.names:
                data = self.tar.extractfile(cand).read().decode('utf-8', 'replace')
                self.cache[path] = data
                return data
        for n in self.names:  # snapshot may carry a top-level folder
            if n.endswith('/' + path):
                data = self.tar.extractfile(n).read().decode('utf-8', 'replace')
                self.cache[path] = data
                return data
        return None

    def apply_edit(self, path: str, old: str, new: str) -> None:
        cur = self.read(path)
        if cur is not None and old in cur:
            self.cache[path.lstrip('./')] = cur.replace(old, new, 1)


# ----------------------------------------------------------------------------- parsing
def parse_native_args(body: str) -> dict | None:
    """Parse `key:value,key:value` where strings are wrapped in <|"|> tokens. None if invalid."""
    args, i, n = {}, 0, len(body)
    while i < n:
        m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', body[i:])
        if not m:
            return None
        key = m.group(1)
        i += m.end()
        if body.startswith(STR, i):
            j = body.find(STR, i + len(STR))
            if j < 0:
                return None
            args[key] = body[i + len(STR):j]
            i = j + len(STR)
        else:
            m2 = re.match(r'\s*(-?\d+(?:\.\d+)?|true|false|null)', body[i:])
            if not m2:
                return None
            tok = m2.group(1)
            args[key] = {'true': True, 'false': False, 'null': None}.get(tok, None) if tok in ('true', 'false', 'null') else (float(tok) if '.' in tok else int(tok))
            i += m2.end()
        m3 = re.match(r'\s*,', body[i:])
        if m3:
            i += m3.end()
        elif body[i:].strip():
            return None
        else:
            break
    return args


def classify(text: str, snap: Snapshot | None) -> dict:
    """Label one raw completion. status: ok | garbled | no_call | multi_call."""
    calls = CALL_RE.findall(text)
    if not calls:
        return {'status': 'no_call'}
    if len(calls) > 1:
        return {'status': 'multi_call', 'tool': calls[0][0]}
    name, body = calls[0]
    args = parse_native_args(body)
    if args is None:
        return {'status': 'garbled', 'tool': name, 'why': 'syntax'}
    if name not in REQUIRED:
        return {'status': 'ok', 'tool': name, 'args': args}
    if not REQUIRED[name] <= set(args):
        return {'status': 'garbled', 'tool': name, 'why': 'missing:' + ','.join(sorted(REQUIRED[name] - set(args)))}
    if name == 'read_file' and not isinstance(args.get('filepath'), str):
        return {'status': 'garbled', 'tool': name, 'why': 'filepath'}
    if name == 'edit_file' and snap is not None:
        content = snap.read(args['filepath']) if isinstance(args.get('filepath'), str) else None
        if content is None:
            return {'status': 'ok', 'tool': name, 'args': args, 'why': 'unverified'}
        if not all(isinstance(args.get(k), str) for k in ('old_string', 'new_string')):
            return {'status': 'garbled', 'tool': name, 'why': 'edit_args_not_strings'}
        # The harness's own tiered matcher (exact -> flexible -> regex -> JSON-unescaped) is the oracle.
        from adk_eval_core.editing import apply_replacement
        res = apply_replacement(content, args['old_string'], args['new_string'],
                                allow_multiple=bool(args.get('allow_multiple', False)))
        if res.error_message:
            return {'status': 'garbled', 'tool': name, 'why': f'edit:{res.error_message[:60]}'}
        return {'status': 'ok', 'tool': name, 'args': args, 'strategy': res.strategy}
    return {'status': 'ok', 'tool': name, 'args': args}


REPAIRS = [
    # a plain backtick or quote where the closing string token belonged, before the next key or the end
    (re.compile(r'`(?=,[A-Za-z_][A-Za-z0-9_]*:|\}<tool_call\|>)'), STR),
    (re.compile(r'"(?=,[A-Za-z_][A-Za-z0-9_]*:|\}<tool_call\|>)'), STR),
    # JSON-style quoted keys
    (re.compile(r'(?<=[{,])\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:'), r'\1:'),
]


def repair(text: str, snap: Snapshot | None) -> dict | None:
    """Apply one repair rule at a time; return the first variant that verifies as well-formed."""
    for pat, rep in REPAIRS:
        fixed, n = pat.subn(rep, text)
        if n == 0 or fixed == text:
            continue
        lab = classify(fixed, snap)
        if lab['status'] == 'ok':
            return {'text': fixed, **lab, 'repair': pat.pattern[:30]}
    return None


# ----------------------------------------------------------------------------- mining
def mine_run(run: Path, out, args, seen: set) -> dict:
    sub_dir = Path(args.submission_dir).resolve() if args.submission_dir else guess_submission(run)
    sampling = yaml.safe_load((sub_dir / 'configs' / 'sampling.yaml').read_text())
    tb = (sampling.get('thinking_config') or {}).get('thinking_budget', 0)
    enable_thinking = bool(tb and int(tb) > 0)
    if enable_thinking and not args.allow_thinking:
        print(f'skip {run.name}: thinking on (budget {tb}); pass --allow-thinking to mine anyway')
        return {}
    agent_name, tools = tool_specs(sub_dir)
    stats = {'contexts': 0, 'mismatch': 0, 'compacted': 0, 'pairs': 0, 'ok': 0, 'garbled': 0, 'other': 0}
    for trace_path in sorted((run / 'traces').glob('trace_*.json')):
        trace = json.loads(trace_path.read_text())
        iid = trace_path.stem.replace('trace_', '')
        snap = None
        for s in trace['steps']:
            if s['source'] != 'agent':
                continue
            obs = (s.get('observation') or {}).get('content') or ''
            calls = s.get('tool_calls') or []
            if not any(garbled_args(c, obs) for c in calls):
                # keep file state current with successful edits
                for c in calls:
                    if c['function_name'] == 'edit_file' and '"status": "ok"' in obs[:40]:
                        a = c.get('arguments') or {}
                        if snap is None:
                            snap = Snapshot(iid)
                        if all(isinstance(a.get(k), str) for k in ('filepath', 'old_string', 'new_string')):
                            snap.apply_edit(a['filepath'], a['old_string'], a['new_string'])
                continue
            key = (run.name, iid, s['step_id'])
            if key in seen:
                continue
            seen.add(key)
            if stats['contexts'] >= args.limit:
                return stats
            if compacted_before(trace, s['step_id']):
                stats['compacted'] += 1
                continue
            messages = steps_to_messages(trace, s['step_id'], agent_name)
            kw = {'enable_thinking': enable_thinking}
            tok = post('/tokenize', {'model': MODEL, 'messages': messages, 'tools': tools,
                                     'chat_template_kwargs': kw, 'add_generation_prompt': True})
            recorded = (s.get('metrics') or {}).get('prompt_tokens')
            if recorded and abs(tok['count'] - recorded) > args.tolerance:
                stats['mismatch'] += 1
                print(f'  {iid} step {s["step_id"]}: rendered {tok["count"]} tokens, trace says {recorded}; skipped')
                continue
            stats['contexts'] += 1
            body = {'model': MODEL, 'prompt': tok['tokens'], 'n': args.samples,
                    'temperature': sampling.get('temperature', 1.0), 'top_p': sampling.get('top_p', 1.0),
                    'max_tokens': args.max_tokens, 'skip_special_tokens': False}
            if sampling.get('top_k'):
                body['top_k'] = sampling['top_k']
            t0 = time.time()
            res = post('/v1/completions', body)
            if snap is None:
                snap = Snapshot(iid)
            samples = []
            for ch in res['choices']:
                lab = {'status': 'truncated'} if ch.get('finish_reason') == 'length' else classify(ch['text'], snap)
                samples.append({'text': ch['text'], **lab})
                stats['ok' if lab['status'] == 'ok' else 'garbled' if lab['status'] == 'garbled' else 'other'] += 1
            oks = [x for x in samples if x['status'] == 'ok']
            bads = [x for x in samples if x['status'] == 'garbled']
            # Minimal pairs: a garbled call with its one format error repaired, if the repair
            # verifies (parser + harness matcher). Flagged synthetic; same prefix, same tool.
            if not any(o['tool'] == b['tool'] for o in oks for b in bads):
                for b in bads:
                    fixed = repair(b['text'], snap)
                    if fixed is not None:
                        oks.append({**fixed, 'synthetic': True})
                        samples.append({**{k: v for k, v in fixed.items() if k != 'args'}, 'synthetic': True})
                        break
            rec = {
                'run': run.name, 'instance_id': iid, 'step_id': s['step_id'],
                'prompt_tokens': tok['count'], 'enable_thinking': enable_thinking,
                'original_tool': calls[0]['function_name'], 'messages': messages, 'tools': tools,
                'sampling': {k: body[k] for k in ('temperature', 'top_p') if k in body} | ({'top_k': body['top_k']} if 'top_k' in body else {}),
                'samples': [{k: v for k, v in x.items() if k != 'args'} for x in samples],
                'pairs': [{'chosen': o['text'], 'rejected': b['text'], 'chosen_tool': o['tool'], 'rejected_tool': b['tool'], 'why': b.get('why')}
                          for o in oks for b in bads if o['tool'] == b['tool']][:args.max_pairs],
            }
            stats['pairs'] += len(rec['pairs'])
            out.write(json.dumps(rec) + '\n')
            out.flush()
            print(f'  {iid} step {s["step_id"]:3d} ({tok["count"]:5d} tok): {len(oks)} ok / {len(bads)} garbled / '
                  f'{len(samples) - len(oks) - len(bads)} other, {len(rec["pairs"])} pairs, {time.time() - t0:.0f}s', flush=True)
    return stats


def guess_submission(run: Path) -> Path:
    """Runs are named after their submission (runs/<sub>_<tasks>_<conc>)."""
    for sub in sorted((ROOT / 'submissions').iterdir(), key=lambda p: -len(p.name)):
        if run.name.startswith(sub.name):
            return sub
    raise SystemExit(f'cannot guess the submission for {run}; pass --submission-dir')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--submission-dir', type=Path, default=None, help='the submission that produced the run(s)')
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'pairs' / 'format_pairs.jsonl')
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--max-pairs', type=int, default=8, help='pairs kept per context')
    ap.add_argument('--limit', type=int, default=10 ** 9, help='contexts per run')
    ap.add_argument('--tolerance', type=int, default=8, help='allowed token-count difference vs trace')
    ap.add_argument('--allow-thinking', action='store_true')
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen.add((r['run'], r['instance_id'], r['step_id']))
    with args.out.open('a') as out:
        for run in args.runs:
            print(f'== {run}')
            stats = mine_run(run, out, args, seen)
            print(f'   {stats}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
