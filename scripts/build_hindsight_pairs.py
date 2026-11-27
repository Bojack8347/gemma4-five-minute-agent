"""Hindsight pairs for the decisiveness direction (IMPLEMENTATION.md 6.18).

On-policy sampling at the state where the model should have edited almost never produces
the correct edit, so the chosen side comes from the trajectory itself: its later successful
`edit_file`, rendered in the native call format exactly as the chat template renders a
tool call (keys sorted, strings in <|"|> tokens, no escaping; verified identical to the
model's own calls up to key order), and verified against the file state at the earlier
step with the harness's own matcher. The rejected side is the sampled explorations the
miner already stored for that state. Records keep the miner's format.

Usage:
  .venv/bin/python scripts/build_hindsight_pairs.py data/pairs/decisive_earliest_pairs.jsonl data/pairs/decisive_pairs.jsonl \
      --out data/pairs/decisive_hindsight_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_decisive_pairs import first_good_edit  # noqa: E402
from mine_format_pairs import STR, Snapshot, classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def render_native_call(name: str, args: dict) -> str:
    """The chat template's rendering of a tool call: `key:value` pairs in sorted key order."""
    parts = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, bool):
            parts.append(f'{k}:{"true" if v else "false"}')
        elif isinstance(v, (int, float)):
            parts.append(f'{k}:{v}')
        elif v is None:
            parts.append(f'{k}:null')
        else:
            parts.append(f'{k}:{STR}{v}{STR}')
    return f'<|tool_call>call:{name}{{' + ','.join(parts) + '}<tool_call|>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'pairs' / 'decisive_hindsight_pairs.jsonl')
    ap.add_argument('--max-pairs', type=int, default=8)
    args = ap.parse_args()
    stats = {'contexts': 0, 'chosen_ok': 0, 'chosen_bad': 0, 'pairs': 0}
    seen = set()
    with args.out.open('w') as out:
        for path in args.inputs:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r['run'], r['instance_id'], r['step_id'])
                if key in seen:
                    continue
                seen.add(key)
                trace_file = next((d / 'traces' / f'trace_{r["instance_id"]}.json' for d in (ROOT / 'runs' / r['run'], ROOT / 'runs' / 'ie_pilot' / r['run'])
                                   if (d / 'traces' / f'trace_{r["instance_id"]}.json').exists()), None)
                if trace_file is None:
                    continue
                trace = json.loads(trace_file.read_text())
                step, good = first_good_edit(trace)
                if good is None:
                    continue
                gargs = {k: v for k, v in (good.get('arguments') or {}).items() if k in ('filepath', 'old_string', 'new_string', 'allow_multiple')}
                if not all(isinstance(gargs.get(k), str) for k in ('filepath', 'old_string', 'new_string')):
                    continue
                # file state at the mined step: base snapshot plus the successful edits before it
                snap = Snapshot(r['instance_id'])
                for s in trace['steps']:
                    if s['step_id'] >= r['step_id'] or s['source'] != 'agent':
                        continue
                    obs = (s.get('observation') or {}).get('content') or ''
                    for c in s.get('tool_calls') or []:
                        a = c.get('arguments') or {}
                        if c['function_name'] == 'edit_file' and obs[:40].startswith('{"status": "ok"') and all(isinstance(a.get(k), str) for k in ('filepath', 'old_string', 'new_string')):
                            snap.apply_edit(a['filepath'], a['old_string'], a['new_string'])
                chosen_text = render_native_call('edit_file', gargs)
                lab = classify(chosen_text, snap)
                stats['contexts'] += 1
                if lab['status'] != 'ok':
                    stats['chosen_bad'] += 1
                    continue
                stats['chosen_ok'] += 1
                explores = [s for s in r['samples'] if s.get('role') == 'explore']
                onpolicy = [s for s in r['samples'] if s.get('role') == 'edit_at_location']
                chosen = [{'text': chosen_text, 'synthetic': True}] + [{'text': s['text'], 'synthetic': False} for s in onpolicy[:1]]
                pairs = [{'chosen': c['text'], 'rejected': x['text'], 'chosen_tool': 'edit_file', 'rejected_tool': x['tool'],
                          'why': 'explore_instead_of_edit', 'synthetic': c['synthetic']} for x in explores for c in chosen][:args.max_pairs]
                stats['pairs'] += len(pairs)
                rec = dict(r)
                rec['pairs'] = pairs
                rec['hindsight_chosen'] = chosen_text[:300]
                out.write(json.dumps(rec) + '\n')
    print(stats)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
