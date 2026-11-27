"""Compare raw completions from an adapter route and the base model on one rendered task.

The quickest check that a mounted adapter behaves (the first format direction looped on
`<|"|>` and was caught this way within a minute, IMPLEMENTATION.md 6.13). Renders the
submission's system prompt and tool declarations exactly as the harness does, then samples
one completion per (thinking on/off) x (base, route) and prints the raw text.

Usage:
  .venv/bin/python scripts/probe_route.py --route merged [--submission-dir submissions/v0_long] [--max-tokens 700]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_format_pairs import MODEL, post, tool_specs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASK = ("You are evaluating a software engineering task for repository fastapi/fastapi.\n\nProblem Statement:\n"
        "Fix typo in error message of `jsonable_encoder` when `exclude_unset` is used with a dataclass.\n\nFix the issue.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--route', default='merged', help='served adapter name')
    ap.add_argument('--submission-dir', type=Path, default=ROOT / 'submissions' / 'v0_long')
    ap.add_argument('--max-tokens', type=int, default=700)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--task', default=TASK)
    args = ap.parse_args()
    agent_name, tools = tool_specs(args.submission_dir.resolve())
    system = (args.submission_dir / 'prompts' / 'system.md').read_text() + f'\n\nYou are an agent. Your internal name is "{agent_name}".'
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': args.task}]
    for thinking in (True, False):
        tok = post('/tokenize', {'model': MODEL, 'messages': msgs, 'tools': tools,
                                 'chat_template_kwargs': {'enable_thinking': thinking}, 'add_generation_prompt': True})
        for model in (MODEL, args.route):
            res = post('/v1/completions', {'model': model, 'prompt': tok['tokens'], 'n': 1, 'temperature': args.temperature,
                                           'top_p': 0.95, 'max_tokens': args.max_tokens, 'skip_special_tokens': False})
            ch = res['choices'][0]
            flag = ' <-- hit the length limit' if ch['finish_reason'] == 'length' else ''
            print(f'=== thinking={thinking} model={model}: {res["usage"]["completion_tokens"]} tokens, finish={ch["finish_reason"]}{flag}')
            print(repr(ch['text'][:400]))
            if len(ch['text']) > 400:
                print('   ...', repr(ch['text'][-160:]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
