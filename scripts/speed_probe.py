"""Measure raw serving speed of an OpenAI-compatible endpoint (stdlib only).

For several prompt sizes it reports prefill time (uncached and prefix-cached) and
decode tokens/second, plus one tool-call round trip. The same code is embedded in the
Kaggle timing notebook so local and L4x4 numbers are directly comparable.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid

MODEL = 'gemma-4-31b-it-qat-w4a16-ct'

FILLER = (
    'def resolve_dependency(graph, node, seen=None):\n'
    '    """Walk the dependency graph depth-first and return nodes in install order."""\n'
    '    seen = set() if seen is None else seen\n'
    '    order = []\n'
    '    for child in sorted(graph.get(node, ())):\n'
    '        if child not in seen:\n'
    '            seen.add(child)\n'
    '            order.extend(resolve_dependency(graph, child, seen))\n'
    '    order.append(node)\n'
    '    return order\n\n'
)


def post(base: str, path: str, body: dict, timeout: float = 900.0) -> tuple[dict, float]:
    req = urllib.request.Request(
        base.rstrip('/') + path,
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, time.perf_counter() - t0


def chat(base: str, model: str, prompt: str, max_tokens: int, thinking: bool = False) -> tuple[dict, float]:
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'min_tokens': max_tokens,
        'temperature': 0.0,
        'chat_template_kwargs': {'enable_thinking': thinking},
    }
    return post(base, '/chat/completions', body)


def probe(base: str = 'http://127.0.0.1:8000/v1', model: str = MODEL, decode_tokens: int = 384) -> dict:
    results: dict = {'model': model, 'sizes': []}
    chat(base, model, 'Say hi.', 8)  # warm up

    for repeats in (8, 60, 120):  # roughly 1k / 8k / 16k prompt tokens
        nonce = uuid.uuid4().hex
        prompt = f'# session {nonce}\n' + FILLER * repeats + '\nExplain this code in great detail.'
        out1, t_prefill = chat(base, model, prompt, 1)
        out_cached, t_cached = chat(base, model, prompt, 1)
        out2, t_full = chat(base, model, prompt, decode_tokens)
        usage = out2.get('usage', {})
        completion = usage.get('completion_tokens', decode_tokens)
        decode_s = max(t_full - t_cached, 1e-6)
        results['sizes'].append({
            'prompt_tokens': usage.get('prompt_tokens'),
            'prefill_seconds_uncached': round(t_prefill, 2),
            'prefill_tok_per_s_uncached': round((usage.get('prompt_tokens') or 0) / max(t_prefill, 1e-6), 1),
            'prefill_seconds_cached': round(t_cached, 2),
            'completion_tokens': completion,
            'decode_tok_per_s': round((completion - 1) / decode_s, 2),
        })

    # One realistic agent-style turn: thinking on, a tool available.
    tools = [{
        'type': 'function',
        'function': {
            'name': 'read_file',
            'description': 'Read a file from the workspace.',
            'parameters': {
                'type': 'object',
                'properties': {'filepath': {'type': 'string'}},
                'required': ['filepath'],
            },
        },
    }]
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Open src/app.py so we can find the bug in the login handler.'}],
        'tools': tools,
        'tool_choice': 'auto',
        'max_tokens': 1024,
        'temperature': 0.0,
    }
    out, t = post(base, '/chat/completions', body)
    msg = out['choices'][0]['message']
    results['tool_turn'] = {
        'seconds': round(t, 2),
        'completion_tokens': out.get('usage', {}).get('completion_tokens'),
        'has_reasoning': bool(msg.get('reasoning_content') or msg.get('reasoning')),
        'tool_calls': [c['function'] for c in (msg.get('tool_calls') or [])],
        'finish_reason': out['choices'][0].get('finish_reason'),
    }
    return results


if __name__ == '__main__':
    base_url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000/v1'
    model_name = sys.argv[2] if len(sys.argv) > 2 else MODEL
    print(json.dumps(probe(base_url, model=model_name), indent=2))
