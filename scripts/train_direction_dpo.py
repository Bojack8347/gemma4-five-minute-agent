"""Train one preference *direction* (a LoRA adapter) with DPO on mined pairs.

Input: the JSONL written by `mine_format_pairs.py` (one record per context, with the
exact prompt messages/tools the harness rendered and same-prefix chosen/rejected raw
completions). Output: a PEFT adapter folder that `adapters/<name>/` in a submission
can ship directly (adapter_config.json + adapter_model.safetensors).

Why this shape (memo section 10.5):
  * The base is the served checkpoint decompressed to bf16 (`run_compressed=False`);
    int4 group dequantisation is exact, so the LoRA is trained on the same weights
    vLLM computes with.
  * The prefix (system, task, tool history, 3-14k tokens) is run once without
    gradient and cached; chosen and rejected completions are scored against that
    cache with gradient. Only the call tokens carry gradient ("prefix-detached" DPO),
    which is what the format direction is about and keeps long contexts affordable.
  * Reference log-probs come from the same model with the adapter disabled.

Run in the GPU environment with the vLLM server stopped (it holds ~67 GB):
  .venv-vllm/bin/python scripts/train_direction_dpo.py \
      --pairs data/pairs/format_pairs.jsonl --out adapters_out/format_r16 \
      --rank 16 --beta 0.1 --epochs 2 --lr 5e-5 --max-prompt-tokens 12000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'models' / 'gemma-4-31b-it-qat-w4a16-ct'


def load_pairs(specs: list[str], max_pairs_per_context: int, max_contexts_per_source: int | None = None,
               seed: int = 0) -> list[dict]:
    """Pairs from one or more sources. A spec is `path`, `path:weight` or `name=path:weight`; the weight
    scales the source's DPO and anchor gradients (the iE outer loop of 6.36 controls it). With
    max_contexts_per_source, a seeded sample of that many contexts is taken from each source."""
    out = []
    for spec in specs:
        name, rest = spec.split('=', 1) if '=' in spec else (None, spec)
        path, w = rest, 1.0
        if ':' in rest:
            head, tail = rest.rsplit(':', 1)
            try:
                w, path = float(tail), head
            except ValueError:
                pass
        name = name or Path(path).stem
        recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        recs = [r for r in recs if r.get('pairs')]
        if max_contexts_per_source:
            random.Random(seed).shuffle(recs)
            recs = recs[:max_contexts_per_source]
        for r in recs:
            for p in r['pairs'][:max_pairs_per_context]:
                out.append({
                    'messages': r['messages'], 'tools': r['tools'], 'enable_thinking': r['enable_thinking'],
                    'chosen': p['chosen'], 'rejected': p['rejected'], 'tool': p['chosen_tool'], 'w_src': w,
                    'key': f'{name}/{r["run"]}/{r["instance_id"]}/{r["step_id"]}',
                })
    return out


def render_prompt(tok, rec: dict) -> list[int]:
    """Token ids of the prompt exactly as vLLM renders it (verified identical on mined contexts).

    Two details make HF's rendering match vLLM's: tool-call `arguments` must be dicts (vLLM
    parses the JSON string before templating, and the template then emits the native
    bare-key form), and the system content must be a list of text parts (vLLM passes parts,
    and the template's sequence branch appends a space before the tool declarations).
    """
    import copy
    msgs = copy.deepcopy(rec['messages'])
    for m in msgs:
        for tc in m.get('tool_calls') or []:
            if isinstance(tc['function'].get('arguments'), str):
                tc['function']['arguments'] = json.loads(tc['function']['arguments'])
    if msgs and msgs[0]['role'] == 'system' and isinstance(msgs[0]['content'], str):
        msgs[0]['content'] = [{'type': 'text', 'text': msgs[0]['content']}]
    out = tok.apply_chat_template(
        msgs, tools=rec['tools'], add_generation_prompt=True, tokenize=True,
        enable_thinking=rec['enable_thinking'],
    )
    ids = out['input_ids'] if hasattr(out, 'keys') else out
    return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)


def completion_ids(tok, text: str) -> list[int]:
    """Raw completion text (special tokens included) -> ids, plus the stop token.

    vLLM strips the stop token from the returned text. After `}<tool_call|>` the model
    stops by emitting `<|tool_response>` (id 50, in generation_config.eos_token_id); a
    plain text turn stops with `<turn|>`. Both are appended here so the adapter learns
    to stop exactly where the served model does.
    """
    ids = tok.encode(text, add_special_tokens=False)
    stop = '<|tool_response>' if text.rstrip().endswith('<tool_call|>') else '<turn|>'
    stop_id = tok.convert_tokens_to_ids(stop)
    if not ids or ids[-1] != stop_id:
        ids.append(stop_id)
    return ids


@torch.no_grad()
def prefix_cache(model, prompt_ids: torch.Tensor):
    """Run the prefix once; return the cache, a snapshot of its state, and the last-position logits."""
    # logits_to_keep=1: the full-vocabulary logits of an 8k prefix would be 4-8 GiB on their own.
    # In train mode with gradient checkpointing HF returns no cache, so run this no-grad pass in
    # eval mode (checkpointing is irrelevant here) and restore the mode afterwards.
    was_training = model.training
    model.eval()
    try:
        out = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    finally:
        model.train(was_training)
    cache = out.past_key_values
    snap = snapshot(cache)
    return cache, snap, out.logits[:, -1:, :].float()


def snapshot(cache) -> list[dict]:
    """Per-layer tensor references after the prefix pass (keys/values are replaced, not mutated,
    by `torch.cat` on update, so holding references is a valid snapshot). Sliding-window layers
    cannot be `crop()`ed once past their window, which is why restore is done this way."""
    snap = []
    for layer in cache.layers:
        snap.append({k: getattr(layer, k) for k in ('keys', 'values', 'cumulative_length', 'is_initialized')
                     if hasattr(layer, k)})
    return snap


def restore(cache, snap: list[dict]) -> None:
    for layer, s in zip(cache.layers, snap):
        for k, v in s.items():
            setattr(layer, k, v)


def detach_cache(cache) -> None:
    for layer in cache.layers:
        for k in ('keys', 'values'):
            t = getattr(layer, k, None)
            if isinstance(t, torch.Tensor) and t.requires_grad:
                setattr(layer, k, t.detach())


def completion_backward_chunked(model, cache, snap, last_logits, comp_ids: torch.Tensor,
                                scale: torch.Tensor, chunk: int) -> float:
    """Back-propagate scale * log p(comp_ids | prefix) one token chunk at a time.

    The attention scores are materialised on this GPU (SDPA math path), about 30 MB per
    completion token across the 60 layers, so a whole completion's graph does not fit next
    to the 60 GiB model at long prefixes. Each chunk is forwarded with gradient against the
    cache, back-propagated at once, and the cache is detached, so earlier chunks act as
    constants for later ones (the same approximation as the detached prefix). Returns the
    summed log-probability.
    """
    restore(cache, snap)
    n = comp_ids.shape[1]
    total = 0.0
    s = 0
    while s < n:
        e = min(s + chunk, n)
        # targets s..e-1 are predicted by the prefix logits (for target 0) and by inputs s-1..e-2
        inputs = comp_ids[:, max(s - 1, 0):e - 1]
        with torch.enable_grad():
            parts = []
            if s == 0:
                parts.append(last_logits)
            if inputs.shape[1] > 0:
                out = model(input_ids=inputs, past_key_values=cache, use_cache=True)
                parts.append(out.logits.float())
            logits = torch.cat(parts, dim=1)
            logp = F.log_softmax(logits, dim=-1)
            lp = logp.gather(-1, comp_ids[:, s:e, None]).squeeze(-1).sum()
            (scale * lp).backward()
        total += lp.item()
        detach_cache(cache)
        s = e
    return total


def completion_logprob(model, cache, snap, last_logits, comp_ids: torch.Tensor, grad: bool) -> torch.Tensor:
    """Sum of log p(comp_ids | prefix), scoring the completion against the cached prefix."""
    restore(cache, snap)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if comp_ids.shape[1] > 1:
            out = model(input_ids=comp_ids[:, :-1], past_key_values=cache, use_cache=True)
            logits = torch.cat([last_logits, out.logits.float()], dim=1)
        else:
            logits = last_logits
        logp = F.log_softmax(logits, dim=-1)
        tgt = comp_ids[:, :, None]
        return logp.gather(-1, tgt).squeeze(-1).sum()


def export_for_vllm(out_dir: Path, targets: list[str]) -> None:
    """Rename `<proj>.linear.lora_X` -> `<proj>.lora_X` and list plain target names, matching the
    competition's sample adapter and vLLM's LoRA loader (which packs q/k/v and gate/up itself)."""
    from safetensors.torch import load_file, save_file
    f = out_dir / 'adapter_model.safetensors'
    tensors = load_file(str(f))
    renamed = {k.replace('.linear.lora_', '.lora_'): v.contiguous() for k, v in tensors.items()}
    save_file(renamed, str(f), metadata={'format': 'pt'})
    cfg_path = out_dir / 'adapter_config.json'
    cfg = json.loads(cfg_path.read_text())
    cfg['target_modules'] = targets
    cfg['inference_mode'] = True
    cfg_path.write_text(json.dumps(cfg, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='+', required=True, help='pair files: path, path:weight or name=path:weight')
    ap.add_argument('--max-contexts-per-source', type=int, default=None, help='seeded sample of contexts per source')
    ap.add_argument('--context-seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', type=Path, default=MODEL_DIR)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--target-modules', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--module-regex', default=r'.*language_model.*\.({projs})$', help='PEFT target regex; {projs} is filled in')
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--nll-weight', type=float, default=0.2,
                    help='SFT anchor: adds nll_weight * mean token NLL of the chosen completion (keeps the policy on the chosen distribution)')
    ap.add_argument('--stop-margin', type=float, default=1.5,
                    help='stop once the running-window mean implicit-reward margin exceeds this (0 disables)')
    ap.add_argument('--sanity-contexts', type=int, default=3,
                    help='after training, generate greedily on this many contexts and refuse to export a degenerate adapter')
    ap.add_argument('--sanity-tokens', type=int, default=256,
                    help='a sanity generation that has not closed its call within this many tokens counts as degenerate '
                         '(raise it for thinking-on contexts, where a thought plus a written script can be long)')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--max-pairs-per-context', type=int, default=3)
    ap.add_argument('--max-prompt-tokens', type=int, default=12000, help='longer prompts are left-truncated after the system message')
    ap.add_argument('--max-completion-tokens', type=int, default=1024)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None, help='debug: use only the first N pairs')
    ap.add_argument('--dry-run', action='store_true', help='tokenise and report sizes only')
    ap.add_argument('--attn', default='sdpa', help='attention implementation (sdpa, eager, flash_attention_2)')
    ap.add_argument('--chunk', type=int, default=96, help='completion tokens back-propagated at a time')
    ap.add_argument('--min-steps', type=int, default=8, help='refuse to export an adapter trained for fewer optimiser steps')
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = load_pairs(args.pairs, args.max_pairs_per_context, args.max_contexts_per_source, args.context_seed)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f'{len(pairs)} pairs from {args.pairs}')

    # Tokenise everything up front; group pairs by prompt so one prefix pass serves several pairs.
    groups: dict[str, dict] = {}
    for rec in pairs:
        if rec['key'] not in groups:
            ids = render_prompt(tok, rec)
            if len(ids) > args.max_prompt_tokens:
                # keep the system+task head and the most recent tail
                head = ids[:512]
                ids = head + ids[-(args.max_prompt_tokens - 512):]
            groups[rec['key']] = {'prompt': ids, 'pairs': []}
        c = completion_ids(tok, rec['chosen'])[:args.max_completion_tokens]
        r = completion_ids(tok, rec['rejected'])[:args.max_completion_tokens]
        groups[rec['key']]['pairs'].append((c, r, rec['tool'], rec.get('w_src', 1.0)))
    lens = [len(g['prompt']) for g in groups.values()]
    print(f'{len(groups)} contexts; prompt tokens median {sorted(lens)[len(lens) // 2]}, max {max(lens)}; '
          f'completion tokens max {max(len(x) for g in groups.values() for p in g["pairs"] for x in p[:2])}')
    if args.dry_run:
        return 0

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, CompressedTensorsConfig

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='cuda',
        quantization_config=CompressedTensorsConfig(run_compressed=False),
        attn_implementation=args.attn,
    )
    print(f'model loaded and decompressed in {time.time() - t0:.0f}s; '
          f'{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated')
    model.config.use_cache = True
    # Plain names would also match the vision tower, whose projections are Gemma4ClippableLinear
    # wrappers PEFT cannot adapt (the first GPU attempt failed on exactly that). The language
    # model's projections are nn.Linear, so restrict the match to it.
    projs = '|'.join(args.target_modules.split(','))
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias='none',
                      target_modules=args.module_regex.format(projs=projs), task_type='CAUSAL_LM')
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    # HF gradient checkpointing drops a supplied cache in Gemma 4, so memory is bounded by
    # back-propagating the completion in token chunks instead (completion_backward_chunked).
    model.eval()  # no dropout anywhere; LoRA layers train fine in eval mode

    t_phase = time.time()

    def mem(tag: str) -> None:
        nonlocal t_phase
        print(f'[mem] {tag}: {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, '
              f'{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak, {time.time() - t_phase:.0f}s since last', flush=True)
        t_phase = time.time()

    print('attention implementation:', getattr(model.config, '_attn_implementation', None),
          '| text:', getattr(getattr(model.config, 'text_config', None), '_attn_implementation', None), flush=True)

    mem('after peft')
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    keys = list(groups)
    n_steps = args.epochs * len(keys)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / max(n_steps, 1))))

    step = acc = 0
    stopped = False
    stats = {'loss': 0.0, 'acc': 0.0, 'margin': 0.0, 'n': 0}
    for epoch in range(args.epochs):
        random.shuffle(keys)
        for key in keys:
            g = groups[key]
            prompt = torch.tensor([g['prompt']], device='cuda')
            # reference log-probs: adapter disabled, no grad
            with model.disable_adapter():
                cache_ref, snap_ref, last_ref = prefix_cache(model, prompt)
                ref = [(completion_logprob(model, cache_ref, snap_ref, last_ref, torch.tensor([c], device='cuda'), grad=False),
                        completion_logprob(model, cache_ref, snap_ref, last_ref, torch.tensor([r], device='cuda'), grad=False))
                       for c, r, _, _ in g['pairs']]
            del cache_ref, snap_ref
            if acc == 0:
                mem('after reference pass')
            cache, snap, last = prefix_cache(model, prompt)
            if acc == 0:
                mem('after policy prefix pass')
            loss_total = 0.0
            for (c, r, _, w_src), (ref_c, ref_r) in zip(g['pairs'], ref):
                c_t, r_t = torch.tensor([c], device='cuda'), torch.tensor([r], device='cuda')
                # DPO gradient = sigma(-m) * beta * (grad lp_c - grad lp_r). Evaluate m without
                # gradient, then back-propagate each completion on its own so only one graph
                # is alive at a time (two live graphs were 42 GiB at a 2k prefix).
                with torch.no_grad():
                    lp_c0 = completion_logprob(model, cache, snap, last, c_t, grad=False)
                    lp_r0 = completion_logprob(model, cache, snap, last, r_t, grad=False)
                margin = args.beta * ((lp_c0 - ref_c) - (lp_r0 - ref_r))
                weight = torch.sigmoid(-margin) * args.beta * w_src / (len(g['pairs']) * args.grad_accum)
                for comp, sign in ((c_t, -1.0), (r_t, +1.0)):
                    scale = sign * weight
                    if sign < 0 and args.nll_weight > 0:
                        # anchor: minimise nll_weight * mean NLL(chosen) = -(nll_weight / n) * lp
                        scale = scale - args.nll_weight * w_src / (comp.shape[1] * args.grad_accum)
                    completion_backward_chunked(model, cache, snap, last, comp, scale, args.chunk)
                    if acc == 0:
                        mem('after one completion backward')
                loss_total += (-F.logsigmoid(margin) / (len(g['pairs']) * args.grad_accum)).item()
                stats['loss'] += -F.logsigmoid(margin).item()
                stats['acc'] += float(margin.item() > 0)
                stats['margin'] += margin.item()
                stats['n'] += 1
            del cache, snap
            if acc == 0:
                mem('after first backward')
            acc += 1
            if acc % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if stats['n'] and acc % 8 == 0:
                window_margin = stats['margin'] / stats['n']
                print(f'epoch {epoch} ctx {acc}/{len(keys)} step {step}: loss {stats["loss"] / stats["n"]:.3f} '
                      f'acc {stats["acc"] / stats["n"]:.2f} margin {window_margin:.2f} '
                      f'| {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB peak', flush=True)
                stats = {'loss': 0.0, 'acc': 0.0, 'margin': 0.0, 'n': 0}
                if args.stop_margin > 0 and window_margin > args.stop_margin and step >= 4:
                    print(f'early stop: window margin {window_margin:.2f} > {args.stop_margin}', flush=True)
                    stopped = True
                    break
        if stopped:
            break
    if step < args.min_steps:
        # A handful of warm-up steps leaves the adapter equal to the base model; the sanity gate
        # cannot tell (day 6 of 2026-10-13: 7 contexts, one step, a null direction was exported).
        print(f'TOO FEW STEPS: {step} optimiser steps < {args.min_steps}; adapter not exported', flush=True)
        return 6
    if args.sanity_contexts > 0:
        bad = sanity_generate(model, tok, groups, args)
        if bad:
            # keep the weights for inspection: a false positive must not cost the training run
            # (2026-10-19: a thinking-on reproduction script ran past the 256-token cap; 3.5 h lost)
            quarantine = args.out.with_name(args.out.name + '_quarantine')
            quarantine.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(quarantine)
            print(f'SANITY FAILED on {bad} of {args.sanity_contexts} contexts; adapter not exported '
                  f'(weights kept in {quarantine})', flush=True)
            return 5
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    export_for_vllm(args.out, args.target_modules.split(','))
    print(f'saved adapter to {args.out} after {step} steps', flush=True)
    return 0


def degenerate(ids: list[int], max_run: int = 8) -> bool:
    run = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1
        if run >= max_run:
            return True
    return False


@torch.no_grad()
def sanity_generate(model, tok, groups: dict, args) -> int:
    """Greedy generations on a few training contexts; a run of 8 identical tokens or a call that
    never closes within --sanity-tokens (default 256) marks the adapter as degenerate (the first format direction
    collapsed into an endless <|"|> loop, which this would have caught before deployment)."""
    model.eval()
    close_id = tok.convert_tokens_to_ids('<tool_call|>')
    bad = 0
    for key in list(groups)[:args.sanity_contexts]:
        prompt = torch.tensor([groups[key]['prompt']], device=next(model.parameters()).device)
        out = model.generate(prompt, max_new_tokens=args.sanity_tokens, do_sample=False, use_cache=True,
                             eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids('<turn|>'),
                                           tok.convert_tokens_to_ids('<|tool_response>')])
        gen = out[0, prompt.shape[1]:].tolist()
        text = tok.decode(gen, skip_special_tokens=False)
        flag = degenerate(gen) or (close_id not in gen and len(gen) >= args.sanity_tokens)
        bad += flag
        print(f'[sanity] {key} {"DEGENERATE" if flag else "ok"} ({len(gen)} tokens): {text[:300]!r}'
              + (f' ... {text[-200:]!r}' if len(text) > 300 else ''), flush=True)
    return bad


if __name__ == '__main__':
    raise SystemExit(main())
