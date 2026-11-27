"""Trajectory-level distillation: supervised fine-tuning of a LoRA on the informed agent's resolved
trajectories, rendered title-only (IMPLEMENTATION.md 6.34; data from build_traj_sft.py).

Each agent step is one example: prompt = the title-only history up to the step (no gradient), target =
the informed agent's thought and call in generation form (gradient, chunked as in train_direction_dpo).
Steps of one trajectory are trained in order with a shared KV cache: after a step, the cache is reset to
its prompt and extended with the next prompt's new tokens (the step's call rendered as history plus the
tool response), so a trajectory costs about one forward pass over its full length instead of one per step.
The loss is the mean token NLL of the target; thoughts that quote the full statement are masked out of
the loss but kept in the target as conditioning.

Usage (server stopped; .venv-vllm):
  .venv-vllm/bin/python scripts/train_traj_sft.py --data data/traj_sft/trajectories.jsonl --out adapters_out/traj_r16
  ... --dry-run   tokenise everything and report sizes, prefix-property violations and mask boundaries
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_direction_dpo import (MODEL_DIR, completion_ids, detach_cache, export_for_vllm, prefix_cache,  # noqa: E402
                                 render_prompt, restore, sanity_generate, snapshot)

ROOT = Path(__file__).resolve().parent.parent


def load_trajectories(path: Path, limit: int | None) -> list[dict]:
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return out[:limit] if limit else out


def tokenise(tok, trajs: list[dict], args) -> dict:
    """Prompt ids per step, target ids and loss masks; checks that each prompt extends the previous one."""
    stats = {'steps': 0, 'trained': 0, 'masked_thoughts': 0, 'mask_boundary_miss': 0, 'prefix_violations': 0,
             'over_cap_steps': 0, 'over_cap_trained': 0, 'prompt_tokens': [], 'target_tokens': []}
    rng = random.Random(args.seed + 1)
    for traj in trajs:
        prev = None
        for st in traj['steps']:
            ids = render_prompt(tok, {'messages': traj['messages'][:st['n_msgs']], 'tools': traj['tools'], 'enable_thinking': True})
            st['prompt_len'] = len(ids)
            stats['steps'] += 1
            if len(ids) > args.max_prompt_tokens:
                # Beyond the cap the shared cache cannot follow (2026-10-21: an 11.4k prompt peaked at 108 GiB
                # and thrashed). Such a step trains on its own, on the system/task head plus the most recent
                # tail (train_direction_dpo's truncation): every edit or write, and a sample of the rest.
                stats['over_cap_steps'] += 1
                keep = not st.get('skip') and (st.get('tool') in ('edit_file', 'write_file') or rng.random() < args.over_cap_sample)
                if not keep:
                    st['drop'] = True
                    continue
                st['over'] = True
                ids = ids[:args.head_tokens] + ids[-(args.max_prompt_tokens - args.head_tokens):]
                stats['over_cap_trained'] += 1
            else:
                if prev is not None and ids[:len(prev)] != prev:
                    stats['prefix_violations'] += 1
                prev = ids
            st['prompt'] = None if args.dry_run else ids
            if st.get('skip'):
                continue
            comp = completion_ids(tok, st['thought'] + st['call'])[:args.max_completion_tokens]
            mask = [1.0] * len(comp)
            if st['thought']:
                head = tok.encode(st['thought'], add_special_tokens=False)
                b = len(head)
                if comp[:b] == head:
                    st['call_start'] = b
                    if st['mask_thought']:
                        mask = [0.0] * b + [1.0] * (len(comp) - b)
                        stats['masked_thoughts'] += 1
                else:
                    st['call_start'] = 0
                    stats['mask_boundary_miss'] += 1
            else:
                st['call_start'] = 0
            st['comp'], st['mask'] = comp, mask
            stats['trained'] += 1
            stats['prompt_tokens'].append(len(ids))
            stats['target_tokens'].append(len(comp))
    return stats


@torch.no_grad()
def extend_cache(model, cache, seg: torch.Tensor):
    was = model.training
    model.eval()
    try:
        out = model(input_ids=seg, past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        model.train(was)
    return snapshot(cache), out.logits[:, -1:, :].float()


def backward_masked(model, cache, snap, last_logits, comp: torch.Tensor, mask: torch.Tensor, call_start: int,
                    scale: float, chunk: int) -> tuple[float, float, int]:
    """Back-propagate scale * sum(mask * log p(comp | prefix)) chunk by chunk (train_direction_dpo's scheme).
    Returns (masked log-prob sum, call-token log-prob sum, call tokens)."""
    restore(cache, snap)
    n = comp.shape[1]
    total = call_lp = 0.0
    s = 0
    while s < n:
        e = min(s + chunk, n)
        inputs = comp[:, max(s - 1, 0):e - 1]
        m = mask[:, s:e]
        live = bool(m.sum() > 0)
        with torch.enable_grad() if live else torch.no_grad():
            parts = []
            if s == 0:
                parts.append(last_logits)
            if inputs.shape[1] > 0:
                out = model(input_ids=inputs, past_key_values=cache, use_cache=True)
                parts.append(out.logits.float())
            logits = torch.cat(parts, dim=1)
            logp = F.log_softmax(logits, dim=-1).gather(-1, comp[:, s:e, None]).squeeze(-1)
            lp = (logp * m).sum()
            if live:
                (scale * lp).backward()
        total += lp.item()
        if e > call_start:
            call_lp += logp[:, max(call_start - s, 0):].sum().item()
        detach_cache(cache)
        s = e
    return total, call_lp, n - call_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'traj_sft' / 'trajectories.jsonl')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', type=Path, default=MODEL_DIR)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--target-modules', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--module-regex', default=r'.*language_model.*\.({projs})$')
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--grad-accum', type=int, default=4, help='steps per optimiser update')
    ap.add_argument('--max-prompt-tokens', type=int, default=8192, help='shared-cache prompts up to this; longer steps are truncated to it')
    ap.add_argument('--head-tokens', type=int, default=512, help='system/task head kept when a prompt is truncated')
    ap.add_argument('--over-cap-sample', type=float, default=0.2, help='share of non-edit over-cap steps trained (edits: all)')
    ap.add_argument('--mem-soft-gib', type=float, default=100.0, help='lower the cap by 1000 tokens after a trajectory peaks above this')
    ap.add_argument('--max-completion-tokens', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=96)
    ap.add_argument('--chunk-long', type=int, default=48, help='chunk once the prompt exceeds 8k tokens')
    ap.add_argument('--limit', type=int, default=None, help='first N trajectories only')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt-every', type=int, default=20, help='save the adapter every N trajectories')
    ap.add_argument('--sanity-contexts', type=int, default=3)
    ap.add_argument('--sanity-tokens', type=int, default=1024)
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    trajs = load_trajectories(args.data, args.limit)
    t0 = time.time()
    stats = tokenise(tok, trajs, args)
    pt, tt = sorted(stats['prompt_tokens']), sorted(stats['target_tokens'])
    print(f'{len(trajs)} trajectories, {stats["steps"]} steps rendered, {stats["trained"]} trained '
          f'({stats["masked_thoughts"]} with the thought masked, {stats["mask_boundary_miss"]} boundary misses; '
          f'{stats["over_cap_trained"]} of {stats["over_cap_steps"]} over-cap steps kept, truncated), {stats["prefix_violations"]} prefix violations; '
          f'prompt tokens median {pt[len(pt) // 2]} p90 {pt[int(0.9 * len(pt))]} max {pt[-1]}; '
          f'target tokens median {tt[len(tt) // 2]} max {tt[-1]}; tokenised in {time.time() - t0:.0f}s', flush=True)
    if args.dry_run:
        return 0

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, CompressedTensorsConfig
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda',
                                                 quantization_config=CompressedTensorsConfig(run_compressed=False),
                                                 attn_implementation=args.attn)
    print(f'model loaded and decompressed in {time.time() - t0:.0f}s; {torch.cuda.memory_allocated() / 2**30:.1f} GiB', flush=True)
    model.config.use_cache = True
    projs = '|'.join(args.target_modules.split(','))
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias='none',
                                             target_modules=args.module_regex.format(projs=projs), task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n_updates = max(1, math.ceil(stats['trained'] / args.grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, n_updates) / n_updates)))

    order = list(range(len(trajs)))
    random.shuffle(order)
    sanity_groups = {}
    acc = step = fallbacks = 0
    win = {'lp': 0.0, 'n': 0, 'call_lp': 0.0, 'call_n': 0}
    t_run = time.time()
    for ti, idx in enumerate(order):
        traj = trajs[idx]
        cache = snap = last = None
        cur: list[int] | None = None
        t_traj = time.time()
        n_tr = 0
        for st in traj['steps']:
            if st.get('drop') or st.get('prompt') is None:
                continue
            ids = st['prompt']
            if st.get('over'):
                ids = ids[:args.head_tokens] + ids[-(args.max_prompt_tokens - args.head_tokens):]  # the cap may have dropped
                if cache is not None:
                    del cache, snap, last
                    cache = snap = last = None
                    cur = None
            elif len(ids) > args.max_prompt_tokens:
                continue  # cap lowered by the memory guard after tokenisation
            t_ids = torch.tensor([ids], device='cuda')
            if st.get('over'):
                cache, snap, last = prefix_cache(model, t_ids)
                cur = None
            elif cache is not None and cur is not None and len(ids) > len(cur) and ids[:len(cur)] == cur:
                restore(cache, snap)
                snap, last = extend_cache(model, cache, t_ids[:, len(cur):])
            elif cache is not None and ids == cur:
                restore(cache, snap)
            else:
                if cache is not None:
                    fallbacks += 1
                del cache
                cache, snap, last = prefix_cache(model, t_ids)
            if not st.get('over'):
                cur = ids
            if len(sanity_groups) < args.sanity_contexts and st.get('comp') is not None and n_tr == 0:
                sanity_groups[f'{traj["run"]}/{traj["instance_id"]}/{st["step_id"]}'] = {'prompt': ids}
            if st.get('skip') or st.get('comp') is None:
                continue
            comp = torch.tensor([st['comp']], device='cuda')
            mask = torch.tensor([st['mask']], device='cuda', dtype=torch.float32)
            n_loss = float(mask.sum().item())
            scale = -1.0 / (n_loss * args.grad_accum)
            chunk = args.chunk_long if len(ids) > 6000 else args.chunk
            lp, call_lp, call_n = backward_masked(model, cache, snap, last, comp, mask, st['call_start'], scale, chunk)
            win['lp'] += lp
            win['n'] += n_loss
            win['call_lp'] += call_lp
            win['call_n'] += call_n
            n_tr += 1
            acc += 1
            if acc % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        del cache, snap, last
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f'[{time.strftime("%F %T")}] traj {ti + 1}/{len(trajs)} {traj["instance_id"]}: {n_tr} steps in {time.time() - t_traj:.0f}s '
              f'(cap {args.max_prompt_tokens}); window NLL {-win["lp"] / max(win["n"], 1):.3f} '
              f'call NLL {-win["call_lp"] / max(win["call_n"], 1):.3f} over {int(win["n"])} tokens; updates {step}; '
              f'fallbacks {fallbacks}; {peak:.0f} GiB peak; '
              f'{(time.time() - t_run) / 60:.0f} min elapsed', flush=True)
        if peak > args.mem_soft_gib and args.max_prompt_tokens > 4096:
            args.max_prompt_tokens -= 1000
            print(f'  memory guard: peak {peak:.0f} GiB > {args.mem_soft_gib:.0f}; prompt cap lowered to {args.max_prompt_tokens}', flush=True)
        torch.cuda.reset_peak_memory_stats()
        if (ti + 1) % 10 == 0:
            win = {'lp': 0.0, 'n': 0, 'call_lp': 0.0, 'call_n': 0}
        if args.ckpt_every and (ti + 1) % args.ckpt_every == 0:
            ck = args.out.with_name(args.out.name + '_ckpt')
            ck.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ck)
    if acc % args.grad_accum:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
    print(f'training done: {acc} steps, {step} updates, {fallbacks} prefix fallbacks, {(time.time() - t_run) / 3600:.2f} h', flush=True)
    if args.sanity_contexts > 0 and sanity_groups:
        bad = sanity_generate(model, tok, sanity_groups, args)
        if bad:
            q = args.out.with_name(args.out.name + '_quarantine')
            q.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(q)
            print(f'SANITY FAILED on {bad} of {len(sanity_groups)} contexts; adapter not exported (weights kept in {q})', flush=True)
            return 5
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    export_for_vllm(args.out, args.target_modules.split(','))
    print(f'saved adapter to {args.out} after {step} updates', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
