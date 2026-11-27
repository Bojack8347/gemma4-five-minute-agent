"""Exact weighted merge of LoRA directions into one shippable adapter (memo section 10.2).

PEFT applies  dW_j = (alpha_j / r_j) B_j A_j  for each direction j. For weights a_j the
merged adapter has rank r = sum_j r_j and, with lora_alpha = r (scaling 1),

    A = [A_1; ...; A_d]            (r x in)
    B = [a_1 s_1 B_1, ..., a_d s_d B_d]   (out x r),  s_j = alpha_j / r_j

so  B A = sum_j a_j s_j B_j A_j  exactly. A module that only some directions target gets
zero blocks for the others. Negative weights are allowed. The rank must stay <= 128
(vLLM max_lora_rank in the scorer): eight rank-16 or four rank-32 directions.

Usage:
  .venv-vllm/bin/python scripts/merge_directions.py --out submissions/x/adapters/merged \
      --direction format=adapters_out/format_r16:1.0 --direction decisive=adapters_out/decisive_r16:0.5
Importable: merge(directions: list[(name, path, weight)], out_dir).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MAX_RANK = 128
KEY_RE = re.compile(r'^(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$')


def load_direction(path: Path) -> tuple[dict, dict[str, dict[str, torch.Tensor]]]:
    cfg = json.loads((path / 'adapter_config.json').read_text())
    tensors = load_file(str(path / 'adapter_model.safetensors'))
    mods: dict[str, dict[str, torch.Tensor]] = {}
    for k, t in tensors.items():
        m = KEY_RE.match(k)
        if not m:
            raise ValueError(f'{path}: unexpected tensor name {k}')
        mods.setdefault(m['module'], {})[m['ab']] = t
    return cfg, mods


def merge(directions: list[tuple[str, Path, float]], out_dir: Path) -> dict:
    loaded = [(name, w, *load_direction(Path(p))) for name, p, w in directions]
    ranks = [cfg['r'] for _, _, cfg, _ in loaded]
    r = sum(ranks)
    if r > MAX_RANK:
        raise ValueError(f'merged rank {r} exceeds {MAX_RANK}; use fewer or lower-rank directions')
    modules = sorted({m for _, _, _, mods in loaded for m in mods})
    out: dict[str, torch.Tensor] = {}
    for mod in modules:
        a_blocks, b_blocks = [], []
        in_dim = out_dim = None
        for _, w, cfg, mods in loaded:
            if mod in mods:
                A, B = mods[mod]['A'], mods[mod]['B']
                in_dim, out_dim = A.shape[1], B.shape[0]
                break
        for _, w, cfg, mods in loaded:
            rj, s = cfg['r'], cfg['lora_alpha'] / cfg['r']
            if mod in mods:
                A, B = mods[mod]['A'], mods[mod]['B']
                a_blocks.append(A.to(torch.float32))
                b_blocks.append((w * s) * B.to(torch.float32))
            else:
                a_blocks.append(torch.zeros(rj, in_dim))
                b_blocks.append(torch.zeros(out_dim, rj))
        out[f'{mod}.lora_A.weight'] = torch.cat(a_blocks, 0).to(torch.bfloat16).contiguous()
        out[f'{mod}.lora_B.weight'] = torch.cat(b_blocks, 1).to(torch.bfloat16).contiguous()
    base_cfg = dict(loaded[0][2])
    targets = sorted({t for _, _, cfg, _ in loaded for t in cfg['target_modules']})
    base_cfg.update({
        'r': r, 'lora_alpha': r, 'target_modules': targets, 'rank_pattern': {}, 'alpha_pattern': {},
        'use_rslora': False, 'use_dora': False, 'inference_mode': True,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    (out_dir / 'adapter_config.json').write_text(json.dumps(base_cfg, indent=2))
    manifest = {'directions': [{'name': n, 'path': str(p), 'weight': w, 'rank': cfg['r'], 'alpha': cfg['lora_alpha']}
                               for (n, p, w), (_, _, cfg, _) in zip(directions, loaded)], 'rank': r}
    (out_dir / 'merge.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--direction', action='append', required=True, help='name=path:weight')
    args = ap.parse_args()
    dirs = []
    for d in args.direction:
        name, rest = d.split('=', 1)
        path, w = rest.rsplit(':', 1)
        dirs.append((name, Path(path), float(w)))
    m = merge(dirs, args.out)
    print(json.dumps(m, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
