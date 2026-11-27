"""Reachability analysis for the informed-target controller (IMPLEMENTATION.md 6.41).

Decomposes the whitened residual W (alpha - z) at the probe base and after each step into the part that
lies in the column space of the whitened Jacobian W G (what the knobs can move) and the part orthogonal
to it (what no weighting of these knobs reaches, to first order). Reports both norms, the singular
structure of W G, the per-coordinate reachable share, and the identity that ties the edit coordinates
to resolution: source_edit_rate = resolution_rate + wrong_edit_rate for every run.

Usage: .venv/bin/python scripts/ie_informed_report.py configs/ie_informed.yaml [--state runs/ie_informed/state.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ie_control import fill_targets, residual_of, state_path, targets_of, z_of, z_target  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def decompose(Gw: np.ndarray, v: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Orthogonal projection of v on the column space of Gw (rank by rcond) and its complement."""
    U, s, _ = np.linalg.svd(Gw, full_matrices=False)
    k = int(np.sum(s > rcond * s[0])) if s.size else 0
    Uk = U[:, :k]
    inside = Uk @ (Uk.T @ v)
    return inside, v - inside, k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', type=Path)
    ap.add_argument('--state', type=Path, default=None)
    ap.add_argument('--json', type=Path, default=None, help='write the numbers here')
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    fill_targets(cfg)
    st = json.loads((a.state or state_path(cfg)).read_text())
    coords = targets_of(cfg)
    sigma = np.array(st['sigma']) if st.get('sigma') else np.ones(len(coords))
    w = 1.0 / sigma
    rcond = cfg['gain'].get('rcond', 0.1)
    out = {'coords': coords, 'alpha': [t['value'] for t in cfg['targets']], 'sigma': sigma.tolist(), 'points': []}

    def report(label: str, G: np.ndarray, r: np.ndarray, x: dict) -> None:
        Gw = G * w[:, None]
        v = w * r
        inside, outside, k = decompose(Gw, v, rcond)
        tot = float(np.linalg.norm(v))
        rec = {'label': label, 'x': x, 'residual': r.tolist(), 'whitened_norm': tot,
               'reachable_norm': float(np.linalg.norm(inside)), 'unreachable_norm': float(np.linalg.norm(outside)),
               'rank': k, 'reachable_by_coord': (inside / np.where(np.abs(v) > 1e-9, v, np.nan)).tolist()}
        out['points'].append(rec)
        print(f'{label}: x={x}')
        print(f'  |W r| = {tot:.2f};  in span(W G) (rank {k}): {rec["reachable_norm"]:.2f};  orthogonal: {rec["unreachable_norm"]:.2f}'
              f'  ({100 * rec["unreachable_norm"] ** 2 / max(tot ** 2, 1e-12):.0f}% of the squared residual is unreachable to first order)')
        for c, ri, vi, ii in zip(coords, r, v, inside):
            print(f'    {c:22} residual {ri:+.3f}  whitened {vi:+.2f}  reachable part {ii:+.2f}')

    G_probe = np.array(st.get('G_probe') or st['G'])
    print('targets alpha:', {c: round(v, 4) for c, v in zip(coords, out['alpha'])})
    print('sigma:', np.round(sigma, 3).tolist())
    s = np.linalg.svd(G_probe * w[:, None], compute_uv=False)
    print('singular values of W G (probe):', np.round(s, 3).tolist())
    if st.get('base_z') is not None:
        zb = z_target(cfg)
        r0 = zb - np.array(st['base_z'])
        for i, t in enumerate(cfg['targets']):
            if t.get('side') == 'max':
                r0[i] = min(0.0, r0[i])
        x0 = {k['name']: k['x0'] for k in cfg['knobs']}
        report('probe base (probe Jacobian)', G_probe, np.nan_to_num(r0), x0)
        if st.get('G') and st.get('G_history'):
            report('probe base (re-estimated Jacobian)', np.array(st['G']), np.nan_to_num(r0), x0)
    for run in st['runs']:
        if run['label'].startswith('step_'):
            report(run['label'] + (f' [{run["gate"]}]' if run.get('gate') else ''), np.array(st['G']), residual_of(cfg, run['y']), run['x'])
    # the identity that ties the edit coordinates to resolution
    print('identity source_edit = resolution + wrong_edit (per run):')
    for run in st['runs']:
        y = run['y']
        print(f'  {run["label"]:18} {y["source_edit_rate"]:.3f} = {y["resolution_rate"]:.3f} + {y["wrong_edit_rate"]:.3f}  '
              f'(diff {y["source_edit_rate"] - y["resolution_rate"] - y["wrong_edit_rate"]:+.3f}); resolved {round(y["resolution_rate"] * y["n_tasks"])}/{y["n_tasks"]}')
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
