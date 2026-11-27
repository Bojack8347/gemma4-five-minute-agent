"""Build the paper's figures from the run folders (no numbers are typed in by hand).

Run with the plotting environment:  .venv-paper/bin/python paper/make_figures.py
Writes paper/figures/{thinking_tradeoff,main_100,trajectories}.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from behaviour_vector import behaviour_vector  # noqa: E402

OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette, light mode, first three categorical slots (validated all-pairs; slot 3 is
# below 3:1 contrast on white, so every series is also direct-labelled).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
})


def run(name: str, n_tasks: int | None = None) -> dict | None:  # noqa: D103
    """Behaviour vector of a run; None if missing or (when n_tasks is given) not finished."""
    d = ROOT / 'runs' / name
    if not (d / 'task_results.jsonl').exists():
        return None
    y = behaviour_vector(d)
    return y if n_tasks is None or y['n_tasks'] == n_tasks else None


def thinking_tradeoff() -> None:
    points = [('512', 'v0_lean_dev24_c1'), ('128', 'v0_think128_dev24_c2x2'), ('64', 'v0_think64_dev24_c2x2'),
              ('32', 'v0_think32_dev24_c2x2'), ('off', 'v0_think0_dev24_c2x2')]
    ys = [run(r) for _, r in points]
    labels = [p for p, _ in points]
    x = np.arange(len(points))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for ax, key, title, fmt, log in [
        (axes[0], 'garbled_call_rate', 'Malformed tool calls (share of calls)', '{:.1%}', True),
        (axes[1], 'source_edit_rate', 'Tasks whose patch touches source', '{:.0%}', False),
    ]:
        vals = [y[key] for y in ys]
        ax.plot(x, vals, color=SERIES[0], lw=1.5, marker='o', ms=5, zorder=3)
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), textcoords='offset points', xytext=(0, 6), ha='center', color=INK, fontsize=7.5)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.4, len(points) - 0.6)
        ax.set_xlabel('thinking budget per turn (tokens)')
        ax.set_title(title, loc='left', color=INK)
        if log:
            ax.set_yscale('log')
            ax.set_ylim(0.003, 0.6)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0%}' if v >= 0.01 else f'{v:.1%}'))
        else:
            ax.set_ylim(0, 0.55)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / 'thinking_tradeoff.pdf')
    plt.close(fig)


def main_100() -> None:
    groups = [
        ('base', ['base_sound100_c8', 'base_sound100_c8_r2'], SERIES[0]),
        ('format direction', ['format1_sound100_c8', 'format1_sound100_c8_r2'], SERIES[1]),
        ('gated end point', ['gated3_end_sound100_c8'], SERIES[2]),
    ]
    data = [(g, [y for y in (run(r, n_tasks=100) for r in runs) if y is not None], c) for g, runs, c in groups]
    panels = [
        ('resolution_rate', 'Resolved', lambda v: f'{v * 100:.1f}'.replace('.0', ''), (0.25, 0.42), matplotlib.ticker.PercentFormatter(1.0, decimals=0)),
        ('garbled_call_rate', 'Malformed calls', lambda v: f'{v * 100:.1f}%', (0, 0.065), matplotlib.ticker.PercentFormatter(1.0, decimals=0)),
        ('repeat_loop_rate', 'Tasks with loops', lambda v: f'{v * 100:.0f}%', (0, 0.15), matplotlib.ticker.PercentFormatter(1.0, decimals=0)),
        ('first_edit_s_kaggle', 'First edit (scorer s)', lambda v: f'{v:.0f}', (200, 290), None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.0), sharey=False)
    for ax, (key, title, lab, ylim, fmtr) in zip(axes.flat, panels):
        for gi, (g, ys, c) in enumerate(data):
            vals = [y[key] for y in ys]
            xs = gi + (np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.zeros(1))
            ax.scatter(xs, vals, s=34, color=c, edgecolor='white', linewidth=1.0, zorder=3)
            if vals:
                m = float(np.mean(vals))
                ax.annotate(lab(m), (gi, m), textcoords='offset points', xytext=(14, -3), ha='left', color=INK, fontsize=7.5)
        ax.set_xticks(range(len(data)), ['base', 'format\ndirection', 'gated\nend point'])
        ax.set_xlim(-0.6, len(data) - 0.4)
        ax.set_ylim(*ylim)
        if fmtr is not None:
            ax.yaxis.set_major_formatter(fmtr)
        ax.set_title(title, loc='left', color=INK)
        ax.grid(axis='x', visible=False)
    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    fig.savefig(OUT / 'main_100.pdf')
    plt.close(fig)


def trajectories() -> None:
    def residuals(state_file: str) -> tuple[list[float], list[str]]:
        st = json.loads((ROOT / 'runs' / state_file / 'state.json').read_text())
        sig = np.array(st['sigma'])
        out, gates = [], []
        for r in st['runs']:
            if r['label'].startswith('step') and 'residual' in r:
                out.append(float(np.linalg.norm(np.array(r['residual']) / sig)))
                gates.append(r.get('gate') or '')
        return out, gates

    series = [
        ('sampling knobs, ungated', 'ie_pilot', SERIES[0], True),
        ('directions, ungated', 'ie_directions', SERIES[1], True),
        ('directions, gated', 'ie_directions_gated3', SERIES[2], False),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for label, sf, c, at_bound in series:
        ys, _ = residuals(sf)
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, color=c, lw=1.5, zorder=2)
        ax.scatter(xs[:-1], ys[:-1], s=30, color=c, edgecolor='white', linewidth=1.0, zorder=3)
        ax.scatter(xs[-1:], ys[-1:], s=34, facecolor='white' if at_bound else c, edgecolor=c, linewidth=1.5, zorder=4)
        ax.annotate(label, (xs[-1], ys[-1]), textcoords='offset points', xytext=(8, -3), color=INK, fontsize=7.5, va='center')
    ax.set_xlabel('controller step')
    ax.set_ylabel(r'whitened residual $\|W^{1/2}(b-y_n)\|$')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.7, 6.2)
    ax.set_ylim(0, 30)
    fig.tight_layout()
    fig.savefig(OUT / 'trajectories.pdf')
    plt.close(fig)


def difficulty_ladder() -> None:
    """The base agent's resolution rate on each task set, against the leaderboard score."""
    def rate(run_name: str, n: int | None = None) -> float | None:
        y = run(run_name, n)
        return None if y is None else y['resolution_rate']

    def kaggle_31b() -> float | None:
        f = ROOT / 'runs' / 'kaggle_calibration_v2' / 'calibration.json'
        if not f.exists():
            return None
        c = json.loads(f.read_text())
        return c['models']['31b']['resolved'] / len(c['task_ids'])

    rows = [
        ('easy held-out\n(95, commit msg)', [rate('heldout_base_c8', 95)]),
        ('public\n(100, PR text)', [rate('base_sound100_c8', 100), rate('base_sound100_c8_r2', 100)]),
        ('public on Kaggle\nhardware (30)', [kaggle_31b()]),
        ('hard held-out\n(76, commit msg)', [rate('heldout_hard_base_c8', 76)]),
        ('hard held-out\n(76, title +\nhalf body)', [rate('proxy_f050_base_c8', 76)]),
        ('hard held-out\n(76, title +\nquarter body)', [rate('proxy_f025_base_c8', 76)]),
        ('public\n(100, title only)', [rate('public_terse_base_c8', 100)]),
        ('hard held-out\n(76, title only)', [rate('heldout_hard_terse_base_c8', 76), rate('proxy_f000_base_c8', 76)]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, (label, vals) in enumerate(rows):
        vals = [v for v in vals if v is not None]
        xs = i + (np.linspace(-0.1, 0.1, len(vals)) if len(vals) > 1 else np.zeros(1))
        ax.scatter(xs, vals, s=40, color=SERIES[0], edgecolor='white', linewidth=1.0, zorder=3)
        if vals:
            m = float(np.mean(vals))
            ax.annotate(f'{m * 100:.0f}%', (i, m), textcoords='offset points', xytext=(12, -3), color=INK, fontsize=7.5)
    ax.axhline(0.13, color=SERIES[1], lw=1.5, zorder=2)
    ax.annotate('leaderboard score on the hidden tasks: 13%', (1.5, 0.13), textcoords='offset points', xytext=(0, -11),
                ha='center', color=INK, fontsize=7.5)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(-0.5, len(rows) - 0.4)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel('resolved')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'difficulty_ladder.pdf')
    plt.close(fig)


def ie_moment() -> None:
    """iE calibration of the statement direction: M(w) at each weight drawn, against alpha."""
    f = ROOT / 'runs' / 'ie_distill' / 'state.json'
    if not f.exists():
        return
    st = json.loads(f.read_text())
    by_w: dict[float, list[dict]] = {}
    for r in st['runs']:
        by_w.setdefault(round(r['w'], 3), []).append(r)
    ws = sorted(by_w)
    m = [float(np.mean([r['y'] for r in by_w[w]])) for w in ws]
    se = [float(np.sqrt(np.mean([r['se'] ** 2 for r in by_w[w]]) / len(by_w[w]))) for w in ws]
    bad = [float(np.mean([r['malformed'] for r in by_w[w]])) for w in ws]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    a, a_se = st['alpha'], st['alpha_se']
    ax.axhspan(a - 1.96 * a_se, a + 1.96 * a_se, color=SERIES[1], alpha=0.12, lw=0)
    ax.axhline(a, color=SERIES[1], lw=1.5)
    ax.annotate(f'\u03b1: teacher agrees with itself ({a:.2f})', (ws[0], a), textcoords='offset points', xytext=(2, 5),
                color=INK, fontsize=7.5)
    ax.plot(ws, m, color=SERIES[0], lw=2, zorder=2)
    ax.errorbar(ws, m, yerr=[1.96 * v for v in se], fmt='o', ms=6, color=SERIES[0], mec='white', mew=1.0,
                elinewidth=1.2, capsize=0, zorder=3)
    for w, y, b in zip(ws, m, bad):
        low = y < 0.1  # a point on the axis: one-line label to its right, below the rising line
        note = f'{y:.2f}' + ((', ' if low else '\n') + f'{b * 100:.0f}% malformed' if b > 0.1 else '')
        ax.annotate(note, (w, y), textcoords='offset points', xytext=(24, 0) if low else (6, -12 if b > 0.1 else -13),
                    va='bottom' if low else 'baseline', color=INK, fontsize=7)
    ax.set_xticks(ws, [f'{w:g}' for w in ws])
    ax.set_xlabel('weight of the statement direction (0 = base agent, 1 = trained)')
    ax.set_ylabel('agreement with teacher')
    ax.set_ylim(0, 0.9)
    ax.set_xlim(ws[0] - 0.3, ws[-1] + 0.5)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'ie_moment.pdf')
    plt.close(fig)


if __name__ == '__main__':
    thinking_tradeoff()
    main_100()
    trajectories()
    difficulty_ladder()
    ie_moment()
    print('figures written to', OUT)
