"""iE calibration of the leaderboard proxy's informativeness knob (IMPLEMENTATION.md 6.40).

Knob x in [0, 1]: the fraction of each task's statement body kept (scripts/heldout/build_frac_tasks.py);
x = 0 is the title-only set, x = 1 the full statement. M(x): the base agent's resolution rate on
the 76 sound hard tasks at that fraction. alpha: the leaderboard score of the submitted agent on
the hidden tasks (0.13). The stationary equation M(x) = alpha asks how much of a statement the
hidden tasks are worth, on this task set.

Analytic iE iteration with a Newton gain from the secant of the two existing end-point draws:
    x_{n+1} = clip(x_n + (alpha - M(x_n)) / G, 0, 1),  G = M(1) - M(0).
A step that lands on a bound where the residual keeps its sign has converged at that bound.

Commands:
  step    read the state, compute the next x from the latest draw; prints the next x or CONVERGED
  record  --x X --run runs/NAME   append a finished draw to the state
  report  table of draws, curve, root and standard errors
State: runs/ie_proxy/state.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'runs' / 'ie_proxy' / 'state.json'
ALPHA = 0.13
N_TASKS = 76


def resolution(run: str) -> tuple[int, int]:
    s = json.loads((ROOT / run / 'summary.json').read_text())
    return int(s['resolved']), int(s['total_tasks'])


def load() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    st = {
        'alpha': ALPHA,
        'draws': [  # prior draws: the end points, measured on 2026-10-18 (6.30)
            {'x': 0.0, 'run': 'runs/heldout_hard_terse_base_c8', 'prior': True},
            {'x': 1.0, 'run': 'runs/heldout_hard_base_c8', 'prior': True},
        ],
        'iterates': [],
    }
    return st


def save(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def rate(st: dict, x: float) -> tuple[float, int, int]:
    """Pooled resolution at x over all draws there."""
    k = n = 0
    for d in st['draws']:
        if abs(d['x'] - x) < 1e-9:
            r, t = resolution(d['run'])
            k += r
            n += t
    return (k / n if n else float('nan')), k, n


def secant_gain(st: dict) -> float:
    m0, _, _ = rate(st, 0.0)
    m1, _, _ = rate(st, 1.0)
    return m1 - m0


def cmd_step(st: dict) -> int:
    alpha = st['alpha']
    g = secant_gain(st)
    it = st['iterates']
    if not it:
        # first iterate: the midpoint of the knob's range
        x = 0.5
        it.append({'n': 0, 'x': x, 'status': 'pending'})
        save(st)
        print(f'NEXT {x:.2f}')
        return 0
    last = it[-1]
    xn = last['x']
    m, k, n = rate(st, xn)
    if math.isnan(m):
        print(f'no draw recorded at x={xn:.2f}', file=sys.stderr)
        return 1
    last.update(M=m, k=k, n=n, status='drawn')
    resid = alpha - m
    raw = xn + resid / g
    x_next = min(1.0, max(0.0, raw))
    x_next = round(x_next, 2)
    conv = abs(x_next - xn) < 1e-9 and (x_next in (0.0, 1.0))
    last.update(residual=resid, raw_next=raw)
    if conv:
        last['status'] = 'converged_at_bound'
        save(st)
        print(f'CONVERGED x={xn:.2f} M={m:.4f} residual={resid:+.4f} raw_next={raw:+.3f}')
        return 0
    it.append({'n': last['n'] + 1, 'x': x_next, 'status': 'pending'})
    save(st)
    print(f'NEXT {x_next:.2f}   (from x={xn:.2f} M={m:.4f} residual={resid:+.4f} gain=1/{g:.4f} raw={raw:+.3f})')
    return 0


def cmd_record(st: dict, x: float, run: str) -> int:
    if not (ROOT / run / 'summary.json').exists():
        print(f'no summary in {run}', file=sys.stderr)
        return 1
    st['draws'].append({'x': x, 'run': run})
    save(st)
    r, t = resolution(run)
    print(f'recorded x={x:.2f} {run}: {r}/{t} = {r / t:.4f}')
    return 0


def cmd_report(st: dict) -> int:
    alpha = st['alpha']
    xs = sorted({d['x'] for d in st['draws']})
    print(f'alpha (leaderboard) = {alpha:.3f}; tasks = {N_TASKS}')
    print(f'{"x":>5} {"draws":>5} {"resolved":>9} {"M(x)":>7} {"se_bin":>7}')
    rows = []
    for x in xs:
        m, k, n = rate(st, x)
        nd = sum(1 for d in st['draws'] if abs(d['x'] - x) < 1e-9)
        se = math.sqrt(m * (1 - m) / n) if n else float('nan')
        rows.append((x, nd, k, n, m, se))
        print(f'{x:5.2f} {nd:5d} {k:4d}/{n:<4d} {m:7.4f} {se:7.4f}')
    # replicate noise at x = 0 (draw-to-draw sd of the rate), if two or more draws
    reps = [resolution(d['run'])[0] / resolution(d['run'])[1] for d in st['draws'] if abs(d['x']) < 1e-9]
    if len(reps) >= 2:
        mean = sum(reps) / len(reps)
        sd = math.sqrt(sum((r - mean) ** 2 for r in reps) / (len(reps) - 1))
        print(f'title-only replicates: {", ".join(f"{r:.4f}" for r in reps)}  sd {sd:.4f} (binomial {math.sqrt(mean * (1 - mean) / N_TASKS):.4f})')
    # root by linear interpolation on the pooled curve
    pts = [(x, m) for x, _, _, _, m, _ in rows]
    root = None
    for (x0, m0), (x1, m1) in zip(pts, pts[1:]):
        if (m0 - alpha) * (m1 - alpha) <= 0 and m1 != m0:
            root = x0 + (alpha - m0) * (x1 - x0) / (m1 - m0)
            break
    m0, k0, n0 = rate(st, 0.0)
    se0 = math.sqrt(m0 * (1 - m0) / n0)
    if root is None:
        print(f'no root in [0, 1]: M(0) - alpha = {m0 - alpha:+.4f} ({(m0 - alpha) / se0:+.2f} binomial se of M(0) on {n0} task-draws); '
              f'the root lies at or below x = 0')
    else:
        print(f'root x* = {root:.3f}')
    # local slopes of the curve
    for (x0, m0_), (x1, m1_) in zip(pts, pts[1:]):
        print(f'slope [{x0:.2f},{x1:.2f}]: {(m1_ - m0_) / (x1 - x0):+.3f} per unit fraction  ({(m1_ - m0_) * N_TASKS:+.1f} tasks)')
    print('iterates:', json.dumps(st['iterates']))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['step', 'record', 'report'])
    ap.add_argument('--x', type=float)
    ap.add_argument('--run')
    a = ap.parse_args()
    st = load()
    if a.cmd == 'step':
        return cmd_step(st)
    if a.cmd == 'record':
        return cmd_record(st, a.x, a.run)
    return cmd_report(st)


if __name__ == '__main__':
    raise SystemExit(main())
