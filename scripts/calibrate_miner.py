#!/usr/bin/env python
"""Calibrate the disagreement-based miner's threshold by the iE stationary equation (IMPLEMENTATION.md 6.39).

A disagreement-based miner (mine_statement_pairs.py) emits a pair where a student sample and a teacher
sample are "materially different". Sampling noise makes two samples from the same side differ too, so a
fraction of every pair set is noise. The criterion has one continuous knob, the Jaccard cutoff t on the
calls' targets (differ if |A&B|/|A|B| < t), plus two rules (edit-vs-not always differs; when a side has no
targets, a different tool differs). The stationary equation M(t) = alpha sets t so that the same-side
disagreement rate M(t) equals a chosen false-pair rate alpha; M is observed directly here (per-state means
of same-side pairs from four samples per side at 168 states from unseen repositories), so the equation is
solved by monotone root-finding rather than by iteration. A split-half check (t chosen on odd states,
evaluated on even states) measures how much choosing t on the same data flatters the result.

At the solution the cross-side rate r(t) and the same-side rate q(t) give the fraction of states at which
the statement changes the next call, pi = (r - q) / (1 - q) under the mixture "either the statement matters
at this state (then the sides disagree) or it does not (then cross-side disagreement is same-side noise)".

Usage: .venv/bin/python scripts/calibrate_miner.py --student runs/ie_distill/draw_calib_base_s4.jsonl
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mine_statement_pairs import targets  # noqa: E402


def different(a: dict, b: dict, t: float, edit_rule: bool = True) -> bool:
    """materially_different with the Jaccard cutoff t as a parameter."""
    if a.get('status') != 'ok' or b.get('status') != 'ok':
        return True
    if edit_rule and (a.get('tool') == 'edit_file') != (b.get('tool') == 'edit_file'):
        return True
    ta, tb = targets(a), targets(b)
    if not ta or not tb:
        return a.get('tool') != b.get('tool')
    return len(ta & tb) / len(ta | tb) < t


def rates(states: list[dict], t: float, edit_rule: bool = True) -> dict:
    """Per-state means, then averages over states: same-side student, same-side teacher, cross-side."""
    qs, qt, r = [], [], []
    for s in states:
        S, T = s['student'], s['teacher']
        if len(S) >= 2:
            qs.append(np.mean([different(a, b, t, edit_rule) for a, b in itertools.combinations(S, 2)]))
        if len(T) >= 2:
            qt.append(np.mean([different(a, b, t, edit_rule) for a, b in itertools.combinations(T, 2)]))
        if S and T:
            r.append(np.mean([different(a, b, t, edit_rule) for a in S for b in T]))
    out = {'q_student': float(np.mean(qs)) if qs else float('nan'), 'q_teacher': float(np.mean(qt)) if qt else float('nan'),
           'r': float(np.mean(r)) if r else float('nan')}
    q = np.nanmean([out['q_student'], out['q_teacher']])
    out['q'] = float(q)
    out['pi'] = float((out['r'] - q) / (1 - q)) if r else float('nan')      # states where the statement changes the call
    out['signal_fraction'] = float(1 - q / out['r']) if r and out['r'] > 0 else float('nan')   # of emitted pairs
    return out


def consensus(states: list[dict], t: float = 0.5) -> dict:
    """Consensus mining: a side's decision at a state is defined when two of its samples agree (not different);
    a pair is emitted when the two sides' consensus decisions differ. Same-side false rate from two disjoint
    halves of the same side; cross rate from a student half against a teacher half; averaged over the three
    ways of splitting four samples into halves."""
    splits = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    q_t, q_s, r, defined_t, defined_s, defined_both = [], [], [], [], [], []
    for st in states:
        S, T = st['student'], st['teacher']
        if len(S) < 4 or len(T) < 4:
            continue
        for h1, h2 in splits:
            ct1 = not different(T[h1[0]], T[h1[1]], t); ct2 = not different(T[h2[0]], T[h2[1]], t)
            cs1 = not different(S[h1[0]], S[h1[1]], t); cs2 = not different(S[h2[0]], S[h2[1]], t)
            defined_t.append(ct1); defined_s.append(cs1); defined_both.append(cs1 and ct1)
            if ct1 and ct2: q_t.append(different(T[h1[0]], T[h2[0]], t))
            if cs1 and cs2: q_s.append(different(S[h1[0]], S[h2[0]], t))
            if cs1 and ct1: r.append(different(S[h1[0]], T[h1[0]], t))
    q = float(np.mean(q_t + q_s)); rr = float(np.mean(r))
    return {'q_teacher': float(np.mean(q_t)), 'q_student': float(np.mean(q_s)), 'q': q, 'r': rr,
            'pi': float((rr - q) / (1 - q)), 'signal_fraction': float(1 - q / rr),
            'defined_teacher': float(np.mean(defined_t)), 'defined_student': float(np.mean(defined_s)),
            'defined_both': float(np.mean(defined_both)), 'pairs_per_100_states': float(100 * np.mean(defined_both) * rr)}


def solve(states, alpha: float, grid, edit_rule=True) -> float | None:
    """Smallest t on the grid with q(t) >= alpha (q is non-decreasing in t); None if never reached."""
    for t in grid:
        if rates(states, t, edit_rule)['q'] >= alpha:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', type=Path, default=ROOT / 'data' / 'ie_distill' / 'teacher.jsonl')
    ap.add_argument('--student', type=Path, default=ROOT / 'runs' / 'ie_distill' / 'draw_calib_base_s4.jsonl')
    ap.add_argument('--alpha', type=float, nargs='+', default=[0.05, 0.10, 0.15])
    ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'ie_distill' / 'miner_calibration.json')
    args = ap.parse_args()
    teach = {(r['instance_id'], r['step_id']): r['labels'] for r in map(json.loads, args.teacher.read_text().splitlines())}
    stud = {}
    if args.student.exists():
        stud = {(r['instance_id'], r['step_id']): r['labels'] for r in map(json.loads, args.student.read_text().splitlines())}
    keys = sorted(teach)
    states = [{'key': k, 'student': stud.get(k, []), 'teacher': teach[k]} for k in keys]
    n_s = sum(1 for s in states if s['student'])
    print(f'{len(states)} states; teacher samples per state {len(states[0]["teacher"])}; student samples at {n_s} states')
    grid = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
    print(f'{"t":>5} {"q_student":>10} {"q_teacher":>10} {"q":>7} {"r":>7} {"pi":>7} {"signal":>7}')
    curve = []
    for t in grid:
        o = rates(states, t); curve.append({'t': t, **o})
        print(f'{t:>5.2f} {o["q_student"]:>10.3f} {o["q_teacher"]:>10.3f} {o["q"]:>7.3f} {o["r"]:>7.3f} {o["pi"]:>7.3f} {o["signal_fraction"]:>7.3f}')
    o = rates(states, 0.5)
    print(f'current criterion (t = 0.5): q_student {o["q_student"]:.3f}, q_teacher {o["q_teacher"]:.3f}, r {o["r"]:.3f}, '
          f'pi {o["pi"]:.3f}, signal fraction of pairs {o["signal_fraction"]:.3f}')
    sols = []
    odd = [s for i, s in enumerate(states) if i % 2]; even = [s for i, s in enumerate(states) if not i % 2]
    for alpha in args.alpha:
        t = solve(states, alpha, grid)
        rec = {'alpha': alpha, 't': t}
        if t is not None:
            o = rates(states, t); rec.update(o)
            t_odd = solve(odd, alpha, grid)
            if t_odd is not None:
                e = rates(even, t_odd); rec.update({'t_odd': t_odd, 'q_even_at_t_odd': e['q'], 'r_even_at_t_odd': e['r'], 'pi_even_at_t_odd': e['pi']})
            print(f'alpha = {alpha:.2f}: t* = {t:.2f}; q {o["q"]:.3f}, r {o["r"]:.3f}, pi {o["pi"]:.3f}, signal fraction {o["signal_fraction"]:.3f}, '
                  f'pairs per 100 states {100 * o["r"]:.0f}' + (f'; split-half: t chosen on odd states {t_odd:.2f}, on even states q {e["q"]:.3f} r {e["r"]:.3f} pi {e["pi"]:.3f}' if t_odd is not None else ''))
        else:
            print(f'alpha = {alpha:.2f}: not reached on the grid')
        sols.append(rec)
    o0 = rates(states, 0.5, edit_rule=False)
    print(f'without the edit rule at t = 0.5: q {o0["q"]:.3f}, r {o0["r"]:.3f}, pi {o0["pi"]:.3f}')
    cons = {}
    if n_s:
        for t in (0.05, 0.5):
            c = consensus(states, t); cons[t] = c
            print(f'consensus mining (two samples per side agree), t = {t:.2f}: same-side false rate {c["q"]:.3f} (student {c["q_student"]:.3f}, teacher {c["q_teacher"]:.3f}), '
                  f'cross {c["r"]:.3f}, pi {c["pi"]:.3f}, signal fraction {c["signal_fraction"]:.3f}; consensus defined on {100 * c["defined_both"]:.0f}% of states, '
                  f'pairs per 100 states {c["pairs_per_100_states"]:.0f}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'curve': curve, 'solutions': sols, 'consensus': {str(k): v for k, v in cons.items()}, 'n_states': len(states), 'n_student_states': n_s}, indent=1))


if __name__ == '__main__':
    main()
