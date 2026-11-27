#!/usr/bin/env python
"""iE de-biasing of the selected configuration's resolution rate (IMPLEMENTATION.md 6.38).

Reporting the best of K configurations evaluated on the same tasks inflates the winner's observed rate
(the winner's curse). iE (dissertation, analytic iteration x_{n+1} = x_n + (alpha - M(x_n)), section 3.3)
removes the bias: the parameter x is the vector of K true rates ordered by observed rank, the statistic
alpha is the vector of observed pooled rates sorted ascending, and M(x) is the expectation of that sorted
vector when outcomes are simulated at x. The simulator is a two-way logistic model,
logit P(resolved) = a_i + b_j (task difficulty + configuration effect) with a_i ~ N(0, tau^2); (b, tau) are
fitted by maximum marginal likelihood, the task effects integrated out by Gauss-Hermite quadrature (a MAP
fit with one to thirteen outcomes per task makes every task deterministic, the incidental-parameter
problem). The simulator keeps the fixed task set: each replicate draws every task's difficulty from its
posterior given the data, then outcomes for the actual design (the same number of draws per configuration),
with common random numbers across iterations so M is smooth in x. The iterate is kept sorted by pooling
adjacent violators, so with tied observed rates the iteration stops at the boundary with a residual.

Usage: .venv/bin/python scripts/ie_debias.py [--family NAME ...] [--reps 4000] [--seed 0]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'

FAMILIES = {
    'public100': {   # 100 sound public tasks, calibrated cap, eight concurrent (6.25)
        'base': ['base_sound100_c8', 'base_sound100_c8_r2'],
        'format direction': ['format1_sound100_c8', 'format1_sound100_c8_r2'],
        'gated end point': ['gated3_end_sound100_c8'],
    },
    'public100_pair': {   # the paper's headline comparison alone
        'base': ['base_sound100_c8', 'base_sound100_c8_r2'],
        'format direction': ['format1_sound100_c8', 'format1_sound100_c8_r2'],
    },
    'hard76_title': {   # the leaderboard-matched proxy: 76 hard held-out tasks, title-only statements
        'base': ['heldout_hard_terse_base_c8'],
        'tools': ['heldout_hard_terse_skills_c8'],
        'thinking off + format': ['heldout_hard_terse_v2_c8'],
        'tools + guidance': ['heldout_hard_terse_v3_c8'],
        'statement-distilled': ['heldout_hard_terse_v4_c8'],
        'trajectory imitation': ['heldout_hard_terse_v6_c8'],
    },
    'dev24': {   # dev24 at the calibrated cap, eight concurrent
        'base': ['noise_v0long_dev24_c8_r1', 'noise_v0long_dev24_c8_r2'],
        'format direction': ['v0long_format1_dev24_c8'],
        'knob controller end point': ['pilot_end_dev24_c8_r1', 'pilot_end_dev24_c8_r2'],
        'directions end point (ungated)': ['directions_end_dev24_c8_r1', 'directions_end_dev24_c8_r2'],
        'gated end point': ['gated3_end_dev24_c8_r1', 'gated3_end_dev24_c8_r2'],
        'decisiveness A': ['decisive_A_dev24_c8_r1', 'decisive_A_dev24_c8_r2'],
        'decisiveness B': ['decisive_B_dev24_c8_r1', 'decisive_B_dev24_c8_r2'],
    },
}


def load_family(fam: dict[str, list[str]]):
    """Returns task ids, config names, and Y of shape (n_tasks, n_draws) with the draw->config map."""
    per_run = {}
    for cfg, runs in fam.items():
        for r in runs:
            d = {}
            for line in (RUNS / r / 'task_results.jsonl').read_text().splitlines():
                j = json.loads(line)
                d[j['instance_id']] = int(bool(j.get('resolved')))
            per_run[(cfg, r)] = d
    ids = sorted(set.intersection(*[set(d) for d in per_run.values()]))
    configs = list(fam)
    cols, col_cfg = [], []
    for cfg in configs:
        for r in fam[cfg]:
            cols.append([per_run[(cfg, r)][i] for i in ids]); col_cfg.append(configs.index(cfg))
    Y = np.array(cols, dtype=float).T          # (n, D)
    return ids, configs, Y, np.array(col_cfg)


def sigmoid(z): return 1 / (1 + np.exp(-z))


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)   # probabilists' Hermite: nodes for N(0,1)
GH_W = GH_W / GH_W.sum()


def task_loglik_nodes(Y, J, b, tau):
    """log p(y_i | a = tau*z_q, b) for every task i and node q: (n, Q)."""
    eta = tau * GH_X[None, :, None] + (J @ b)[None, None, :]          # (1, Q, D)
    logp = np.log(sigmoid(eta)); log1p = np.log(sigmoid(-eta))
    return (Y[:, None, :] * logp + (1 - Y[:, None, :]) * log1p).sum(2)   # (n, Q)


def marginal_loglik(Y, J, b, tau):
    L = task_loglik_nodes(Y, J, b, tau)
    m = L.max(1, keepdims=True)
    return float((m[:, 0] + np.log((np.exp(L - m) * GH_W[None, :]).sum(1))).sum())


def fit_marginal(Y, J, K):
    """Maximise the marginal likelihood over b (Newton with numerical derivatives) and tau (grid)."""
    best = None
    for tau in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        b = np.zeros(K)
        f = lambda v: marginal_loglik(Y, J, v, tau)
        for _ in range(60):
            h = 1e-4
            g = np.array([(f(b + h * e) - f(b - h * e)) / (2 * h) for e in np.eye(K)])
            H = np.zeros((K, K))
            for i in range(K):
                for k in range(i, K):
                    ei, ek = np.eye(K)[i], np.eye(K)[k]
                    H[i, k] = H[k, i] = (f(b + h * ei + h * ek) - f(b + h * ei - h * ek) - f(b - h * ei + h * ek) + f(b - h * ei - h * ek)) / (4 * h * h)
            step = np.linalg.solve(H - 1e-6 * np.eye(K), -g)
            step = np.clip(step, -2, 2)
            b = b + step
            if np.abs(step).max() < 1e-6:
                break
        ll = f(b)
        if best is None or ll > best[0]:
            best = (ll, tau, b.copy())
    ll, tau, b = best
    L = task_loglik_nodes(Y, J, b, tau)
    post = np.exp(L - L.max(1, keepdims=True)) * GH_W[None, :]
    post /= post.sum(1, keepdims=True)                                   # (n, Q) posterior over nodes per task
    return b, tau, ll, post


def pava(x):
    """Pool adjacent violators: the closest non-decreasing vector."""
    x = list(map(float, x)); blocks = [[v, 1] for v in x]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]; del blocks[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    return np.array([v for v, c in blocks for _ in range(c)])


def rate_fn(post, tau, b):
    """Expected resolution rate on the fixed task set at configuration effect b, difficulties from their posteriors."""
    return float((post * sigmoid(tau * GH_X[None, :] + b)).sum(1).mean())


def b_of_rates(post, tau, x):
    """Invert rate_fn(b) = x_j by bisection."""
    out = np.zeros(len(x))
    for j, t in enumerate(x):
        lo, hi = -25.0, 25.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if rate_fn(post, tau, mid) < t: lo = mid
            else: hi = mid
        out[j] = (lo + hi) / 2
    return out


def sample_difficulties(post, tau, V):
    """One difficulty per task per replicate from the posterior over nodes. V: (B, n) uniforms."""
    cdf = np.cumsum(post, axis=1)                                      # (n, Q)
    idx = (V[:, :, None] > cdf[None, :, :]).sum(2)                     # (B, n)
    idx = np.minimum(idx, len(GH_X) - 1)
    return tau * GH_X[idx]                                             # (B, n)


def simulate_sorted(A, b_by_cfg, J, U):
    """Sorted pooled rates per replicate. A: (B, n) difficulties; U: (B, n, D) uniforms (common random numbers)."""
    P = sigmoid(A[:, :, None] + (J @ b_by_cfg)[None, None, :])        # (B, n, D)
    Ys = (U < P).astype(np.float32)
    pooled = (Ys.mean(1) @ J) / J.sum(0)                               # (B, K)
    return np.sort(pooled, axis=1), pooled


def run_family(name, fam, reps, seed):
    ids, configs, Y, col_cfg = load_family(fam)
    n, D = Y.shape; K = len(configs)
    J = np.zeros((D, K)); J[np.arange(D), col_cfg] = 1
    b_hat, tau, ll, post = fit_marginal(Y, J, K)
    obs = (Y.mean(0) @ J) / J.sum(0)                                   # observed pooled rate per config
    order = np.argsort(obs, kind='stable')                             # ascending; rank -> config
    alpha = obs[order]
    rng = np.random.default_rng(seed)
    U = rng.random((reps, n, D), dtype=np.float32); V = rng.random((reps, n))
    A = sample_difficulties(post, tau, V)
    x = alpha.copy(); hist = []
    for it in range(100):
        b_rank = b_of_rates(post, tau, x)
        b_cfg = np.zeros(K); b_cfg[order] = b_rank
        S, _ = simulate_sorted(A, b_cfg, J, U)
        M = S.mean(0); resid = alpha - M
        hist.append(float(np.abs(resid).max()))
        if np.abs(resid).max() < 1.5e-3:
            break
        x = pava(np.clip(x + resid, 1e-3, 1 - 1e-3))
    b_rank = b_of_rates(post, tau, x); b_cfg = np.zeros(K); b_cfg[order] = b_rank
    S, pooled = simulate_sorted(A, b_cfg, J, U); M = S.mean(0); se = S.std(0)
    base_j = configs.index('base'); base_rank = int(np.where(order == base_j)[0][0])
    # model check: expected tasks differing between two draws of the base, under the posterior
    P0 = sigmoid(tau * GH_X[None, :] + b_cfg[base_j])
    exp_diff = float((post * 2 * P0 * (1 - P0)).sum())
    cols = np.where(col_cfg == base_j)[0]
    obs_diff = int((Y[:, cols[0]] != Y[:, cols[1]]).sum()) if len(cols) >= 2 else None
    # the winner's curse alone: expected best observed rate if every configuration had the base's de-biased rate
    S_eq, _ = simulate_sorted(A, np.full(K, b_rank[base_rank]), J, U)
    print(f'\n=== {name}: {n} tasks, {K} configurations, {D} draws; tau={tau}, marginal loglik={ll:.1f}; '
          f'{len(hist)} iE iterations, final max residual {np.abs(alpha - M).max():.4f}')
    print(f'{"rank":>4}  {"configuration":32} {"draws":>5} {"observed":>9} {"de-biased":>10} {"shift":>7} {"se":>6}')
    for rank in range(K - 1, -1, -1):
        j = order[rank]
        print(f'{K - rank:>4}  {configs[j]:32} {int(J[:, j].sum()):>5} {alpha[rank]:>9.3f} {x[rank]:>10.3f} {x[rank] - alpha[rank]:>+7.3f} {se[rank]:>6.3f}'
              f'   ({alpha[rank] * n:.1f} -> {x[rank] * n:.1f} of {n})')
    top_j = order[-1]
    sd_diff = float((pooled[:, top_j] - pooled[:, base_j]).std()) if top_j != base_j else 0.0
    print(f'top minus base: observed {alpha[-1] - alpha[base_rank]:+.3f} ({(alpha[-1] - alpha[base_rank]) * n:+.1f} tasks), '
          f'de-biased {x[-1] - x[base_rank]:+.3f} ({(x[-1] - x[base_rank]) * n:+.1f} tasks); sd of that difference under the model {sd_diff:.3f} ({sd_diff * n:.1f} tasks)')
    print(f"winner's curse alone: if all {K} had the base's de-biased rate {x[base_rank]:.3f}, the expected best observed rate would be "
          f"{S_eq.mean(0)[-1]:.3f} ({S_eq.mean(0)[-1] * n:.1f} of {n}), sd {S_eq.std(0)[-1]:.3f}")
    print(f'model check: tasks differing between two draws of the base: expected {exp_diff:.1f}' + (f', observed {obs_diff}' if obs_diff is not None else ''))
    return {'family': name, 'tasks': n, 'tau': tau, 'configs': [configs[j] for j in order], 'observed': alpha.tolist(),
            'debiased': x.tolist(), 'se': se.tolist(), 'iterations': len(hist), 'final_residual': float(np.abs(alpha - M).max()),
            'sd_top_minus_base': sd_diff, 'expected_best_if_equal': float(S_eq.mean(0)[-1]), 'sd_best_if_equal': float(S_eq.std(0)[-1]), 'expected_diff_two_base_draws': exp_diff, 'observed_diff_two_base_draws': obs_diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', nargs='*', default=list(FAMILIES))
    ap.add_argument('--reps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=RUNS / 'ie_debias' / 'results.json')
    args = ap.parse_args()
    results = [run_family(f, FAMILIES[f], args.reps, args.seed) for f in args.family]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
