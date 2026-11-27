"""Public-leaderboard structure from a downloaded snapshot (IMPLEMENTATION.md 6.42).

    kaggle competitions leaderboard gemma-4-developer-agent --download -p data/kaggle_lb_YYYYMMDD
    unzip data/kaggle_lb_YYYYMMDD/*.zip -d data/kaggle_lb_YYYYMMDD/csv
    python -I scripts/leaderboard_stats.py data/kaggle_lb_YYYYMMDD/csv

Prints the task count the displayed scores imply, each score in tasks, and how a team's displayed (best)
score depends on its number of submissions. The snapshot holds user names and is not released.
"""
import collections
import csv
import glob
import math
import statistics as st
import sys


def trunc2(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def round2(x: float) -> float:
    return math.floor(x * 100 + 0.5 + 1e-9) / 100


def consistent_counts(observed: list[float], f, lo: int = 20, hi: int = 300) -> list[int]:
    """Task counts n for which every observed score is f(k/n) and no achievable value below a quarter is unoccupied
    (a board with thousands of teams fills every value in its bulk)."""
    out = []
    for n in range(lo, hi + 1):
        grid = {f(k / n): k for k in range(n + 1)}
        if not all(v in grid for v in observed):
            continue
        top = max(grid[v] for v in observed if grid[v] <= 0.25 * n)
        if all(f(k / n) in observed for k in range(top + 1)):
            out.append(n)
    return out


def main() -> None:
    path = glob.glob(sys.argv[1].rstrip('/') + '/*.csv')[0]
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    scores = [float(r['Score']) for r in rows]
    observed = sorted(set(scores))
    print(f'{len(rows)} teams, {len(observed)} distinct scores: {observed}')
    print('task counts consistent with truncation:', consistent_counts(observed, trunc2))
    print('task counts consistent with rounding:  ', consistent_counts(observed, round2))
    n = consistent_counts(observed, trunc2)[0]
    tasks = {s: next(k for k in range(n + 1) if trunc2(k / n) == s) for s in observed}
    k = [tasks[s] for s in scores]
    m = [int(r['SubmissionCount']) for r in rows]
    print(f'n = {n}; score -> tasks: {tasks}')
    hist = collections.Counter(k)
    print('teams per task count:', {i: hist[i] for i in sorted(hist)})
    print(f'all teams: median {st.median(k)} tasks, mean {st.mean(k):.2f}; median submissions {st.median(m)}')
    print('submissions  teams  mean  median  share >= 13 tasks')
    for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 40), (41, 10 ** 6)]:
        g = [x for x, mm in zip(k, m) if lo <= mm <= hi]
        if g:
            print(f'{lo:4d}-{hi if hi < 10 ** 6 else "+":<5} {len(g):6d} {st.mean(g):6.2f} {st.median(g):6.1f} {sum(x >= 13 for x in g) / len(g):8.3f}')
    top = [mm for x, mm in zip(k, m) if x >= 13]
    print(f'teams at 13 tasks or more: {len(top)}, median submissions {st.median(top)}')
    print('top six (tasks, submissions):', sorted(zip(k, m), reverse=True)[:6])


if __name__ == '__main__':
    main()
