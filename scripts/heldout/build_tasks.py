"""Turn mined candidates into harness tasks: tasks.jsonl plus one snapshot per task.

Each task mirrors the public format: the gold `patch` is the commit's package-source diff, the
`test_patch` its test diff, the problem statement the commit message (squash merges carry the
PR title and body). The snapshot is the repository tree at the parent commit as a single-commit
git repository (public snapshots carry full history; ours do not, so `git log` shows nothing).

Usage: .venv/bin/python scripts/heldout/build_tasks.py --per-repo 10 [--repos tenacity,...]
Writes data/heldout/{tasks.jsonl,snapshots/,graphs/,embeddings/}.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELD = ROOT / 'data' / 'heldout'
HELD_OUT = HELD


def git(repo: Path, *args: str, **kw) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True, **kw).stdout


def select(cands: list[dict], per_repo: int, repos: set[str] | None, hard: bool = False) -> list[dict]:
    by_repo = defaultdict(list)
    for c in cands:
        short = c['repo'].split('/')[1]
        if repos and short not in repos:
            continue
        if len(c['problem_statement']) < 30:
            continue
        if hard:
            # larger fixes: at least 40 changed source lines or at least two source files
            if not (c['src_lines'] >= 40 or len(c['src_files']) >= 2) or c['src_lines'] > 600:
                continue
        elif not (3 <= c['src_lines'] <= 150):
            continue
        by_repo[c['repo']].append(c)
    out = []
    for repo, cs in by_repo.items():
        cs.sort(key=lambda c: c['created_at'], reverse=True)
        step = max(1, len(cs) // per_repo)
        out.extend(cs[::step][:per_repo])         # spread over the year, newest first
    return out


def instance_id(c: dict) -> str:
    short = re.sub(r'[^a-z0-9]+', '', c['repo'].split('/')[1].lower())
    m = re.search(r'\(#(\d+)\)', c['subject'])
    return f'{short}_{m.group(1)}' if m else f'{short}_{c["sha"][:8]}'


def snapshot(repo: Path, base: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory(dir=HELD) as tmp:
        tree = Path(tmp) / 'tree'
        tree.mkdir()
        archive = subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', base], capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(tree)], input=archive, check=True)
        env = {'GIT_AUTHOR_NAME': 'snapshot', 'GIT_AUTHOR_EMAIL': 'snapshot@local', 'GIT_COMMITTER_NAME': 'snapshot',
               'GIT_COMMITTER_EMAIL': 'snapshot@local', 'GIT_AUTHOR_DATE': '2000-01-01T00:00:00Z',
               'GIT_COMMITTER_DATE': '2000-01-01T00:00:00Z', 'PATH': '/usr/bin:/bin'}
        for cmd in (['git', 'init', '-q', '-b', 'main'], ['git', 'add', '-A'], ['git', 'commit', '-q', '-m', 'snapshot']):
            subprocess.run(cmd, cwd=tree, env=env, check=True, capture_output=True)
        subprocess.run(['tar', '-czf', str(dest), '-C', str(tree), '.'], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-repo', type=int, default=10)
    ap.add_argument('--repos', default=None, help='comma-separated short repo names')
    ap.add_argument('--candidates', type=Path, default=HELD / 'candidates.jsonl')
    ap.add_argument('--out-dir', type=Path, default=HELD)
    ap.add_argument('--hard', action='store_true', help='select larger fixes (>=40 source lines or >=2 files)')
    ap.add_argument('--terse', action='store_true', help='problem statement = PR title only, PR number stripped')
    ap.add_argument('--snapshots-from', type=Path, default=None, help='reuse snapshot files from this data dir')
    args = ap.parse_args()
    global HELD_OUT
    HELD_OUT = args.out_dir
    cands = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]
    chosen = select(cands, args.per_repo, set(args.repos.split(',')) if args.repos else None, hard=args.hard)
    for d in ('snapshots', 'graphs', 'embeddings'):
        (HELD_OUT / d).mkdir(parents=True, exist_ok=True)
    tasks, ids = [], set()
    for c in chosen:
        repo = HELD / 'repos' / c['repo'].replace('/', '__')
        iid = instance_id(c)
        if iid in ids:
            continue
        ids.add(iid)
        patch = git(repo, 'diff', c['base_commit'], c['sha'], '--', *c['src_files'])
        test_patch = git(repo, 'diff', c['base_commit'], c['sha'], '--', *c['test_files'])
        if not patch.strip() or not test_patch.strip():
            continue
        snap = HELD_OUT / 'snapshots' / f'{iid}.tgz'
        if not snap.exists():
            src = (args.snapshots_from / 'snapshots' / f'{iid}.tgz') if args.snapshots_from else None
            if src is not None and src.exists():
                shutil.copy(src, snap)
            else:
                snapshot(repo, c['base_commit'], snap)
        statement = c['problem_statement']
        if args.terse:
            statement = re.sub(r'\s*\(#\d+\)', '', c['subject']).strip()
        tasks.append({'instance_id': iid, 'repo': c['repo'], 'base_commit': c['base_commit'], 'patch': patch,
                      'test_patch': test_patch, 'problem_statement': statement, 'hints_text': '',
                      'created_at': c['created_at']})
        print(f'{iid:28} {c["created_at"][:10]} src {c["src_lines"]:4d} lines  {c["subject"][:70]}')
    (HELD_OUT / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    print(f'{len(tasks)} tasks written to {HELD_OUT / "tasks.jsonl"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
