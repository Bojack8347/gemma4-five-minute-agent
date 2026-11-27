"""Mine candidate held-out tasks from repositories the model is unlikely to know well.

For each repo: blobless clone, require a flat layout (the harness's local sandbox puts only the
workspace on PYTHONPATH), then walk first-parent commits on the default branch since --since and
keep those that change package source (.py inside a top-level package) and test files, within size
limits, with a usable message. Writes one JSON line per candidate to data/heldout/candidates.jsonl.

Usage: .venv/bin/python scripts/heldout/mine_commits.py owner/repo [...] --since 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOS = ROOT / 'data' / 'heldout' / 'repos'
TEST_RE = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$')
SKIP_SUBJECT = re.compile(r'^(release|bump|prepare|version|\[pre-commit|chore|ci[:(]|build[:(]|docs?[:(]|style[:(]|merge |revert )', re.I)
SKIP_ANYWHERE = re.compile(r'dependabot|pre-commit-ci|update (the )?changelog', re.I)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout


def clone(slug: str) -> Path:
    dest = REPOS / slug.replace('/', '__')
    if not dest.exists():
        subprocess.run(['git', 'clone', '--quiet', '--filter=blob:none', '--no-checkout',
                        f'https://github.com/{slug}.git', str(dest)], check=True)
    return dest


def packages(repo: Path) -> list[str]:
    files = git(repo, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines()
    if any(f.startswith('src/') for f in files):
        return []
    return sorted({f.split('/')[0] for f in files if re.match(r'^[A-Za-z_][\w]*/__init__\.py$', f)}
                  - {'tests', 'test', 'docs', 'examples', 'scripts', 'benchmarks', 'tools'})


def mine(slug: str, since: str, max_src_lines: int, max_src_files: int) -> list[dict]:
    repo = clone(slug)
    pkgs = packages(repo)
    if not pkgs:
        print(f'{slug}: not a flat layout, skipped')
        return []
    out = []
    commits = git(repo, 'log', '--first-parent', f'--since={since}', '--format=%H%x09%P%x09%cI', 'HEAD').splitlines()
    for line in commits:
        sha, parents, date = line.split('\t')
        parents = parents.split()
        if len(parents) != 1:
            continue
        msg = git(repo, 'log', '-1', '--format=%B', sha).strip()
        subject = msg.splitlines()[0] if msg else ''
        if SKIP_SUBJECT.search(subject) or SKIP_ANYWHERE.search(msg):
            continue
        stats = [l.split('\t') for l in git(repo, 'diff', '--numstat', parents[0], sha).splitlines() if l.strip()]
        src, tests, other = [], [], []
        for added, deleted, path in stats:
            if added == '-':          # binary
                other.append(path)
                continue
            n = int(added) + int(deleted)
            if TEST_RE.search(path) and path.endswith('.py'):
                tests.append((path, n))
            elif path.endswith('.py') and path.split('/')[0] in pkgs:
                src.append((path, n))
            else:
                other.append(path)
        src_lines = sum(n for _, n in src)
        if not src or not tests or src_lines > max_src_lines or len(src) > max_src_files:
            continue
        # the test change must add or modify test functions, not only reformat
        if sum(n for _, n in tests) < 3:
            continue
        trailers = re.compile(r'^(co-authored-by|signed-off-by|reviewed-by):', re.I)
        statement = '\n'.join(l for l in msg.splitlines() if not trailers.match(l.strip())).strip()
        if len(subject) < 12:
            continue
        out.append({'repo': slug, 'sha': sha, 'base_commit': parents[0], 'created_at': date,
                    'subject': subject, 'problem_statement': statement,
                    'src_files': [p for p, _ in src], 'src_lines': src_lines,
                    'test_files': [p for p, _ in tests], 'other_files': other, 'packages': pkgs})
    print(f'{slug}: {len(commits)} commits since {since}, {len(out)} candidates, packages {pkgs}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('repos', nargs='+')
    ap.add_argument('--since', default='2026-01-01')
    ap.add_argument('--max-src-lines', type=int, default=200)
    ap.add_argument('--max-src-files', type=int, default=4)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'heldout' / 'candidates.jsonl')
    args = ap.parse_args()
    REPOS.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.out.exists():
        seen = {json.loads(l)['sha'] for l in args.out.read_text().splitlines() if l.strip()}
    with args.out.open('a') as f:
        for slug in args.repos:
            try:
                for c in mine(slug, args.since, args.max_src_lines, args.max_src_files):
                    if c['sha'] not in seen:
                        f.write(json.dumps(c) + '\n')
            except subprocess.CalledProcessError as e:
                print(f'{slug}: git failed: {(e.stderr or "")[:200]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
