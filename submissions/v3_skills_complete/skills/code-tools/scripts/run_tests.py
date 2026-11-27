"""Run the tests related to the current edits and print a compact result.

Usage: run_skill_script(skill_name="code-tools", file_path="scripts/run_tests.py")
       run_skill_script(..., args=["tests/test_x.py", "tests/test_y.py::test_name"])  # explicit targets
Without args, finds the source files changed since the task started (git), picks up to 5 test files
that import or are named after them, runs pytest -x with a time limit, and prints the summary plus
the first failure only (at most ~45 lines).
"""

import os
import re
import subprocess
import sys

TIME_LIMIT = 100


def workspace() -> str:
    for cand in ['/workspace', *os.environ.get('PYTHONPATH', '').split(os.pathsep)]:
        if cand and os.path.isdir(os.path.join(cand, '.git')):
            return cand
    return '/workspace'


def git(root: str, *args: str) -> str:
    try:
        return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ''


def is_test(rel: str) -> bool:
    base = os.path.basename(rel)
    return base.startswith('test_') or base.endswith('_test.py') or '/tests/' in '/' + rel or '/test/' in '/' + rel


def changed_sources(root: str) -> list[str]:
    files = set(git(root, 'diff', '--name-only', 'HEAD').split())
    files |= set(git(root, 'ls-files', '--others', '--exclude-standard').split())
    return sorted(f for f in files if f.endswith('.py') and not is_test(f))


def related_tests(root: str, sources: list[str]) -> list[str]:
    all_tests = [f for f in git(root, 'ls-files').split() if f.endswith('.py') and is_test(f)
                 and os.path.basename(f).startswith('test')]
    scored = {}
    for src in sources:
        mod_path = src[:-3].replace('/', '.')
        if mod_path.endswith('.__init__'):
            mod_path = mod_path[: -len('.__init__')]
        stem = os.path.basename(src)[:-3]
        pkg_mod = mod_path.split('.')
        for t in all_tests:
            score = 0
            tname = os.path.basename(t)[:-3]
            if stem != '__init__' and (tname == f'test_{stem}' or stem in tname):
                score += 3
            try:
                text = open(os.path.join(root, t), encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            if mod_path and re.search(r'\b' + re.escape(mod_path) + r'\b', text):
                score += 2
            elif len(pkg_mod) > 1 and re.search(r'from\s+' + re.escape('.'.join(pkg_mod[:-1])) + r'\s+import[^\n]*\b' + re.escape(pkg_mod[-1]) + r'\b', text):
                score += 2
            if score:
                scored[t] = scored.get(t, 0) + score
    return [t for t, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]


def main() -> int:
    root = workspace()
    targets = [a for a in sys.argv[1:] if a and a != '--']
    if not targets:
        sources = changed_sources(root)
        if not sources:
            print('No edited source files yet. Pass test paths explicitly: args=["tests/test_x.py"]')
            return 0
        targets = related_tests(root, sources)
        if not targets:
            print(f'Edited: {sources}. No related test files found; pass them explicitly: args=["tests/test_x.py"]')
            return 0
        print(f'Edited: {sources}\nRunning: {targets}')
    # -p no:pretty keeps the plain summary line if a table-printing plugin happens to be installed
    cmd = [sys.executable, '-m', 'pytest', '-x', '-q', '--tb=short', '-p', 'no:cacheprovider', '-p', 'no:pretty',
           '--no-header', *targets]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    try:
        res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=TIME_LIMIT, env=env)
    except subprocess.TimeoutExpired:
        print(f'pytest did not finish in {TIME_LIMIT}s; run a narrower target (a single test file or test).')
        return 0
    out = (res.stdout or '') + (res.stderr or '')
    lines = out.splitlines()
    final = re.compile(r'(\d+ (passed|failed|errors?|skipped)|no tests ran).* in [\d.]+s')
    summary = next((l for l in reversed(lines) if final.search(l)),
                   next((l for l in reversed(lines) if re.search(r'\b(passed|failed|error|no tests ran)\b', l)), lines[-1] if lines else ''))
    print(f'exit code {res.returncode}: {summary.strip()}')
    if res.returncode != 0:
        start = next((i for i, l in enumerate(lines) if l.startswith(('____', 'E ', 'ERROR', 'FAILED')) or '___ ' in l), 0)
        print('\n'.join(lines[start:start + 40]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
