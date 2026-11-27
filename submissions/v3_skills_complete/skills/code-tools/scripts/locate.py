"""Rank the repository's source files for a set of identifiers, with enclosing definitions.

Usage: run_skill_script(skill_name="code-tools", file_path="scripts/locate.py",
                        args=["identifier_or_phrase", "another", ...])
Prints at most ~60 lines: the best-matching source files (outside tests and docs), each with the
matched terms and a few `path:line  [enclosing def]  code` hits, then test files that mention them.
"""

import ast
import os
import subprocess
import sys

SKIP_DIRS = ('/tests/', '/test/', '/docs/', '/doc/', '/examples/', '/benchmarks/', '/.git/', '/build/', '/dist/')


def workspace() -> str:
    for cand in ['/workspace', *os.environ.get('PYTHONPATH', '').split(os.pathsep)]:
        if cand and os.path.isdir(os.path.join(cand, '.git')):
            return cand
    return '/workspace'


def is_test(rel: str) -> bool:
    base = os.path.basename(rel)
    return base.startswith('test_') or base.endswith('_test.py') or base == 'conftest.py' or \
        any(seg in ('/' + rel) for seg in ('/tests/', '/test/', '/testing/'))


def grep(root: str, term: str) -> list[tuple[str, int, str]]:
    try:
        out = subprocess.run(['grep', '-rnF', '--include=*.py', '--', term, '.'], cwd=root,
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    hits = []
    for line in out.splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3 and parts[1].isdigit():
            hits.append((parts[0][2:] if parts[0].startswith('./') else parts[0], int(parts[1]), parts[2].strip()))
    return hits


def enclosing(root: str, rel: str, cache: dict) -> dict:
    """Map line number -> innermost 'class.def' name for a file (best effort)."""
    if rel in cache:
        return cache[rel]
    spans = []
    try:
        tree = ast.parse(open(os.path.join(root, rel), encoding='utf-8', errors='replace').read())

        def walk(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = f'{prefix}.{child.name}' if prefix else child.name
                    spans.append((child.lineno, getattr(child, 'end_lineno', child.lineno), name))
                    walk(child, name)
                else:
                    walk(child, prefix)
        walk(tree, '')
    except Exception:
        pass
    cache[rel] = spans
    return spans


def name_at(spans, line: int) -> str:
    best = ''
    for start, end, name in spans:
        if start <= line <= end and len(name) >= len(best):
            best = name
    return best


def main() -> int:
    terms = [t for t in sys.argv[1:] if t and t != '--']
    if not terms:
        print('usage: args=["identifier", ...] (identifiers, error messages or option names from the task)')
        return 0
    root = workspace()
    per_file: dict[str, dict] = {}
    test_files: dict[str, set] = {}
    for term in terms[:8]:
        for rel, line, code in grep(root, term):
            if any(s in '/' + rel for s in SKIP_DIRS[2:]) and not is_test(rel):
                continue
            if is_test(rel):
                test_files.setdefault(rel, set()).add(term)
                continue
            entry = per_file.setdefault(rel, {'terms': set(), 'hits': []})
            entry['terms'].add(term)
            entry['hits'].append((line, code, term))
    if not per_file:
        print(f'No source matches for {terms}. Try shorter or different identifiers.')
        return 0
    ranked = sorted(per_file.items(), key=lambda kv: (-len(kv[1]['terms']), -len(kv[1]['hits']), kv[0]))
    cache: dict = {}
    lines_out = [f'Source files ranked by matched terms (of {len(terms[:8])}):']
    for rel, entry in ranked[:8]:
        lines_out.append(f'\n{rel}  terms={sorted(entry["terms"])}  hits={len(entry["hits"])}')
        spans = enclosing(root, rel, cache)
        shown = set()
        for line, code, _ in sorted(entry['hits'])[:40]:
            where = name_at(spans, line)
            if (where, code) in shown:
                continue
            shown.add((where, code))
            lines_out.append(f'  {rel}:{line}  [{where or "module"}]  {code[:110]}')
            if len(shown) >= 4:
                break
    if test_files:
        tests = sorted(test_files.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
        lines_out.append('\nTest files mentioning these terms: ' + ', '.join(t for t, _ in tests))
    print('\n'.join(lines_out[:70]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
