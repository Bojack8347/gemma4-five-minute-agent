"""Statement-fraction variants of the hard held-out set, for the proxy calibration (IMPLEMENTATION.md 6.40).

The knob is the fraction f of the statement body kept: the task's statement becomes the terse
title (PR number stripped, exactly as data/heldout_hard_terse) followed by the first ceil(f * L)
non-empty body lines of the full statement, where L is the number of non-empty body lines. f = 0
reproduces the title-only set, f = 1 the full statement without its PR number. Blank lines between
kept lines are preserved so paragraphs stay readable; trailing partial paragraphs are kept as is.

Snapshots, graphs and embeddings are symlinked from data/heldout_hard (the base tree is identical).

Usage:
  python scripts/heldout/build_frac_tasks.py 0.5          # writes data/heldout_hard_f050/
  python scripts/heldout/build_frac_tasks.py 0.25 0.75     # several
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / 'data' / 'heldout_hard'
TERSE = ROOT / 'data' / 'heldout_hard_terse'


def body_lines(statement: str) -> list[str]:
    """Lines after the title line, with leading blank lines removed."""
    lines = statement.split('\n')[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def cut(statement: str, title: str, f: float) -> str:
    lines = body_lines(statement)
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    keep = math.ceil(f * len(nonempty) - 1e-9) if nonempty else 0
    if keep <= 0:
        return title
    last = nonempty[keep - 1]
    kept = lines[: last + 1]
    return title + '\n\n' + '\n'.join(kept).rstrip()


def build(f: float) -> Path:
    tag = f'heldout_hard_f{int(round(f * 100)):03d}'
    out = ROOT / 'data' / tag
    out.mkdir(exist_ok=True)
    terse = {json.loads(l)['instance_id']: json.loads(l)['problem_statement'] for l in (TERSE / 'tasks.jsonl').open()}
    rows = []
    n_kept, n_total = 0, 0
    for line in (FULL / 'tasks.jsonl').open():
        t = json.loads(line)
        title = terse[t['instance_id']]
        st = cut(t['problem_statement'], title, f)
        nb = len([l for l in body_lines(t['problem_statement']) if l.strip()])
        kb = len([l for l in body_lines(st) if l.strip()])
        n_kept += kb
        n_total += nb
        t = dict(t, problem_statement=st)
        rows.append(t)
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'sound_tasks.txt').write_text((FULL / 'sound_tasks.txt').read_text())
    for sub in ('snapshots', 'graphs', 'embeddings'):
        dst = out / sub
        if not dst.exists():
            os.symlink(os.path.relpath(FULL / sub, out), dst)
    print(f'{tag}: {len(rows)} tasks, body lines kept {n_kept}/{n_total} = {n_kept / max(n_total, 1):.3f}')
    return out


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        build(float(arg))
