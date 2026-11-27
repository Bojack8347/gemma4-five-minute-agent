"""Does the fixed-state check predict agent runs? (IMPLEMENTATION.md 6.33)

For every direction with agent runs, compare the fixed-state paired change (reach_check.py draws) with
the agent-level change on the matching behaviour coordinate, and count sign agreements among the changes
that are distinguishable from noise at fixed states (|change| > 2 paired se).

Usage: .venv/bin/python scripts/reach_validate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behaviour_vector import behaviour_vector  # noqa: E402
from reach_check import load_draw, paired  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# fixed-state statistic -> agent coordinate (edit propensity also predicts fewer empty patches)
MAP = [('malformed', 'garbled_call_rate', 1), ('repeat', 'repeat_loop_rate', 1), ('edit', 'edit_attempt_rate', 1),
       ('edit', 'empty_patch_rate', -1), ('tokens', 'tokens_per_turn', 1)]

CASES = [
    ('format 1 vs base', 'public', 'base', 'fmt1',
     ['base_sound100_c8', 'base_sound100_c8_r2'], ['format1_sound100_c8', 'format1_sound100_c8_r2']),
    ('decisiveness +1 vs format 1', 'public', 'fmt1', 'fmt1_dec1',
     ['v0long_format1_dev24_c8'], ['decisive_A_dev24_c8_r1', 'decisive_A_dev24_c8_r2']),
    ('decisiveness -0.5 vs format 1', 'public', 'fmt1', 'fmt1_decm05',
     ['v0long_format1_dev24_c8'], ['decisive_B_dev24_c8_r1', 'decisive_B_dev24_c8_r2']),
    ('statement 1 vs base', 'hard_terse', 'base', 'stmt1',
     ['heldout_hard_terse_base_c8'], ['heldout_hard_terse_v4_c8']),
]


def agent_mean(runs: list[str]) -> dict:
    vs = [behaviour_vector(ROOT / 'runs' / r) for r in runs]
    return {k: sum(v[k] for v in vs) / len(vs) for k in vs[0] if isinstance(vs[0][k], (int, float))}


def main() -> int:
    rows, hits, calls = [], 0, 0
    for name, st_set, ref, cfg, a_ref, a_cfg in CASES:
        try:
            r, c = load_draw(st_set, ref), load_draw(st_set, cfg)
        except FileNotFoundError:
            print(f'{name}: draws missing')
            continue
        ar, ac = agent_mean(a_ref), agent_mean(a_cfg)
        print(f'\n{name}  (fixed states: {st_set}, {ref} -> {cfg}; agent: {"+".join(a_ref)} -> {"+".join(a_cfg)})')
        for stat, coord, sign in MAP:
            m, se, n = paired(r, c, stat)
            da = ac[coord] - ar[coord]
            sig = abs(m) > 2 * se
            agree = (m * sign > 0) == (da > 0) if sig and da != 0 else None
            if agree is not None:
                calls += 1
                hits += agree
            verdict = {True: 'same sign', False: 'OPPOSITE', None: 'not called'}[agree]
            print(f'  {stat:9} {m:+.4f} (se {se:.4f}){"*" if sig else " "}   agent {coord:18} {ar[coord]:8.3f} -> {ac[coord]:8.3f} '
                  f'({da:+.3f})   {verdict}')
            rows.append({'case': name, 'stat': stat, 'fixed': m, 'se': se, 'coord': coord, 'agent_from': ar[coord],
                         'agent_to': ac[coord], 'agree': agree})
    print(f'\nsign agreement where the fixed-state change is called (|change| > 2 se): {hits}/{calls}')
    out = ROOT / 'runs' / 'reach' / 'validation.json'
    out.write_text(json.dumps({'rows': rows, 'hits': hits, 'calls': calls}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
