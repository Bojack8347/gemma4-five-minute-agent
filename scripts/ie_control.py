"""iE controller for the agent's behaviour vector (memo section 10, Usage 1 and 2).

The dissertation's accelerated iE iterates  x_{n+1} = x_n + A_n (alpha - y_n)  with one
noisy forward draw y_n per step and Robbins-Monro gains. Here x is a vector of
submission knobs, one draw is a batch of tasks run through `run_guarded.py`, y is the
behaviour vector from `behaviour_vector.py`, and alpha = b is the target operating
point. The gain is  A_n = (P / n^beta) * pinv(W^1/2 G) W^1/2  on transformed scales,
where G is a prior Jacobian from a one-time probe (matrix gain carrying Jacobian
information, dissertation section 4.4) and W whitens by the noise-floor variance.

Modes
  probe    run x0 and x0 + h_i e_i for each knob on one fixed task batch; fit G
  iterate  take --steps RM steps, rotating task batches
  status   print the state

Usage
  .venv/bin/python scripts/ie_control.py configs/ie_pilot.yaml probe
  .venv/bin/python scripts/ie_control.py configs/ie_pilot.yaml iterate --steps 5
All runs go under runs/ie_<name>/ and the state is runs/ie_<name>/state.json.

Added 2026-11-02 for the informed-target experiment (IMPLEMENTATION.md 6.41):
  data_dir / runner / python / harness_pythonpath   evaluate another task set under the 0.2.11 harness
  targets_from_run: runs/X      target values alpha = the behaviour vector measured on run X
  targets[i].side: max          a ceiling: the residual is clipped at zero once the coordinate is below it
  knobs[i].probe_from: runs/X   the probe column for this knob comes from an existing run at x0 + probe_step
  probe_base_from: [runs/X, ..] the probe base from existing runs (their mean z; also the gate reference)
  jacobian: {refit: true, min_runs: 6, ridge: 1.0}
                                re-estimate G by ridge regression over every draw once enough exist
  iterate --dry-run             compute the next move from the state without drawing
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behaviour_vector import COORDS, behaviour_vector, transform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / '.venv' / 'bin' / 'python'


# ----------------------------------------------------------------------------- knobs
def to_u(knob: dict, x: float) -> float:
    return math.log(x) if knob.get('scale') == 'log' else float(x)


def to_x(knob: dict, u: float):
    x = math.exp(u) if knob.get('scale') == 'log' else u
    lo, hi = knob['bounds']
    x = min(max(x, lo), hi)
    return int(round(x)) if knob.get('integer') else round(float(x), 4)


def apply_knobs(cfg: dict, x: dict, out_dir: Path) -> None:
    """Materialise a submission directory with the knob values x applied."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(ROOT / cfg['base_submission'], out_dir)
    for knob in cfg['knobs']:
        v = x[knob['name']]
        kind = knob['kind']
        if kind in ('sampling', 'eval'):
            path = out_dir / ('configs/sampling.yaml' if kind == 'sampling' else 'eval_config.yaml')
            doc = yaml.safe_load(path.read_text()) or {}
            node = doc
            keys = knob['key'].split('.')
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = v
            path.write_text(yaml.safe_dump(doc, sort_keys=False))
        elif kind == 'prompt_regex':
            path = out_dir / knob.get('file', 'prompts/system.md')
            text = path.read_text()
            new, n = re.subn(knob['pattern'], knob['replacement'].replace('{v}', str(v)), text)
            if n == 0:
                raise RuntimeError(f'prompt pattern for {knob["name"]} not found in {path}')
            path.write_text(new)
        elif kind == 'adapter_weight':
            pass  # merged below, once all weights are known
        else:
            raise ValueError(f'unknown knob kind {kind}')
    weights = {k['direction']: x[k['name']] for k in cfg['knobs'] if k['kind'] == 'adapter_weight'}
    if weights:
        # Exact merge of the trained directions (memo 10.2) into one adapter the agent routes to.
        # merge_directions needs torch, which lives in the GPU environment, so run it there.
        dirs = {d['name']: ROOT / d['path'] for d in cfg['directions']}
        cmd = [str(ROOT / '.venv-vllm' / 'bin' / 'python'), str(ROOT / 'scripts' / 'merge_directions.py'),
               '--out', str(out_dir / 'adapters' / 'merged')]
        for n, w in weights.items():
            cmd += ['--direction', f'{n}={dirs[n]}:{float(w)}']
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        (out_dir / 'adapters' / 'merged' / 'merge.json').unlink(missing_ok=True)  # only .json/.safetensors adapter files ship
        agent_yaml = out_dir / 'agent.yaml'
        text = agent_yaml.read_text()
        if re.search(r'^adapter:', text, flags=re.M):
            text = re.sub(r'^adapter:.*$', 'adapter: merged', text, flags=re.M)
        else:
            text = re.sub(r'^(model:.*)$', r'\1\nadapter: merged', text, count=1, flags=re.M)
        agent_yaml.write_text(text)


def restart_server(sub_dir: Path) -> None:
    """Adapters are mounted at server start, so a new merged adapter needs a restart (~4.5 min)."""
    import os
    import signal
    pid_file = ROOT / 'logs' / 'vllm_server.pid'
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            for _ in range(60):
                time.sleep(2)
                os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    subprocess.call(['pkill', '-9', '-f', 'vllm.entrypoints|VLLM::EngineCore'])
    time.sleep(5)
    pid_file.unlink(missing_ok=True)  # wait_server.sh must not read the previous server's pid
    log = (ROOT / 'logs' / 'vllm_server.log').open('a')
    subprocess.Popen([str(PY), str(ROOT / 'scripts' / 'serve.py'), '--submission-dir', str(sub_dir)],
                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT)
    rc = subprocess.call([str(ROOT / 'scripts' / 'wait_server.sh')], cwd=ROOT)
    if rc != 0:
        raise RuntimeError('server failed to start with the merged adapter')


# ----------------------------------------------------------------------------- state
def load_state(cfg: dict) -> dict:
    p = state_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    x0 = {k['name']: k['x0'] for k in cfg['knobs']}
    return {'name': cfg['name'], 'x': x0, 'runs': [], 'G': None, 'sigma': None, 'n': 0}


def save_state(cfg: dict, state: dict) -> None:
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1))


def state_path(cfg: dict) -> Path:
    return ROOT / 'runs' / f'ie_{cfg["name"]}' / 'state.json'


# ----------------------------------------------------------------------------- tasks
def task_pool(cfg: dict) -> list[str]:
    lines = (ROOT / cfg['task_pool']).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith('#')]


def batch_tasks(cfg: dict, index: int) -> list[str]:
    """Fixed batch for the probe (index 0); rotating chunks of a seeded permutation after."""
    pool = task_pool(cfg)
    size = cfg['batch_size']
    rng = random.Random(cfg.get('seed', 0))
    perm = pool[:]
    rng.shuffle(perm)
    chunks = max(1, len(perm) // size)
    start = (index % chunks) * size
    return sorted(perm[start:start + size])


# ----------------------------------------------------------------------------- one draw
def run_batch(cfg: dict, state: dict, label: str, x: dict, tasks: list[str]) -> dict:
    """Build the submission for x, run it, return the behaviour vector (one noisy draw)."""
    name = cfg['name']
    sub_dir = ROOT / 'submissions' / 'ie' / name / label
    apply_knobs(cfg, x, sub_dir)
    if (sub_dir / 'adapters').exists():
        restart_server(sub_dir)
    results_dir = ROOT / 'runs' / f'ie_{name}' / label
    tasks_file = results_dir.parent / f'{label}.tasks'
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text('\n'.join(tasks) + '\n')
    ev = cfg['eval']
    cmd = [str(PY), str(ROOT / 'scripts' / 'run_guarded.py'), '--name', f'ie_{name}_{label}']
    if cfg.get('runner'):
        cmd += ['--runner', cfg['runner']]
    if cfg.get('python'):
        cmd += ['--python', str(ROOT / cfg['python'])]
    cmd += [
        '--',
        '--submission-dir', str(sub_dir), '--results-dir', str(results_dir),
        '--task-ids-file', str(tasks_file),
        '--max-time-minutes', str(ev['max_time_minutes']), '--max-turns', str(ev.get('max_turns', 250)),
        '--concurrency', str(ev['concurrency']),
    ]
    if cfg.get('data_dir'):
        cmd += ['--data-dir', str(ROOT / cfg['data_dir'])]
    if not any(k['kind'] == 'eval' and k['key'] == 'max_tool_calls' for k in cfg['knobs']):
        cmd += ['--max-tool-calls', str(ev.get('max_tool_calls', 100))]
    import os
    env = dict(os.environ)
    if cfg.get('harness_pythonpath'):
        env['PYTHONPATH'] = str(ROOT / cfg['harness_pythonpath'])
    t0 = time.time()
    print(f'[{time.strftime("%F %T")}] {label}: x={x} tasks={len(tasks)}', flush=True)
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    if rc == 3:
        raise RuntimeError('watchdog killed the run; server degraded')
    if rc != 0:
        raise RuntimeError(f'run_guarded exited {rc}')
    y = behaviour_vector(results_dir)
    rec = {'label': label, 'x': x, 'tasks': tasks, 'y': y, 'wall_s': round(time.time() - t0), 'ts': time.time()}
    state['runs'].append(rec)
    save_state(cfg, state)
    print(f'[{time.strftime("%F %T")}] {label}: ' + ', '.join(f'{k}={y[k]:.3f}' for k in targets_of(cfg)), flush=True)
    return rec


def targets_of(cfg: dict) -> list[str]:
    return [t['coord'] for t in cfg['targets']]


def fill_targets(cfg: dict) -> None:
    """alpha measured on a run: every target without a value takes that run's coordinate."""
    src = cfg.get('targets_from_run')
    if not src:
        return
    y = behaviour_vector(ROOT / src)
    for t in cfg['targets']:
        if 'value' not in t:
            t['value'] = float(y[t['coord']])


def residual_of(cfg: dict, y: dict) -> np.ndarray:
    """alpha - z(y), with ceilings (side: max) contributing nothing once satisfied."""
    r = np.nan_to_num(z_target(cfg) - z_of(cfg, y), nan=0.0)
    for i, t in enumerate(cfg['targets']):
        if t.get('side') == 'max':
            r[i] = min(0.0, r[i])
    return r


def record_existing(cfg: dict, state: dict, label: str, x: dict, run: str) -> dict:
    """An existing run used as a draw (probe column or base) without re-running it."""
    y = behaviour_vector(ROOT / run)
    rec = {'label': label, 'x': x, 'tasks': None, 'y': y, 'from': run, 'ts': time.time()}
    state['runs'].append(rec)
    save_state(cfg, state)
    print(f'[{time.strftime("%F %T")}] {label}: from {run}: ' + ', '.join(f'{k}={y[k]:.3f}' for k in targets_of(cfg)), flush=True)
    return rec


def z_of(cfg: dict, y: dict) -> np.ndarray:
    n_tasks, n_calls = y.get('n_tasks', 100), y.get('n_calls', 1000)
    out = []
    for t in cfg['targets']:
        n = n_calls if t['coord'] == 'garbled_call_rate' else n_tasks
        out.append(transform(t['coord'], y[t['coord']], n))
    return np.array(out, dtype=float)


def z_target(cfg: dict) -> np.ndarray:
    return np.array([transform(t['coord'], t['value']) for t in cfg['targets']], dtype=float)


def u_of(cfg: dict, x: dict) -> np.ndarray:
    return np.array([to_u(k, x[k['name']]) for k in cfg['knobs']], dtype=float)


def x_of(cfg: dict, u: np.ndarray) -> dict:
    return {k['name']: to_x(k, float(u[i])) for i, k in enumerate(cfg['knobs'])}


# ----------------------------------------------------------------------------- modes
def probe(cfg: dict, state: dict) -> None:
    """Forward differences on one fixed batch: G[:, i] = (z(x0 + h_i e_i) - z(x0)) / h_i."""
    tasks = batch_tasks(cfg, 0)
    done = {r['label']: r for r in state['runs']}
    base_from = cfg.get('probe_base_from') or []
    if base_from:
        bases = [done.get(f'probe_base_{j}') or record_existing(cfg, state, f'probe_base_{j}', state['x'], run)
                 for j, run in enumerate(base_from)]
        z0 = np.mean(np.stack([z_of(cfg, b['y']) for b in bases]), axis=0)
        gate = cfg.get('gate')
        if gate:
            vals = [b['y'][gate['coord']] for b in bases]
            state['last_accepted_x'] = dict(state['x'])
            state['gate_reference'] = float(np.mean(vals))
            state['gate_ref_n'] = len(vals)
        state['base_z'] = z0.tolist()
    else:
        base = done.get('probe_base') or run_batch(cfg, state, 'probe_base', state['x'], tasks)
        z0 = z_of(cfg, base['y'])
        state['base_z'] = z0.tolist()
    u0 = u_of(cfg, state['x'])
    cols = []
    for i, knob in enumerate(cfg['knobs']):
        label = f'probe_{knob["name"]}'
        u = u0.copy()
        u[i] += knob['probe_step']
        xi = x_of(cfg, u)
        h = to_u(knob, xi[knob['name']]) - u0[i]  # actual step after rounding/bounds
        if knob.get('probe_from'):
            rec = done.get(label) or record_existing(cfg, state, label, xi, knob['probe_from'])
        else:
            rec = done.get(label) or run_batch(cfg, state, label, xi, tasks)
        zi = z_of(cfg, rec['y'])
        cols.append(np.nan_to_num((zi - z0) / h, nan=0.0))
    G = np.stack(cols, axis=1)  # targets x knobs
    state['G'] = G.tolist()
    state['G_probe'] = G.tolist()
    save_state(cfg, state)
    print_G(cfg, G)


def print_G(cfg: dict, G: np.ndarray) -> None:
    names = [k['name'] for k in cfg['knobs']]
    print('G (d z_target / d u_knob):')
    print(f'{"":22}' + ''.join(f'{n:>16}' for n in names))
    for t, row in zip(cfg['targets'], G):
        print(f'{t["coord"]:22}' + ''.join(f'{v:16.3f}' for v in row))
    s = np.linalg.svd(G, compute_uv=False)
    print('singular values:', np.round(s, 3), ' condition:', round(s[0] / max(s[-1], 1e-12), 1))


def set_sigma(cfg: dict, state: dict, noise_runs: list[Path]) -> None:
    """Per-target SD on the transformed scale from repeated runs at one x (whitening)."""
    ys = [behaviour_vector(r) for r in noise_runs]
    zs = np.stack([z_of(cfg, y) for y in ys])
    sd = zs.std(axis=0, ddof=1) if len(noise_runs) > 1 else np.full(zs.shape[1], np.nan)
    # Floor: with one or two repeats the empirical sd is unreliable, so never go below the
    # sampling sd of the coordinate on the transformed scale (binomial for rates, 0.15 for logs).
    floor = []
    for t in cfg['targets']:
        kind = COORDS.get(t['coord'], 'identity')
        if kind == 'logit':
            n = np.mean([y['n_calls'] if t['coord'] == 'garbled_call_rate' else y['n_tasks'] for y in ys])
            p = np.clip(np.mean([y[t['coord']] for y in ys]), 1.0 / n, 1 - 1.0 / n)
            floor.append(1.0 / math.sqrt(n * p * (1 - p)))
        elif kind == 'log':
            floor.append(0.15)
        else:
            floor.append(0.1)
    sd = np.where(np.isfinite(sd), np.maximum(sd, floor), floor)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    state['sigma'] = sd.tolist()
    save_state(cfg, state)
    for t, s in zip(cfg['targets'], sd):
        print(f'sigma[{t["coord"]}] = {s:.3f}')


def refit_jacobian(cfg: dict, state: dict) -> np.ndarray | None:
    """Ridge regression of z on u over every draw, shrunk toward the probe Jacobian (item 5 of 2026-11-01:
    columns re-estimated as steps accumulate instead of trusted from one draw)."""
    jc = cfg.get('jacobian') or {}
    if not jc.get('refit'):
        return None
    runs = [r for r in state['runs'] if r.get('y')]
    if len(runs) < jc.get('min_runs', 6):
        return None
    G0 = np.array(state.get('G_probe') or state['G'])
    U = np.stack([u_of(cfg, r['x']) for r in runs])
    Z = np.stack([np.nan_to_num(z_of(cfg, r['y']), nan=0.0) for r in runs])
    Uc = U - U.mean(axis=0)
    Zc = Z - Z.mean(axis=0)
    lam = float(jc.get('ridge', 1.0))
    # G = G0 + (Zc - G0 Uc^T)^T-style ridge update: minimise |Zc - Uc G^T|^2 + lam |G - G0|^2
    A = Uc.T @ Uc + lam * np.eye(U.shape[1])
    B = Zc.T @ Uc + lam * G0
    G = B @ np.linalg.inv(A)
    state['G'] = G.tolist()
    state.setdefault('G_history', []).append({'n_runs': len(runs), 'G': G.tolist()})
    save_state(cfg, state)
    return G


def iterate(cfg: dict, state: dict, steps: int, dry_run: bool = False) -> None:
    if state['G'] is None:
        raise SystemExit('no Jacobian prior: run probe first (or set G in state.json)')
    sigma = np.array(state['sigma']) if state.get('sigma') else np.ones(len(cfg['targets']))
    w = np.array([t.get('weight', 1.0) for t in cfg['targets']]) / sigma
    zb = z_target(cfg)
    trust = np.array([k['trust'] for k in cfg['knobs']])
    lo = np.array([to_u(k, k['bounds'][0]) for k in cfg['knobs']])
    hi = np.array([to_u(k, k['bounds'][1]) for k in cfg['knobs']])
    P, beta = cfg['gain']['P'], cfg['gain']['beta']
    n_probe = 1 + len(cfg['knobs'])

    def gain_matrix():
        G = np.array(state['G'])
        return np.linalg.pinv(G * w[:, None], rcond=cfg['gain'].get('rcond', 0.1))

    if state['n'] == 0 and state.get('base_z') is not None and not state.get('first_move_done'):
        # The probe base is known (its draws exist): the first move needs no new draw.
        z0 = np.array(state['base_z'])
        r = zb - z0
        for i, t in enumerate(cfg['targets']):
            if t.get('side') == 'max':
                r[i] = min(0.0, r[i])
        r = np.nan_to_num(r, nan=0.0)
        du = (P / 1.0) * (gain_matrix() @ (w * r))
        du = np.clip(du, -trust, trust)
        u = np.clip(u_of(cfg, state['x']) + du, lo, hi)
        new_x = x_of(cfg, u)
        print(f'[{time.strftime("%F %T")}] first move from the probe base: |W r| = {np.linalg.norm(w * r):.3f}, '
              f'residual = {np.round(r, 3).tolist()}, du = {np.round(du, 3).tolist()}, next x = {new_x}', flush=True)
        if dry_run:
            return
        state['first_move'] = {'residual': r.tolist(), 'du': du.tolist(), 'next_x': new_x}
        state['first_move_done'] = True
        state['x'] = new_x
        save_state(cfg, state)
    elif dry_run:
        print(f'dry run: n={state["n"]} x={state["x"]} (next draw would be at this x)')
        return

    for _ in range(steps):
        state['n'] += 1
        n = state['n']
        label = f'step_{n:03d}'
        batch_index = n_probe + n  # rotate past the probe batch
        tasks = batch_tasks(cfg, batch_index)
        rec = run_batch(cfg, state, label, state['x'], tasks)
        r = residual_of(cfg, rec['y'])
        pinv = gain_matrix()
        # Merit gate (memo section 3.3, step 7; absent from the pilot, see 6.17): a draw whose
        # protected coordinate falls below the reference by more than the allowance rejects the
        # move that produced it, and the iteration resumes from the last accepted point.
        gate = cfg.get('gate')
        val = rec['y'][gate['coord']] if gate else None
        at_accepted = gate and state.get('last_accepted_x') == state['x']
        if gate and at_accepted:
            # A re-draw of the accepted point is a measurement, not a move: fold it into the
            # running mean that serves as the reference (common random tasks make the mean
            # converge), and never reject it. Round 2 of 2026-10-14 stalled on exactly this.
            n_ref = state.get('gate_ref_n', 0)
            ref = state.get('gate_reference')
            state['gate_reference'] = val if ref is None else (ref * n_ref + val) / (n_ref + 1)
            state['gate_ref_n'] = n_ref + 1
            rec['gate'] = f'measured ({state["gate_reference"]:.3f} over {state["gate_ref_n"]})'
        elif gate and state.get('last_accepted_x') is not None and state.get('gate_reference') is not None \
                and val < state['gate_reference'] + gate.get('min_delta', -0.10):
            rec['gate'] = 'rejected'
            state['x'] = state['last_accepted_x']
            state['trust_scale'] = state.get('trust_scale', 1.0) * 0.5  # do not re-propose the same move
            print(f'[{time.strftime("%F %T")}] {label}: {gate["coord"]}={val:.3f} below reference '
                  f'{state["gate_reference"]:.3f}{gate.get("min_delta", -0.10):+.2f}; move rejected, back to {state["x"]}', flush=True)
            save_state(cfg, state)
            continue
        elif gate:
            # a new point that passes becomes the accepted point; its draw starts the new reference
            rec['gate'] = 'accepted'
            state['last_accepted_x'] = dict(state['x'])
            # The bar never ratchets down: an accepted point whose draw is below the old reference
            # becomes the iterate, but the reference keeps the better value (round 3 of 2026-10-14
            # reset it to a 7/24 draw after a 0.358 mean, which would have loosened the gate).
            old_ref = state.get('gate_reference')
            if old_ref is None or val >= old_ref:
                state['gate_reference'] = val
                state['gate_ref_n'] = 1
            state['trust_scale'] = min(1.0, state.get('trust_scale', 1.0) * 2.0)  # recover trust after progress
        du = (P / n ** beta) * (pinv @ (w * r))
        scale = state.get('trust_scale', 1.0)
        du = np.clip(du, -trust * scale, trust * scale)
        u = np.clip(u_of(cfg, state['x']) + du, lo, hi)
        new_x = x_of(cfg, u)
        rec['residual'] = r.tolist()
        rec['du'] = du.tolist()
        rec['next_x'] = new_x
        state['x'] = new_x
        save_state(cfg, state)
        print(f'[{time.strftime("%F %T")}] {label}: |W r| = {np.linalg.norm(w * r):.3f}, next x = {new_x}', flush=True)
        if refit_jacobian(cfg, state) is not None:
            print(f'[{time.strftime("%F %T")}] Jacobian re-estimated over {len(state["runs"])} draws:', flush=True)
            print_G(cfg, np.array(state['G']))


def status(cfg: dict, state: dict) -> None:
    print(f'name={state["name"]} n={state["n"]} x={state["x"]}')
    if state['G'] is not None:
        print_G(cfg, np.array(state['G']))
    if state.get('sigma'):
        print('sigma:', np.round(state['sigma'], 3))
    cols = targets_of(cfg)
    sigma = np.array(state['sigma']) if state.get('sigma') else np.ones(len(cols))
    print('alpha:', {t['coord']: round(t['value'], 4) for t in cfg['targets']})
    print(f'{"label":16}' + ''.join(f'{c[:14]:>15}' for c in cols) + f'{"|W r|":>8}   x')
    for r in state['runs']:
        wr = np.linalg.norm(residual_of(cfg, r['y']) / sigma)
        print(f'{r["label"]:16}' + ''.join(f'{r["y"][c]:15.3f}' for c in cols) + f'{wr:8.2f}   {r["x"]}'
              + (f'   [{r["gate"]}]' if r.get('gate') else ''))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', type=Path)
    ap.add_argument('mode', choices=['probe', 'iterate', 'status', 'build'])
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--noise-runs', nargs='*', type=Path, default=None, help='repeat runs at one x, for sigma')
    ap.add_argument('--dry-run', action='store_true', help='iterate: print the next move without drawing')
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    fill_targets(cfg)
    state = load_state(cfg)
    if args.noise_runs:
        set_sigma(cfg, state, args.noise_runs)
    if args.mode == 'probe':
        probe(cfg, state)
    elif args.mode == 'iterate':
        iterate(cfg, state, args.steps, dry_run=args.dry_run)
    elif args.mode == 'build':
        out = ROOT / 'submissions' / 'ie' / cfg['name'] / 'current'
        apply_knobs(cfg, state['x'], out)
        print(f'built {out} with x={state["x"]}')
    else:
        status(cfg, state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
