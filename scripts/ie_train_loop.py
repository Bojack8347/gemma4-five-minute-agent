"""iE as the outer loop of DPO (IMPLEMENTATION.md 6.36).

The controller of ie_control.py moves merge weights of fixed, separately trained directions. Here the
knobs are training settings, and each draw retrains one adapter from scratch on a weighted mixture of
pair sources, serves it, and evaluates it on a fixed batch of title-only tasks:

    x = (source loss weights, early-stop margin)   ->   train   ->   y = behaviour vector of the batch
    x_{n+1} = x_n + A_n (b - y_n),   A_n = (P / n^beta) * pinv(W^1/2 G) W^1/2

with G from a one-time forward-difference probe (one retrain per knob), W from the spread of three
base-like agents on the batch, a trust region, box bounds, and the merit gate of ie_control (a draw
that loses more than the allowance on resolution rejects the move). A retrain that the trainer's
gates refuse (collapse, too few steps) counts as an infeasible point: the move is rejected without an
evaluation. This is the primal-dual pattern of constrained RL with DPO as the primal step and iE as
the dual update, in the dissertation's single-draw Robbins-Monro form.

Usage (runs the trainer in .venv-vllm and the evaluator in .venv-heldout; stops and restarts the server):
  .venv/bin/python scripts/ie_train_loop.py configs/ie_train.yaml probe
  .venv/bin/python scripts/ie_train_loop.py configs/ie_train.yaml iterate --steps 4
  .venv/bin/python scripts/ie_train_loop.py configs/ie_train.yaml status
  .venv/bin/python scripts/ie_train_loop.py configs/ie_train.yaml final     # the current x on the whole task set
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behaviour_vector import behaviour_vector  # noqa: E402
from ie_control import (load_state, print_G, restart_server, save_state, targets_of, to_u, u_of, x_of,  # noqa: E402
                        z_of, z_target)

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / '.venv' / 'bin' / 'python'
PY_TRAIN = ROOT / '.venv-vllm' / 'bin' / 'python'
PY_EVAL = ROOT / '.venv-heldout' / 'bin' / 'python'


# ----------------------------------------------------------------------------- tasks and references
def batch_ids(cfg: dict) -> list[str]:
    pool = [l.strip() for l in (ROOT / cfg['task_pool']).read_text().splitlines() if l.strip() and not l.startswith('#')]
    rng = random.Random(cfg.get('seed', 0))
    rng.shuffle(pool)
    return sorted(pool[:cfg['batch_size']])


def subset_run(run: Path, ids: list[str], out: Path) -> Path:
    """A run folder restricted to ids: filtered results, everything else linked."""
    out.mkdir(parents=True, exist_ok=True)
    keep = set(ids)
    lines = [l for l in (run / 'task_results.jsonl').read_text().splitlines() if l.strip() and json.loads(l)['instance_id'] in keep]
    (out / 'task_results.jsonl').write_text('\n'.join(lines) + '\n')
    for entry in run.iterdir():
        if entry.name == 'task_results.jsonl':
            continue
        dst = out / entry.name
        if not dst.exists():
            dst.symlink_to(entry.resolve())
    return out


def references(cfg: dict, state: dict) -> None:
    """Whitening and the gate reference from base-like runs restricted to the batch (once)."""
    if state.get('sigma') and state.get('gate_reference') is not None and state.get('sigma_floored'):
        return
    ids = batch_ids(cfg)
    zs, res = [], []
    for i, r in enumerate(cfg['noise_runs']):
        sub = subset_run(ROOT / r, ids, ROOT / 'runs' / f'ie_{cfg["name"]}' / 'noise' / Path(r).name)
        y = behaviour_vector(sub)
        zs.append(z_of(cfg, y))
        res.append(y['resolution_rate'])
        print(f'noise run {Path(r).name} on the batch: ' + ', '.join(f'{k}={y[k]:.3f}' for k in targets_of(cfg)) + f', resolution {y["resolution_rate"]:.3f}')
    zs = np.stack(zs)
    sd = zs.std(axis=0, ddof=1) if len(zs) > 1 else np.ones(zs.shape[1])
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    # three runs give a noisy spread; a per-coordinate floor (config: sigma_floor) keeps one coordinate
    # from dominating the whitened residual (the first edit's three values were within 10%)
    floor = np.array([cfg.get('sigma_floor', {}).get(t['coord'], 0.0) for t in cfg['targets']])
    sd = np.maximum(sd, floor)
    state['sigma'] = sd.tolist()
    state['sigma_floored'] = True
    if state.get('gate_reference') is None:
        state['gate_reference'] = float(res[0])   # the first noise run is the base agent
        state['gate_ref_n'] = 1
    state['base_z'] = zs[0].tolist()
    save_state(cfg, state)
    for t, s in zip(cfg['targets'], sd):
        print(f'sigma[{t["coord"]}] = {s:.3f}')


# ----------------------------------------------------------------------------- one draw
def stop_server() -> None:
    pid_file = ROOT / 'logs' / 'vllm_server.pid'
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    for _ in range(30):
        if subprocess.call(['pgrep', '-f', 'vllm.entrypoints|VLLM::EngineCore'], stdout=subprocess.DEVNULL) != 0:
            break
        time.sleep(2)
    subprocess.call(['pkill', '-9', '-f', 'vllm.entrypoints|VLLM::EngineCore'])
    pid_file.unlink(missing_ok=True)
    for _ in range(30):  # wait for the unified memory to come back
        used = int(subprocess.run(['free', '-g'], capture_output=True, text=True).stdout.splitlines()[1].split()[2])
        if used < 15:
            break
        time.sleep(2)


def train(cfg: dict, label: str, x: dict) -> Path | None:
    """Retrain the adapter at x. Returns the adapter directory, or None if the trainer refused it."""
    out = ROOT / 'adapters_out' / f'ie_{cfg["name"]}' / label
    if (out / 'adapter_model.safetensors').exists():
        return out
    tr = cfg['train']
    cmd = [str(PY_TRAIN), str(ROOT / 'scripts' / 'train_direction_dpo.py'), '--out', str(out), '--pairs']
    for name, src in cfg['sources'].items():
        w = next((x[k['name']] for k in cfg['knobs'] if k['kind'] == 'source_weight' and k['source'] == name), 1.0)
        cmd.append(f'{name}={ROOT / src["path"]}:{float(w)}')
    cmd += ['--max-contexts-per-source', str(tr['max_contexts_per_source']), '--context-seed', str(tr.get('context_seed', 0)),
            '--max-pairs-per-context', str(tr['max_pairs_per_context']), '--epochs', str(tr['epochs']),
            '--max-prompt-tokens', str(tr['max_prompt_tokens']), '--max-completion-tokens', str(tr.get('max_completion_tokens', 768)),
            '--chunk', str(tr['chunk']), '--grad-accum', str(tr['grad_accum']), '--lr', str(tr['lr']),
            '--nll-weight', str(tr['nll_weight']), '--sanity-tokens', str(tr.get('sanity_tokens', 1024)),
            '--min-steps', str(tr.get('min_steps', 8))]
    for k in cfg['knobs']:
        if k['kind'] == 'train_arg':
            cmd += [f'--{k["arg"]}', str(x[k['name']])]
    log_dir = ROOT / 'logs' / f'ie_{cfg["name"]}'
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f'train_{label}.log'
    print(f'[{time.strftime("%F %T")}] {label}: training at x={x} (log {log.relative_to(ROOT)})', flush=True)
    t0 = time.time()
    with log.open('w') as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
    tail = [l for l in log.read_text().splitlines() if re.search(r'early stop|TOO FEW|SANITY|saved adapter|contexts;', l)]
    print(f'[{time.strftime("%F %T")}] {label}: trainer rc={rc} after {(time.time() - t0) / 60:.0f} min; ' + ' | '.join(t[:90] for t in tail[-3:]), flush=True)
    if rc != 0 or not (out / 'adapter_model.safetensors').exists():
        return None
    return out


def package(cfg: dict, label: str, adapter: Path) -> Path:
    sub = ROOT / 'submissions' / 'ie' / cfg['name'] / label
    if sub.exists():
        shutil.rmtree(sub)
    shutil.copytree(ROOT / cfg['base_submission'], sub)
    (sub / 'adapters').mkdir(exist_ok=True)
    subprocess.run([str(PY_TRAIN), str(ROOT / 'scripts' / 'merge_directions.py'), '--out', str(sub / 'adapters' / 'merged'),
                    '--direction', f'mix={adapter}:1.0'], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (sub / 'adapters' / 'merged' / 'merge.json').unlink(missing_ok=True)
    agent_yaml = sub / 'agent.yaml'
    text = agent_yaml.read_text()
    if re.search(r'^adapter:', text, flags=re.M):
        text = re.sub(r'^adapter:.*$', 'adapter: merged', text, flags=re.M)
    else:
        text = re.sub(r'^(model:.*)$', r'\1\nadapter: merged', text, count=1, flags=re.M)
    agent_yaml.write_text(text)
    return sub


def evaluate(cfg: dict, label: str, sub: Path, ids: list[str]) -> Path:
    results = ROOT / 'runs' / f'ie_{cfg["name"]}' / label
    tasks_file = results.parent / f'{label}.tasks'
    results.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text('\n'.join(ids) + '\n')
    ev = cfg['eval']
    cmd = [str(PY), str(ROOT / 'scripts' / 'run_guarded.py'), '--name', f'ie_{cfg["name"]}_{label}',
           '--runner', 'run_eval_v2.py', '--python', str(PY_EVAL), '--',
           '--submission-dir', str(sub), '--results-dir', str(results), '--data-dir', str(ROOT / cfg['data_dir']),
           '--task-ids-file', str(tasks_file), '--max-time-minutes', str(ev['max_time_minutes']),
           '--max-tool-calls', str(ev.get('max_tool_calls', 100)), '--max-turns', str(ev.get('max_turns', 250)),
           '--concurrency', str(ev['concurrency'])]
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'data' / 'harness_v2_site')}
    rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 3:
        raise RuntimeError('watchdog killed the run; server degraded')
    if rc != 0:
        raise RuntimeError(f'run_guarded exited {rc}')
    return results


def draw(cfg: dict, state: dict, label: str, x: dict, ids: list[str] | None = None) -> dict | None:
    """Train, serve, evaluate at x. Returns the record, or None if the trainer refused x (infeasible)."""
    ids = ids or batch_ids(cfg)
    t0 = time.time()
    stop_server()
    adapter = train(cfg, label, x)
    if adapter is None:
        rec = {'label': label, 'x': x, 'infeasible': True, 'ts': time.time()}
        state['runs'].append(rec)
        save_state(cfg, state)
        return None
    sub = package(cfg, label, adapter)
    restart_server(sub)
    results = evaluate(cfg, label, sub, ids)
    y = behaviour_vector(results)
    rec = {'label': label, 'x': x, 'tasks': ids, 'y': y, 'wall_s': round(time.time() - t0), 'ts': time.time()}
    state['runs'].append(rec)
    save_state(cfg, state)
    print(f'[{time.strftime("%F %T")}] {label}: ' + ', '.join(f'{k}={y[k]:.3f}' for k in targets_of(cfg))
          + f', resolution {y["resolution_rate"]:.3f} ({(time.time() - t0) / 3600:.1f} h)', flush=True)
    return rec


# ----------------------------------------------------------------------------- modes
def residual_of(cfg: dict, y: dict) -> np.ndarray:
    """Target minus measurement on the transformed scale. A target with `side: max` is a ceiling: a
    coordinate already below it contributes nothing, so the controller does not spend a step pushing a
    rate back up to an equality target it has overshot (the transforms are monotone, so lower z is better)."""
    r = np.nan_to_num(z_target(cfg) - z_of(cfg, y), nan=0.0)
    for i, t in enumerate(cfg['targets']):
        if t.get('side') == 'max':
            r[i] = min(0.0, r[i])
    return r


def probe(cfg: dict, state: dict) -> None:
    references(cfg, state)
    done = {r['label']: r for r in state['runs'] if not r.get('infeasible')}
    base = done.get('probe_base') or draw(cfg, state, 'probe_base', state['x'])
    if base is None:
        raise SystemExit('the base point itself is infeasible; change x0')
    z0 = z_of(cfg, base['y'])
    u0 = u_of(cfg, state['x'])
    cols = []
    for i, knob in enumerate(cfg['knobs']):
        label = f'probe_{knob["name"]}'
        rec = done.get(label)
        step = knob['probe_step']
        for attempt in range(2):
            if rec is not None:
                break
            u = u0.copy()
            u[i] += step
            xi = x_of(cfg, u)
            rec = draw(cfg, state, label if attempt == 0 else f'{label}_half', xi)
            if rec is None:
                step *= 0.5   # an infeasible perturbation: try half the step once
        if rec is None:
            print(f'{label}: infeasible twice; column set to zero', flush=True)
            cols.append(np.zeros_like(z0))
            continue
        h = to_u(knob, rec['x'][knob['name']]) - u0[i]
        cols.append(np.nan_to_num((z_of(cfg, rec['y']) - z0) / h, nan=0.0))
    G = np.stack(cols, axis=1)
    state['G'] = G.tolist()
    state['last_accepted_x'] = dict(state['x'])
    save_state(cfg, state)
    print_G(cfg, G)


def iterate(cfg: dict, state: dict, steps: int) -> None:
    if state.get('G') is None:
        raise SystemExit('no Jacobian: run probe first')
    references(cfg, state)
    G = np.array(state['G'])
    sigma = np.array(state['sigma'])
    w = np.array([t.get('weight', 1.0) for t in cfg['targets']]) / sigma
    pinv = np.linalg.pinv(G * w[:, None], rcond=cfg['gain'].get('rcond', 0.1))
    zb = z_target(cfg)
    trust = np.array([k['trust'] for k in cfg['knobs']])
    lo = np.array([to_u(k, k['bounds'][0]) for k in cfg['knobs']])
    hi = np.array([to_u(k, k['bounds'][1]) for k in cfg['knobs']])
    P, beta = cfg['gain']['P'], cfg['gain']['beta']
    gate = cfg.get('gate')
    # the base draw of the probe is the first draw at the accepted point; start the residual from it
    if state.get('n', 0) == 0:
        base = next(r for r in state['runs'] if r['label'] == 'probe_base')
        r0 = residual_of(cfg, base['y'])
        du = np.clip((P / 1.0) * (pinv @ (w * r0)), -trust, trust)
        state['x'] = x_of(cfg, np.clip(u_of(cfg, state['last_accepted_x']) + du, lo, hi))
        print(f'first move from the probe base: |W r| = {np.linalg.norm(w * r0):.3f}, x = {state["x"]}', flush=True)
        save_state(cfg, state)
    for _ in range(steps):
        state['n'] = state.get('n', 0) + 1
        n = state['n']
        label = f'step_{n:03d}'
        rec = draw(cfg, state, label, state['x'])
        if rec is None:
            state['x'] = state['last_accepted_x']
            state['trust_scale'] = state.get('trust_scale', 1.0) * 0.5
            print(f'[{time.strftime("%F %T")}] {label}: infeasible training; back to {state["x"]}, trust halved', flush=True)
            save_state(cfg, state)
            continue
        r = residual_of(cfg, rec['y'])
        val = rec['y'][gate['coord']] if gate else None
        at_accepted = gate and state.get('last_accepted_x') == state['x']
        if gate and at_accepted:
            n_ref = state.get('gate_ref_n', 0)
            ref = state.get('gate_reference')
            state['gate_reference'] = val if ref is None else (ref * n_ref + val) / (n_ref + 1)
            state['gate_ref_n'] = n_ref + 1
            rec['gate'] = f'measured ({state["gate_reference"]:.3f} over {state["gate_ref_n"]})'
        elif gate and state.get('gate_reference') is not None and val < state['gate_reference'] + gate.get('min_delta', -0.10):
            rec['gate'] = 'rejected'
            state['x'] = state['last_accepted_x']
            state['trust_scale'] = state.get('trust_scale', 1.0) * 0.5
            print(f'[{time.strftime("%F %T")}] {label}: {gate["coord"]}={val:.3f} below reference {state["gate_reference"]:.3f}'
                  f'{gate.get("min_delta", -0.10):+.2f}; move rejected, back to {state["x"]}', flush=True)
            save_state(cfg, state)
            continue
        elif gate:
            rec['gate'] = 'accepted'
            state['last_accepted_x'] = dict(state['x'])
            old_ref = state.get('gate_reference')
            if old_ref is None or val >= old_ref:
                state['gate_reference'] = val
                state['gate_ref_n'] = 1
            state['trust_scale'] = min(1.0, state.get('trust_scale', 1.0) * 2.0)
        du = (P / (n + 1) ** beta) * (pinv @ (w * r))
        scale = state.get('trust_scale', 1.0)
        du = np.clip(du, -trust * scale, trust * scale)
        new_x = x_of(cfg, np.clip(u_of(cfg, state['x']) + du, lo, hi))
        rec['residual'] = r.tolist()
        rec['du'] = du.tolist()
        rec['next_x'] = new_x
        state['x'] = new_x
        save_state(cfg, state)
        print(f'[{time.strftime("%F %T")}] {label}: |W r| = {np.linalg.norm(w * r):.3f}, next x = {new_x}', flush=True)


def status(cfg: dict, state: dict) -> None:
    print(f'name={state["name"]} n={state.get("n", 0)} x={state["x"]} accepted={state.get("last_accepted_x")}')
    if state.get('G') is not None:
        print_G(cfg, np.array(state['G']))
    if state.get('sigma'):
        print('sigma:', np.round(state['sigma'], 3))
    w = None
    if state.get('sigma'):
        w = np.array([t.get('weight', 1.0) for t in cfg['targets']]) / np.array(state['sigma'])
    cols = targets_of(cfg)
    print(f'{"label":22}' + ''.join(f'{c[:12]:>13}' for c in cols) + '   resolved   |W r|   x')
    for r in state['runs']:
        if r.get('infeasible'):
            print(f'{r["label"]:22} infeasible (trainer refused)   {r["x"]}')
            continue
        res = np.linalg.norm(w * residual_of(cfg, r['y'])) if w is not None else float('nan')
        print(f'{r["label"]:22}' + ''.join(f'{r["y"][c]:13.3f}' for c in cols)
              + f'   {r["y"]["resolution_rate"]:8.3f}   {res:5.2f}   {r["x"]}' + (f'  [{r["gate"]}]' if r.get('gate') else ''))


def final(cfg: dict, state: dict) -> None:
    """The current x on the whole task pool (the deciding comparison against the base's 13 of 76)."""
    pool = [l.strip() for l in (ROOT / cfg['task_pool']).read_text().splitlines() if l.strip() and not l.startswith('#')]
    x = state.get('last_accepted_x') or state['x']
    label = f'final_n{state.get("n", 0):03d}'
    draw(cfg, state, label, x, ids=pool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', type=Path)
    ap.add_argument('mode', choices=['probe', 'iterate', 'status', 'final'])
    ap.add_argument('--steps', type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    state = load_state(cfg)
    {'probe': lambda: probe(cfg, state), 'iterate': lambda: iterate(cfg, state, args.steps),
     'status': lambda: status(cfg, state), 'final': lambda: final(cfg, state)}[args.mode]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
