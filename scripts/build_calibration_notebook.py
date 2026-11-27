"""Build the private Kaggle notebook that calibrates local results against the scoring hardware.

Question it answers: is the drop from 31-38% (local, public tasks) to 0.13 (leaderboard, hidden
tasks) caused by the scoring environment or by the hidden tasks? It runs, on Kaggle's 4x L4 with
the current official wheelhouse and the starter notebook's server and evaluator settings:

  1. a gold-patch control on 30 public sound tasks (does Kaggle grade them correctly?);
  2. our v0_long agent with the 31B, the five-minute cap, tasks one at a time like the scorer;
  3. the same agent with the 12B (allowed since 2026-10-13), plus a raw speed probe of each model.

Everything goes to /kaggle/working (calibration.json, results_31b/, results_12b/).
Usage: .venv/bin/python scripts/build_calibration_notebook.py   -> kaggle_nb/calibration/
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'kaggle_nb' / 'calibration'
STARTER = sorted((ROOT / 'kaggle_nb' / 'starter_2026-10-17').glob('*.ipynb'))
SUB = ROOT / 'submissions' / 'v0_long'


def starter_cell(i: int) -> str:
    nb = json.loads(STARTER[0].read_text())
    return ''.join(nb['cells'][i]['source'])


CELL_RESULTS = r'''
import json, time
T0 = time.time()
CALIB = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'models': {}}
CALIB_PATH = Path('/kaggle/working/calibration.json')

def save():
    CALIB['elapsed_seconds'] = round(time.time() - T0, 1)
    CALIB_PATH.write_text(json.dumps(CALIB, indent=2))

save()
'''

CELL_DATA = r'''
from swegemma.models import load_tasks

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORKING_DIR = Path('/kaggle/working')
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
GRAPH_DIR = str(DATA_DIR / 'graphs')
EMBEDDINGS_DIR = str(DATA_DIR / 'embeddings')
tasks = load_tasks(TASKS_PATH)
TASK_IDS = __TASK_IDS__
SELECTED = [t for t in tasks if t.instance_id in TASK_IDS]
assert len(SELECTED) == len(TASK_IDS), (len(SELECTED), len(TASK_IDS))

FILES = __FILES__


def write_submission(directory: Path, model_name: str) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    for rel, text in FILES.items():
        path = directory / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == 'agent.yaml':
            text = text.replace('model: gemma-4-31b-it-qat-w4a16-ct', f'model: {model_name}')
        path.write_text(text, encoding='utf-8')
    return directory


SUBS = {
    '31b': write_submission(WORKING_DIR / 'sub_31b', 'gemma-4-31b-it-qat-w4a16-ct'),
    '12b': write_submission(WORKING_DIR / 'sub_12b', 'gemma-4-12b-it-qat-w4a16-ct'),
}
CALIB['task_ids'] = TASK_IDS
save()
print(len(SELECTED), 'tasks;', {k: (v / 'agent.yaml').read_text().splitlines()[1] for k, v in SUBS.items()})
'''

CELL_EVAL_HELPERS = r'''
import asyncio
import concurrent.futures
import yaml
from adk_eval_core.tracing import SessionTrace
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator


def run_sync(coro_or_fn, *args, **kwargs):
    if callable(coro_or_fn) and not asyncio.iscoroutine(coro_or_fn):
        fn = lambda: coro_or_fn(*args, **kwargs)
    else:
        fn = lambda: coro_or_fn
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(fn())).result()
    return asyncio.run(fn())


class GoldEvaluator(Evaluator):
    """Grades each task's reference patch instead of running the agent (grader control)."""

    async def _run_agent_sandbox(self, task, snapshot_path, *args, **kwargs):
        return task.patch, None, SessionTrace()


def make_eval_config(sub_dir: Path, models, adapters, results_dir: Path) -> EvalConfig:
    ev = yaml.safe_load((sub_dir / 'eval_config.yaml').read_text())['evaluation']
    limits, gen_constraints = build_submission_limits()
    # Starter-notebook / HARNESS_README (2026-10-13) settings: compaction every 15 events.
    return EvalConfig(
        tasks_path=TASKS_PATH,
        snapshots_dir=DATA_DIR / 'snapshots',
        results_dir=results_dir,
        submission_dir=sub_dir,
        models=models,
        sandbox='subprocess',
        timeout_seconds=int(ev.get('timeout_seconds', 300)),
        verification_timeout_seconds=int(ev.get('verification_timeout_seconds', 300)),
        max_time_minutes=float(ev.get('max_time_minutes', 60.0)),
        max_tool_calls=int(ev.get('max_tool_calls', 100)),
        max_turns=int(ev['max_turns']) if ev.get('max_turns') is not None else None,
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5,
        ),
        graph_dir=GRAPH_DIR,
        embeddings_dir=EMBEDDINGS_DIR,
        wheels_dir=DATA_DIR / 'wheels',
        verbose=False,
    )


def run_tasks(evaluator, label: str) -> list[dict]:
    rows = []
    CALIB.setdefault('runs', {})[label] = rows
    for idx, task in enumerate(SELECTED, start=1):
        t0 = time.time()
        try:
            r = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(SELECTED))
            row = {'instance_id': task.instance_id, 'resolved': bool(r.resolved),
                   'wall_seconds': round(time.time() - t0, 1),
                   'duration_seconds': round(float(getattr(r, 'duration_seconds', 0) or 0), 1),
                   'tool_calls': getattr(r, 'tool_calls', None), 'llm_calls': getattr(r, 'total_llm_calls', None),
                   'patch_chars': len(getattr(r, 'agent_patch', '') or ''), 'error': (getattr(r, 'error', '') or '')[:200],
                   'test_exit_code': getattr(r, 'test_exit_code', None)}
            if not row['resolved']:
                row['test_output_tail'] = (getattr(r, 'test_output', '') or '')[-1500:]
        except Exception as e:
            row = {'instance_id': task.instance_id, 'resolved': False, 'wall_seconds': round(time.time() - t0, 1),
                   'exception': repr(e)[:300]}
        rows.append(row)
        save()
        print(label, idx, row)
    return rows
'''

CELL_GOLD = r'''
# 1. Gold-patch control: does Kaggle grade these tasks? (no model needed)
from adk_submission import discover_adapters
from swegemma.cli import _build_cli_model_registry
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

t0 = time.time()
gold_models = _build_cli_model_registry(SUBS['31b'])
gold_adapters = discover_adapters(str(SUBS['31b']), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gold_eval = GoldEvaluator(make_eval_config(SUBS['31b'], gold_models, gold_adapters, WORKING_DIR / 'results_gold'))
gold_rows = run_tasks(gold_eval, 'gold')
CALIB['gold_resolved'] = sum(r['resolved'] for r in gold_rows)
CALIB['gold_seconds'] = round(time.time() - t0, 1)
save()
print('gold control:', CALIB['gold_resolved'], '/', len(gold_rows))
'''

CELL_SERVER_AND_RUN = r'''
# 2-3. For each model: start the server exactly as the starter does, probe speed, run the agent.
import urllib.request

import litellm
import torch
from adk_submission import VllmConfig, VllmServer
from adk_submission.discovery import discover_declared_models

litellm.drop_params = True
MODEL_VLLM_EXTRA_ARGS = {
    'gemma-4-12b-it-qat-w4a16-ct': (
        '--hf-overrides',
        '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
    ),
}
INFERENCE_API_KEY = os.environ.get('LOCAL_API_KEY', 'EMPTY')

__SPEED_PROBE__


def run_model(tag: str) -> None:
    sub_dir = SUBS[tag]
    declared = next(iter(discover_declared_models(sub_dir)))
    model_path = Path(f'/kaggle/input/models/google/gemma-4/other/{declared}/2')
    adapters = discover_adapters(str(sub_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, 'adapters', None) or {}
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
    cfg = VllmConfig(
        model=str(model_path), port=8000, host='127.0.0.1',
        tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768,
        dtype='bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'auto',
        gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
        enable_lora=bool(adapter_map), max_loras=max(1, len(adapter_map)), max_lora_rank=128,
        scheduler_reserve_full_isl=False, tensor_parallel_size=tp_size, startup_timeout=60 * 20,
        extra_args=MODEL_VLLM_EXTRA_ARGS.get(declared, ()),
    )
    info = CALIB['models'].setdefault(tag, {'declared': declared})
    server = VllmServer(cfg, adapter_manifest=adapters)
    t0 = time.time()
    server.start()
    info['vllm_startup_seconds'] = round(time.time() - t0, 1)
    save()
    try:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=30) as resp:
                served_ids = [m['id'] for m in json.loads(resp.read())['data']]
            info['served_model_ids'] = served_ids
            base_ids = [m for m in served_ids if m not in set(adapter_map)]
            info['speed'] = probe(model=base_ids[0])
        except Exception as e:  # the probe is a side measurement; never let it stop the agent runs
            info['speed_error'] = repr(e)[:300]
        save()
        models = server.create_model_registry(aliases=[declared], model_prefix='openai/', api_key=INFERENCE_API_KEY)
        evaluator = Evaluator(make_eval_config(sub_dir, models, adapters, WORKING_DIR / f'results_{tag}'))
        t0 = time.time()
        rows = run_tasks(evaluator, tag)
        info['agent_seconds'] = round(time.time() - t0, 1)
        info['resolved'] = sum(r['resolved'] for r in rows)
        gradable = {g['instance_id'] for g in CALIB['runs'].get('gold', []) if g['resolved']}
        info['resolved_on_gradable'] = sum(r['resolved'] for r in rows if r['instance_id'] in gradable)
        info['gradable'] = len(gradable)
        save()
    finally:
        server.stop()
        time.sleep(20)


for tag in ('31b', '12b'):
    try:
        run_model(tag)
    except Exception as e:
        CALIB['models'].setdefault(tag, {})['exception'] = repr(e)[:500]
        save()
        print(tag, 'failed:', e)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'speed'} for k, v in CALIB['models'].items()}, indent=2))
'''


def speed_probe_source() -> str:
    """The stdlib-only probe from scripts/speed_probe.py, minus its __main__ block."""
    src = (ROOT / 'scripts' / 'speed_probe.py').read_text()
    src = src.split("if __name__ == '__main__':")[0]
    body = [l for l in src.splitlines() if not l.startswith(('from __future__',))]
    return '\n'.join(body)


def code(src: str) -> dict:
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': src.strip('\n').splitlines(keepends=True)}


def main() -> None:
    files = {rel: (SUB / rel).read_text() for rel in ('agent.yaml', 'prompts/system.md', 'configs/sampling.yaml', 'eval_config.yaml')}
    task_ids = [l.strip() for l in (ROOT / 'data' / 'kaggle_calib_tasks.txt').read_text().splitlines() if l.strip()]
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Calibration: local results against the scoring hardware\n']},
        code(starter_cell(2)),
        code(CELL_RESULTS),
        code(CELL_DATA.replace('__TASK_IDS__', json.dumps(task_ids)).replace('__FILES__', json.dumps(files, indent=1))),
        code(CELL_EVAL_HELPERS),
        code(CELL_GOLD),
        code(CELL_SERVER_AND_RUN.replace('__SPEED_PROBE__', speed_probe_source())),
    ]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'calibration.ipynb').write_text(json.dumps(nb, indent=1))
    meta = json.loads((sorted((ROOT / 'kaggle_nb' / 'starter_2026-10-17').glob('kernel-metadata.json'))[0]).read_text())
    meta.update({'id': 'bojackkk/gemma4-calibration', 'title': 'gemma4-calibration', 'code_file': 'calibration.ipynb',
                 'is_private': True, 'enable_internet': False, 'keywords': []})
    meta['model_sources'] = ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2', 'google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2']
    meta.pop('id_no', None)
    (OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print('wrote', OUT, '|', len(task_ids), 'tasks |', len(cells), 'cells')


if __name__ == '__main__':
    main()
