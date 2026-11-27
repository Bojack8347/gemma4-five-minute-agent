"""Build the private Kaggle notebook that measures real L4x4 timing.

It follows the official starter notebook (same wheel install, same vLLM config, same
Evaluator call) and adds: server startup time, the raw speed probe from
scripts/speed_probe.py, and the sample agent on a few public tasks with a 5-minute cap.
Everything is written to /kaggle/working so `kaggle kernels output` can fetch it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'kaggle_nb' / 'timing_probe'
STARTER_META = ROOT / 'data' / 'notebooks' / 'starter' / 'kernel-metadata.json'

PROBE_TASK_IDS = [
    'fastapi_14258', 'fastapi_14479', 'fastapi_15589',
    'rich_3470', 'rich_3676', 'rich_4077',
]

CELL_ENV = r'''
import glob, importlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

T_NOTEBOOK_START = time.time()
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '1200'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')

for pth_pattern in (
    '/usr/local/lib/python*/dist-packages/*cutlass*.pth',
    '/usr/local/lib/python*/site-packages/*cutlass*.pth',
):
    for pth in glob.glob(pth_pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass

tmp_whl = Path('/tmp/wheelhouse')
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    target_name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    target = tmp_whl / target_name
    if not target.exists():
        os.symlink(w, target)

wheels = sorted(str(w) for w in tmp_whl.glob('*.whl'))
t0 = time.time()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

WORKING_DIR = Path('/kaggle/working')
SUMMARY_PATH = WORKING_DIR / 'timing_summary.json'
TIMING = {'wheel_install_seconds': round(time.time() - t0, 1), 'python': sys.version.split()[0]}

def save_timing():
    TIMING['notebook_elapsed_seconds'] = round(time.time() - T_NOTEBOOK_START, 1)
    SUMMARY_PATH.write_text(json.dumps(TIMING, indent=2))

save_timing()
print(TIMING)
'''

CELL_DATA = r'''
from swegemma.models import load_tasks

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
AGENT_DIR = WORKING_DIR / 'sample_submission'
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
shutil.copytree(DATA_DIR / 'sample_submission', AGENT_DIR)

(AGENT_DIR / 'configs' / 'sampling.yaml').write_text(
    'temperature: 0.2\n'
    'top_p: 0.95\n'
    'max_output_tokens: 16384\n'
    'thinking_config:\n'
    '  thinking_budget: 4096\n'
    '  include_thoughts: true\n',
    encoding='utf-8',
)
(AGENT_DIR / 'eval_config.yaml').write_text(
    'evaluation:\n'
    '  timeout_seconds: 300\n'
    '  max_tool_calls: 100\n'
    '  max_time_minutes: 5\n'
    '  max_turns: 250\n',
    encoding='utf-8',
)

TASKS_PATH = DATA_DIR / 'tasks.jsonl'
tasks = load_tasks(TASKS_PATH)
PROBE_TASK_IDS = __PROBE_TASK_IDS__
probe_tasks = [t for t in tasks if t.instance_id in PROBE_TASK_IDS]
print(len(tasks), 'tasks;', [t.instance_id for t in probe_tasks])
'''

CELL_SERVER = r'''
import litellm
import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
from swegemma.models.discovery import validate_single_declared_model

litellm.drop_params = True

TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')

declared_model = validate_single_declared_model(AGENT_DIR)
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
TIMING['gpus'] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
TIMING['tensor_parallel_size'] = tp_size

# gpu_memory_utilization 0.80 is what HARNESS_README section 3.1 states for the scorer.
vllm_cfg = VllmConfig(
    model=str(MODEL_PATH),
    port=8000,
    host='127.0.0.1',
    tool_call_parser='gemma4',
    reasoning_parser='gemma4',
    default_chat_template_kwargs={'enable_thinking': True},
    max_model_len=32768,
    dtype='bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'auto',
    gpu_memory_utilization=0.80,
    enable_auto_tool_choice=True,
    enable_lora=True,
    max_loras=8,
    max_lora_rank=128,
    tensor_parallel_size=tp_size,
    startup_timeout=60 * 20,
)
server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
t0 = time.time()
server_instance.start()
TIMING['vllm_startup_seconds'] = round(time.time() - t0, 1)

try:
    log_text = Path(server_instance.log_path).read_text(errors='replace')
    TIMING['vllm_log_facts'] = [
        line.strip()[-160:] for line in log_text.splitlines()
        if any(k in line for k in ('GPU KV cache size', 'Maximum concurrency', 'Model loading took', 'Available KV cache memory'))
    ][:12]
except Exception as e:
    TIMING['vllm_log_facts'] = f'unavailable: {e}'

models = server_instance.create_model_registry(
    aliases=[declared_model, TARGET_MODEL_NAME],
    model_prefix='openai/',
    api_key='EMPTY',
)
save_timing()
print(json.dumps(TIMING, indent=2))
'''

CELL_PROBE_PREFIX = r'''
# ---- raw serving speed (same code as scripts/speed_probe.py in the local repo) ----
'''

CELL_PROBE_SUFFIX = r'''
import urllib.request as _ur
with _ur.urlopen('http://127.0.0.1:8000/v1/models', timeout=30) as _resp:
    served_ids = [m['id'] for m in json.loads(_resp.read())['data']]
adapter_names = set(adapters.adapters.keys()) if adapters and getattr(adapters, 'adapters', None) else set()
base_ids = [m for m in served_ids if m not in adapter_names]
TIMING['served_model_ids'] = served_ids

TIMING['speed_base'] = probe(model=base_ids[0])
save_timing()
if adapter_names:
    lora_name = sorted(adapter_names)[0]
    TIMING['speed_lora_route'] = probe(model=lora_name)
    TIMING['speed_lora_route']['adapter'] = lora_name
save_timing()
print(json.dumps({k: TIMING[k] for k in TIMING if k.startswith('speed')}, indent=2))
'''

CELL_AGENT = r'''
import asyncio
import concurrent.futures
import yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator


def run_sync(fn, *args, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(fn(*args, **kwargs))).result()
    return asyncio.run(fn(*args, **kwargs))


ev = yaml.safe_load((AGENT_DIR / 'eval_config.yaml').read_text())['evaluation']
limits, gen_constraints = build_submission_limits()
eval_config = EvalConfig(
    tasks_path=TASKS_PATH,
    snapshots_dir=DATA_DIR / 'snapshots',
    results_dir=WORKING_DIR / 'results',
    submission_dir=AGENT_DIR,
    models=models,
    sandbox='subprocess',
    timeout_seconds=int(ev['timeout_seconds']),
    max_time_minutes=float(ev['max_time_minutes']),
    max_tool_calls=int(ev['max_tool_calls']),
    max_turns=int(ev['max_turns']),
    limits=limits,
    generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5, overlap_size=2, token_threshold=14336, event_retention_size=5
    ),
    graph_dir=str(DATA_DIR / 'graphs'),
    embeddings_dir=str(DATA_DIR / 'embeddings'),
    wheels_dir=DATA_DIR / 'wheels',
    verbose=False,
    display_mode='quiet',
)
evaluator = Evaluator(eval_config)

TIMING['tasks'] = []
for idx, task in enumerate(probe_tasks, start=1):
    t0 = time.time()
    try:
        result = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(probe_tasks))
        row = {
            'instance_id': task.instance_id,
            'wall_seconds': round(time.time() - t0, 1),
            'harness_duration_seconds': round(float(result.duration_seconds or 0), 1),
            'resolved': bool(result.resolved),
            'tool_calls': result.tool_calls,
            'llm_calls': result.total_llm_calls,
            'patch_chars': len(result.agent_patch or ''),
            'error': (result.error or '')[:200],
        }
    except Exception as e:  # keep going: timing of the remaining tasks is still useful
        row = {'instance_id': task.instance_id, 'wall_seconds': round(time.time() - t0, 1), 'exception': repr(e)[:300]}
    TIMING['tasks'].append(row)
    save_timing()
    print(row)
'''

CELL_END = r'''
try:
    server_instance.stop()
except Exception as e:
    print('server stop:', e)
save_timing()
print(json.dumps(TIMING, indent=2))
'''


def code_cell(src: str) -> dict:
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': src.strip('\n') + '\n',
    }


def main() -> None:
    probe_src = (ROOT / 'scripts' / 'speed_probe.py').read_text()
    # Drop the module docstring's __future__ import position issue and the CLI entry point.
    probe_body = probe_src.split("if __name__ == '__main__':")[0]
    probe_body = probe_body.replace('from __future__ import annotations\n', '')

    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': '# Gemma 4 agent: L4x4 timing probe\nPrivate measurement notebook. No submission is produced.'},
        code_cell(CELL_ENV),
        code_cell(CELL_DATA.replace('__PROBE_TASK_IDS__', repr(PROBE_TASK_IDS))),
        code_cell(CELL_SERVER),
        code_cell(CELL_PROBE_PREFIX + probe_body + CELL_PROBE_SUFFIX),
        code_cell(CELL_AGENT),
        code_cell(CELL_END),
    ]
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }

    starter = json.loads(STARTER_META.read_text())
    meta = {
        'id': 'bojackkk/gemma4-timing-probe',
        'title': 'gemma4-timing-probe',
        'code_file': 'timing_probe.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': False,
        'keywords': [],
        'dataset_sources': starter['dataset_sources'],
        'kernel_sources': [],
        'competition_sources': starter['competition_sources'],
        'model_sources': starter['model_sources'],
        'docker_image': starter.get('docker_image', ''),
        'machine_shape': starter.get('machine_shape', 'NvidiaL4'),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'timing_probe.ipynb').write_text(json.dumps(nb, indent=1))
    (OUT_DIR / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print('wrote', OUT_DIR)


if __name__ == '__main__':
    main()
