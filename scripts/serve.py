"""Start the local vLLM server with the same flags the Kaggle scorer uses.

The command line is built by the harness's own `VllmServer.build_cmd()` (so LoRA
flags and rank rounding match), then run with the patched vLLM in `.venv-vllm`.
Run with the harness interpreter:  .venv/bin/python scripts/serve.py [--submission-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=MODEL_NAME, help='served model name; 31b/12b/e4b/e2b are accepted as shorthands')
    ap.add_argument('--model-path', type=Path, default=None, help='defaults to models/<model name>')
    ap.add_argument('--submission-dir', type=Path, default=None, help='mount its adapters/ as LoRA modules')
    ap.add_argument('--port', type=int, default=8000)
    # Kaggle uses 0.80 of 4x24 GB; this machine has one 121 GB unified pool.
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.55)
    ap.add_argument('--vllm-python', type=Path, default=ROOT / '.venv-vllm' / 'bin' / 'python')
    ap.add_argument('--print-only', action='store_true')
    args, extra = ap.parse_known_args()
    short = {'31b': 'gemma-4-31b-it-qat-w4a16-ct', '12b': 'gemma-4-12b-it-qat-w4a16-ct',
             'e4b': 'gemma-4-e4b-it-qat-w4a16-ct', 'e2b': 'gemma-4-e2b-it-qat-w4a16-ct'}
    model_name = short.get(args.model, args.model)
    if args.model_path is None:
        args.model_path = ROOT / 'models' / model_name
    # Same per-model flags as the scorer's MODEL_VLLM_EXTRA_ARGS (HARNESS_README 3, 2026-10-13).
    if model_name == 'gemma-4-12b-it-qat-w4a16-ct':
        extra = ['--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}', *extra]

    adapters = None
    if args.submission_dir is not None:
        adapters = discover_adapters(
            str(args.submission_dir.resolve()), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS
        )

    cfg = VllmConfig(
        model=str(args.model_path),
        port=args.port,
        host='127.0.0.1',
        tool_call_parser='gemma4',
        reasoning_parser='gemma4',
        default_chat_template_kwargs={'enable_thinking': True},
        max_model_len=32768,
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_auto_tool_choice=True,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=128,
        tensor_parallel_size=1,
        extra_args=['--served-model-name', model_name, *extra],
    )
    cmd = VllmServer(cfg, adapter_manifest=adapters).build_cmd()
    cmd[0] = str(args.vllm_python)

    print(shlex.join(cmd), flush=True)
    if args.print_only:
        return 0

    env = dict(os.environ)
    env.setdefault('VLLM_NO_USAGE_STATS', '1')
    env.setdefault('VLLM_ENGINE_READY_TIMEOUT_S', '1200')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # Triton compiles a C launcher at startup and needs Python.h. python3.12-dev is not
    # installed system-wide here, so point gcc at headers unpacked under tools/pydev.
    local_inc = ROOT / 'tools' / 'pydev' / 'root' / 'usr' / 'include'
    if (local_inc / 'python3.12' / 'Python.h').exists():
        extra_inc = f'{local_inc / "python3.12"}:{local_inc}'
        env['CPATH'] = f'{extra_inc}:{env["CPATH"]}' if env.get('CPATH') else extra_inc

    # exec keeps this PID, so the pid file identifies the server process.
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'logs' / 'vllm_server.pid').write_text(str(os.getpid()))
    os.execve(cmd[0], cmd, env)


if __name__ == '__main__':
    sys.exit(main())
