"""Local evaluation runner that mirrors the Kaggle scoring configuration.

`swegemma eval` does not enable context compaction or read the submission's
eval_config.yaml, both of which the Kaggle scorer does (HARNESS_README 7.1/7.2).
This runner applies them so local numbers are comparable.

Expects an OpenAI-compatible server on --api-base serving the base model as
`gemma-4-31b-it-qat-w4a16-ct` and each LoRA adapter under its directory name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import yaml

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')

import litellm  # noqa: E402
from adk_eval_core.tracing import SessionTrace  # noqa: E402
from adk_submission import discover_adapters  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from swegemma.config import (  # noqa: E402
    ALLOWED_ADAPTER_EXTENSIONS,
    EvalConfig,
    build_submission_limits,
)
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import setup_gemma_model_registry  # noqa: E402
from swegemma.models.discovery import validate_single_declared_model  # noqa: E402

litellm.drop_params = True

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'competition'


def install_sandbox_limits(memory: str, cpu: str) -> None:
    """Run every sandbox command in its own transient cgroup, like Kaggle's 4 GiB / 2 vCPU box.

    The subprocess sandbox has no limits of its own. On 2026-10-11 one agent-run `python`
    reached 41.8 GB RSS and the OOM kill took the whole evaluator scope down. A per-command
    `systemd-run --user --scope` makes a runaway command die alone, with the error returned
    to the agent as it would be on Kaggle.
    """
    import shutil as _shutil

    import swegemma.sandbox.subprocess as sb

    if not _shutil.which('systemd-run'):
        print('sandbox limits: systemd-run not found; running unlimited', flush=True)
        return
    original = sb.execute_subprocess_command

    bus_env = {k: os.environ[k] for k in ('XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS') if k in os.environ}

    def limited(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and command and command[0] == '/bin/bash':
            # systemd-run needs the user bus, which the sandbox's sanitised env omits; give it
            # the two variables and strip them again for the command itself.
            env = dict(kwargs.get('env') or {})
            env.update(bus_env)
            kwargs['env'] = env
            command = ['systemd-run', '--user', '--scope', '--quiet', '-p', f'MemoryMax={memory}',
                       '-p', 'MemorySwapMax=0', '-p', f'CPUQuota={cpu}', '--',
                       'env', *[f'-u{k}' for k in bus_env], *command]
        return original(command, *args, **kwargs)

    sb.execute_subprocess_command = limited


class GoldPatchEvaluator(Evaluator):
    """Grades each task's reference patch instead of running the agent (grader control)."""

    async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, **kwargs):
        return task.patch, None, SessionTrace()


def load_eval_section(submission_dir: Path) -> dict:
    cfg_file = submission_dir / 'eval_config.yaml'
    if not cfg_file.exists():
        return {}
    raw = yaml.safe_load(cfg_file.read_text(encoding='utf-8')) or {}
    return raw.get('evaluation', raw) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submission-dir', type=Path, required=True)
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=DATA)
    ap.add_argument('--tasks', type=Path, default=None, help='defaults to <data-dir>/tasks.jsonl')
    # The bundled wheels include x86_64-only builds; this copy swaps in aarch64 ones.
    ap.add_argument('--wheels-dir', type=Path, default=ROOT / 'data' / 'wheels_aarch64')
    ap.add_argument('--task-ids', nargs='*', default=None)
    ap.add_argument('--task-ids-file', type=Path, default=None, help='one instance_id per line')
    ap.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    ap.add_argument('--sandbox', choices=['docker', 'subprocess'], default='subprocess')
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--compaction-interval', type=int, default=5)
    ap.add_argument('--no-compaction', action='store_true')
    ap.add_argument('--skip-agent-patch', action='store_true', help='control: grade an empty patch')
    ap.add_argument('--gold-patch', action='store_true', help='control: grade the reference patch')
    # Overrides for the submission's eval_config.yaml (Kaggle defaults: 300 / 100 / 60 / 500)
    ap.add_argument('--timeout-seconds', type=int, default=None)
    ap.add_argument('--max-tool-calls', type=int, default=None)
    ap.add_argument('--max-time-minutes', type=float, default=None)
    ap.add_argument('--max-turns', type=int, default=None)
    ap.add_argument('--sandbox-memory', default='4G', help="per-command cgroup MemoryMax ('' disables)")
    ap.add_argument('--sandbox-cpu', default='200%', help='per-command cgroup CPUQuota')
    args = ap.parse_args()
    if args.sandbox and args.sandbox == 'subprocess' and args.sandbox_memory:
        install_sandbox_limits(args.sandbox_memory, args.sandbox_cpu)

    submission_dir = args.submission_dir.resolve()
    ev = load_eval_section(submission_dir)

    def pick(cli_value, key, default):
        return cli_value if cli_value is not None else ev.get(key, default)

    task_ids = list(args.task_ids or [])
    if args.task_ids_file:
        task_ids += [
            line.strip()
            for line in args.task_ids_file.read_text().splitlines()
            if line.strip() and not line.startswith('#')
        ]

    validate_single_declared_model(submission_dir)
    adapters = discover_adapters(str(submission_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    models = setup_gemma_model_registry(api_base=args.api_base, adapter_manifest=adapters)
    limits, gen_constraints = build_submission_limits()

    turns = pick(args.max_turns, 'max_turns', ev.get('max_llm_calls', 500))
    config = EvalConfig(
        tasks_path=args.tasks or args.data_dir / 'tasks.jsonl',
        snapshots_dir=args.data_dir / 'snapshots',
        results_dir=args.results_dir,
        submission_dir=submission_dir,
        models=models,
        sandbox=args.sandbox,
        timeout_seconds=int(pick(args.timeout_seconds, 'timeout_seconds', 300)),
        max_tool_calls=int(pick(args.max_tool_calls, 'max_tool_calls', 100)),
        max_time_minutes=float(pick(args.max_time_minutes, 'max_time_minutes', 60.0)),
        max_turns=int(turns) if turns is not None else None,
        task_ids=task_ids or None,
        skip_agent_patch=args.skip_agent_patch,
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=None
        if args.no_compaction
        else EventsCompactionConfig(
            compaction_interval=args.compaction_interval,
            overlap_size=2,
            token_threshold=14336,
            event_retention_size=5,
        ),
        graph_dir=str(args.data_dir / 'graphs'),
        embeddings_dir=str(args.data_dir / 'embeddings'),
        wheels_dir=args.wheels_dir,
        concurrency=args.concurrency,
        display_mode='quiet',
    )

    started = time.time()
    evaluator_cls = GoldPatchEvaluator if args.gold_patch else Evaluator
    result = asyncio.run(evaluator_cls(config).run())
    elapsed = time.time() - started
    summary = {
        'resolved': result.resolved,
        'total': result.total,
        'resolution_rate': result.resolution_rate,
        'wall_seconds': round(elapsed, 1),
        'budget': {
            'max_time_minutes': config.budget.time_minutes,
            'max_tool_calls': config.budget.tool_calls,
            'max_turns': config.budget.turns,
        },
    }
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
