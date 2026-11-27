"""Run `scripts/run_eval.py` under resource limits and a server watchdog.

Why: on 2026-10-05 the vLLM server's generation rate collapsed from ~40 to
2-9 tokens/s fifteen minutes into a 100-task run at concurrency 8 and stayed
there for ten hours; both overnight runs were lost (IMPLEMENTATION.md 6.10).
The local subprocess sandbox has no CPU or memory limit (Kaggle's has 2 vCPU
and 4 GiB), so eight concurrent `pytest` runs can starve the engine process.

This launcher
  * runs the evaluator inside a `systemd-run --user --scope` with a memory
    and CPU cap, at lower scheduling priority than the server;
  * polls the server log once a minute, logging generation rate, running
    requests, MemAvailable, swap and load average to logs/guard_<name>.log;
  * kills the whole run (process group) if the server stays degraded for
    --grace-minutes while requests are running, and exits with code 3.

Usage:
  .venv/bin/python scripts/run_guarded.py --name noise_a -- \
      --submission-dir submissions/v0_long --results-dir runs/noise_a ...
Everything after `--` is passed to run_eval.py unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_RE = re.compile(
    r'(\d\d-\d\d \d\d:\d\d:\d\d).*prompt throughput: ([\d.]+) tokens/s.*'
    r'generation throughput: ([\d.]+) tokens/s.*Running: (\d+) reqs.*Waiting: (\d+) reqs'
)


def meminfo() -> tuple[float, float]:
    vals = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':', 1)
        vals[k] = int(v.split()[0]) / 1024 / 1024  # GiB
    return vals['MemAvailable'], vals['SwapTotal'] - vals['SwapFree']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='tag for the guard log')
    ap.add_argument('--server-log', type=Path, default=ROOT / 'logs' / 'vllm_server.log')
    ap.add_argument('--memory-max', default='40G', help='cgroup MemoryMax for the evaluator and its sandboxes')
    ap.add_argument('--cpu-quota', default='1200%', help='cgroup CPUQuota (1200%% = 12 cores)')
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--min-gen-tps', type=float, default=4.0, help='aggregate generation rate considered degraded')
    ap.add_argument('--max-prompt-tps', type=float, default=400.0, help='ignore samples with heavy prefill')
    ap.add_argument('--grace-minutes', type=int, default=6, help='consecutive degraded samples before killing')
    ap.add_argument('--no-limits', action='store_true', help='skip systemd-run (plain subprocess)')
    ap.add_argument('--runner', default='run_eval.py', help='runner script in scripts/ (run_eval_v2.py for the 0.2.11 harness)')
    ap.add_argument('--python', default=str(ROOT / '.venv' / 'bin' / 'python'),
                    help='interpreter for the runner; its environment is what task sandboxes see (.venv-heldout for held-out tasks)')
    ap.add_argument('eval_args', nargs=argparse.REMAINDER, help='arguments for run_eval.py after --')
    args = ap.parse_args()

    eval_args = args.eval_args[1:] if args.eval_args[:1] == ['--'] else args.eval_args
    if not eval_args:
        ap.error('pass run_eval.py arguments after --')

    guard_log = ROOT / 'logs' / f'guard_{args.name}.log'
    runner = [args.python, str(ROOT / 'scripts' / args.runner), *eval_args]
    cmd = ['nice', '-n', str(args.nice), *runner]
    if not args.no_limits:
        cmd = [
            'systemd-run', '--user', '--scope', '--quiet',
            '-p', f'MemoryMax={args.memory_max}', '-p', f'CPUQuota={args.cpu_quota}',
            # an OOM kill inside the scope must not take the whole batch down (2026-10-11)
            '-p', 'OOMPolicy=continue',
            '--', *cmd,
        ]

    # New session: the evaluator, its sandboxes and their pytest children share one
    # process group, so one killpg stops everything.
    proc = subprocess.Popen(cmd, start_new_session=True, cwd=ROOT)

    with guard_log.open('a') as g:
        g.write(f'# {time.strftime("%Y-%m-%d %H:%M:%S")} start pid={proc.pid} cmd={" ".join(eval_args)}\n')
        g.write('# time  gen_tps  prompt_tps  running  waiting  mem_avail_GiB  swap_used_GiB  load1  degraded_streak\n')
        g.flush()
        streak = 0
        last_pos = args.server_log.stat().st_size if args.server_log.exists() else 0
        last_sample = None
        rc = None
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(60)
            # read new server-log lines and keep the latest throughput sample
            if args.server_log.exists():
                size = args.server_log.stat().st_size
                if size < last_pos:
                    last_pos = 0  # log rotated or server restarted
                with args.server_log.open('rb') as f:
                    f.seek(last_pos)
                    chunk = f.read().decode('utf-8', 'replace')
                    last_pos = f.tell()
                for m in LOG_RE.finditer(chunk):
                    last_sample = m
            avail, swap = meminfo()
            load1 = os.getloadavg()[0]
            if last_sample is None:
                g.write(f'{time.strftime("%H:%M:%S")}  -  -  -  -  {avail:.1f}  {swap:.2f}  {load1:.1f}  {streak}\n')
                g.flush()
                continue
            _, prompt_tps, gen_tps, running, waiting = last_sample.groups()
            prompt_tps, gen_tps, running, waiting = float(prompt_tps), float(gen_tps), int(running), int(waiting)
            degraded = running >= 1 and gen_tps < args.min_gen_tps and prompt_tps < args.max_prompt_tps
            streak = streak + 1 if degraded else 0
            g.write(
                f'{time.strftime("%H:%M:%S")}  {gen_tps:.1f}  {prompt_tps:.1f}  {running}  {waiting}  '
                f'{avail:.1f}  {swap:.2f}  {load1:.1f}  {streak}\n'
            )
            g.flush()
            last_sample = None  # a sample counts once; a silent log means no new evidence
            if streak >= args.grace_minutes:
                g.write(f'# {time.strftime("%Y-%m-%d %H:%M:%S")} DEGRADED for {streak} min: killing process group {proc.pid}\n')
                g.flush()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    for _ in range(30):
                        if proc.poll() is not None:
                            break
                        time.sleep(1)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                print(f'WATCHDOG_KILLED name={args.name} streak={streak}', flush=True)
                return 3
        g.write(f'# {time.strftime("%Y-%m-%d %H:%M:%S")} runner exited rc={rc}\n')
    print(f'RUNNER_EXIT name={args.name} rc={rc}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
