"""Package a local submission directory as a competition `submission.zip` and validate it.

Stages a copy, drops files the harness does not allow (adapter READMEs, merge manifests),
runs the harness's own `validate_directory` with the official limits, compiles the agent
tree with `compile_submission` (so a bad YAML or adapter reference fails here, not on
Kaggle), checks the 3 GiB total, and zips with `agent.yaml` at the root.

Usage:
  .venv/bin/python scripts/build_submission.py submissions/v0_long_format1 --out bundles/format1.zip
Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('submission_dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from adk_submission import compile_submission, discover_adapters, validate_directory
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, build_submission_limits
    from swegemma.models import setup_gemma_model_registry
    from swegemma.tools import create_tools

    limits, gen_constraints = build_submission_limits()
    allowed = set(limits.allowed_file_extensions)
    with tempfile.TemporaryDirectory(dir=str(ROOT / 'bundles')) as tmp:
        stage = Path(tmp) / 'submission'
        shutil.copytree(args.submission_dir, stage)
        dropped = []
        for f in list(stage.rglob('*')):
            if f.is_file() and (f.suffix not in allowed or f.name == 'merge.json' or f.name.startswith('.')):
                dropped.append(str(f.relative_to(stage)))
                f.unlink()
        total = sum(f.stat().st_size for f in stage.rglob('*') if f.is_file())
        print(f'staged {args.submission_dir} -> {total / 2**30:.2f} GiB (limit {limits.max_total_size_bytes / 2**30:.0f} GiB); dropped {dropped or "nothing"}')
        sub = validate_directory(stage, limits=limits)
        print('validate_directory: ok')
        adapters = discover_adapters(str(stage), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        models = setup_gemma_model_registry(api_base='http://127.0.0.1:8000/v1', adapter_manifest=adapters)
        agent = compile_submission(stage, tool_registry=create_tools(None), model_registry=models,
                                   limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters)
        names = [a.name for a in getattr(adapters, 'adapters', {}).values()] if hasattr(adapters, 'adapters') else list(getattr(adapters, 'adapters', {}) or {})
        print(f'compile_submission: ok (root agent {agent.name!r}; adapters {list(adapters.adapters) if adapters and getattr(adapters, "adapters", None) else []})')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(stage.rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(stage))
        print(f'wrote {args.out} ({args.out.stat().st_size / 2**20:.0f} MiB)')
        with zipfile.ZipFile(args.out) as z:
            top = sorted({n.split('/')[0] for n in z.namelist()})
            assert 'agent.yaml' in top, 'agent.yaml must be at the zip root'
            print('zip root entries:', top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
