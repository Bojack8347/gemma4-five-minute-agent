"""Write the FILES dictionary of a public notebook's bundle cell to a submission folder, without running the cell."""
import ast, hashlib, json, sys
from pathlib import Path
nb = json.load(open(sys.argv[1])); out = Path(sys.argv[2])
files = None
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    tree = ast.parse(''.join(c['source']))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'FILES' for t in node.targets):
            files = ast.literal_eval(node.value)
assert isinstance(files, dict) and set(files) == {'agent.yaml', 'configs/sampling.yaml', 'eval_config.yaml', 'prompts/system.md'}, files and list(files)
for name, text in files.items():
    p = out / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
print('files_sha256', hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16], {k: len(v) for k, v in files.items()})
