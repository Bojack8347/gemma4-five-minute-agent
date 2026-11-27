# Held-out task sets

Three task sets in the competition harness's format, mined from 2026 commits of less-known, actively maintained, pure-Python repositories (`scripts/heldout/mine_commits.py`, `scripts/heldout/build_tasks.py`). Each record in `tasks.jsonl` has the instance id, the repository, the base commit, the gold patch (the commit's source diff), the test patch (its test diff), the problem statement and the tests to run. `sound_tasks.txt` lists the tasks that grade soundly in our environment: the gold patch passes and an empty patch fails.

| set | built | sound | statement |
|---|---|---|---|
| `heldout` | 120 | 95 | the commit message (title and body) |
| `heldout_hard` | 106 | 76 | the commit message; fixes of at least 40 source lines or two source files |
| `heldout_hard_terse` | 106 | 76 | the same tasks with the statement cut to the commit's title |
| `heldout_hard_f025`, `heldout_hard_f050` | 106 | 76 | the title followed by the first quarter or half of the commit body's lines (`scripts/heldout/build_frac_tasks.py 0.25 0.5`; the proxy calibration of notes section 6.40) |

Snapshots (the repository tree at the base commit) are not included; they are about 250 MB per set and are rebuilt from the public repositories:

```
python scripts/heldout/build_tasks.py --candidates tasks/heldout/candidates.jsonl --out-dir data/heldout
python scripts/heldout/build_tasks.py --candidates tasks/heldout/candidates_wide.jsonl --out-dir data/heldout_hard --hard
python scripts/heldout/build_tasks.py --candidates tasks/heldout/candidates_wide.jsonl --out-dir data/heldout_hard_terse --hard --terse --snapshots-from data/heldout_hard
```

The build clones each repository once (blobless) and checks the mined commits out; the commits are listed in `candidates*.jsonl`. The patches and statements derive from these repositories and remain under their licenses:

| repository | license | sound tasks (easy) | sound tasks (hard) |
|---|---|---|---|
| jd/tenacity | Apache-2.0 | 9 | 10 |
| getsentry/responses | Apache-2.0 | 9 | 2 |
| jpadilla/pyjwt | MIT | 10 | 5 |
| wntrblm/nox | Apache-2.0 | 7 | 4 |
| hgrecco/pint | BSD-3-Clause (repository LICENSE; GitHub reports no SPDX id) | 9 | 8 |
| pyparsing/pyparsing | MIT | 6 | 6 |
| scrapy/w3lib | BSD-3-Clause | 9 | 3 |
| simonw/sqlite-utils | Apache-2.0 | 10 | 8 |
| pydantic/pydantic-settings | MIT | 8 | 3 |
| mahmoud/boltons | BSD-3-Clause (repository LICENSE; GitHub reports no SPDX id) | 10 | 7 |
| msiemens/tinydb | MIT | 8 | 2 |
| executablebooks/markdown-it-py | MIT | 0 | 6 |
| pydantic/pydantic-extra-types | MIT | 0 | 8 |
| more-itertools/more-itertools | MIT | 0 | 1 |
| python-jsonschema/jsonschema | MIT | 0 | 3 |
