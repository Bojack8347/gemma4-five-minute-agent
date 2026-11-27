# Gemm4Agent: implementation guide

State as of 2026-10-05, morning. This explains what has been built in this folder, how each piece works, and what the measurements so far say.

**The short version:** most of what exists is the measurement rig: a local copy of the competition's scoring pipeline that runs on this machine, a model server that matches the competition's, a validated list of tasks we can trust, and a baseline. On top of it there is one first agent of our own, v0, which changes only the prompt and settings. Everything that follows (skill scripts, the LoRA) will be judged with this rig.

---

## 1. The competition in one page

You need this mental model to follow the code.

**What is submitted.** A `submission.zip` containing only declarative files: `agent.yaml` (a Google ADK agent config), prompts, optional sub-agents, optional skills (a `SKILL.md` plus Python scripts), optional LoRA adapters, and an optional `eval_config.yaml`. No Python entry point of ours runs outside the sandbox. The only allowed model is `gemma-4-31b-it-qat-w4a16-ct`.

**How one task is scored.** Two separate sandboxes per task:

```
Phase 1 (agent sandbox)                      Phase 2 (verification sandbox)
-----------------------                      ------------------------------
extract repo at base commit                  extract the same repo again, clean
install it, commit as "baseline"             apply the agent's patch
send the problem statement to the agent      reset any test/config files the patch touched
agent calls tools until it submits,          apply the hidden test patch
  times out, or runs out of budget           run pytest on the target tests
harness takes `git diff` as the patch   -->  all required tests pass  =>  resolved
```

The score is the fraction of tasks resolved.

**Budgets.** Per task the defaults are 60 minutes, 100 tool calls, 500 model turns. But all ~120 hidden tasks must finish inside 12 hours, run one after another, so in practice teams cap each task at 4.5 to 5 minutes.

**The nine tools the agent gets.** `run_command`, `read_file`, `edit_file`, `write_file`, `submit_patch`, `get_status`, and three code-graph tools (`search_similar_code`, `get_code_neighbors`, `get_code_subgraph`).

**Harness behaviours that matter for design.** All verified by reading the harness source in `data/wheelhouse/src/`:

| Behaviour | Consequence |
|---|---|
| `submit_patch` ends the session once the current turn finishes | It must be the last action, never an early checkpoint |
| On timeout or exhausted budget, the harness still takes the working-tree diff and grades it | An edit on disk counts even if the agent never submits |
| New untracked files are included in the diff (`git add -N .`) | Scratch files left in `/workspace` pollute the patch; use `/tmp` |
| An exception in the agent loop (for example calling an undeclared tool) skips the fallback | The patch is lost; declare every tool the model might call |
| Edits to tests and runner config (`conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`) are reset before grading | Only source changes count |
| `submit_patch` and `get_status` are free; other tools and skill-script runs each cost one tool call | A skill script can do a lot of deterministic work for one call |
| When the conversation passes 14,336 tokens, ADK compacts older events | The agent's working memory is far smaller than the 32,768-token model limit |

**What the tasks look like.** The 129 public tasks come from four repos (fastapi 67, rich 48, requests 13, httpx 1). The "problem statement" is a pull-request title and description, often one line such as `fix fonts` plus a link the agent cannot open. Fixes are small: 91 of 129 reference patches touch a single file, and the median is 12 changed lines. The hidden test set comes from different, private repos.

---

## 2. Folder layout

```
Gemm4Agent/
  kaggle.json            Kaggle access token (plain text, owner-only, gitignored)
  .venv/                 Kaggle CLI + evaluation harness
  .venv-vllm/            model server (vLLM + PyTorch)
  scripts/               the tooling (section 4)
  submissions/v0_lean/   our first agent (section 6.3)
  docs/                  this file
  data/
    competition/         the unpacked competition dataset (21 GB)
    zip/                 the original archive (21 GB, safe to delete)
    raw/                 a few files fetched early (duplicates of competition/)
    wheelhouse/          official harness wheels, their unpacked source, the patched vLLM
    wheels_aarch64/      ARM copy of the offline wheels the sandbox installs
    notebooks/           public Kaggle notebooks pulled for reference
    sound_tasks.txt      the 100 tasks that grade correctly here (section 5)
    dev24.txt            fixed 24-task development subset
  models/gemma-4-31b-it-qat-w4a16-ct/   model weights (23 GB)
  runs/                  one folder per evaluation run
  kaggle_nb/timing_probe/               the notebook pushed to Kaggle (section 7)
  tools/pydev/           Python headers unpacked locally (section 3.3)
  logs/                  server and run logs
```

Disk use is about 76 GB: `data/` 43 GB, `models/` 22 GB, `.venv-vllm/` 11 GB.

---

## 3. The environment

### 3.1 Why two virtual environments

| Environment | Holds | Why separate |
|---|---|---|
| `.venv` | Kaggle CLI; the harness (`swegemma` 0.2.7, `adk_submission` 0.2.12, `adk_eval_core` 0.1.0, `google_adk` 1.36.1); pytest and test libraries | The sandbox for each task inherits this environment's packages, so it must look like a plain test machine |
| `.venv-vllm` | vLLM 0.19.1, PyTorch 2.10.0+cu128, transformers 5.13.1, tokenizers 0.22.2 | Heavy GPU stack with strict version pins that would conflict with the harness |

The two talk over HTTP: the harness sends OpenAI-style requests to `http://127.0.0.1:8000/v1`.

### 3.2 Making the model server match the competition

The competition ships a patched vLLM wheel, but only for x86. This machine is ARM. What I did:

1. Installed the stock ARM build of the same version, vLLM 0.19.1.
2. Unpacked the competition wheel and diffed every Python file against stock. Only three files differ, 119 changed lines in total:
   - `vllm/lora/model_manager.py`
   - `vllm/model_executor/models/gemma4.py`
   - `vllm/model_executor/models/gemma4_mm.py`

   These add LoRA support for Gemma 4.
3. Copied those three files over the stock ones. A re-diff shows zero differing Python files. The originals are kept in `data/wheelhouse/vllm_stock_backup/`.
4. Pinned `transformers` and `tokenizers` to the competition's versions.

So the Python serving code is identical to the scorer's. The compiled extensions are the stock ARM ones, and the hardware is different (section 6).

### 3.3 Three machine-specific fixes

| Problem | Symptom | Fix |
|---|---|---|
| vLLM's compiled extension needs CUDA 12, but I first installed the CUDA 13 PyTorch | `libcudart.so.12: cannot open shared object file` | Replaced PyTorch with the `+cu128` build |
| Python development headers are not installed system-wide | Server crashed after loading the model: `Python.h: No such file or directory` | Downloaded the `libpython3.12-dev` package without installing it, unpacked it into `tools/pydev/`, and `serve.py` points the compiler at it through `CPATH` |
| Unified memory: file cache counts as used GPU memory | CUDA saw 18.8 GB free out of 121 GB after the downloads | Asked the kernel to drop cached pages for this project's files (`posix_fadvise`), which needs no admin rights; CUDA then saw 95 GB |

The third one can recur after any large file operation. If the server refuses to start for lack of memory, that is the first thing to check.

### 3.4 Sandbox choice

The harness has two sandbox backends: Docker containers, or plain subprocesses with a per-task virtual environment. Docker is not usable by this account (permission denied on the socket), so everything here uses the subprocess backend. The official starter notebook uses the same one.

One side effect: the subprocess sandbox inherits packages from `.venv`. That is why `.venv` contains pytest and a few test libraries (section 5).

---

## 4. The scripts

### 4.1 `scripts/serve.py`: start the model server

It does not hand-write the vLLM command line. It asks the harness's own `VllmServer.build_cmd()` to build it, so flags, LoRA arguments and rank rounding are whatever the scorer would produce. Then it:

- swaps the interpreter for the one in `.venv-vllm`,
- adds `--served-model-name gemma-4-31b-it-qat-w4a16-ct` so the harness's default model alias resolves,
- sets the `CPATH` header fix,
- writes its process ID to `logs/vllm_server.pid`,
- replaces itself with the server process.

The resulting flags: `--max-model-len 32768 --dtype bfloat16 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4`, thinking enabled by default, and for each adapter folder in the submission, `--enable-lora --lora-modules name=path`.

```bash
# start (detached), mounting the adapters of a submission
setsid nohup .venv/bin/python scripts/serve.py \
  --submission-dir data/competition/sample_submission \
  > logs/vllm_server.log 2>&1 < /dev/null &

# block until it is ready or has died
scripts/wait_server.sh
```

Startup takes about 4.5 minutes. Adapters are mounted at startup, so testing a submission with different adapters means restarting the server with that `--submission-dir`.

`--gpu-memory-utilization` defaults to 0.55 (about 67 GB of this machine's 121 GB), which leaves room for the sandboxes.

### 4.2 `scripts/wait_server.sh`: readiness check

Polls `/health` and exits when the server answers or its process is gone. It tracks the server by the PID file. The first version matched on the command line, which also matched the waiting shell itself, so it never noticed the server had crashed. That cost a night.

### 4.3 `scripts/run_eval.py`: run an evaluation the way Kaggle does

The stock `swegemma eval` command differs from the Kaggle scorer in two ways: it ignores the submission's `eval_config.yaml`, and it does not turn on context compaction. This runner fixes both, so local runs behave like scored ones.

What it sets up:

- budgets read from the submission's `eval_config.yaml`, overridable on the command line,
- context compaction at 14,336 tokens with interval 5 (the README's values),
- the subprocess sandbox with the ARM wheels folder,
- adapter discovery, so agents that name a LoRA route to it.

```bash
.venv/bin/python scripts/run_eval.py \
  --submission-dir data/competition/sample_submission \
  --results-dir runs/my_run \
  --task-ids-file data/dev24.txt \
  --max-time-minutes 5 --max-tool-calls 100 --max-turns 250 \
  --concurrency 1
```

It also has two modes that need no model at all, used to validate the grader:

| Flag | What gets graded | Expected result |
|---|---|---|
| `--skip-agent-patch` | an empty patch | every task fails |
| `--gold-patch` | the task's reference patch | every task passes |

Each run folder contains:

```
summary.json         overall and per-repo resolution rate
task_results.jsonl   one line per task: resolved, duration, tool calls, model calls, error
patches/             the diff taken from the agent sandbox
test_outputs/        pytest output from verification
traces/              full step-by-step trace as JSON
logs/                readable transcript of each agent session
```

### 4.4 `scripts/speed_probe.py`: raw serving speed

Standard library only, so the identical code runs here and inside the Kaggle notebook. For prompts of roughly 1k, 6k and 13k tokens it measures:

- **prefill**: time to read a prompt the server has not seen,
- **cached prefill**: the same prompt again (vLLM reuses the prefix),
- **decode**: tokens generated per second.

It also sends one agent-style request with a tool available, to confirm thinking and tool-call parsing work.

### 4.5 `scripts/analyze_run.py`: what happened in a run

Reads a run folder's results, patches and traces and prints:

- the outcome of every task (resolved; timed out with or without a real source edit; wrong fix),
- turns, generated tokens and seconds per task,
- time and tokens broken down by tool,
- context compactions and what they cost.

It treats a patch that contains only files like `repro.py` as "no source edit". Given two run folders it also lists which tasks each one resolved that the other did not.

```bash
.venv/bin/python scripts/analyze_run.py runs/baseline_sample_dev24_c1
```

### 4.6 `scripts/build_timing_notebook.py` and `scripts/wait_kernel.sh`

The first generates the Kaggle notebook described in section 7. The second polls a Kaggle notebook until it finishes and downloads its output.

### 4.7 `scripts/run_guarded.py`: the evaluator under limits and a watchdog (added 2026-10-11)

Written after the overnight failure in section 6.10. It runs `run_eval.py` inside a `systemd-run --user --scope` with `MemoryMax=40G` and `CPUQuota=1200%` (the local subprocess sandbox has no limits of its own; Kaggle's has 2 vCPU and 4 GiB), at `nice 10` so the model server wins CPU contention, and polls the server log once a minute. It writes `logs/guard_<name>.log` (generation rate, running requests, MemAvailable, swap, load) and kills the whole process group if the server stays below 4 tokens/s with requests running for 6 consecutive minutes (exit code 3). Everything after `--` goes to `run_eval.py` unchanged. `scripts/exp_noise_floor.sh` and `scripts/exp_day*_chain.sh` are the chained experiments built on it.

One trap in the chains, found on 2026-10-13: a chain that waits for its predecessor with `pgrep -f "scripts/exp_dayN_chain.sh"` also matches the shell that launched it when that shell's command line contains the script text (the Claude Code Bash tool's wrapper does), and then waits forever. The wait patterns are anchored to the real process, `^bash scripts/exp_dayN_chain.sh`, and lingering launcher shells can be killed without affecting the chains, which run in their own sessions.

### 4.8 `scripts/behaviour_vector.py`: the behaviour vector of a run

One JSON object per run: resolution, garbled-call rate (argument keys that are not identifiers, a path that swallowed `,start_line:...`, or a "mandatory input parameters are not present" reply), fraction of patches touching real source, wrong-edit rate, empty-patch rate, median seconds to the first `edit_file` in Kaggle-equivalent time (local / 3.2), median turns, median generated tokens per turn, repeat-loop rate (4+ identical calls in a row), scratch-write rate, compactions per task. Each is also given on a logit or log scale (`z_` prefix) for the controller. It reproduces the hand counts in 6.6 to 6.8 (for example 185/770 = 0.242 garbled calls for thinking off at temperature 0.2).

### 4.9 `scripts/ie_control.py` and `configs/ie_pilot.yaml`: the iE controller

The accelerated iE iteration of the dissertation, `x_{n+1} = x_n + A_n (b - y_n)`, over submission knobs (memo `iE_DPO_EVAE_connections.md`, section 10). A knob is a sampling key, an `eval_config.yaml` key, or a regex substitution in the prompt, each with a scale (identity or log), bounds, a probe step and a trust region. `probe` runs the base point and one forward-difference per knob on a fixed 24-task batch and fits a prior Jacobian `G`; `iterate` takes Robbins–Monro steps with gain `(P / n^beta) pinv(W^1/2 G) W^1/2`, where `W` whitens by the per-coordinate repeat standard deviation from two noise-floor runs, rotating through disjoint 24-task chunks of the 100 sound tasks. Each draw is one `run_guarded.py` batch; the state (`runs/ie_<name>/state.json`) makes it resumable. Added 2026-10-13 after the pilot's end point proved worse than its start (6.17): an optional merit gate (`gate: {coord: resolution_rate, min_delta: -0.05}` in the config, memo section 3.3 step 7). When present, a draw at a *new* point whose protected coordinate falls more than the allowance below the reference rejects the move that produced it, the iteration resumes from the last accepted point and the trust region halves; a draw that passes makes the new point the accepted one and starts a new reference. Revised after round 2 (6.23): the reference is the running mean of the draws at the accepted point (common random tasks make it converge), a re-draw of the accepted point is folded into that mean and never rejected, the allowance defaults to 0.10 (about two tasks on 24, the resolution rate's per-draw noise), and trust recovers on an accepted move. Round 1 of the directions experiment had no gate and tested the method as specified; rounds 2 and 3 use it. The pilot's knobs are temperature, thinking budget (log, kept in the thinking-on branch because its response is non-monotone), the prompt's read window, and the tool-call cap; its targets are garbled-call rate 0.01, repeat-loop rate 0.02, source-edit rate 0.85, first edit at 120 Kaggle seconds, 0.4 compactions per task, empty-patch rate 0.08.

### 4.10 `scripts/mine_format_pairs.py`: on-policy preference pairs for the call-format direction

Traces keep only parsed arguments, so the raw text of a garbled call cannot be recovered from them. The miner instead rebuilds the exact context at every turn where a run produced a garbled call, samples eight raw completions from the server (`/v1/completions` on the rendered token ids, special tokens kept), labels each with a strict parser of the native call format plus the harness's own `apply_replacement` for `edit_file`, and pairs well-formed with garbled calls of the same tool from the same prefix. When no well-formed call of that tool is sampled, a minimal pair is synthesised by repairing the one format error (a backtick or quote where the closing `<|"|>` belonged; JSON-style quoted keys) and re-verifying; those are flagged `synthetic`. Pre-compaction turns only. Output: `data/pairs/format_pairs.jsonl`, one record per context with messages, tool specs, all samples and the pairs.

What had to be matched to render the context exactly as the harness does (verified by token count against the trace, and by token-id identity against vLLM's `/tokenize`):

- ADK appends `You are an agent. Your internal name is "swe_agent".` to the instruction, after a blank line (`flows/llm_flows/identity.py`), and the trace records only the instruction.
- Tool declarations are derived from the harness's Python signatures and docstrings through `FunctionTool(...)._get_declaration()` and ADK's `_function_declaration_to_tool_param`, in `agent.yaml` order.
- Tool results use role `tool` (the model string `openai/gemma-4-31b-it-qat-w4a16-ct` does not contain `gemma4`, so ADK's `tool_responses` branch is not taken) with the JSON observation as content; assistant tool calls carry JSON-serialised arguments.
- `thinking_budget` is **not** passed by ADK's LiteLlm wrapper at all. `adk_submission/resolvers/generation.py` bridges it into `extra_body = {chat_template_kwargs: {enable_thinking: bool}, thinking_token_budget: N}`, so thinking off and thinking on render different prompts (`<|think|>` in the system turn).
- When rendering with the HF tokenizer (for training), tool-call `arguments` must be dicts and the system content a list of text parts; vLLM does both before templating, and the template's sequence branch emits a space before the first `<|tool>`. With those two changes the HF rendering is token-for-token identical to vLLM's.
- After `}<tool_call|>` the model stops by emitting `<|tool_response>` (id 50, in `generation_config.eos_token_id`), not `<turn|>`; vLLM strips it from the returned text.

### 4.11 `scripts/train_direction_dpo.py`: one preference direction as a LoRA (untested on the GPU)

DPO on the mined pairs, producing a PEFT adapter folder a submission can ship. The base is the served checkpoint decompressed to bf16 (`CompressedTensorsConfig(run_compressed=False)`; the checkpoint is compressed-tensors int4, group 32, symmetric, on every language-model Linear, and that dequantisation is exact, so the adapter is trained on the values vLLM computes with). The prefix is run once without gradient and its KV cache snapshotted; chosen and rejected completions are scored against the cache with gradient, so only the call tokens carry gradient, which is what the format direction is about and keeps 10k-token contexts affordable. The sliding-window cache layers (window 1024) cannot be cropped once past the window, so restore works by re-pointing each layer's tensors to the snapshot. Reference log-probabilities come from the same model with the adapter disabled. The mechanics (cached scoring equals a full forward, restore, LoRA backward) are verified on a tiny random Gemma 4 on the CPU; the data path is verified on real mined contexts. It needs the server stopped (the server holds about 67 GB; the bf16 model is 62 GB). `peft` 0.21.2 is installed in `.venv-vllm`.

**What the first GPU runs taught (2026-10-12).** Loading and decompressing takes about 160 s and 60.0 GiB; a rank-16 LoRA on all seven projections of the 60 language-model layers is 122M parameters. Timing is not the problem (at a 2k prefix: reference pass 34 s, policy prefix pass 18 s, completion forwards 14 s, backward 9 s). Memory is: the two completion forwards with gradient held 42 GiB of activations (62 to 104 GiB) for about 480 tokens each, because this GPU takes SDPA's math path and materialises the attention scores, roughly 30 MB per completion token over 60 layers, and both graphs stayed alive until `backward()`. At an 8k prefix that exceeds the 121 GB pool, which is what the earlier NVRM out-of-memory errors were. HF's gradient checkpointing is not an option: Gemma 4 drops a supplied cache when it is on. The trainer therefore (a) computes the DPO margin without gradient and back-propagates each completion separately with the weight `sigma(-m) * beta` detached, and (b) back-propagates each completion in chunks of 96 tokens, detaching the cache between chunks, so earlier chunks act as constants for later ones (the same approximation as the detached prefix; on the tiny model the chunked gradient has cosine 0.92 with the exact one at a 4-token chunk). The prefix pass runs in eval mode with `logits_to_keep=1`. With these, two-context smoke runs complete end to end (including the vLLM-format export) with peaks of **81.3 GiB at a 4k prefix and 97.2 GiB at 8k**, and cost about 170 s and 260 s per context for two pairs (reference pass, policy prefix pass, four chunked completion backwards). The first real run (`exp_day3_chain.sh`, 2026-10-12 09:04 to 11:59) used a 4k prefix, two pairs per context and one epoch: 205 pairs from the 103 contexts that had a same-tool or repaired pair, about 103 s per context, peak 82 GiB. The DPO loss went from ln 2 (a zero-initialised LoRA is the reference) to 0.01-0.02 within 60 contexts, with pair accuracy 1.00 and an implicit-reward margin of about 5 by the end, so the direction is strongly separated on its own data; its strength on the deployed model is what the merge weight scales. The adapter (`adapters_out/format_r16`, 490 MB, rank 16 on seven projections) is served as `submissions/v0_long_format1` (v0_long routed through it at weight 1) and evaluated on dev24 against the two noise-floor runs; see 6.13 once that lands.

---

## 5. Validating the grader

A local score means nothing unless the local grader is right. The check: every task should fail with an empty patch and pass with its reference patch. A task that does both is called **sound**.

The first control run exposed three problems, each of which would have silently corrupted every later measurement:

| Round | Sound tasks | What was wrong |
|---|---|---|
| 1 (3-task smoke test) | none | The sandbox had no `pytest`; all three tasks "failed" for the wrong reason |
| 2 (all 129) | 69 | fastapi tests import `inline_snapshot` and `dirty_equals`, which were missing |
| 3 (all 129) | **100** | final state |

Final counts: fastapi 60 of 67, rich 40 of 48, requests 0 of 13, httpx 0 of 1.

The 29 excluded tasks:

- 7 pass with an empty patch (two fastapi, five requests), so they cannot measure a fix.
- 22 fail even with the reference patch: eight requests tasks (SSL-related errors), the one httpx task, eight rich tasks (rendered colour output differs, most likely a Pygments version mismatch; not confirmed), and five fastapi tasks (not yet diagnosed).

Public audits on x86 machines found 114 to 119 sound tasks, so this machine loses a few more. 100 is enough to work with. The list is `data/sound_tasks.txt`.

`data/dev24.txt` is a fixed random draw of 12 fastapi and 12 rich tasks from that list, used for quick comparisons.

---

## 6. Measurements so far

### 6.1 Local serving speed

| Quantity | Value |
|---|---|
| Decode, one request at a time | 10.6 tokens/s |
| Decode, 4 requests in parallel | 6.8 tokens/s each (27 total) |
| Decode, 8 requests in parallel | 5.0 tokens/s each (40 total) |
| Prefill, unseen prompt | about 700 tokens/s (13k tokens takes 19 s) |
| Prefill, cached prefix | 0.1 to 0.2 s |
| Model memory | 19.7 GB |
| KV cache | 47,664 tokens |
| Server startup | about 4.5 minutes |

What this means: at 10.6 tokens/s, a 5-minute task can generate about 3,000 tokens in total, across all its thinking and tool calls. That is the whole budget.

### 6.2 Baseline: the unmodified sample agent

The sample agent is one root agent with all nine tools plus a read-only "code analyzer" sub-agent, a thinking budget of 4,096 tokens, and two placeholder LoRA adapters whose weights are zero (they change nothing).

Run: `dev24`, 5-minute cap, one task at a time (`runs/baseline_sample_dev24_c1`).

| Outcome | Tasks |
|---|---|
| Resolved | 3 |
| Timed out without ever editing real source | 20 |
| Timed out with a source edit that was wrong | 1 |

**Result: 3 of 24 (12.5%).** Every failure is a timeout. Only 4 of the 24 tasks produced a real source edit, and 3 of those 4 were correct. So when this agent reaches the edit it is usually right; it almost never reaches it.

Where the 5 minutes went, summed over all tasks:

| Activity | Turns | Share of time | Note |
|---|---|---|---|
| `run_command` and `read_file` (exploring) | 283 | 74% | about 15 s per turn, about 125 generated tokens each |
| `write_file` | 18 | 19% | 62 s per call; 17 of the 18 were reproduction scripts, not fixes |
| `edit_file` | 5 | 2% | the only tool that actually fixes anything |
| everything else | 18 | 5% | |

Other measurements from the traces:

- A turn costs about 4 s of fixed overhead plus 0.1 s per generated token. Prompt length costs nothing extra, so prefix caching works inside the agent loop.
- The median task used 14 turns and 2,100 generated tokens.
- Nine of the thirteen non-empty "patches" contained only a scratch file such as `repro.py`.
- Three tasks hit context compaction, which cost about 75 s each time (a quarter of the budget).
- A few single turns spent 90 to 210 s thinking before a trivial command.

**Important caveat, added after the Kaggle timing run (section 7):** this machine is about 3.2 times slower per agent turn than Kaggle's four L4s. A 5-minute cap here is worth about 1.6 Kaggle minutes. Everything in sections 6.2 to 6.8 was measured at that much harsher budget. The comparisons between variants are still fair to each other, but "the agent cannot reach the edit in time" has to be re-tested at the real budget.

### 6.3 Lean agent v0

`submissions/v0_lean/` is the first agent of our own. It changes only prompt, sampling and budgets, each aimed at a measured cost above:

| Change | Aimed at |
|---|---|
| One agent, no analyzer sub-agent, no placeholder adapters | The sub-agent was never called in 24 tasks |
| Prompt forbids scratch and reproduction files | 19% of time spent writing them |
| Prompt asks for one chained `grep` call, then short ranged reads | 74% of time spent exploring one small step at a time |
| Prompt requires an `edit_file` by the sixth call | 20 of 24 tasks never edited |
| `thinking_budget` 512 (was 4,096), `max_output_tokens` 2,048 (was 16,384) | Runaway 90 to 210 s thinking turns |
| Command timeout 120 s (was 300 s), 40 tool calls, 80 turns | One slow command must not consume the whole task |
| Prompt explains that the statement is a PR description | Statements such as "fix fonts" read oddly as bug reports |

**Result: v0 did not help.** Same 24 tasks, same conditions (`runs/v0_lean_dev24_c1`):

| | Sample agent | v0 |
|---|---|---|
| Resolved | 3 | 2 |
| Timed out without a real source edit | 20 | 19 |
| Source edit made but wrong | 1 | 3 |
| Tasks whose patch touches real source | 4 | 5 |
| `write_file` calls | 18 | 3 |
| Exploration turns (`run_command` + `read_file`) | 283 | 344 |
| Median turns per task | 14 | 16 |

A difference of one task out of 24 is noise, so the honest reading is "no change". What the traces show:

- The scratch-file instruction worked (18 `write_file` calls down to 3).
- The time saved went straight into more exploring. The model takes many tiny steps (a one-line `grep`, then a 15-line read, repeated) and does not converge on an edit.
- "Edit by your sixth call" was ignored: `edit_file` was used in 5 tasks, as before.
- The 512-token thinking cap is enforced, but the model often thinks right up to it. Turns of 30 to 55 s of thinking before a one-line `grep` are common, despite the prompt asking for a sentence or two.

The lesson: prompt wording changes what the model avoids, but not how decisively it acts. That points at three levers that do not rely on instructions: cutting thinking in the sampling config, doing the localization deterministically in a skill script, and fine-tuning on short successful trajectories.

Compare any two runs with:

```bash
.venv/bin/python scripts/analyze_run.py runs/baseline_sample_dev24_c1 runs/v0_lean_dev24_c1
```

### 6.4 Thinking-budget test

Two copies of v0 that differ only in `thinking_budget`: `submissions/v0_think0` (thinking off) and `submissions/v0_think128`. Both ran at once, two tasks in parallel each, so four requests shared the server and each got about 6.8 tokens/s instead of 10.6. To keep the token budget equal to the earlier runs, the time cap was scaled to 7.8 minutes. These two runs are directly comparable with each other; against the earlier one-at-a-time runs they are only roughly comparable.

| | Thinking off | Thinking 128 |
|---|---|---|
| Resolved | 5 | 3 |
| Timed out without a real source edit | 7 | 18 |
| Ended (calls used up or submitted) without a real source edit | 7 | 0 |
| Source edit made but wrong | 5 | 3 |
| Median turns per task | 38 | 21 |
| Generated tokens per turn | about 50 | about 120 |
| Tasks that used `edit_file` | 14 | 7 |
| Tasks with 4 or more identical calls in a row | 14 | 0 |

What this shows:

- **Any thinking at all keeps the slow behaviour.** A 128-token cap acts like the 512-token one: about 20 turns, mostly exploring, 18 timeouts without an edit.
- **Thinking off makes the model decisive.** Turns are short, it edits in 14 of 24 tasks, and its five wins took 68 to 244 s.
- **Thinking off also causes repetition loops.** In 14 of 24 tasks the model issued the same call four or more times in a row, and in the worst case 48 times. Seven tasks ended with the tool-call budget spent on repeats and no edit.

So the next question is how to keep thinking off and remove the loops.

### 6.5 Sampling test

The sample submission sets `temperature: 0.2`, and v0 inherited it. The model card recommends `temperature=1.0, top_p=0.95, top_k=64` for all uses. Low temperature was a plausible cause of the exact-repeat loops, so two more thinking-off variants ran under the same conditions as 6.4: `submissions/v0_think0_t06` (temperature 0.6, top-k 64) and `submissions/v0_think0_t10` (temperature 1.0, top-k 64).

| Thinking off, temperature | 0.2 | 0.6 | 1.0 |
|---|---|---|---|
| Resolved | 5 | 5 | 4 |
| Tasks with 4 or more identical calls in a row | 14 | 7 | 0 |
| Patches that touch real source | 10 | 12 | 13 |
| Source edit made but wrong | 5 | 7 | 9 |
| Ended without a real source edit | 14 | 12 | 11 |

Higher temperature removes the exact-repeat loops, but the resolved count does not move. Temperature is not the fix.

### 6.6 The real problem with thinking off: malformed tool calls

Counting tool calls across every run so far:

| Run | `edit_file` calls | Succeeded | Calls with missing or garbled arguments |
|---|---|---|---|
| Sample agent (thinking 4,096) | 5 | 5 | 5 |
| v0 (thinking 512) | 6 | 6 | 2 |
| Thinking 128 | 10 | 9 | 4 |
| Thinking off, temp 0.2 | 139 | 11 | 185 |
| Thinking off, temp 0.6 | 50 | 21 | 60 |
| Thinking off, temp 1.0 | 45 | 15 | 66 |

Two patterns account for almost all of it:

- **`edit_file` with broken structure.** When the model has to put multi-line code into `old_string` and `new_string`, the argument boundaries come out wrong (one argument swallows the next), the tool answers "mandatory parameter not present", and the model retries the same thing. Most of the "loops" in 6.4 were this.
- **`read_file` with broken keys.** Calls arrive with a key such as `end_line"`. The range is ignored and the tool returns the top of the file, so the model never sees the lines it asked for.

With thinking on, tool calls are almost always well formed, but the agent is too slow to reach the edit. With thinking off, it is fast but cannot reliably write the one call that matters.

A second effect: once turns are cheap, the agent runs out of its 40 tool calls before it runs out of time, and the harness's "few calls remaining" warning pushes it to submit an empty patch.

**What the model actually sees and writes.** Probing the server directly (rendering a short history through its `/tokenize` endpoint, and sampling raw completions) shows the native format:

```
<|tool_call>call:read_file{end_line:20,filepath:<|"|>a.py<|"|>,start_line:10}<tool_call|>
<|tool_response>response:read_file{value:<|"|>{"status": "ok", "content": "def f(x):\n    return x[\"k\"]\n", ...}<|"|>}<tool_response|>
```

- Calls use bare keys in alphabetical order, and strings are wrapped in a special `<|"|>` token rather than ordinary quotes.
- Tool results arrive as one JSON string, so file contents reach the model JSON-escaped: newlines as `\n`, quotes as `\"`.
- To edit, the model must copy code out of that escaped form into raw text between `<|"|>` tokens.

In a clean single turn with thinking off, the model writes a perfectly formed `edit_file` call (3 of 3 samples). The breakage appears only deep in a session, after many JSON-formatted tool results. The garbled keys (`end_line"`) look like the model drifting into JSON-style arguments, which the parser then misreads. This is my reading of the evidence, not something I have isolated in a controlled test.

It matters beyond this experiment: these two formats are exactly what LoRA training data has to reproduce, and robust call formatting is one of the clearest things fine-tuning could buy.

### 6.7 Tiny thinking budgets

`submissions/v0_think32` and `submissions/v0_think64`: temperature 0.2, thinking capped at 32 and 64 tokens, tool-call limit raised to 100 (so these change two things at once relative to 6.4). The question was whether a few tokens of thinking keep tool calls well formed while staying nearly as fast as thinking off.

They do not combine the two benefits. The 32-token variant resolved 4 of 24 with mostly well-formed calls, but went back to exploring without editing (14 timeouts with no source edit). The 64-token variant resolved 2 of 24, and 25 of its 33 `edit_file` calls failed.

### 6.8 All variants side by side

Every row is the same 24 dev tasks. The first two ran one task at a time with a 5-minute cap; the rest ran four requests in parallel with the cap scaled to 7.8 minutes, so compare across those two groups only roughly.

| Variant | Resolved | Median turns | `edit_file` ok / attempted | Garbled calls | Patches touching real source |
|---|---|---|---|---|---|
| Sample agent (thinking 4,096) | 3 | 14 | 5 / 5 | 5 | 4 |
| v0 (thinking 512) | 2 | 16 | 6 / 6 | 2 | 5 |
| Thinking 128 | 3 | 21 | 9 / 10 | 4 | 6 |
| Thinking 64 | 2 | 27 | 8 / 33 | 27 | 7 |
| Thinking 32 | 4 | 38 | 11 / 16 | 6 | 10 |
| Thinking off, temp 0.2 | 5 | 38 | 11 / 139 | 185 | 10 |
| Thinking off, temp 0.6 | 5 | 34 | 21 / 50 | 60 | 12 |
| Thinking off, temp 1.0 | 4 | 34 | 15 / 45 | 66 | 13 |

How to read this:

- **Resolved counts of 2 to 5 out of 24 are within noise of each other.** No variant is a demonstrated improvement over the sample agent.
- **Only 7 of the 24 tasks were solved by any variant.** Two were solved by all eight and one by five; the rest were solved once to three times. Each variant's score is mostly "the easy tasks plus luck".
- **The behavioural trend is real even though the scores are not.** Less thinking means more turns, more edits attempted and more patches that touch real source, but also more broken tool calls and more wrong fixes.

Conclusion for the plan: prompt and sampling settings have been pushed about as far as they go on this model. The two failure modes (slow to reach an edit with thinking on; cannot format the edit with thinking off) are behaviours of the base model, which is what fine-tuning is for.

### 6.9 Long-budget run: the first Kaggle-equivalent number

`submissions/v0_long` is v0 (thinking 512, well-formed calls) with the urgency wording removed and 100 tool calls. It ran on all 100 sound tasks, eight in parallel, with a 30-minute cap (`runs/v0_long_sound100_c8`). After the calibration in section 7, that cap turns out to be worth about 4.4 Kaggle minutes, slightly under the real 5, so this is the first result that approximates real scoring.

**Result: 36 of 100 resolved (36%).** fastapi 24 of 60, rich 12 of 40. On the 24 dev tasks it resolved 8, against 2 at the harsh budget.

| Outcome | Tasks |
|---|---|
| Resolved | 36 |
| Timed out without a real source edit | 23 |
| Timed out with a wrong source edit | 21 |
| Finished with a wrong source edit | 20 |

- **At the real budget the agent reaches the edit.** 77 of 100 patches touch real source. The first edit comes at a median of 816 s here, about 2.2 Kaggle minutes.
- **Successes are not fast.** Resolved tasks took a median of 993 s here (about 2.7 Kaggle minutes) and 19 tool calls; a quarter took more than 1,589 s. Nothing finishes with time to spare, which is why a faster model should convert more of the near misses.
- **Wrong fixes are now the largest failure class**: 41 of the 64 failures include a source edit that did not pass the tests.
- **Context compaction happened in 95 of 100 tasks**, costing a median of 181 s each at this concurrency, about 10% of the budget. Keeping reads small and avoiding repeats is worth real time.

Two cautions. These are public repos the model may know well, and public agents of this kind score 0.10 to 0.15 on the leaderboard, so expect the hidden set to be much harder. And the 36 successful runs (28 outside the dev set) are the first LoRA harvest.

### 6.10 Overnight runs: invalid, the server degraded

Two runs queued back to back at the calibrated cap (34 minutes, eight in parallel):

1. `runs/v0_think0_t06_sound100_c8`: the thinking-off agent (temperature 0.6) on all 100 sound tasks. Thinking off was the most decisive setting at the harsh budget; this tests it at the real one, against the 36% above.
2. `runs/v0_long_pass2_failed64_c8`: v0 again on the 64 tasks it failed, for more training trajectories (sampling is not deterministic, so a second pass recovers some).

**Neither result can be used.** Checked on 2026-10-11: the server's aggregate generation rate (`logs/vllm_server.log`, eight requests running throughout) was about 40 to 45 tokens/s until 20:05 on Oct 5, then 2 to 9 tokens/s from 21:13 onward and never recovered. The first run therefore scored 8 of 100 (81 timeouts) mostly on a crippled server, and the second scored 0 of 64 with every task timing out after 2 to 17 model calls at roughly 100 s per call. The second harvest of winning trajectories does not exist. The cause is not diagnosed; candidates are the unified-memory page-cache effect of section 3.3, sandboxes competing for CPU at concurrency 8, or a vLLM fault. Before any further long run: restart the server, and have the runner (or a watchdog) abort when the logged generation rate stays below about a quarter of the expected value for several minutes.

---

### 6.11 Noise floor at the calibrated budget (2026-10-11)

v0_long on dev24, concurrency 8, 34-minute cap, run twice under `run_guarded.py` (`runs/noise_v0long_dev24_c8_r1`, `_r2`; 98 and 93 minutes wall). Both runs were healthy throughout (35 to 48 tokens/s aggregate, memory flat).

| | run 1 | run 2 |
|---|---|---|
| Resolved | 10/24 | 9/24 |
| Garbled-call rate | 0.020 | 0.029 |
| Patches touching real source | 0.83 | 0.75 |
| First edit, median Kaggle seconds | 297 | 342 |
| Median turns | 30.5 | 35.5 |
| Repeat-loop rate | 0.04 | 0.13 |
| Compactions per task | 1.04 | 1.00 |
| Empty-patch rate | 0.17 | 0.21 |

Two readings. First, the Kaggle-equivalent score on dev24 is about 40%, against 8/24 in the earlier 30-minute run (6.9), so a few minutes of cap matter near this operating point. Second, the behaviour coordinates repeat far more tightly than the score; their standard deviations on the transformed scale (the controller's whitening, `runs/ie_pilot/state.json`) are 0.25 (garbled, logit), 0.84 (repeat loops, logit: one versus three tasks), 0.36 (source edits, logit), 0.10 (first edit, log), 0.03 (compactions), 0.19 (empty patches, logit). Two repeats give a rough estimate; it is enough to set gain scales.

### 6.12 Format-pair mining and the first iE probe (2026-10-11)

Mining (4.10) over the three thinking-off dev runs, 50 contexts each, took 35 minutes: 148 contexts (27 + 20 + 3 skipped as post-compaction, 3 for a token-count mismatch), 1,184 samples of which 330 well-formed and 852 garbled, **738 pairs** (289 `edit_file`, 449 `read_file`); 55 contexts needed a repaired chosen. Almost every rejected call fails the parser on syntax (732), the signature being a backtick or quote where the closing `<|"|>` belonged.

The pilot's Jacobian probe (4.9) then ran four of its five batches on one fixed 24-task batch drawn from the 100 sound tasks:

| batch | x | resolved | garbled | source edit | first edit (s) | turns | compactions | empty |
|---|---|---|---|---|---|---|---|---|
| base | T 0.2, think 512, read 60, calls 100 | 6/24 | 0.021 | 0.75 | 369 | 31 | 1.21 | 0.17 |
| temperature 0.6 | | 7/24 | 0.044 | 0.67 | 240 | 28 | 1.12 | 0.29 |
| thinking 188 | | 8/24 | 0.033 | 0.88 | 220 | 42 | 1.12 | 0.08 |
| read window 121 | | 8/24 | 0.026 | 0.79 | 309 | 33 | 1.12 | 0.08 |

Each row is one draw, so read the directions, not the digits: a smaller thinking budget reaches the first edit sooner and edits more often at the cost of more garbled calls; higher temperature also edits sooner but leaves more empty patches. The fifth batch (tool-call cap 50) was lost: 26 minutes in, one agent-run `python` in a sandbox reached 41.8 GB RSS, hit the evaluator scope's 40 GiB `MemoryMax`, was OOM-killed, and systemd's default `OOMPolicy=stop` then sent SIGTERM to the whole scope (journal: `run-....scope: Failed with result 'oom-kill'`). The watchdog was not involved. Two fixes (2026-10-12): the outer scope now has `OOMPolicy=continue`, and `run_eval.py` runs **every sandbox command in its own transient cgroup of 4 GiB and 2 cores**, Kaggle's sandbox limits, so a runaway command dies alone and the agent gets the error (verified: a 6 GB allocation is killed, 2 GB runs). `systemd-run` needs the user bus, which the sandbox's sanitised environment omits; the wrapper passes `XDG_RUNTIME_DIR` and `DBUS_SESSION_BUS_ADDRESS` to `systemd-run` and strips them again for the command. Consequence for comparisons: runs from 2026-10-12 on have per-command CPU limits that the noise-floor and probe runs did not, so timing coordinates are not strictly comparable across that date.

The first trainer attempt (4.11) stopped at PEFT: the plain target names also matched the vision tower, whose projections are `Gemma4ClippableLinear` wrappers PEFT cannot adapt; the language model's projections are plain `nn.Linear`. The target regex is now restricted to `language_model`. The load path itself worked: the checkpoint decompressed to bf16 in 174 s at 60 GiB.

### 6.13 The first format direction collapsed (2026-10-12)

The adapter from 4.11's first run (`adapters_out/format_r16_collapsed`: lr 5e-5, beta 0.1, one epoch, final margin about 5) was mounted and v0 routed through it on dev24. Five minutes in, no task had completed its first model turn: all eight requests were generating continuously with no prefill arriving (the guard log shows 50 tokens/s and zero prompt throughput), so the run was stopped. A direct probe of the `merged` route on one rendered task shows why: the adapter emits `call:run_command{command<|"|>grep ... <|"|><|"|><|"|>…` and loops on the string-delimiter token until the length limit, with thinking on or off, while the base model stops cleanly after 40 to 106 tokens.

The cause is the objective, not the pipeline. Every mined pair differs in the same way (the chosen call has `<|"|>` where the rejected has a backtick or quote), and whole-sequence DPO driven to a 5-nat margin generalised that to "`<|"|>` is always good". The retrain (`exp_day4_chain.sh`, after the pilot) uses lr 1e-5, an NLL anchor on the chosen completion (weight 0.2, which penalises exactly the delimiter runs the chosen completions do not contain), early stopping once the window margin passes 1.5, and the trainer now generates greedily on three contexts after training and refuses to export an adapter with an 8-token repeat or a call that never closes. The collapsed adapter would have failed that gate, saving the restart-and-evaluate cycle.

The retrain ran 20:15 to 23:15 on the same 205 pairs: the margin rose slowly (0.02 at 40 contexts, 0.60 at 88, 1.50 at 96, where the early stop fired after 25 optimiser steps; loss 0.27, pair accuracy 1.00). The sanity gate passed: greedy generations on three training contexts are correctly closed `read_file` and `edit_file` calls of 42 to 87 tokens that stop on `<|tool_response>`. The adapter (`adapters_out/format_r16_v2`) was mounted and v0 routed through it on dev24 from 23:19; see 6.16.

### 6.14 The pilot's Jacobian (2026-10-12)

With the fifth batch rerun (tool-call cap 50: garbled 0.036, source edit 0.83, first edit 314 s), the controller fitted the prior Jacobian `G` (rows: targets on the transformed scale; columns: temperature per unit, thinking budget per log unit, read window per log unit, tool-call cap per call):

| target | temperature | thinking | read window | tool calls |
|---|---|---|---|---|
| garbled calls (logit) | +1.89 | −0.47 | +0.28 | −0.01 |
| repeat loops (logit) | −1.84 | 0.00 | 0.00 | 0.00 |
| source edits (logit) | −1.01 | −0.85 | +0.34 | −0.01 |
| first edit (log s) | −1.07 | +0.52 | −0.25 | 0.00 |
| compactions | −0.21 | +0.08 | −0.12 | 0.00 |
| empty patches (logit) | +1.81 | +0.79 | −1.12 | +0.03 |

Singular values 3.57, 1.69, 0.45, 0.006 (condition 591). Three readings. First, the signs agree with the earlier sweeps: temperature trades garbled calls and empty patches against repeat loops and time to the first edit; thinking trades garbled calls against speed and edit rate; a larger read window mainly reduces empty patches. Second, each entry is one forward difference, so only the larger ones are above the noise floor (the temperature column's garbled entry is about three standard deviations; the repeat-loop entries are within one). Third, the tool-call cap has no measurable effect at this budget (the agent never approaches 50 calls), which is exactly the rank diagnostic of memo section 10.5: that direction is uncontrollable and the pseudo-inverse (rcond 0.1) drops it rather than taking a large step along it. Four RM steps follow, rotating through the other 24-task chunks of the sound set.

### 6.15 The pilot's RM steps (2026-10-12, in progress)

Each step is one 24-task draw on a fresh chunk of the sound set, so consecutive rows differ by task difficulty as well as by the knobs. `|W r|` is the whitened residual norm the controller minimises.

| step | temperature | thinking | read window | resolved | garbled | loops | source edit | first edit (s) | compactions | empty | `|W r|` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (base) | 0.2 | 512 | 60 | 8/24 | 0.018 | 0.125 | 0.63 | 305 | 1.08 | 0.29 | 26.7 |
| 2 | 0.5 | 1031 | 99 | 10/24 | 0.056 | 0.042 | 0.75 | 255 | 0.75 | 0.25 | 17.4 |
| 3 | 0.58 | 1275 | 150 | 6/24 | 0.068 | 0.083 | 0.75 | 279 | 1.00 | 0.17 | 24.0 |
| 4 | 0.88 | 2048 | 150 | 11/24 | 0.046 | 0.042 | 0.79 | 267 | 0.75 | 0.13 | 16.0 |
| next | 1.0 | 2048 | 150 | | | | | | | | |

The first move hit the trust region on all three live knobs (the gain is `P/n^0.6` with `n = 1`) and the tool-call cap stayed put, as its Jacobian column is numerically zero. Step 3 ran on the same chunk as the probe's base batch, so it is the one like-for-like comparison: same resolution (6/24), same loops, edit rate and empty-patch rate, a faster first edit (279 s against 369) and fewer compactions, but three times the garbled-call rate (0.068 against 0.021). Over the four steps the whitened residual went 26.7, 17.4, 24.0, 16.0 and the controller drove all three live knobs to their upper bounds (temperature 1.0, thinking 2048, read window 150). That is the method's honest answer: the target vector is not attainable inside this box with these knobs; what remains of the residual is the garbled-call rate (0.046 against 0.01), the first-edit time (267 s against 120) and empty patches, which are the coordinates the memo expects an adapter direction, not a sampling knob, to move. The resolution counts (8, 10, 6, 11 of 24 on different chunks) are single draws and should not be read as a trend. Whether the end point is better than the base point will be settled only by a repeated run on a fixed task set; it is a candidate, not a result.

The pilot's purpose was the machinery, and that worked: five probe batches, a Jacobian with a usable rank diagnostic, four RM steps, one sandbox OOM survived without losing a batch, all unattended.

### 6.16 The format direction on dev24 (2026-10-13)

v0_long routed through `adapters_out/format_r16_v2` at weight 1 (`submissions/v0_long_format1`, `runs/v0long_format1_dev24_c8`), same 24 tasks and budget as the noise floor, one draw. The noise-floor columns are the base model.

| | noise run 1 | noise run 2 | + format direction |
|---|---|---|---|
| Resolved | 10/24 | 9/24 | 9/24 |
| Garbled-call rate | 0.020 | 0.029 | **0.012** |
| Repeat-loop rate | 0.04 | 0.13 | **0.00** |
| First edit, median Kaggle seconds | 297 | 342 | **211** |
| Patches touching real source | 0.83 | 0.75 | 0.71 |
| Wrong-edit rate | 0.42 | 0.38 | 0.33 |
| Empty-patch rate | 0.17 | 0.21 | 0.25 |
| Compactions per task | 1.04 | 1.00 | 1.13 |
| Median turns | 30.5 | 35.5 | 29.5 |

On the controller's whitened scale the garbled-call rate moved by about three standard deviations of the repeat noise and the first-edit time by about four; the rest is within noise. The direction therefore does what it was trained for (fewer malformed calls, no exact-repeat loops) without costing resolution, and the agent reaches its first edit sooner, plausibly because fewer turns are wasted on failed calls. Two caveats. It is one draw, so the resolution comparison says nothing yet. And this run had the per-command sandbox limits (4 GiB, 2 cores) that the noise-floor runs did not; those slow test commands, which would make the first edit later, not earlier, so the timing gain is if anything understated, while the garbled-call rate is unaffected by them.

What this establishes for the plan: the whole path from mined on-policy pairs to a deployable direction works, and the direction moves the coordinate it targets. The first attempt also showed the failure mode to guard against (6.13). The merge weight on this direction is now a controllable coordinate for the memo's Usage 2 once a second direction exists (the decisiveness pairs are being mined by `exp_day6_chain.sh`).

### 6.17 The pilot's end point on a fixed task set (2026-10-13)

The end point of the RM steps (temperature 1.0, thinking budget 2048, read window 150) run on dev24 twice under the same budget as the noise floor (`runs/pilot_end_dev24_c8_r1`, `_r2`), against the base point and the format direction:

| | base run 1 | base run 2 | end point run 1 | end point run 2 | format direction |
|---|---|---|---|---|---|
| Resolved | 10/24 | 9/24 | 7/24 | 8/24 | 9/24 |
| Garbled-call rate | 0.020 | 0.029 | 0.075 | 0.058 | 0.012 |
| Repeat-loop rate | 0.04 | 0.13 | 0.13 | 0.08 | 0.00 |
| First edit, Kaggle seconds | 297 | 342 | 221 | 295 | 211 |
| Patches touching real source | 0.83 | 0.75 | 0.63 | 0.88 | 0.71 |
| Empty-patch rate | 0.17 | 0.21 | 0.38 | 0.08 | 0.25 |
| Wrong-edit rate | 0.42 | 0.38 | 0.33 | 0.54 | 0.33 |

**The end point is not an improvement.** Both of its draws resolve fewer tasks than both base draws, its garbled-call rate is two to three times the base (temperature 1.0, as the Jacobian's first column predicted), loops are no better, and only the time to the first edit moves the right way. The RM steps therefore did what a stochastic-approximation controller does when the target is unattainable in its box: they kept pushing along the directions that reduce the largest whitened residuals (loops, empty patches, first-edit time) until the bounds, and single draws on rotating chunks disguised the cost on the other coordinates. Two consequences for the method. The target vector must be attainable, or the controller needs a merit gate (the memo's step 7: accept a move only if a held-out draw does not lose resolution), which the pilot deliberately did not have. And the comparison with the format direction is the point of the exercise: a trained direction moved the targeted coordinate by three noise standard deviations at no resolution cost, where a 24-batch sweep of the sampling knobs could not. This is the pilot's result for the paper: the machinery runs unattended, the rank diagnostic is correct, and the knob-only controller shows why directions are needed.

### 6.18 Decisiveness direction, first attempt: a null adapter (2026-10-13)

Mining at the pre-edit states of 51 resolved trajectories (`mine_decisive_pairs.py`, 30 minutes) labelled 88 samples as edits at the known-good location and 286 as further exploration, but same-context pairs need both, and only 7 contexts had them: 42 pairs in all, 13 after the trainer's two-per-context cap. With gradient accumulation of 4 that was **one optimiser step at warm-up learning rate**. The sanity gate passed, as it must for an adapter that equals the base model, and the chain exported it; it is set aside as `adapters_out/decisive_r16_null`. Two fixes: the trainer now refuses to export after fewer than 8 optimiser steps (`--min-steps`), and the fallback chain (`exp_day6b_chain.sh`) mines the earliest informed states (87 contexts, where the trajectory itself kept exploring after seeing the target text) and trains on the union with the pre-edit pairs, four pairs per context, accumulation 2. The lesson for the mining design: the pre-edit state is where the model is already decisive, so it rarely yields the contrast; the contrast lives earlier in the trajectory.

The fallback (05:12 to 05:48) showed the mirror image: at the 87 earliest informed states the model sampled an edit at the right location 35 times against 617 explorations, so same-context pairs were again scarce (15; 57 in the union, below the 60 threshold) and nothing was trained. The contrast is real, the model explores where it could edit, but on-policy sampling cannot supply the chosen side at that state. The next construction takes the chosen from the trajectory itself: its later successful edit rendered in the native call format, paired with sampled explorations at the earlier state (a hindsight pair: the chosen is still the model's own output, conditioned on more context). `build_hindsight_pairs.py` does this offline from the miner's stored samples: the rendering rule (keys sorted, strings in `<|"|>` tokens, no escaping) was checked against the chat template on 51 of 51 sampled `edit_file` calls and against the model's own raw calls on 204 of 256, the rest differing only in key order; each rendered chosen is verified against the file state at the earlier step with the harness matcher (137 of 138 pass). Result: **910 pairs from 137 contexts** (`data/pairs/decisive_hindsight_pairs.jsonl`), trained by `exp_day8_chain.sh` with three pairs per context, accumulation 2, the min-steps and sanity gates, followed by the Usage 2 probe and RM steps over `w_format`, `w_decisive` and temperature (`configs/ie_directions.yaml`). Training (05:52 to 07:33) used 359 pairs over 122 contexts and early-stopped at a window margin of 1.54 after 24 optimiser steps (loss 0.20, pair accuracy 1.00); the gates passed and `adapters_out/decisive_r16` was exported. The greedy sanity generations at training contexts are still `read_file` calls, so the direction is a mild tilt toward editing rather than a switch, which is the right strength for a merge weight to scale.

### 6.19 iE over the two directions: the probe (2026-10-13, in progress)

`configs/ie_directions.yaml`: knobs `w_format` (merge weight of the format direction, base 1.0), `w_decisive` (base 0.0) and temperature (base 0.2); same six targets and the pilot's whitening. Each draw rebuilds the merged adapter (exact rank-32 stack, memo 10.2) and restarts the server. One fixed 24-task chunk (seed 1), single draws:

| batch | w_format | w_decisive | T | resolved | garbled | loops | source edit | first edit (s) | wrong edits | compactions | empty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.0 | 0.0 | 0.2 | 6/24 | 0.001 | 0.00 | 0.88 | 322 | 0.63 | 1.00 | 0.08 |
| w_format 0.5 | 0.5 | 0.0 | 0.2 | 4/24 | 0.018 | 0.08 | 0.79 | 311 | 0.63 | 1.13 | 0.17 |
| w_decisive 1.0 | 1.0 | 1.0 | 0.2 | 3/24 | 0.000 | 0.04 | 0.83 | 225 | 0.71 | 1.04 | 0.17 |
| temperature 0.6 | 1.0 | 0.0 | 0.6 | 4/24 | 0.004 | 0.00 | 0.75 | 382 | 0.58 | 0.96 | 0.17 |

The fitted Jacobian (rows: targets on the transformed scale; columns: `w_format`, `w_decisive` per unit weight, temperature per unit):

| target | w_format | w_decisive | temperature |
|---|---|---|---|
| garbled calls (logit) | −5.27 | −0.56 | +2.80 |
| repeat loops (logit) | −2.91 | +0.72 | 0.00 |
| source edits (logit) | +1.22 | −0.34 | −2.12 |
| first edit (log s) | +0.07 | −0.36 | +0.42 |
| compactions | −0.25 | +0.04 | −0.10 |
| empty patches (logit) | −1.58 | +0.79 | +1.97 |

Singular values **7.2, 2.2, 1.3, condition 5.6**, against 3.6, 1.7, 0.45, 0.006 (condition 591) for the knob pilot. This is the memo's claim in numbers: a basis of trained directions is well conditioned where the sampling knobs were not, and each direction has a dominant coordinate (format: garbled calls and loops; decisiveness: time to the first edit, with a cost in loops and empty patches; temperature: the opposite of format on garbled calls). Single-difference entries, so the small ones are noise.

Two things the knob pilot could not show. The format weight is a clean, monotone control of its coordinate: halving it takes garbled calls from 0.001 to 0.018 and brings loops back; at weight 1 the rate is one call in 800. And the decisiveness direction moves the time to the first edit by about 100 s, which no sampling knob did, with the edit rate unchanged. The cost side is visible in the same draw: wrong edits rise and resolution falls (3/24), the premature-edit risk the memo names in 10.5; one draw on a hard chunk (the base model has none here) cannot separate that from noise, which is what the fixed-set repeat after the RM steps is for. This chunk resolves 3 to 6 of 24 across all points, against 8 to 11 on the pilot's chunks, so compare rows, not columns, with other sections.

### 6.20 iE over the two directions: the RM steps (2026-10-13, in progress)

Each step is one 24-task draw on a fresh chunk; the merged adapter is rebuilt and the server restarted for each point.

| step | w_format | w_decisive | T | resolved | garbled | loops | source edit | first edit (s) | wrong edits | empty | `|W r|` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (base) | 1.0 | 0.0 | 0.2 | 8/24 | 0.002 | 0.00 | 0.79 | 165 | 0.46 | 0.21 | 24.0 |
| 2 | 1.5 | −0.5 | 0.5 | 13/24 | 0.004 | 0.00 | 0.83 | 289 | 0.29 | 0.17 | 26.7 |
| 3 | 1.97 | −0.5 | 0.8 | 12/24 | 0.003 | 0.00 | 0.71 | 275 | 0.21 | 0.25 | 27.6 |
| next | 2.0 | −0.5 | 1.0 | | | | | | | | |

The controller's first move went *against* the decisiveness direction (−0.5, the lower bound of its trust step) and up on the format weight and temperature: with garbled calls and loops already near zero at the base, the residual is dominated by empty patches and loops, which the Jacobian attributes to `w_decisive`, while the first-edit coordinate (165 s at step 1, close to its 120 s target) gave it little to gain. Step 2 resolved 13 of 24, the highest single draw of any run so far, with wrong edits down to 0.29, but on a different chunk; the first-edit time went back up. Whether a negative decisiveness weight is genuinely better, or step 2's chunk is simply easier, is exactly what the fixed-set repeat of the end point (`exp_day9_chain.sh`) is for. Step 3 kept the pattern: resolution 12/24 and wrong edits down to 0.21, but the whitened residual rose again (24.0, 26.7, 27.6) and the controller ended at the bounds on all three knobs (format 2.0, decisiveness −0.5, temperature 1.0), the same signature as the knob pilot (6.15). The garbled-call rate stayed at 0.002 to 0.004 at every point, so the format direction's effect is robust to the other knobs; the interesting and unconfirmed part is the monotone fall in wrong edits (0.46, 0.29, 0.21) as the decisiveness weight went negative and temperature rose, which three chunks cannot separate from task difficulty.

### 6.21 The directions controller's end point on a fixed task set (2026-10-13)

The end point (format 2.0, decisiveness −0.5, temperature 1.0) on dev24 twice (`runs/directions_end_dev24_c8_r1`, `_r2`), against the base, the format-only point and, for contrast, the knob pilot's end point (6.17), which also sat at temperature 1.0:

| | base 1 | base 2 | format only (w 1, T 0.2) | directions end 1 | directions end 2 | knob end 1 | knob end 2 |
|---|---|---|---|---|---|---|---|
| Resolved | 10/24 | 9/24 | 9/24 | **10/24** | **9/24** | 7/24 | 8/24 |
| Garbled-call rate | 0.020 | 0.029 | 0.012 | **0.001** | **0.005** | 0.075 | 0.058 |
| Repeat-loop rate | 0.04 | 0.13 | 0.00 | 0.04 | 0.00 | 0.13 | 0.08 |
| First edit, Kaggle seconds | 297 | 342 | 211 | 274 | 348 | 221 | 295 |
| Patches touching real source | 0.83 | 0.75 | 0.71 | 0.75 | 0.88 | 0.63 | 0.88 |
| Empty-patch rate | 0.17 | 0.21 | 0.25 | 0.13 | 0.13 | 0.38 | 0.08 |
| Wrong-edit rate | 0.42 | 0.38 | 0.33 | 0.33 | 0.50 | 0.33 | 0.54 |

Three readings. First, the directions controller's end point **does not lose resolution** (10 and 9 against the base's 10 and 9), where the knob pilot's end point at the same temperature did (7 and 8). Second, the format weight at 2.0 holds the garbled-call rate at 0.001 to 0.005 *despite* temperature 1.0, which on its own took it to 0.06 to 0.08: the direction compensates the knob, which is the mechanism the memo's Usage 2 is built on. Empty patches and loops also end below the base. Third, against the format-only point the end point is mixed (fewer garbled calls and empty patches, a slower first edit, same resolution), and the negative decisiveness weight cannot be judged from this run because temperature moved with it; 6.22 isolates it.

For the plan this closes the loop the memo asked for in section 10.5: trained directions plus the iE controller reach an operating point with better behaviour coordinates at no resolution cost, which the sampling knobs alone could not do; what the controller still lacks is a merit gate and a target vector it can attain, since it again ran to its bounds.

### 6.22 The decisiveness weight isolated on dev24 (2026-10-14)

Format weight 1 and temperature 0.2 throughout, so only the decisiveness weight varies; the format-only run (6.16) is the zero point. Two draws per point (`runs/decisive_A_dev24_c8_r*`, `runs/decisive_B_dev24_c8_r*`).

| | base 1 | base 2 | decisive 0 | decisive +1, run 1 | +1, run 2 | decisive −0.5, run 1 | −0.5, run 2 |
|---|---|---|---|---|---|---|---|
| Resolved | 10/24 | 9/24 | 9/24 | 9/24 | 8/24 | 8/24 | 10/24 |
| Garbled-call rate | 0.020 | 0.029 | 0.012 | 0.002 | 0.001 | 0.001 | 0.001 |
| Patches touching real source | 0.83 | 0.75 | 0.71 | 0.79 | 0.92 | 0.67 | 0.79 |
| First edit, Kaggle seconds | 297 | 342 | 211 | 233 | 283 | 201 | 195 |
| Empty-patch rate | 0.17 | 0.21 | 0.25 | 0.13 | 0.04 | 0.29 | 0.08 |
| Wrong-edit rate | 0.42 | 0.38 | 0.33 | 0.42 | 0.58 | 0.33 | 0.38 |
| Repeat-loop rate | 0.04 | 0.13 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 |

The direction is a tilt, as its training margin (1.54, early-stopped) suggested. At +1 the agent edits more often and leaves fewer empty patches, the objective it was trained on, but more of those edits are wrong, and the first edit is not sooner on these tasks; at −0.5 the first edit is the earliest of any point and the edit rate is lower. Resolution is within noise at every point (8 to 10 of 24), so neither sign changes the score on dev24; the weight trades empty patches against wrong edits, which is the trade the memo expected (10.5, "premature edits"). One unexplained detail: every merged point has fewer garbled calls (1 to 2 of about 800) than the format-only run (9 of 773), at either sign of the decisiveness weight; the counts are small enough to be noise and the merge was checked to preserve the format block exactly (6.19), so it is noted rather than interpreted.

What this means for the controller: with the format weight fixed, the decisiveness weight moves the edit-rate and empty-patch coordinates in one direction and the wrong-edit coordinate in the other, and resolution not at all on dev24; a controller that targets empty patches will push it positive, as round 1's residual did not (round 1 was dominated by loops, which the Jacobian attributes to the same weight). Round 2 (gated, fixed dev24, attainable targets) is the test of whether the controller finds a better point than format-only rather than a bound.

### 6.23 Round 2: the gated directions controller on fixed dev24 (2026-10-14, in progress)

`configs/ie_directions_gated.yaml`: merit gate on resolution (reference starts at the noise floor's 9.5/24, allowance one task, trust halves on a rejection), targets moved to where round 1 says they are attainable (garbled 0.005, loops 0.02, source edits 0.85, first edit 200 s, compactions 1.0, empty 0.10), every draw on dev24, Jacobian and whitening reused from round 1, four steps from the format-only point.

| step | w_format | w_decisive | T | resolved | gate | garbled | loops | source edit | first edit (s) | wrong edits | empty | `|W r|` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (format only) | 1.0 | 0.0 | 0.2 | 9/24 | accepted | 0.018 | 0.04 | 0.88 | 279 | 0.50 | 0.08 | 9.5 |
| 2 | 1.4 | 0.4 | 0.4 | 6/24 | rejected | 0.001 | 0.00 | 0.67 | 270 | 0.42 | 0.21 | |
| 3 (back to 1) | 1.0 | 0.0 | 0.2 | 8/24 | rejected | 0.001 | 0.08 | 0.67 | 253 | 0.33 | 0.25 | |
| 4 (same) | 1.0 | 0.0 | 0.2 | 8/24 | rejected | 0.009 | 0.04 | 0.75 | 225 | 0.42 | 0.17 | |

The gate did its job once and then misfired. Step 2's draw (6 of 24, three below the reference) was rejected and the controller fell back to the format-only point, as intended. But the reference was a fixed number (the noise floor's mean, 9.5/24) with a one-task allowance, which sits inside the per-draw noise: the format-only point's own draws on dev24 are 9, 9, 8, 8. So the two re-draws of the accepted point were "rejected" too, the trust region halved twice (to 0.125), and the round ended where it began. The round therefore says nothing about whether a better point exists; it says the gate needs a noise-aware reference: a running mean of the draws at the accepted point, an allowance sized by the resolution noise (about two tasks on 24), and no penalty for re-measuring the accepted point. Round 3 runs with that gate.

**Round 3 (revised gate, 2026-10-14 12:08 onward).** Step 1 re-drew the format-only point: 9/24, folded into the reference (now 0.358 over five draws), garbled 0.005, loops 0, source edits 0.88, first edit 288 s, empty patches 0.13, whitened residual **4.8**. With the targets set where the responses say they are attainable, the controller is already close: the residual is mostly the first-edit coordinate (288 s against 200), and its move is small and sensible (format 1.05, decisiveness +0.13, temperature 0.20), a nudge toward the direction the Jacobian says shortens the first edit, rather than a jump. Steps 2 to 4 follow on the same tasks.

| round 3 step | w_format | w_decisive | T | resolved | gate | garbled | loops | source edit | first edit (s) | wrong edits | empty | `|W r|` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (format only) | 1.0 | 0.0 | 0.2 | 9/24 | measured | 0.005 | 0.00 | 0.88 | 288 | 0.50 | 0.13 | 4.8 |
| 2 | 1.05 | 0.13 | 0.20 | 7/24 | accepted | 0.010 | 0.00 | 0.83 | 336 | 0.54 | 0.08 | 8.3 |
| 3 | 1.22 | 0.52 | 0.32 | 9/24 | accepted | 0.003 | 0.00 | 0.88 | 240 | 0.50 | 0.08 | **3.6** |
| 4 | 1.16 | 0.58 | 0.33 | 8/24 | accepted | 0.001 | 0.04 | 0.79 | 302 | 0.46 | 0.17 | 9.8 |
| next | 1.14 | 0.34 | 0.34 | | | | | | | | | |

Round 3 ended at 18:50 after four steps. The whitened residual went 4.8, 8.3, 3.6, 9.8, which is noise around a small value: the controller stayed in a small region (format 1.0 to 1.22, decisiveness 0 to 0.58, temperature 0.2 to 0.34) and never touched a bound, and its last move pulled decisiveness back after step 4's slower first edit. Resolution draws were 9, 7, 9, 8 against the format-only point's 9, 9, 8, 8 on the same tasks, so nothing separates the points on score; the end point (format 1.14, decisiveness 0.34, temperature 0.34) is being drawn twice on dev24 (6.24). Compared with rounds 1 and 2, this is what a working controller looks like: with a gate it can trust and targets it can reach, it hovers near the operating point instead of running at a wall.

Step 3 is the best operating point drawn so far on the behaviour vector: the first edit 48 s sooner than the format-only point with garbled calls at 0.003, no loops, the edit rate and empty-patch rate at their targets, and resolution 9/24, the format-only point's typical draw. The controller's moves stayed small and inside the trust region throughout. One gate detail surfaced: an accepted point's draw replaced the reference even when lower (step 2's 7/24 replaced the 0.358 mean), which would let the bar ratchet down; the rule is now that the reference keeps the better value. Whether step 3's point is genuinely better than format-only is, as always, a question for a repeated draw (`exp_day13_chain.sh`).

Two early readings on round 2. The whitened residual at the start is 9.5 against 24 to 27 in round 1: with attainable targets the controller has something to converge to rather than a wall to run at, and its first move is a moderate one inside the trust region on all three knobs instead of a jump to the bounds. And the format-only point drawn again on dev24 gives 9 garbled calls in about 800 (0.018), matching its earlier draw (9 of 773) and not the 1 to 2 of the merged points in 6.22; the latter was noise, as suspected there.

### 6.24 Round 3's end point on dev24, and what the controller phase established (2026-10-14)

The end point (format 1.14, decisiveness 0.34, temperature 0.34) drawn twice on dev24 (`runs/gated3_end_dev24_c8_r*`):

| | base 1 | base 2 | format only | round-3 end 1 | round-3 end 2 |
|---|---|---|---|---|---|
| Resolved | 10/24 | 9/24 | 9/24 (9, 8, 8 on re-draws) | 10/24 | 7/24 |
| Garbled-call rate | 0.020 | 0.029 | 0.012 | **0.000** | **0.000** |
| Repeat-loop rate | 0.04 | 0.13 | 0.00 | 0.04 | 0.08 |
| First edit, Kaggle seconds | 297 | 342 | 211 | 298 | 271 |
| Patches touching real source | 0.83 | 0.75 | 0.71 | 0.83 | 0.67 |
| Empty-patch rate | 0.17 | 0.21 | 0.25 | 0.13 | 0.21 |
| Wrong-edit rate | 0.42 | 0.38 | 0.33 | 0.42 | 0.38 |

The gated controller's output is as good as the format-only point on score (8.5 against 8.5 on average, inside the noise of the base's 9.5), holds the format coordinate at zero garbled calls, and did not deliver the shorter first edit it was steering toward (that coordinate's single-draw gains in round 3 were noise). What the whole controller phase (6.14 to 6.24) established:

1. **The machinery works unattended**: probes, Jacobians, RM steps, gates, merged-adapter rebuilds and server restarts, dozens of guarded runs, with every failure (sandbox OOM, launcher match, null adapter, gate misfire) diagnosed and fixed.
2. **The memo's rank argument holds in numbers**: the direction basis is well conditioned (5.6) where the sampling knobs are not (591), and a direction compensates a knob (format weight against temperature, 6.21).
3. **Directions move behaviour coordinates reliably and at no score cost**: garbled calls from 2 to 3% to 0 to 0.5%, loops to zero, empty patches down, on fixed task sets.
4. **Neither direction moves resolution on 24 tasks**, and no controller output beats the format-only point on score. A one-task difference is invisible at this size; the format point against the base on the 100 sound tasks (section 6.25) is the test that can see two or three points.
5. **A controller needs a gate and attainable targets**: without them it runs to its bounds (6.15, 6.20); with them it hovers near the operating point (6.23).

### 6.25 Base against the format point on the 100 sound tasks (2026-10-15, in progress)

Both at the calibrated 34-minute cap, concurrency 8, guarded, with the per-command sandbox limits (`exp_day14_chain.sh`), about 5.6 hours each.

| | base (v0_long) | format point, draw 1 | format point, draw 2 | round-3 gated end point |
|---|---|---|---|---|
| Resolved | 31/100, then 38/100 (second draw) | 38/100 | 38/100 | 33/100 |
| Garbled-call rate | 0.042 (127 calls) | 0.003 (9) | 0.006 (18) | 0.002 (6) |
| Repeat-loop rate | 0.10 | 0.02 | 0.01 | 0.01 |
| First edit, median Kaggle seconds | 270 | 230 | 229 | 269 |
| Wrong-edit rate | 0.45 | 0.39 | 0.40 | 0.46 |
| Patches touching real source | 0.76 | 0.77 | 0.78 | 0.79 |
| Empty-patch rate | 0.17 | 0.16 | 0.19 | 0.16 |
| Compactions per task | 1.04 | 1.03 | 1.08 | 0.96 |
| fastapi / rich | 21/60, 10/40 | 24/60, 14/40 | | |

The base's 31 compares with 36 in 6.9, which ran at a 30-minute cap, without per-command limits, before the guarded runner existed; the binomial standard deviation at this rate is about 4.6 tasks, so the two are consistent, and the per-command CPU limit (two cores, as on Kaggle) plausibly costs a task or two on test-heavy commands. Task by task the two base runs agree on 27 resolved; 9 tasks resolved only in the earlier run and 4 only in this one, so about 13 of 100 tasks flip between draws of the same agent. That churn is the yardstick for the format point's run on the same tasks: a difference of three or four tasks is within it, and only a larger one would count. Turn counts, tokens, edit timing and compactions are the same in both base runs (median 30 turns, first edit at 816 and 865 s, 95 and 104 compactions).

**The format point resolves 38 against 31 on the same tasks.** Its behaviour coordinates reproduce the dev24 picture at scale: garbled calls fourteen times rarer (9 against 127 of about 3,000), loops five times rarer, the first edit 40 Kaggle seconds sooner, fewer wrong edits, with turns, compactions and edit rate unchanged. Task by task: 28 resolved by both, 3 only by the base, 10 only by the format point (fastapi 21 to 24 of 60, rich 10 to 14 of 40); an exact two-sided test on the 13 discordant tasks gives p = 0.09, so it is suggestive rather than settled; what makes it credible is that it comes with the mechanism the direction was built for and that the memo expected to convert near-misses (10.5: fewer turns wasted on failed calls). This looked like the first measured score gain of the project; the correction below shows it was not established (a second base draw also resolved 38). The format bundle (`bundles/format1.zip`) remains the candidate on behaviour and is no worse on score.

The round-3 gated end point, run third on the same tasks (`runs/gated3_end_sound100_c8`), resolves 33: it keeps the format direction's behaviour (garbled calls 0.002, loops 0.01) but its added decisiveness and temperature bring no first-edit gain at scale and more wrong edits, and it lands between the base and the format-only point on score. Task by task the gated end point and the base differ on 10 tasks (6 to 4 in its favour, p = 0.75), and it loses 9 to 4 against the format-only point (p = 0.27); 24 tasks are resolved by all three. The ranking at 100 tasks is therefore format-only 38, gated end point 33, base 31, and the controller phase's conclusion stands: the single format direction is the best operating point found, and the controller's job was to establish that rather than to beat it. A second draw of the format point on the same tasks (`runs/format1_sound100_c8_r2`, 15:13 to 20:54) resolved **38 again**: the two format draws share 32 resolved tasks and swap 6 each way, the same churn as the base's two draws, and this draw wins 12 to 5 against the base on the discordant tasks (p = 0.14; the first draw was 10 to 3, p = 0.09). The behaviour coordinates reproduce as well (garbled 0.006, loops 0.01, first edit 229 s). Two independent draws at 38 against 31 looked like a replicated seven-point gain; the base side had one draw at the current settings, so a second base draw ran next (`exp_day17_chain.sh`).

**Correction (2026-10-16): the second base draw resolved 38 as well.** `runs/base_sound100_c8_r2` (20:59 to 02:37, same settings, same server throughput, median 43 tokens/s, and the same number of timeouts, 39 against 40) resolved 38 of 100; its two draws differ by 2 to 9 on discordant tasks (p = 0.07), so the first base draw at 31 was a low draw, not a confound we can find. Against it the format draws tie exactly (5 discordant tasks each way, both). Pooling both draws per arm and pairing by task: base 34.5%, format 38.0%, difference **+3.5 points, 95% paired-bootstrap interval −1.5 to +8.5**; 13 tasks lean to the format direction, 5 to the base (sign test p = 0.10). The score effect is therefore at most suggestive. The time-to-first-edit gain does not replicate either (base draw 2: 227 s, the same as the format draws). What replicates against both base draws is the behaviour the direction was trained for: malformed calls 4.2% and 5.7% (base) against 0.3% and 0.6% (format), and tasks with loops 10% and 13% against 2% and 1%. The lesson is the one this section opened with, at a larger scale than expected: a seven-point swing in the total happens between two draws of the same agent on the same 100 tasks.

### 6.26 Pre-submission checks against the scorer's current build (2026-10-16)

The official wheelhouse was updated on 2026-10-12 (swegemma 0.2.7 → 0.2.11, adk_submission 0.2.12 → 0.2.13, same vLLM 0.19.1 with a new build); every local measurement in this document used the old harness. Before submitting `bundles/format1.zip`:

- **vLLM**: all 1,554 Python files of the new competition wheel were diffed against our patched install. One line differs: the text model's weight loader also skips `vision_embedder.` weights. The LoRA path and the alias fix (6.13's adapters demonstrably change outputs) are identical.
- **Harness**: the changes are robustness fixes. A `ToolErrorPlugin` returns undeclared-tool and runtime tool errors to the model instead of ending the loop; protected and test files are stripped from the patch inside `submit_patch`; verification gets a 300 s timeout; with `include_thoughts: false` thoughts are now stripped from history (we use `true`). The model registry moved from `swegemma.models` into `swegemma.cli` and `swegemma.harness.agent_runner`, so `scripts/run_eval.py` cannot import 0.2.11; `scripts/run_eval_v2.py` is the same runner on the new API, run with `PYTHONPATH=data/harness_v2_site` (the three new wheels installed there with `--no-deps`, shadowing `.venv`), and `run_guarded.py --runner run_eval_v2.py` selects it.
- **Bundle under the new harness**: `validate_directory` and `compile_submission` pass on the unzipped `format1.zip`; the adapter is discovered (`merged`, rank 16), the root agent routes to `openai/merged`, and each request carries `chat_template_kwargs.enable_thinking: true` and `thinking_token_budget: 512`.
- **Smoke run under the new harness** (`exp_day18_chain.sh`, `runs/bundle_format1_v2harness_dev12`, 2026-10-16 02:42 to 03:27): the unzipped bundle served as is, the first 12 dev24 tasks at the calibrated cap. **Passed**: vLLM's `lora_requests_info` gauge (which appears once a LoRA request arrives) shows `running_lora_adapters="merged"` during the run, so requests route through the adapter; 372 requests completed with no errors and one length stop; the harness exited cleanly; 0 malformed calls in 352, no loops, a source edit in all 12 tasks, 5 of 12 resolved.
- **Submission** (2026-10-16 12:53 UTC, with the user's go-ahead): `bundles/format1.zip` submitted as ref 57269654. The upload must be **named `submission.zip`** (the API answers a bare 400 otherwise; the body says "Submission files must be named \"submission.zip\""), so the same bytes were copied to `bundles/upload/submission.zip` (sha256 prefix ca8374c39b71e37a) and submitted with `kaggle competitions submit -c gemma-4-developer-agent -f bundles/upload/submission.zip`. The daily limit is one. **Result: public score 0.13** (scored within 11 hours of submission).

### 6.27 A held-out task set from unfamiliar repositories (2026-10-17)

The submission scored 0.13 on the hidden private repositories against 34.5–38% locally on fastapi and rich, so local scores were not tracking the leaderboard. To get a local set that behaves more like the hidden one:

- **Repositories**: 24 candidates screened by popularity (GitHub API) and activity; kept pure-Python, flat-layout repositories (the local sandbox puts only the workspace on `PYTHONPATH`, so `src/` layouts cannot import), with 338 to 8.8k stars and active in 2026: tenacity, responses, pyjwt, nox, pint, pyparsing, w3lib, sqlite-utils, pydantic-settings, boltons, tinydb, astroid. Very popular repositories (loguru, llm, pre-commit) and inactive ones were left out.
- **Mining** (`scripts/heldout/mine_commits.py`): blobless clones under `data/heldout/repos/`; first-parent commits since 2026-01-01 that change package source (3–150 lines, at most 4 files) and test files, excluding release, dependency, CI and docs commits: 690 candidates.
- **Tasks** (`scripts/heldout/build_tasks.py`): up to 10 per repository spread over the year, 120 in all. Gold `patch` is the source diff, `test_patch` the test diff, the problem statement the commit message (squash merges carry the PR title and body). Snapshots are the base tree as a **single-commit** git repository, unlike the public snapshots which carry full history (the agent's `git log` shows nothing). No code-graph files are built: our agent called those tools twice in 200 public tasks, and a missing graph returns a tool error.
- **Environment**: `.venv-heldout` is a copy of `.venv` with script shebangs repointed and the held-out repositories' runtime and test dependencies installed (plus `typeguard` and `cowsay`, found missing by the first controls). The harness itself runs from it, so task sandboxes see its packages; four held-out packages (tenacity, pyjwt, pydantic-settings, pyparsing) are also harness dependencies and are shadowed by the workspace copy during tests.
- **Soundness controls** under the 0.2.11 harness: 95 of 120 tasks pass with the gold patch and fail with an empty one (`data/heldout/sound_tasks.txt`); 9 pass with an empty patch (type-hint-only, comment-only and revert commits) and 16 fail with the gold patch, including all 10 astroid tasks. 6 to 10 sound tasks per repository over 11 repositories.
- **Evaluation** (`scripts/exp_heldout_chain.sh`, from 2026-10-17 01:12): the base agent (v0_long) and the submitted format bundle on the 95 tasks, 0.2.11 harness, calibrated cap, concurrency 8, guarded (`run_guarded.py --runner run_eval_v2.py --python .venv-heldout/bin/python`).
- **Result for the base agent: 56 of 95 resolved (59%)**, against 31–38% on the public sound tasks and 0.13 on the leaderboard. Per repository from 8/9 (responses) and 7/9 (tenacity) down to 3/9 (w3lib) and 1/8 (pydantic-settings). The agent finds these tasks easier on every coordinate: first edit at 166 Kaggle seconds against 227, median 26 turns against 30, wrong edits 30% against 43%. Unresolved tasks have larger gold patches (median 27 changed lines against 11 for resolved ones). **So unfamiliar repositories do not reproduce the leaderboard gap; this set is easier than the public one**, most plausibly because its statements are commit messages that name the fix precisely (median 213 characters, against 418 for the public PR descriptions) and recent commits in these repositories are small, well-scoped fixes. Repository familiarity is ruled out as the main cause of the 34.5–38% → 0.13 drop; the candidates left are the difficulty of the hidden tasks and the scoring environment itself, and the second can be tested directly by running public tasks on Kaggle's hardware.

### 6.28 The competition's rules changed on 2026-10-13: four allowed base models

The harness README (`data/raw_v2/HARNESS_README.md`, dated 2026-10-13; the copy in `data/raw/` is the original) and the official starter notebook now permit one base model per submission from four: `gemma-4-31b-it-qat-w4a16-ct`, **`gemma-4-12b-it-qat-w4a16-ct`** (the starter's new default; vLLM needs `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'`), `gemma-4-e4b-it-qat-w4a16-ct` and `gemma-4-e2b-it-qat-w4a16-ct`. Everything in this document so far used the 31B, which was the only option before. Other scorer changes in the same update: `gpu_memory_utilization` 0.90 (was 0.80); `scheduler_reserve_full_isl=False`; LoRA settings sized from the bundle (`enable_lora` only when adapters ship, `max_lora_rank` = the largest adapter rank); with `include_thoughts: true` thoughts are kept in the context of later tool calls; the starter compacts every 15 events (the README earlier said 5, which our runner uses).

Under a five-minute cap the 12B is the obvious lever: on bandwidth-bound int4 decoding it should run roughly twice as fast as the 31B, so it gets more turns per task, at some cost in capability. Its weights are in `models/gemma-4-12b-it-qat-w4a16-ct/` (Kaggle `google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2`). The local-to-Kaggle time calibration (7.1) was measured for the 31B only; a 12B comparison at the 34-minute local cap assumes the same speed ratio, which a Kaggle timing run would have to confirm.

### 6.29 Calibration on the scoring hardware (pushed 2026-10-17)

`scripts/build_calibration_notebook.py` builds the private notebook `bojackkk/gemma4-calibration` (pushed with the user's go-ahead, queued from 2026-10-17). Built on the *current* official starter (its install cell verbatim, its server settings: `gpu_memory_utilization` 0.90, `scheduler_reserve_full_isl=False`, LoRA sized to the bundle, the 12B's `--hf-overrides`; compaction every 15 events), it runs on 30 public sound tasks (`data/kaggle_calib_tasks.txt`, 18 fastapi and 12 rich, seed 20261017): a gold-patch control, then v0_long with the 31B and the same agent with the 12B, each after a raw speed probe, tasks one at a time with the five-minute cap like the scorer. Its non-server cells were executed locally against the 0.2.11 harness before pushing (the gold control resolved both test tasks). Output goes to `runs/kaggle_calibration_v1/` via `scripts/wait_kernel.sh`. Reading: if the 31B resolves about a third of these tasks on Kaggle, the leaderboard's 0.13 reflects harder hidden tasks; if it resolves about 0.13 here, the scoring conditions cost the score, and local measurements need recalibrating.

**Version 1 (2026-10-17 09:05–09:20, `runs/kaggle_calibration_v1_probe_bug/`)** ran at once without queueing but stopped after the gold control: the speed probe asked for the model by its short name, which the Kaggle server does not serve (it serves the model path; the earlier timing notebook read `/v1/models` for this reason), and the 404 aborted both agent runs. Its gold control is already a finding: **on Kaggle only 22 of the 30 reference patches pass**, all eight failures being fastapi tasks that grade soundly locally; the output kept no test logs, so the cause is not yet known. Version 2 (pushed 10:24) reads the served ID, makes the probe non-fatal, keeps the test-output tail of every unresolved task, and scores the agents on all 30 tasks and on the Kaggle-gradable subset.

**Version 2 result (2026-10-17 10:25–15:21, `runs/kaggle_calibration_v2/`).**

| on the same 30 public tasks | all 30 | Kaggle-gradable 22 |
|---|---|---|
| Kaggle, 31B (v0_long), five-minute cap, one task at a time | **10** | 10 |
| Kaggle, 12B (same agent) | 6 | 6 |
| local base draws 1 and 2 (31B, calibrated cap) | 8, 12 | 7, 11 |
| local format-direction draws 1 and 2 | 11, 10 | 10, 8 |

- **The scoring environment reproduces local results.** Every task the 31B resolved on Kaggle was resolved locally in at least one base draw. The local cap calibration (7.2) holds, and the drop from about 35% (public, local or Kaggle) to 0.13 (leaderboard) belongs to the hidden tasks themselves. With 6.27 ruling out repository familiarity, the hidden tasks are simply harder than the public ones.
- **Gold control:** the 8 ungradable tasks all fail on `ModuleNotFoundError: No module named 'inline_snapshot'`, the same test library this project had to install locally at the start (5). The Kaggle notebook environment lacks it for the public fastapi tasks; it says nothing about the hidden set.
- **Speed on L4×4:** 31B decode 35–39 tokens/s, prefill 1,250–1,480 (as in 7.1); 12B decode 67–78, prefill about 2,950 (2.0× the 31B). vLLM start-up 714 s for the 31B in this session (454 s in 7.1) and 305 s for the 12B.
- **The 12B is worse under the real budget**: 6 against 10 resolved, 19 against 12 timeouts, median 53 tool calls against 27. It spends its speed on more steps without converging. A 12B submission is therefore not supported (the user's go-ahead was conditional on the comparison supporting it), and none was made.

**Local 12B, 2026-10-17.** The 12B checkpoint failed to load in our vLLM with `KeyError: 'vision_embedder.patch_dense.weight'`, which is exactly the one-line change in the scorer's 2026-10-12 build (6.26); its `gemma4.py` is now installed locally (previous file in `data/wheelhouse/vllm_stock_backup/gemma4_v1_competition.py`), so the local serving code again matches the scorer's. Local speed: decode 24 tokens/s against 10.6 for the 31B (2.3×), prefill about 1,950 against 700 tokens/s, a tool-calling turn 2.5 s against 6.1 s; KV cache 153k tokens against 48k.

### 6.30 A hard held-out set, and the first agent change aimed at it (2026-10-18, in progress)

The easy held-out set (6.27) is easier than the public tasks, and the calibration (6.29) puts the leaderboard gap in the hidden tasks' difficulty, so a local proxy has to be harder.

- **Mining** (`mine_commits.py --since 2025-11-01 --max-src-lines 600 --max-src-files 8`, `data/heldout/candidates_wide.jsonl`): 542 candidates from 18 repositories (the 11 usable ones plus markdown-it-py, parsel, lark, pydantic-extra-types, more-itertools, jsonschema, glom), 197 of them larger fixes (at least 40 changed source lines or at least two source files).
- **Tasks** (`build_tasks.py --hard`, `data/heldout_hard/`): up to 10 per repository, 106 tasks; gold patches a median 47.5 changed lines over 2 files, against 16 lines for the easy set. Controls under the 0.2.11 harness: **76 sound** (85 gold-pass after installing `w3lib`; 9 pass with an empty patch; the azure-source tests of pydantic-settings and 6 nox tasks fail with the gold patch here). Some tasks overlap the easy set (two-file fixes under 40 lines).
- **Terse variant** (`build_tasks.py --terse`, `data/heldout_hard_terse/`): the same 76 tasks with the statement cut to the PR title, PR number stripped, to test whether short statements are what makes hidden tasks hard.
- **Environment**: `.venv-heldout` gained the new repositories' dependencies; pydantic moved from 2.13.5 to 2.14.1 (pydantic-extra-types requires it), so runs before and after 2026-10-18 differ in that package.
- **Runs** (`exp_hard_chain.sh`): the 31B base agent on the hard set, full statements then terse, calibrated cap, 0.2.11 harness.

**First agent change: the `code-tools` skill** (`submissions/v1_skills/`): v0_long plus one skill with two scripts, and the prompt's locate and check steps pointing at them.
- `scripts/locate.py` takes identifiers from the problem statement and prints the best-matching source files (tests and docs excluded) with matched terms, `path:line [Class.method] code` hits and the test files mentioning the terms, at most about 60 lines. On `tenacity_741` it ranked the two files of the gold patch first and second and surfaced the buggy line.
- `scripts/run_tests.py` finds the source files changed since the baseline (git), picks up to five related test files by name and import, runs `pytest -x` with a 100 s limit and prints the summary line plus the first failure. It disables table-printing plugins (`-p no:pretty`), which otherwise replace pytest's summary line.
- Skill scripts run from a temporary directory (ADK materialises them there), so both find the workspace through `/workspace` or `PYTHONPATH`. ADK requires a **kebab-case** skill name (`code-tools`); a snake_case name fails compilation.
- `exp_skills_chain.sh` (queued behind the hard chain) runs a 4-task smoke test, stops unless `run_skill_script` calls succeed inside the sandbox, then evaluates the 76 hard tasks under the same settings as the base.

**Base on the hard set (full statements): 28 of 76 resolved (37%)**, about the public level (34.5–38%) and still far above the leaderboard's 0.13, so larger fixes alone do not reproduce the hidden tasks' difficulty. Outcomes: 28 resolved, 28 timed out after a wrong edit, 11 finished with a wrong edit, 9 timed out without a source edit. Against the easy held-out set: first edit at a median 261 scorer-seconds (166 easy) of the 300 available, wrong edits 51% (30%), compactions 1.03 per task (0.67), loops 13% (8%). The agent finds the fix too late to test and iterate, so the second change aims at speed.

**Second change: `submissions/v2_think0_format`** = v0_long's prompt, thinking off (temperature 0.6, top-k 64, as `v0_think0_t06`) and the format direction (`adapters/merged`, rank 16). Thinking off was always the fastest and most decisive setting and failed on malformed calls (6.4–6.6); the format direction was trained on exactly those thinking-off contexts and removes them. Queued in `exp_v2_chain.sh` after the skills chain.

**Title-only statements look like the missing factor (partial, 2026-10-18 08:56).** With 37 of the 76 title-only tasks finished, the base agent resolved 5 (13.5%, the leaderboard's level) where the full-statement run resolved 14 of the same 37: 9 tasks lost, none gained. The hidden tasks' statements are most likely as uninformative as a PR title, which would explain most of the gap from about 35% locally to 0.13. The title-only hard set is therefore the best local proxy for the leaderboard, and both queued agent changes were retargeted to it (`runs/heldout_hard_terse_skills_c8`, `runs/heldout_hard_terse_v2_c8`).

**Final (10:46): 13 of 76 resolved with title-only statements (17%) against 28 of 76 with full statements, on the same tasks; 15 lost, none gained (exact sign test p = 0.0001).** Wrong edits rise from 51% to 62% and empty patches double (10% to 20%); time to the first edit barely changes (241 against 261 s). With only a title the agent guesses what to change and guesses wrong more often. Agent changes should now be judged on this set, whose 17% sits near the leaderboard's 13%.

**What the title-only losses look like.** Of the 15 tasks lost, about 6 changed only part of what the fix needed: sync but not async (`tenacity/__init__.py` without `tenacity/asyncio/__init__.py`), the implementation but not the package export of a new function, one of the four nox modules that read an option. 7 timed out mid-work. The rest edited the right files with the wrong semantics.

**First change, result (`runs/heldout_hard_terse_skills_c8`, 11:15–15:40): 13 of 76, the same as the base**, swapping 3 tasks each way. The scripts worked in the sandbox (smoke: 10 `run_skill_script` calls, no errors) and were used heavily: `locate.py` in 66 tasks, `run_tests.py` in 41, the skill loaded in 59 (about one extra turn). Behaviour: malformed calls 2.7% (4.7%), loops 9% (15%), but compactions 1.42 per task (1.01) and the first edit later (267 against 241 s). Finding code and running tests faster does not change the score on title-only tasks; the bottleneck is working out what the title asks for. v3 adds exactly that guidance on top of v1_skills, so the v3–v1 comparison isolates it.

**Second change, result (`runs/heldout_hard_terse_v2_c8`, 15:45–19:27): 4 of 76, worse than the base's 13** (10 tasks lost, 1 gained). Thinking off made the agent fast (first edit at 147 against 241 scorer-seconds, 43 against 125 tokens per turn) and the format direction kept malformed calls low (2.2%), but turns doubled (69 against 35), exact-repeat loops rose to 41% of tasks (15%), empty patches to 34% (20%) and source edits fell to 63% (79%); 17 timeouts, and the median task used 71 tool calls. On title-only tasks the model needs its thinking to work out what to change; buying speed by removing it is counterproductive. This also closes the question left open in 6.4–6.9: thinking off does not pay at the real budget.

**Third change: `submissions/v3_skills_complete`** = v1_skills plus a prompt section for title-only statements: read the existing tests of the code first; make the change everywhere it applies (sync and async variants, sibling functions, every reader of a changed option or caller of a changed signature), re-running `locate.py` with the changed names to find the rest; export new names where their siblings are. Queued in `exp_v3_chain.sh` after the v2 chain.

**Third change, result (`runs/heldout_hard_terse_v3_c8`, 19:31–00:09): 13 of 76, the same as the base and v1** (3 swapped each way with the base, 2 with v1). The guidance changed behaviour in two directions that cancelled: wrong edits fell to 51% (62% base, 59% v1), empty patches rose to 30% (20%); turns, first-edit time and compactions matched v1.

**Summary of the three changes on the title-only hard set** (76 tasks, calibrated cap, 0.2.11 harness):

| agent | resolved | vs base (lost / gained) |
|---|---|---|
| base (v0_long) | 13 | |
| v1: `code-tools` skill | 13 | 3 / 3 |
| v2: thinking off + format direction | 4 | 10 / 1 |
| v3: v1 + title-only guidance | 13 | 3 / 3 |

Better code search and test running, and explicit guidance on what a terse title implies, each shift behaviour but not the score; removing thinking costs two thirds of it. Together with the earlier prompt and sampling sweeps (6.3–6.8), the format direction (6.25) and the controllers (6.17, 6.24), this points to the same limit throughout: on underspecified tasks under a five-minute budget, the 31B with its thinking is the binding constraint, and changes to the scaffold around it trade tasks rather than add them.

### 6.31 Paper revision and statement distillation (2026-10-19)

**Paper.** `paper/main.tex` is now titled *What Limits a Five-Minute Coding Agent? Preference Directions, Query-Only Control, and the Cost of Terse Problem Statements* (15 pages): a fourth contribution, section "Where the leaderboard gap comes from" (environment, familiarity, fix size, statement informativeness, the 12B) with the figure `figures/difficulty_ladder.pdf`, section "Scaffold changes on a leaderboard-matched proxy" (Table 4), and revised lessons, limitations and conclusion.

**Statement distillation.** The agent resolves 28 of 76 hard tasks with the full statement and 13 with the title, so the question is whether a direction can teach it to act from a title as it does from a full statement. Design:
- *Privileged teacher, student states.* Building contexts from informed trajectories would leak the statement: the agent's thoughts, kept in later context, quote it ("Wait, the problem statement says ..."). Instead the student (base agent, title only) is run, and at its states the same model with the full statement swapped into the user message is the teacher (`scripts/mine_statement_pairs.py`). Only tasks the informed agent resolved in at least one draw qualify.
- *Pairs.* At each early student state, two student and two teacher completions are sampled; when a teacher call differs from a student call (another tool, file or command), the pair is (student thought + teacher call, student thought + student call). The shared thought cancels in the DPO margin, so only the call is learned and the teacher's statement-quoting thought never enters training text. Edit calls on both sides are checked with the harness matcher against the replayed file state.
- *Split.* Training data comes only from the public fastapi and rich tasks (`data/public_terse/`, statements cut to the PR title, snapshots linked); evaluation is the 76 title-only hard held-out tasks from 15 other repositories.
- *Student run* (`exp_public_terse_chain.sh`, 02:25–08:40): the base agent title-only on the 100 sound public tasks, same runner and settings as `base_sound100_c8`/`_r2`.

**The title-only effect replicates on the public tasks: 21 of 100 resolved, against 31 and 38 with the full PR text** (paired against draw 1: 16 lost, 6 gained, p = 0.05; against draw 2: 21 lost, 4 gained, p = 0.0009). Empty patches rise to 33% (17% and 12%) and the first edit slips to 294 of the 300 scorer-seconds (270 and 227): with a title alone the agent more often never decides what to change. Two independent task sets now show the same drop (34.5% → 21% public, 37% → 17% hard held-out).

*Mining* (`mine_statement_pairs.py`, from 08:42): the first version sent one request at a time (about 75 s per context, nearly 7 h for 320 contexts); at 08:48 it was restarted with `--workers 4` (tasks mined in parallel threads, one shared lock for the output file and the context limit; resumes from the records already written), about 2.5 contexts a minute. *Signal check* on the first 30 contexts (5 tasks): two student samples are materially different 37% of the time, two teacher samples 30%, a student and a teacher sample 56%. So roughly 40% of the mined pairs reflect what the statement adds and the rest are sampling noise; since both sides of a noise pair come from nearly the same distribution, under DPO they add variance rather than bias (the expected gradient is the difference between the teacher's and the student's call distributions). With the full statement the teacher runs commands more (77% of its calls against the student's 62%) and reads files less: it searches for identifiers the statement names instead of opening files by guess. *Prompt length*: the mined contexts run 4.1k–7.7k tokens (median 5.3k), the system block with tool declarations takes 1,707 tokens and the title ends near token 1,750. The chain's `--max-prompt-tokens 4096` (keep the first 512 tokens and the most recent tail) would have cut the title out of 14 of the first 26 contexts with pairs, training the student on contexts without its task. The chain was handed at 08:57 to `exp_distill_chain2.sh` (waits for the running miner, then the same steps with `--max-prompt-tokens 8192`; no mined context exceeds it). An earlier 8192 smoke test peaked at 97 GiB with the server stopped; each context costs about twice as long as at 4096.

**Mining result (08:42–10:05): 307 pairs at 156 contexts over 39 of the 40 qualifying tasks** (306 contexts mined, at most 8 early steps per task before any compaction; the other 150 contexts had student and teacher agree). On the full set the signal is weaker than the first 30 contexts suggested: two student samples are materially different 25% of the time, two teacher samples 30%, a student and a teacher sample 41%, so about a third of the pairs carry what the statement adds. The tool mix is the same on both sides (59% commands, 40% reads); the difference is in where the agent looks, not in what kind of action it takes (163 pairs are command against command, 115 swap a read for a command or back, 8 have the teacher write or edit where the student searched). Training (`logs/train_statement.log`, from 10:05) sees 156 contexts with median 5,431 prompt tokens.

*Training, attempt 1 (10:05–13:36, `logs/train_statement_attempt1.log`)*: margins rose steadily (accuracy 0.31 → 0.94, window margin 0.00 → 1.81) and training early-stopped at 80 of 156 contexts (step 40). The sanity gate then refused the adapter: one of three greedy generations was flagged because it had not closed its call within 256 tokens. The flagged text is a coherent thought ("I'll try to reproduce this with a small script") opening a call that writes a reproduction script, which is what the base agent does at that state too; the cap was set for think-off format training, where a thought plus call is short. The gate then discarded the weights, so the 3.5 hours were lost. Fixes in `train_direction_dpo.py`: `--sanity-tokens` (default 256, used at 1024 here) and a refused adapter is saved to `<out>_quarantine` instead of dropped; the log also shows the generation's tail. *Attempt 2* (`exp_distill_chain3.sh`, from 13:45): same pairs and settings plus `--sanity-tokens 1024`, then the same packaging, evaluation and server restore. Result (13:40–17:09): the same trajectory (window margin 1.79 at 80 of 156 shuffled contexts, step 40, early stop) and all three sanity generations close their calls; the context refused in attempt 1 closes at 332 tokens, confirming the false positive. Saved as `adapters_out/statement_r16`, packaged as `submissions/v4_distill` (v0_long plus the direction at weight 1, merged), evaluated from 17:13 as `runs/heldout_hard_terse_v4_c8` (76 title-only hard tasks, 0.2.11 runner, `.venv-heldout`, cap 34, concurrency 8). The deciding comparison is against the base agent's 13 of 76 on the same set (`runs/heldout_hard_terse_base_c8`).

**Result (`runs/heldout_hard_terse_v4_c8`, 17:13–21:51): 13 of 76, the same as the base** (4 lost: boltons_458, markdownitpy_416, pydanticextratypes_372, sqliteutils_96fab692; 4 gained: markdownitpy_428, pydanticextratypes_376, _410, _446; sign test p = 1.0). Behaviour against the base: malformed calls 2.3% (4.7%), exact-repeat loops 7.9% (14.5%), first edit 302 (241) scorer-seconds, turns 32 (35), tokens per turn 114 (124.5), compactions 1.12 (1.01), empty patches 19.7% and edit attempts 78% (both unchanged), wrong edits 60.5% (61.8%). The direction acts mainly like a weaker format direction plus a later first edit; it does not carry the full statement's information into title-only behaviour on unseen repositories. Likely reasons, none tested: only a third of the pairs carry statement signal (25–30% within-side against 41% cross disagreement); the pairs come from two public repositories (fastapi, rich) and the evaluation from 18 others; and a teacher that differs from the student in one step's call cannot convey a multi-step plan. Server restored to `submissions/v0_long` at 21:54 (`distill chain done`).

Paper updated (2026-10-19): abstract, fourth contribution, section 7 (now "Scaffold and adapter changes", with the distillation as the fourth change and a table row), two lessons (sanity check sized to the mode; measure a pair set's noise floor), limitations, conclusion (the future-work sentence now reports this attempt), related work (DAgger, learning by cheating; two `refs.bib` entries). Public title-only replication added to section 6 and the difficulty-ladder figure. 16 pages.

### 6.32 iE calibration of the statement direction (2026-10-20)

The user asked to try iE on the distillation. The direction from 6.31 changed behaviour but not score, and only about a third of its pairs carry statement signal, so the trained adapter is plausibly a *shrunk* estimate of the teacher's shift: the bias-reduction setting of the dissertation. `scripts/ie_distill.py` solves the moment equation M(w) = alpha for the direction's merge weight w, in the dissertation's form (stationary equation M(θ*) = α, section 3.3; accelerated iteration with one noisy draw per step, sections 3.2–3.3; matrix gain, here scalar, 4.4):

- *States*: 168 title-only agent states (first 6 agent steps before any compaction) from `runs/heldout_hard_terse_base_c8` on the 28 hard held-out tasks the informed agent solved (`runs/heldout_hard_base_c8`); rebuilt with the 0.2.11 harness on the path (`PYTHONPATH=data/harness_v2_site`; 3391 rendered against 3397 recorded tokens; the 0.2.7 tool specs are 75 tokens off). Median 4,472 prompt tokens. None of these repositories was in the direction's training pairs.
- *alpha* (observed): the probability that two independent teacher samples agree (not materially different), the teacher being the base model with the full commit message in place of the title; 4 samples per state, U-statistic over the 6 pairs.
- *M(w)* (simulated): the probability that a student sample (title-only context, v0_long plus w times the direction, served as one merged LoRA) agrees with a teacher sample; 2 student samples per state against the 4 fixed teacher samples. If the student's next call had the teacher's distribution, M = alpha exactly; a malformed or truncated sample counts as disagreement on both sides so the equation stays exact.
- *Iteration*: probe at w = 0 and w = 1 for the slope g; then w_{n+1} = w_n + P/(n^0.6 g) (alpha − y_n), trust region 1.0, bounds [0, 4], guard: a weight whose student samples are more than 25% malformed is rejected and the trust halved. One draw is a merge, a server restart and about 336 student samples (about 15 minutes), against 1.7 hours for one agent draw of the behaviour controller.
- *Then*: if the root differs from w = 1, the agent at the calibrated weight runs on the 76 title-only hard tasks (against 13 for the base and for w = 1). If M does not move with w, or saturates below alpha, the finding is that the gap is information the title does not carry, which no weight on this direction can supply.

**Probe results (02:25–03:48).** Teacher sampling: 168 states in 20 minutes, alpha = 0.779 (se 0.025), teacher malformed 3.7%. Student draws (about 13–16 minutes each plus a 5-minute restart):

| w | M(w) | se | student malformed |
|---|---|---|---|
| −1 | 0.687 | 0.031 | 3.0% |
| 0 (base) | 0.682 | 0.031 | 2.7% |
| 1 (the 6.31 direction) | 0.641 | 0.030 | 1.8% |

Paired over the 168 states: M(1) − M(0) = −0.041 (se 0.020), M(0) − M(−1) = −0.005 (se 0.016); least-squares slope over the three points −0.023 (se 0.011). **On the held-out repositories the trained direction moves the student's next call away from the informed teacher's, not towards it**, and reversing it does not help: M is flat from −1 to 0 and drops towards +1, with the base close to its maximum and 0.09 below alpha. So the moment equation has no root along this direction, the scalar gain is negative, and iE steps towards the lower bound (first iterate: a Newton step from the base asks for w = −4.25, clipped by the trust region to −1). The first chain stopped at the w = 1 probe because the script then required a positive slope and w ≥ 0; it was rewritten for signed weights in [−3, 3], a third probe and a paired noise test (the old chain was stopped after an in-place edit of its own script: bash reads a running script by offset). The six RM steps run (about 20 minutes each) to show whether M rises further out; the agent run on the 76 tasks is now gated on the draws at the final weight beating M(0) by more than two paired standard errors (`exp_ie_distill_chain2.sh`).

**Result (03:31–09:05): no root.** All draws (`runs/ie_distill/state.json`, per-state files `draw_*.jsonl`):

| w | draws | M(w) | student malformed |
|---|---|---|---|
| −3 | 4 (steps 3–6) | 0.005–0.010 | 96–99% |
| −2 | 1 (step 2) | 0.512 | 20.5% |
| −1 | 2 (probe, step 1) | 0.687, 0.698 | 3.0%, 2.7% |
| 0 | 1 (probe) | 0.682 | 2.7% |
| 1 | 1 (probe) | 0.641 | 1.8% |

alpha = 0.779. Pooled M(−1) − M(0) = +0.010 (paired se 0.015). The iteration: least-squares slope −0.023 (paired se 0.011), first iterate −1 (Newton step −4.25 clipped), then −2, then the bound −3, where the guard rejected all four remaining draws. **The guard had a sign bug**: on rejection it stepped to `w − trust`, which assumed an upward iteration; moving downwards that is further down, so it stayed at the bound and re-drew it four times (each about an hour, since malformed samples run to the token limit). Fixed in `ie_distill.py`: a rejected weight now goes halfway back to the last accepted weight. A rerun with the fix would only hover between −2 and −1, where M ≤ 0.69 < alpha, so it was not repeated. The gate found no weight worth an agent run (`calibrated weight: none`); the base server was restored at 09:05.

**Reading.** On unseen repositories the statement direction moves the agent's next call *away* from the informed teacher's (consistent with its unchanged score in 6.31), reversing it does not help, and large weights of either sign break the call format. The base model sits near the best agreement any weight on this direction reaches, 0.09 below the teacher's self-agreement: the gap between the title-only and the informed agent is information the title lacks, not a policy bias a weight can remove. The calibration costs a fraction of an agent run and would have predicted 6.31's null result before its 4.5-hour evaluation.

Paper updated: abstract (closing sentence), the iE contribution, a paragraph and Figure 5 (`paper/figures/ie_moment.pdf`, `make_figures.py: ie_moment()`) in section 10, a lesson (check reachability at fixed states before an agent run), limitations (states are the base agent's, agreement criterion is ours), conclusion. 17 pages.

### 6.33 A fixed-state reachability check for any new direction (2026-10-20)

The user asked to make the 6.32 check a standing gate: before an agent run on a new direction, measure at fixed agent states what the direction does to the next call, and whether a target is within reach. `scripts/reach_check.py` generalises `ie_distill.py`; `scripts/exp_reach_chain.sh` validates it against the agent runs we already have; `scripts/reach_validate.py` scores the predictions.

- **Sampling must be the agent's own.** `/v1/completions` ignores `thinking_token_budget` (a 16-token budget left a 343-token thought at 349 tokens), while `/v1/chat/completions` enforces it (14 reasoning tokens against 2,997 unbudgeted). The mining (6.31) and the iE calibration (6.32) therefore sampled hotter (0.6) and with unbounded thoughts; that is consistent within a moment equation but not the agent's regime. `reach_check.py` uses the chat endpoint with the submission's own sampling (`configs/sampling.yaml`: temperature 0.2, top_p 0.95, 2,048 output tokens, thinking budget 512) and classifies the parsed tool call the harness receives.
- **Statistics per state** (mean over 4 samples): malformed (no call, truncated, unparseable, missing required arguments) for the malformed-call rate; exact repeat of the previous call for the loop rate; edit or write for edit attempts and empty patches; completion tokens for tokens per turn; agreement with a full-statement teacher where one exists. Changes are paired by state against a reference configuration.
- **States**: three per task, evenly spread over each trace's pre-compaction steps (median 19 such steps; most first edits fall inside). `public`: 300 states from `runs/base_sound100_c8` (0.2.7 rendering, 4,994 against 5,000 recorded tokens); `hard_terse`: 224 states from `runs/heldout_hard_terse_base_c8` (0.2.11 rendering), 84 of them with a teacher context (tasks the informed agent solved).
- **One server for every configuration**: base by name plus six merged LoRAs (format 1, format 2, format 1 + decisiveness 1, format 1 + decisiveness −0.5, statement 1, statement 2). The draws are prefill-bound (about 500–800 prompt tokens/s with adapters mounted, median public prompt about 7k tokens), so a public draw takes about an hour and parallel requests do not help.
- **Validation targets** (agent runs, mean over draws): format 1 against base on the 100 public tasks: malformed 5.0% → 0.4%, loops 11.5% → 1.5%, edit attempts 82% → 77%, empty patches 14.5% → 17.5%, tokens per turn 120 → 111. Decisiveness +1 against format 1 on dev24: edit attempts 75% → 88%, empty patches 25% → 8%. Decisiveness −0.5: 75% → 73%, 25% → 19% (within dev24 noise). Statement 1 against base on the title-only hard set: malformed 4.7% → 2.3%, loops 14.5% → 7.9%, tokens per turn 125 → 114.

**Result (`logs/exp_reach_chain.log`, 11:12–19:42; draws in `runs/reach/`, scoring `runs/reach/validation.json`).** Fixed-state means (public set: base, then paired changes; hard set likewise):

| config | malformed | repeat | edit | tokens/call | agree |
|---|---|---|---|---|---|
| public base | 2.9% | 14.0% | 4.8% | 138 | |
| format 1 (vs base) | −0.8 (se 0.5) | +0.9 (0.8) | +0.5 (0.6) | −11.8 (6.3) | |
| format 2 (vs base) | −1.4 (0.8) | −0.1 (0.6) | +1.3 (0.8) | −10.6 (6.5) | |
| format 1 + decisiveness 1 (vs format 1) | +0.3 (0.4) | −1.2 (0.6)* | +0.8 (0.6) | +8.6 (6.7) | |
| format 1 + decisiveness −0.5 (vs format 1) | +0.5 (0.3) | −0.2 (0.5) | −0.2 (0.5) | +12.1 (6.5) | |
| hard base | 1.6% | 15.1% | 5.8% | 113 | 0.821 |
| statement 1 (vs base) | −0.2 (0.2) | −2.2 (1.3) | −0.3 (0.5) | +29.0 (7.3)* | −0.037 (0.019) |
| statement 2 (vs base) | +0.6 (0.4) | −5.6 (1.6)* | −1.1 (1.0) | +57.7 (12.6)* | −0.143 (0.032)* |

(percentage points except tokens and agreement; * beyond two paired se). Teacher self-agreement under the agent's sampling: alpha = 0.859 at 84 states (16 minutes).

**The check does not predict agent-level behaviour.** Of 20 comparisons with a matching agent coordinate, one fixed-state change is beyond noise (statement tokens per call, +29) and its sign is opposite to the agent's (tokens per turn 125 → 114); the uncalled changes match the agent's sign about half the time. The gate rejected the format direction ("no effect detected" on malformed calls), the one direction with a robust agent-level effect (5.0% → 0.4% malformed, 11.5% → 1.5% loops, replicated). Reading: those effects act through the trajectory (avoiding failure states, small per-call nudges that accumulate into earlier or more edits), and single next calls at the base agent's states, before any compaction, cannot show them; the per-call malformed rate under the vLLM parser (2.9%) is also below the harness's garbled-call rate (5.0%), so part of what the harness counts is not visible in a parsed call. What the check does measure is per-call statistics: under the agent's own sampling, agreement with the teacher falls 0.821 → 0.784 → 0.679 at w = 0, 1, 2 against alpha 0.859, confirming 6.32.

**Consequences.** `reach_check.py` now says so in its docstring and verdicts: "no effect detected" replaces "not reachable" (it rules out a per-call effect, not a trajectory-level one), and a root on the unprobed sign is reported as such instead of "extrapolated". Use it to gate per-call objectives (teacher agreement, distillation); keep agent runs for loops, malformed calls and edit timing. A version that could gate trajectory-level effects would need short rollouts with tool execution from fixed states, or states drawn from the adapted agent's own runs; neither is cheap. Paper updated (the iE contribution, a paragraph in section 10, the lesson rewritten, a limitation; 18 pages).

### 6.34 Trajectory-level distillation (2026-10-21 to 10-24)

The paper's open question, run at the user's request: instead of correcting single calls (6.31), imitate whole informed trajectories in title-only contexts. The informed agent's resolved runs are the teacher; the student sees the same history with the statement cut to its title and is trained to emit what the informed agent emitted at each step.

- **Data** (`scripts/build_traj_sft.py` → `data/traj_sft/trajectories.jsonl`): 153 resolved trajectories from four informed runs: `base_sound100_c8` (31), `base_sound100_c8_r2` (38), `v0_long_sound100_c8` (36; 30-minute cap) on the public tasks, and `heldout_base_c8` (48 after excluding the 8 resolved tasks that are also in the hard evaluation set) on the easy held-out set; 93 distinct tasks, 33 with more than one trajectory. 11 of the easy set's 12 repositories also appear in the hard set (different commits); the paper's familiarity result says this matters little, but it is a caveat. Steps before the first compaction only (2,679); per step the prompt is `build_messages` of the trace with the full statement replaced by its title (first line minus `(#N)`; verified equal to the public and hard title-only statements) and the target is the generation form `<|channel>thought\n{thought}<channel|><|tool_call>call:{name}{args}<tool_call|>` with arguments serialised as the chat template does (sorted keys, `<|"|>` strings). 351 steps whose thought quotes a 5-gram of the statement that the title lacks keep the thought as conditioning but mask it from the loss; 130 steps whose edit the harness rejected are skipped (kept in the history); 178 text-only steps are skipped. Public traces render under 0.2.7 and the easy held-out traces under 0.2.11 (6 tokens from the recorded prompt lengths in both), so the builder ran once per harness path.
- **Trainer** (`scripts/train_traj_sft.py`, `.venv-vllm`): rank-16 LoRA on the same seven projections, mean token NLL of the target, lr 5e-5 with 10-step warm-up and cosine decay, 4 steps per update, one epoch over shuffled trajectories. The steps of one trajectory share a KV cache: after a step, the cache is reset to its prompt and extended (no gradient) with the next prompt's new tokens, which the dry run verified is exact (0 prefix violations over 2,167 prompts: each step's prompt extends the previous one token for token, because Gemma 4 keeps calls and responses inside one model turn). Prompts capped at 12,000 tokens (86 trajectories cut there; 1,961 trainable steps remain; median prompt 7.2k, p90 10.9k; target median 118 tokens), chunked backward (96 tokens, 48 beyond 8k). Checkpoint every 20 trajectories; the sanity gate at 1,024 tokens; a refused adapter is quarantined.
- **Chain** (`scripts/exp_traj_chain.sh`, from 2026-10-21 about 10:50): train (`logs/train_traj.log`) → `submissions/v6_traj` (v0_long + adapter at weight 1) → per-call check at the 224 hard title-only states (`runs/reach/hard_terse/v6_traj.jsonl`; agreement with the teacher at 84 states, the objective class the check is valid for) → `runs/heldout_hard_terse_v6_c8` on the 76 title-only hard tasks (base 13, single-step distillation 13) → behaviour vector → base server restored.

**Attempt 1 (2026-10-21 08:26 – 10-22 08:37, `logs/train_traj_attempt1.log`) failed on memory.** Trajectory 1 (19 steps, prompts to 9.8k tokens) trained in 248 s, 13 s per step, with call NLL 0.058 and thought NLL about 0.3, and peaked at 101 GiB. Trajectory 2 reached an 11.4k-token prompt, peaked at 108 GiB, took 100 minutes for 7 steps, and the process then made no progress for 22 hours before it was killed (rc 137): the machine's 121 GiB are unified, and allocations past about 100 GiB spill into the 16 GiB swap file. The cost is about 4 GiB per 1k prompt tokens: the HF cache is duplicated while a chunk is concatenated onto it (the snapshot keeps the old tensors) and the full-attention layers' scores grow with the prefix.

**Attempt 2 (from 10-23 21:58, `logs/train_traj.log`).** Shared-cache prompts capped at 8,192 tokens (1,227 steps; the DPO runs trained at this length at 99 GiB); the 1,383 later steps, which the cache cannot follow, are trained alone on a truncated context (512-token system/task head plus the most recent 7,680 tokens, the DPO trainer's rule): every edit or write (68) and a 20% sample of the rest, 303 in all. Chunks 64 (48 beyond 6k), the expandable-segments allocator, and a memory guard that lowers the cap by 1,000 tokens whenever a trajectory peaks above 100 GiB. Estimated 4.4 h for the shared-cache steps and about 6 h for the truncated ones (one 8k prefix pass each), then the 1-hour per-call check and the 4.5-hour agent run.

**Attempt 2 trained (10-23 21:58 – 10-24 12:14, 14.2 h).** 1,530 steps, 383 updates, no prefix fallbacks, peak 95–96 GiB throughout (the guard never fired). Mean token NLL over 10-trajectory windows fell from 0.32 (trajectories 1–10) to 0.14 (50), 0.14 (100), 0.19 (150) and 0.13 at the end; call-token NLL from 0.07 to 0.03–0.09. Note the base model already assigned the informed agent's calls a mean NLL of only about 0.07 per token in title-only context: at most steps the informed agent's next call is nearly what the title-only agent would do anyway, which is the single-step finding of 6.31–6.32 seen from the likelihood side. All three sanity generations closed their calls (110–180 tokens); the adapter is `adapters_out/traj_r16` (`submissions/v6_traj`).

*Per-call check* (12:19–12:57, `runs/reach/hard_terse/v6_traj.jsonl`, the agent's own sampling, paired against the base at the 224 hard title-only states): agreement with the full-statement teacher 0.794 against 0.821 (−0.028, se 0.024; the single-step direction gave −0.037), tokens per call −19.6 (se 7.5), repeats +1.9 points (se 1.1), edit share +0.8 points (se 0.6), malformed unchanged at 1.6%. Imitating whole informed trajectories did not make the next call more teacher-like at the base agent's states; it made the thoughts shorter. Since per-call statistics are what this check can read (6.33), this is a real null on the per-call objective, and the trajectory-level effect is left to the agent run.

**Result (`runs/heldout_hard_terse_v6_c8`, 12:57–17:09): 13 of 76, the same as the base (13) and the single-step direction (13)**; against the base 3 lost (boltons_458, markdownitpy_416, pydanticextratypes_372) and 3 gained (pydanticextratypes_376, _410, tenacity_614), p = 1.0; against v4 2 lost, 2 gained. Behaviour against the base: malformed calls 1.1% (4.7%), exact-repeat loops 7.9% (14.5%), tokens per turn 107 (124.5), turns 31.5 (35), edit attempts 82% (79%), empty patches 18.4% (19.7%), scratch writes 7.9% (13.2%), first edit 241 scorer-seconds (241), wrong edits 62% (62%), compactions 1.05 (1.01). The imitation adapter is the cleanest title-only agent so far (fewest malformed calls of any run on this set, half the loops, fewer scratch files, slightly more edits) and resolves exactly as many tasks. Server restored to `submissions/v0_long` at 17:12.

**Reading.** Both forms of statement distillation, single-call (6.31) and whole-trajectory (this section), and the iE calibration between them (6.32) agree: the title-only agent's policy is not what separates it from the informed agent. The training loss says the same from the likelihood side: before any training, the base model assigned the informed agent's calls a mean 0.07 nats per token in title-only context, so at most steps the informed agent does what the title-only agent would do; the statement decides a few steps (which file to change, whether the fix has a second half), and no imitation of the rest supplies that. What remains for the leaderboard is the format bundle already submitted; the title-only proxy has now rejected scaffold changes, speed, two distillations and a weight calibration.

### 6.35 Paper-track format and the writeup (2026-10-24)

The paper track's official pages are client-rendered and unreadable by fetch; two independent secondary sources (a competition guide and a competition listing) agree on the format: **a Kaggle Writeup of at most 3,000 words** (title and subtitle, abstract, introduction, methods and experiments, related work with citations), with an optional public link to an arXiv-ready PDF or notebook; judged on novelty, quality, relevance, verifiability and clarity (each 0–5, 20%); awards for Overall Best Paper, Best New Resource and Best New Application; deadline 2026-11-12 23:59 UTC (one listing says 11-13); entering the main competition is not required; a purely LLM-generated paper "will not score well". These are unverified against the official page and the user should confirm them there.

Consequence: the judged artifact is a condensed writeup, with the 18-page paper as the attachment. `paper/writeup.md` is that draft (sections as required, two tables, 16 references); `paper/main.tex` keeps the full account (abstract cut from 593 to 314 words; the iE figure moved to float near its text). Still needed from the user: the author block, a public link for the PDF and code (the held-out task sets are built from public commits and can be released), and the official rules check.

### 6.36 iE as the outer loop of DPO (2026-10-28 to 10-29): feasible, and the same null as the directions

The user asked how iE could update parameters alongside DPO or other RL algorithms. In parameter space the iE iteration needs a direction, which only gradients (policy gradient) or a basis (the DPO directions) can supply; the controller rounds of 6.19–6.24 were the basis form. The form that is both feasible here and new is the primal-dual one: DPO is the primal step and iE the dual update of the training settings, in the dissertation's single-draw Robbins–Monro form. `scripts/ie_train_loop.py` (config `configs/ie_train.yaml`, chain `scripts/exp_ie_train_chain.sh`):

- **Knobs** (all log-scale): the loss weights of two pair sources, format (`format_pairs.jsonl`) and decisiveness (`decisive_hindsight_pairs.jsonl`), and the early-stop margin (0.8–3.0; x0 = 1.5, the anchored run's value). The statement pairs are left out (two thirds noise, no useful direction). `train_direction_dpo.py` now takes several `--pairs name=path:weight` sources, scales each source's DPO and anchor gradients by its weight, and samples a fixed, seeded 30 contexts per source (`--max-contexts-per-source`), 2 pairs each: 120 pairs per retrain, prompts truncated to 4,096 tokens as for the original directions, up to 2 epochs, lr 1e-5, anchor 0.2, sanity gate at 1,024 tokens, min 8 steps.
- **Draw**: stop the server, retrain from scratch at x (about 1–2 h), package v0_long + the adapter (merge at weight 1), serve, run the fixed 24-task batch of the title-only hard set (seed 0; the base resolves 3 of them) with the 0.2.11 runner at the calibrated cap (about 1.4 h), take the behaviour vector. A retrain the trainer's gates refuse is an infeasible point: the move is rejected without an evaluation and the trust region halved.
- **Targets** on the batch: malformed 0.01, loops 0.03, first edit 200 s, empty patches 0.12, compactions 1.0. Whitening from the base, v4 and v6 runs restricted to the batch (base: malformed 1.9%, loops 12.5%, first edit 301 s, empty 20.8%, compactions 0.88), with per-coordinate floors (0.35 logits for rates, 0.15 log for the first edit, 0.10 for compactions) because three runs put the first edit's spread at 0.045 and it would have dominated the residual; base whitened residual 5.6 after the floors (11.0 without). Gate on resolution with a two-task allowance, reference the base's 0.125 with the running-mean rule of 6.23.
- **Schedule**: probe (base retrain plus one retrain per knob: source weights halved, margin ×1.6; a refused perturbation is retried at half the step), four RM steps (P 1, β 0.6, trust 0.5 for the weights and 0.3 for the margin in log units), then the accepted end point on all 76 tasks only if its whitened residual on the batch is below the probe base's; the base server is restored at the end. About 3–4 h per draw, eight draws, roughly 30 h; launched 2026-10-28 (`logs/exp_ie_train_chain.log`, per-draw trainer logs under `logs/ie_train_mix/`, state `runs/ie_train_mix/state.json`).

**Probe (2026-10-28, 00:16–15:36).** Four draws, 3.5–4.5 h each (retrain 115–167 min, serve 4 min, batch 1.5 h). Every retrain reached the stop margin inside the second epoch (32–48 steps) and passed the sanity gate; none was refused.

| draw | x (w_format, w_decisive, margin) | malformed | loops | first edit | empty | compactions | resolved (24) |
|---|---|---|---|---|---|---|---|
| base agent, v4, v6 (noise runs) | no adapter / earlier adapters | 1.9% | 12.5% | 301 s | 20.8% | 0.88 | 3, 4, 4 |
| probe_base | 1.0, 1.0, 1.5 | 0.4% | 0.0% | 274 s | 25.0% | 0.88 | 5 |
| probe_w_format | 0.50, 1.0, 1.5 | 0.3% | 0.0% | 240 s | 20.8% | 1.17 | 5 |
| probe_w_decisive | 1.0, 0.50, 1.5 | 0.5% | 8.3% | 265 s | 29.2% | 1.21 | 2 |
| probe_stop_margin | 1.0, 1.0, 2.4 | 0.0% | 4.2% | 275 s | 20.8% | 1.38 | 5 |

Jacobian (d z / d log knob; z is the logit of a rate, the log of the first edit, compactions as is), condition 11.4, singular values 4.63, 2.07, 0.41:

| | w_format | w_decisive | stop_margin |
|---|---|---|---|
| malformed | 0.70 | −0.12 | −4.07 |
| loops | 0.00 | −2.11 | 1.52 |
| first edit | 0.20 | 0.05 | 0.01 |
| empty | 0.34 | −0.31 | −0.50 |
| compactions | −0.42 | −0.48 | 1.06 |

Readings. The stop margin is the malformed-call knob (a longer run cuts them from 0.4% to 0, and at the 0.4 singular value it is nearly the only thing the third direction does). Halving the decisiveness weight brings loops back (0 → 8%) and costs three resolved tasks, the first time that source has shown a resolution effect; so at this mixture the decisiveness pairs are doing the loop suppression, not the format pairs (whose weight halved changes loops not at all). The first edit responds to nothing: its largest entry is 0.20 per log unit of the format weight, which is the same answer the directions gave on dev24 (6.22). Both retrains at the anchored settings and the format-halved retrain resolve 5 of 24 against 3, 4, 4 for the noise runs, inside the two-task allowance, so not evidence yet.

**Equality targets would have pushed the wrong way.** The retrain at x0 is already below three of the five targets (malformed 0.4% vs 1%, loops 0 vs 3%, compactions 0.88 vs 1.0), so the equality residual at the probe base, whitened, is [+1.30, +1.07, −2.10, −2.45, +0.65]: two fifths of its squared norm asks for *more* malformed calls and loops. The first move it produced (x = 0.61, 0.69, 1.11) lowered the decisiveness weight mainly because loops were "too low". I stopped the chain 25 min into that retrain (`train_step_001_equality_abandoned.log`), made every target a ceiling (`side: max` in the config; `residual_of` clips the residual at zero on the satisfied side, so the equation is M(θ)=α on the active coordinates and "hold" on the rest), reset the state to the accepted point, and relaunched the four steps from the same Jacobian (`scripts/exp_ie_train_chain2.sh`, `logs/exp_ie_train_chain2.log`). The one-sided residual at the probe base is 3.23, all of it first edit and empty patches, and the first move is x = (0.61, 0.96, 1.20): the format weight down by the full trust step, the margin down a little, the decisiveness weight held. The equality-target first move and this one agree on the format weight and the margin; they differ on the decisiveness weight (0.69 against 0.96). Linear prediction for the move: first edit −2.10 → −1.43, empty −2.45 → −2.32, i.e. the model expects the first edit to move and the empty-patch rate hardly at all, consistent with no knob having a large entry on either. The controller rounds of 6.19–6.24 used equality targets placed at attained values, which avoided this by hand; the ceiling form is the same thing done by the residual.

**Steps.** Each row is one retrain and one batch draw; the gate's reference starts at the base's 0.125 and follows the running-mean rule.

| step | x (w_format, w_decisive, margin) | malformed | loops | first edit | empty | compactions | resolved (24) | residual | gate | next x |
|---|---|---|---|---|---|---|---|---|---|---|
| probe base | 1.00, 1.00, 1.50 | 0.4% | 0.0% | 274 s | 25.0% | 0.88 | 5 | 3.23 | reference | 0.61, 0.96, 1.20 |
| 1 | 0.61, 0.96, 1.20 | 0.0% | 8.3% | 325 s | 20.8% | 1.17 | 4 | 4.89 | accepted (0.167) | 0.37, 1.20, 0.93 |
| 2 | 0.37, 1.20, 0.93 | 0.1% | 8.3% | 338 s | 25.0% | 1.33 | 4 | 5.54 | accepted (0.167) | 0.22, 1.45, 0.80 (bound) |
| 3 | 0.22, 1.45, 0.80 | 0.3% | 0.0% | 316 s | 29.2% | 1.08 | 4 | 4.32 | accepted (0.167) | 0.14, 1.42, 0.80 (bound) |
| 4 | 0.14, 1.42, 0.80 | 0.3% | 0.0% | 318 s | 25.0% | 1.00 | 4 | 3.95 | accepted (0.167) | 0.10 (bound), 1.37, 0.80 (bound) |

Step 1 moved the first edit the wrong way (274 → 325 s; the linear model expected 274 → 248) and brought loops back (0 → 8.3%). Against the probe draws, the first edit at 0.61 sits between the 240 s of 0.50 and the 274 s of 1.00, so a swing of a third between neighbouring weights is this coordinate's draw-to-draw noise on 24 tasks, not a response; the 0.15 floor on its whitening spread was, if anything, too low. Loops at 8.3% match the decisiveness-halved probe, though that weight barely moved (0.96), which says loops too are noisy at the single-draw level on this batch (noise runs: 12.5%, probe draws: 0, 0, 8.3, 4.2%). The RM step after it raises the decisiveness weight (loops are above their ceiling and that column is the only one that moves them) and cuts the format weight again by the full trust step.

Step 2 continued in the same direction and the residual rose again (3.23 → 4.89 → 5.54): the first edit 338 s, empty patches back at 25%, compactions 1.33, loops unchanged at 8.3%, resolution 4 of 24 (accepted; the reference stays 0.167). The three draws along the format-weight axis now read 274, 325, 338 s at weights 1.00, 0.61, 0.37, the opposite sign to the probe's 240 s at 0.50 and to the Jacobian entry built from it; the controller is following a column whose sign one draw set. The next move cuts the format weight by the full trust step a third time (0.22), raises the decisiveness weight to 1.45 and puts the margin on its lower bound.

Step 3 (format 0.22, decisiveness 1.45): loops back to zero, which is what the raised decisiveness weight was for, but the first edit stays at 316 s and empty patches rise to 29%, the worst of the run; residual 4.32, resolution 4 of 24 again. With loops satisfied the remaining residual is again first edit and empty, and the move for step 4 is a smaller cut of the format weight (0.14) with the other two knobs held, the gain now at 1/4^0.6 = 0.44.

Step 4 (format 0.14): first edit 318 s, empty 25%, compactions 1.00, loops 0, resolution 4 of 24; residual 3.95. The chain's decision was `none` (best accepted point is the probe base itself, 3.23) so no 76-task run; the base server was restored at 06:39 on 2026-10-29. Total: 8 retrains and 8 batch draws, 30.4 h.

**Result (2026-10-29).** Mechanically the primal-dual loop works: eight retrains, none refused by the trainer's gates, every draw accepted by the merit gate, a Jacobian with condition 11.4. As a controller it did not move the residual below its starting value: 3.23 at the probe base, then 4.89, 5.54, 4.32, 3.95 over the four steps. What happened is simple to state. The two format coordinates (malformed calls, loops) were at or under their ceilings from the first retrain and stayed there (loops came back at 8% in steps 1–2 and went again when the decisiveness weight rose). The two that carried the whole residual, first edit and empty patches, did not respond to the knobs at all at the resolution of one 24-task draw: over the eight retrains the first edit's spread is 0.12 in log units (mean 294 s) and the empty rate's 0.19 in logits, the same size as the three noise runs' spread on the batch (276–301 s; 17–29%), and the correlation between the log format weight and the log first edit across the eight retrains is −0.56, the opposite sign to the Jacobian entry (+0.20) that one probe draw at weight 0.50 (240 s, the lowest first edit of the eleven draws) had set. The accelerated iteration keeps that Jacobian fixed, so it walked the format weight down by the full trust step three times in a row on the strength of one low draw, and the first edit went from 274 s to 316–338 s. Resolution: 5, 5, 2, 5 on the probe, 4, 4, 4, 4 on the steps, against 3, 4, 4 for the noise runs; nothing outside the allowance.

What this adds to 6.24 and 6.33. The directions controller (6.24) reached the same end on dev24 through the merge weights of fixed adapters; this round reaches it through the training settings that make the adapters, which is the stronger form of the question, since the pairs themselves are re-weighted and the adapter is rebuilt each time. The answer is unchanged: the DPO pairs we have reach the format coordinates, which the base retrain already satisfies, and do not reach the planning coordinates (first edit, empty patches) that decide resolution on the hard set, so there is nothing for the outer loop to steer. The fixed-state check (6.33) said the same thing at the per-call level. For the paper: the primal-dual coupling is feasible and cheap to add to any DPO pipeline (one config, one script, the trainer's weighted sources), its diagnostics are informative (the Jacobian localises each knob's effect and the staircase in the stop margin), and its null result is the same null as the directions', obtained a different way. Two method notes for anyone repeating it: use ceiling targets from the start, and estimate a Jacobian column from more than one draw when a coordinate's draw-to-draw spread is comparable to the response being measured (here, a two-draw column would have cost 4 h more per knob and would have caught the sign).

**The stop-margin knob is a staircase.** The trainer checks the window margin every four steps and the window mean jumps from about 0.6 at step 28 to 1.6 at step 32, when the second epoch starts and the pairs are seen again. So every margin in [0.62, 1.64] stops at step 32 (the four runs at 1.2–1.5 all did) and margins up to about 2.6 stop at 48 (the 2.4 probe). The knob's lower bound of 0.8 cannot shorten training below x0's 32 steps, so the controller's downward moves on it (1.5 → 1.20 → 0.93) are no-ops and its Jacobian column is the 32 → 48 step difference. Downward, the controller is effectively over the two source weights. Kept as is (a new knob means a new probe column, 4 h); recorded for the write-up.

### 6.37 Release bundle for the paper track (2026-11-01)

The paper and writeup say the scripts and held-out task sets are released with the code, so `release/` assembles what a public repository needs: `scripts/`, `configs/`, the submission configurations without weights, `paper/` (source, figures, PDF, writeup), these notes, the three task sets as definitions plus candidate commits (snapshots are rebuilt from the public repositories by `build_tasks.py`; `tasks/README.md` lists the 15 source repositories with their licenses, looked up through the GitHub API), the three pair files and the trajectory file, `summary.json` and per-task results for 53 runs with the controller states, the notebooks, and `assets/` (the format adapter, 490 MB, and the submitted bundle) kept out of the tree by `.gitignore` for upload as release assets. About 35 MB without the assets. A scan found no absolute paths or credentials in the release; the Kaggle username appears in notebook metadata. Not included: the organisers' harness and data, model weights, traces, `kaggle.json`, the dissertation, and the iE/DPO memo (the user's own documents). Still the user's: the license, the author block and citation, and creating and pushing the repository.

The Kaggle API (`competitions/list`) confirms the paper track's deadline as 2026-11-12 23:59 UTC and its total reward as 35,000 USD; the overview and rules pages remain client-rendered and unreadable by fetch (tried again on 2026-11-01), so the word limit and the format are still the secondary sources' account (6.35).

### 6.38 iE de-biasing of the selected configuration's score (2026-11-01)

The user asked what other iE applications remain; the first of the list was the one with no GPU cost: reporting the best of several configurations evaluated on the same tasks inflates the winner's rate (the winner's curse that produced the "replicated seven-point gain" of 6.25), and iE in its original role removes that bias. `scripts/ie_debias.py`:

- **Statistic and parameter.** For K configurations with d_j draws each on n shared tasks, α is the vector of observed pooled rates sorted ascending, and x the vector of true rates by observed rank; M(x) is the expectation of the sorted vector when outcomes are simulated at x. The analytic iteration x_{k+1} = x_k + (α − M(x_k)) (dissertation §3.3) runs with M from 20,000 simulated replicates under common random numbers, so M is smooth in x; the iterate is kept sorted by pooling adjacent violators, so with tied observed rates the iteration stops at the boundary with a residual instead of crossing.
- **Simulator.** Two-way logistic model logit P(resolved) = a_i + b_j (task difficulty, configuration effect), a_i ~ N(0, τ²), with (b, τ) by maximum marginal likelihood, the a_i integrated out by 40-node Gauss–Hermite quadrature. A MAP fit was the first attempt and failed: with one to thirteen outcomes per task every task became deterministic (the incidental-parameter problem; the Laplace criterion kept pushing τ up) and the iteration diverged. Each replicate draws the fixed tasks' difficulties from their posteriors given the data, then outcomes for the actual design (same tasks, same draws per configuration). Rates are inverted to b by bisection. The fitted τ is 4–6 logits: most tasks resolve always or never, and the draw-to-draw disagreement comes from a few mid-probability tasks.
- **Model check.** The model is not fitted to the between-draw disagreement, and reproduces it: 11.1 tasks expected to differ between two base draws on the public 100 against 11 observed; 3.9 against 3 on dev24.

| family (n tasks, K) | observed → de-biased (of n) | top − base, observed → de-biased (model sd) | best observed if all K had the base's rate (sd) |
|---|---|---|---|
| public 100, base vs format (2 draws each) | format 38.0 → 38.0; base 34.5 → 34.5 (converged in one iteration: no bias) | +3.5 → +3.5 (2.5) | 35.4 (1.8) |
| public 100, base, format, gated end point | format 38.0 → 37.9; base 34.5 → 33.8; gated 33.0 → 33.8 (pooled) | +3.5 → +4.1 (2.4) | 35.4 (1.8) |
| hard 76 title-only, 6 configurations | unchanged: 13, 13, 13, 13, 13 and 4 (no root; residual 0.029) | 0 → 0 (2.7) | 15.4 (1.5) |
| dev24 at the calibrated cap, 7 configurations (13 draws) | all pool to 8.8 of 24 (observed 7.5–9.5) | 0 → 0 (1.4) | 10.2 (0.7) |

**Reading.** The paired design leaves little room for selection: with two arms separated by 3.5 tasks the expected maximum is the larger true rate to within a tenth of a task, so the format direction's +3.5 is unchanged, and the question is its model standard deviation of 2.5 tasks, which says what the paper already says (not distinguishable from zero). With three arms the two lower ones pool, and the winner's curse for the best of three at this noise is 1.6 tasks (35.4 against 33.8). On the proxy and on dev24 the observed spreads are *smaller* than noise alone would produce (the best of six equal configurations would read 15.4 on average, not 13; the best of seven, 10.2, not 9.5), so the equation has no root with distinct rates and the honest de-biased statement is that the configurations share one rate: no change is above the base on the proxy, and every point the controllers found on dev24 is the base's rate. That is the quantitative form of "within noise". Results in `runs/ie_debias/results.json` (seed 0, 20,000 replicates; `public_seed{0,1}.json` agree to a tenth of a task). About two hours of work, six seconds of compute.

### 6.39 Calibrating the disagreement miner's threshold, and consensus mining (2026-11-01)

The second item of the list. The statement miner (6.31) emits a pair where one student and one teacher sample are materially different, and two samples from the same side differ too, so a share of every pair set is noise; `materially_different` has one continuous knob, the Jaccard cutoff t on the calls' targets (differ if J < t), beside the edit rule and the tool rule. Stationary equation M(t) = α with M the same-side disagreement rate and α a chosen false-pair rate. Data: the 168 fixed title-only states of 6.32 (28 hard tasks the informed agent solved, repositories absent from the pairs), the four teacher samples per state already drawn, and a fresh four-sample student draw at the base (`ie_distill.py draw --base --student-samples 4`, 17 min; the draw records now keep the labels). `scripts/calibrate_miner.py` computes per-state same-side rates (6 pairs per side) and cross-side rates (16 pairs), averaged over states; M is observed directly, so the equation is solved by monotone search, with a split-half check (t chosen on odd states, evaluated on even). Results in `runs/ie_distill/miner_calibration.json`.

| cutoff t | same-side, student | same-side, teacher | cross-side r | π = (r − q)/(1 − q) | signal fraction of pairs | pairs per 100 states |
|---|---|---|---|---|---|---|
| 0 (targets ignored) | 4.0% | 6.3% | 5.7% | 0.5% | 9% | 6 |
| 0.05 (disjoint targets) | 14.3% | 14.9% | 24.7% | 11.8% | 41% | 25 |
| 0.5 (the miner's) | 18.6% | 22.1% | 32.5% | 15.3% | 37.5% | 32 |
| 1 (any difference) | 26.1% | 31.1% | 41.8% | 18.5% | 32% | 42 |
| consensus, t = 0.05 | 3.0% | 1.0% | 12.2% | 10.4% | 83% | 10 |
| consensus, t = 0.5 | 4.1% | 3.3% | 16.5% | 13.3% | 78% | 11 |

Findings. (1) The curve is a staircase: the same-side rate jumps when disjoint targets start to count (t = 0.05) and again at 0.35 and 0.55; between the steps the cutoff changes nothing. (2) α = 5% is reached only at t = 0, where the criterion ignores targets and the cross-side rate is 5.7%: no signal left. The statement's effect is in where the agent looks, and no cutoff separates it from the student's own stochasticity at these states (same-side 15–26% at every t ≥ 0.05); the best signal fraction any cutoff gives is 41% (t = 0.05, false-pair rate 14.6%), against 37.5% at the cutoff used. The split-half check agrees (t chosen on odd states 0.05; on even states q 11.9%, r 26.6%). (3) π, the fraction of early states at which the statement changes the next call under the mixture "either it matters here or cross-side disagreement is same-side noise", is 12–15%: about one early state in seven. This is the per-call number behind "the statement decides a few steps" (6.34). (4) Consensus mining does what the cutoff cannot: define each side's decision at a state only where two of its samples agree, and emit a pair where the two consensus decisions differ. Measured with disjoint halves of the four samples (three splits averaged): false-pair rate 2.0% (t = 0.05) or 3.7% (t = 0.5), signal fraction 83% or 78%, consensus defined at 78% or 69% of states, 10–11 pairs per 100 states — the same number of *signal* pairs as single-sample mining (12 per 100 states) at four times the purity, for four samples per side instead of one.

Not retrained: the distilled direction's failure (6.31–6.32) is that the title lacks the information, and purer pairs do not supply it; the result is a method note for disagreement-based miners (the paper's lesson "measure how much of a pair set is signal" now has a constructive half). CPU work plus one 17-minute draw.

### 6.40 iE calibration of the proxy's informativeness knob (2026-11-02)

Item 3 of the 2026-11-01 list (6.38 is item 1, 6.39 item 2). The paper infers that the hidden statements carry "about a title's worth" of information; this makes the inference a measurement with the one-dimensional iE iteration.

- **Knob** (`scripts/heldout/build_frac_tasks.py` → `data/heldout_hard_fNNN/`): the statement becomes the terse title (PR number stripped, exactly as `heldout_hard_terse`) followed by the first ⌈f·L⌉ of the commit body's L non-empty lines; f = 0 reproduces the title-only set line for line, f = 1 the full statement without its PR number (10 of the 76 sound tasks have no body). Snapshots, graphs and embeddings are symlinks to `heldout_hard`. f = 0.25 keeps 27% of the body lines, f = 0.5 keeps 51% (median statement 330 characters against 58 and 495).
- **Equation** (`scripts/ie_proxy.py`, state `runs/ie_proxy/state.json`): M(f) is the base agent's resolution rate on the 76 sound hard tasks at fraction f, α = 0.13 the leaderboard score of the submitted agent. Analytic iteration f_{n+1} = clip(f_n + (α − M(f_n)) / G, 0, 1) with the Newton gain from the secant of the two existing end-point draws, G = M(1) − M(0) = 0.197 per unit fraction; a step that lands on a bound where the residual keeps its sign has converged at that bound.
- **Chain** (`scripts/exp_ie_proxy_chain.sh`, 08:32–22:28, `logs/exp_ie_proxy_chain.log`): first draw at f = 0.5, then iE steps; on convergence at a bound, one more draw at f = 0.25 for the shape of the curve. Runs `runs/proxy_f{050,000,025}_base_c8`, the same runner and settings as `heldout_hard_terse_base_c8` (v0_long, 0.2.11 harness, held-out environment, calibrated cap, concurrency 8); 4.6 hours each.

| body kept | resolved | rate | binomial se | which draw |
|---|---|---|---|---|
| 0 (title only) | 13, 13 | 17.1% | 3.1% (152 task-draws) | 6.30's draw, and step 1's iterate |
| 0.25 (27% of lines) | 15 | 19.7% | 4.6% | curve point |
| 0.5 (51% of lines) | 21 | 27.6% | 5.1% | the first iterate |
| 1 (full statement) | 28 | 36.8% | 5.5% | 6.30 |

**Iteration.** f_0 = 0.5: M = 0.276, residual −0.146, Newton step −0.74, raw iterate −0.24, clipped to 0. Step 1 at the bound: 13 of 76 again; pooled M(0) = 0.171, residual −0.041, raw next −0.21: converged at the lower bound after two draws. **No root in [0, 1]**: M(0) − α = +4.1 points, 1.34 pooled binomial standard errors of M(0), and the leaderboard's own error is unknown because its task count is hidden (3.4 points if it were 100 tasks). [2026-11-05: the public score is 8 of 58 tasks; against that the gap is 3.3 points, 0.6 standard errors of the difference. See 6.42.]

**Findings.**
1. **No fraction of the statement reproduces the leaderboard's level; the title-only set is the nearest and sits about one standard error above it.** On this task set the hidden statements are worth at most a title, and what remains of the gap is inside the noise of either number. The proxy cannot be made harder along this knob, and needs no further handicap.
2. **The body's information is not front-loaded.** The first quarter of the body lines adds 2 tasks, the second quarter 6, the second half 7. The first lines of a commit body usually restate the title; the specifics (which call, which case, what should happen instead) come later. The resolved sets are not nested (14 of the quarter-body's 15 and 19 of the half-body's 21 are in the full-statement set; 4 and 6 lie outside the title-only union), as draw noise of 3 tasks each way predicts.
3. **The proxy now has a noise floor.** The two title-only base draws resolve 13 and 13, share 10 tasks and swap 3 each way. 6.30's agent variants at 13 each (v1, v3, v4, v6) are indistinguishable from re-draws of the base, which is what the paper's "trade tasks rather than add them" needed measured rather than argued. One draw can see a change of about 4 tasks on 76.
4. **Behaviour along the curve** (`behaviour_vector.py`): empty patches 22% and 20% at title-only, 12% at a quarter, 18% at a half, 10.5% at full; wrong edits 57–62%, 64%, 51%, 51%; source edits 74–79%, 84%, 79%, 88%. At half the body the wrong-edit rate already equals the informed agent's, and the remaining resolution difference is in edit coverage.

Cost: three draws, 14 GPU-hours; the iteration spent two draws (the midpoint and the bound) and the third bought the curve's shape. Paper: a paragraph after the informativeness paragraph of the gap section, the ladder figure gains the body-fraction points and the replicate, contribution 4 extended; writeup: the gap paragraph.

### 6.41 The informed agent's behaviour as the controller's target (2026-11-02 to 11-03)

Items 4 and 5 of the 2026-11-01 list. The paper's controllers (6.2x, 6.36) aimed at hand-set targets and at ceilings; the question left open was whether the behaviour vector of an agent that *sees* the statements is a point the title-only agent's knobs can reach. This run makes that behaviour the target and lets iE say how much of the distance is inside the knobs' span, with the Jacobian columns estimated on all 76 tasks (item 5) and re-estimated as steps accumulate.

- **Target**: the measured behaviour of `heldout_hard_base_c8` (full statements, 28 of 76): source-edit rate 0.882, wrong-edit rate 0.513, empty-patch rate 0.105 as equalities; first edit 260.9 scorer-seconds, malformed calls 4.6%, loops 13.2% as ceilings (`side: max`, residual zero when the draw is under the ceiling). `configs/ie_informed.yaml`, `targets_from_run`.
- **Knobs**: the merge weights of the decisiveness, trajectory-imitation and statement-distillation directions (`merge_directions.py`, bounds [−1, 1.5] and [−1, 2]) and the thinking budget on the log scale (512, bounds [128, 2048]); trust region 0.5 per step on every knob; gain P = 1, β = 0.6, rcond 0.1; merit gate on resolution with allowance −0.06 against the mean of the base draws.
- **Draws**: every draw is the full 76-task title-only set at concurrency 8 under the calibrated cap (4.6 h), so each column of the Jacobian is a 76-task difference rather than the 24-task differences of the earlier controllers. The two title-only base draws (`heldout_hard_terse_base_c8`, `proxy_f000_base_c8`) give the base point and the whitening (per-coordinate sd of the transformed coordinates with binomial floors: 0.270, 0.233, 0.281, 0.15, 0.168, 0.332); the trajectory and statement columns come from 6.30's v6 and v4 draws, which are those directions at weight 1 on the same tasks; the decisiveness and thinking-budget columns were drawn fresh (22:33–02:52 and 02:58–07:44 on 11-03, `logs/exp_ie_informed_chain.log`). `scripts/ie_control.py` gained `targets_from_run`, `probe_base_from`, per-knob `probe_from`, ceiling targets, a first move computed from the probe base without a draw, and a ridge refit of the Jacobian from all accumulated runs after each step (`jacobian: {refit, min_runs: 6, ridge: 1.0}`).

| draw | resolved | source edit | wrong edit | empty patch | first edit s | malformed | loops |
|---|---|---|---|---|---|---|---|
| base (title only), two draws | 13, 13 | 0.789, 0.737 | 0.618, 0.566 | 0.197, 0.224 | 241, 273 | 4.7%, 5.9% | 14.5%, 13.2% |
| decisiveness 1 | 14 | 0.803 | 0.618 | 0.184 | 281 | 0.3% | 2.6% |
| trajectory 1 (v6) | 13 | 0.789 | 0.618 | 0.184 | 241 | 1.1% | 7.9% |
| statement 1 (v4) | 13 | 0.776 | 0.605 | 0.197 | 302 | 2.3% | 7.9% |
| thinking 1024 | 13 | 0.724 | 0.553 | 0.250 | 275 | 6.8% | 13.2% |
| **target** (full statements) | **28** | **0.882** | **0.513** | **0.105** | **≤ 261** | **≤ 4.6%** | **≤ 13.2%** |

**Probe.** The Jacobian on the transformed scale (logit, log) per unit knob:

```
                     decisive   traj   statement   log-thinking
source_edit_rate       0.227   0.146     0.069       -0.307
wrong_edit_rate        0.109   0.109     0.054       -0.234
empty_patch_rate      -0.165  -0.165    -0.079        0.325
first_edit_s           0.092  -0.064     0.163        0.101
garbled_call_rate     -2.843  -1.585    -0.840        0.408
repeat_loop_rate      -1.779  -0.625    -0.625       -0.080
```

Singular values 3.93, 0.61, 0.23, 0.08 (condition 47); whitened, 21.0, 2.37, 1.05, 0.39. The first singular direction is the format lever (decisiveness, trajectory and statement in the ratio 0.84 : 0.46 : 0.26, acting on malformed calls and loops); the three direction columns are nearly collinear there, and the edit coordinates, where the target differs from the base, load onto the second to fourth directions.

**Residual at the base and what the knobs can reach** (`scripts/ie_informed_report.py`): the residual α − M(x₀) is +0.83 logit of source edits, −0.32 of wrong edits, −0.82 of empty patches, 0 on the first edit (the base mean, 256 s, is under the ceiling), −0.13 of malformed calls and −0.06 of loops; |W r| = 4.53, against about 2.4 expected for a re-draw of the base (six coordinates at one sd each). Projected on the span of the whitened Jacobian, 77% of the squared residual is orthogonal at the controller's rank 2, and 54% at full rank 4. More than half of the distance between the title-only agent and the informed agent's behaviour is outside what the four knobs do to first order, and the reachable half is on the two weakest directions.

**First move** (from the probe base, no draw): du = (−0.14, +0.32, −0.22, −0.68 → −0.50 by the trust region), x₁ = decisiveness −0.14, trajectory 0.32, statement −0.22, thinking 311. The pseudo-inverse spends the thinking budget because its column is the only one whose edit coordinates point the right way together (less thinking: more source edits and fewer empty patches, at the price of more wrong edits), and the tilt among the three directions is set by their small differences on the edit coordinates, since on the format coordinates they are collinear. The rank-2 model predicts |W r| = 4.06 after the move, a tenth of the residual.

**Step 1** (x₁ = decisiveness −0.14, trajectory 0.32, statement −0.22, thinking 311; 07:49–12:13): 13 of 76, accepted (reference 0.171, allowance −0.06). Source edits 0.697, wrong edits 0.526, empty patches 0.289, first edit 252 s, malformed calls 3.4%, loops 10.5%. All three ceilings are now met, so the residual lies on the edit coordinates alone: +1.17 logit of source edits, −0.05 of wrong edits, −1.24 of empty patches; |W r| = 6.20, up from 4.53. The move was predicted to raise source edits and cut empty patches; the draw did the opposite (source edits 0.76 → 0.70, empty patches 0.21 → 0.29), a sign error of the first-order model on exactly the coordinates it was spending the knobs on. The refit over 7 draws (ridge 1 toward the probe) halved the thinking-budget column's edit entries (−0.31 → −0.16 source edits, 0.33 → 0.15 empty patches per log unit) and the whitened Jacobian's second singular value fell from 2.37 to 1.56, below the rank cutoff (0.1 × 20.8): from step 3 on, the pseudo-inverse keeps the format lever only. Step 2's move, computed from the probe Jacobian before the refit, continues the same direction with two knobs at the trust limit: x₂ = decisiveness −0.45, trajectory 0.82, statement −0.64, thinking 189.

**Step 2** (x₂ = decisiveness −0.45, trajectory 0.82, statement −0.64, thinking 189; 12:19–16:58): 12 of 76, accepted (allowance −0.06 is 4.5 tasks). Source edits 0.658, wrong edits 0.500, empty patches 0.276, first edit 256 s, malformed calls 4.3%, loops 26.3%: the loop ceiling is now broken by a factor of two, which is what a negative decisiveness weight does (6.22's isolated draws had the same sign), and the edit coordinates moved further from the target. |W r| = 7.03. The refit over 8 draws cut the thinking-budget column's edit entries to a sixth of the probe's (−0.045 source edits per log unit) and the fourth singular value to 0.02 (condition 192): with two draws at low budgets both showing fewer edits, the budget column has lost its edit response and keeps only its malformed-call entry. The step-3 move from the rank-1 pseudo-inverse is (+0.03, +0.02, +0.01, −0.01): the controller has stopped moving, at x₃ = decisiveness −0.41, trajectory 0.84, statement −0.63, thinking 188, and step 3 is in effect a re-draw of step 2's point.

**Step 3** (x₃ = decisiveness −0.41, trajectory 0.84, statement −0.63, thinking 188; 17:04–21:31): 11 of 76, accepted. Source edits 0.737, wrong edits 0.592, empty patches 0.224, first edit 268 s, malformed calls 5.3%, loops 19.7%; |W r| = 5.29. Steps 2 and 3 are the same point within 0.03 on every knob, so their difference is the proxy's re-draw noise there: 12 and 11 resolved with 7 shared, source edits 0.66 and 0.74, loops 26% and 20%, residual 7.0 and 5.3. The final refit over 9 draws has whitened singular values 20.9, 1.43, 0.86, 0.07; the next move would be (+0.03, +0.02, +0.01, 0.00). The chain restored the base server at 21:34 (`runs/ie_informed/`, `runs/ie_informed/report.json`).

| point | resolved | source edit | wrong edit | empty patch | first edit s | malformed | loops | \|W r\| |
|---|---|---|---|---|---|---|---|---|
| probe base (two draws, mean) | 13, 13 | 0.763 | 0.592 | 0.210 | 256 | 5.3% | 13.8% | 4.53 |
| step 1: −0.14, 0.32, −0.22, 311 | 13 | 0.697 | 0.526 | 0.289 | 252 | 3.4% | 10.5% | 6.20 |
| step 2: −0.45, 0.82, −0.64, 189 | 12 | 0.658 | 0.500 | 0.276 | 256 | 4.3% | 26.3% | 7.03 |
| step 3: −0.41, 0.84, −0.63, 188 | 11 | 0.737 | 0.592 | 0.224 | 268 | 5.3% | 19.7% | 5.29 |
| target (informed agent) | 28 | 0.882 | 0.513 | 0.105 | ≤ 261 | ≤ 4.6% | ≤ 13.2% | 0 |

**Findings.**
1. **The informed agent's behaviour is not a point these knobs reach, and the equation says so before the first step.** At the probe base 54% of the squared whitened residual lies outside the span of all four Jacobian columns and 77% outside the controller's rank-2 span; with the Jacobian re-estimated from nine draws, 91–99% along the run. The residual went 4.5 → 6.2 → 7.0 → 5.3 and never fell below the base's. The target's edit coordinates are the score by another name: source edits = resolution + wrong edits holds exactly in all nine draws, so the equalities ask for 28 of 76 in disguise, and nothing in these knobs moves resolution. Over the seven title-only draws (two base, two probes, three steps) the resolved pool is 20 distinct tasks, 17 of them among the informed agent's 28; 11 of the informed agent's tasks were never resolved by any title-only draw.
2. **Re-estimating the Jacobian (item 5) stopped the controller after one wrong step instead of at a bound.** The probe's thinking-budget column, one 76-task difference, said less thinking brings more source edits; the step-1 draw at 311 showed fewer. The ridge refit over 7, 8 and 9 draws cut that column's edit entries to a sixth of the probe's and dropped the whitened Jacobian's second singular value from 2.37 to 1.56 and 1.43, below the rank cutoff, so from step 2 the pseudo-inverse kept the format lever only and the move went to a few hundredths. The pilots of 6.17–6.2x followed a wrong sign to the box bound for the rest of the run; this one took one step (the second, computed before the refit, carried the overshoot). What stopping costs is that the one lever left acts on coordinates whose residual was already near zero.
3. **The sign structure the run exposed.** A negative decisiveness weight brings loops back (10.5% at −0.14; 26% and 20% at −0.45 and −0.41), the sign 6.22's isolated draws gave; the thinking budget below 512 costs edits on this set (source edits 0.70, 0.66, 0.74 at 311, 189, 188 against 0.74–0.79 at 512 and 0.72 at 1024), so the base's 512 sits near the top of a flat maximum, and the earlier controllers' push toward 2,048 was no better founded than this one's toward 128.
4. **The merit gate is not a safeguard against drift.** All three steps passed, 13, 12, 11 against a reference of 13 and an allowance of 4.5 tasks; a loss of a task per step is inside the allowance every time and also inside what two draws of the base show (13 and 13 sharing 10). The gate guards one large loss, and only replicated draws tell a drift from the noise, which is what the stationary controller did at step 3 in effect.

Cost: five fresh draws (two probe columns, three steps), 23 GPU-hours from 22:33 on 11-02 to 21:31 on 11-03; the base and two columns came from existing runs. Items 4 and 5 of the 2026-11-01 list are done, which closes that list. Paper: a paragraph and table after "iE as the outer loop of DPO" in the scaffold section, the controller lesson extended, a limitation and a conclusion sentence; writeup: the outer-loop paragraph.

### 6.42 The public leaderboard in tasks, the public budget frame, and a test of it (2026-11-05)

The plan of 2026-11-05 rested on "our levers do not move score, yet others score nearly double" (0.24 against our 0.13). Checked against the board itself, the second half does not hold as stated, and the check exposed the one lever every comparison of ours held fixed.

**The board in tasks** (`scripts/leaderboard_stats.py` on the snapshot of 2026-11-05 23:43 UTC, `data/kaggle_lb_20261105/`, 4,849 teams; the snapshot holds user names and is not in the release). The public score is k/58 truncated to two decimals: 58 is the only task count in 20–300 for which every one of the 16 distinct scores is achievable and no achievable value in the bulk is unoccupied, and no count fits under rounding. Our 0.13 is 8 of 58.

| score | 0.10 | 0.12 | 0.13 | 0.15 | 0.17 | 0.18 | 0.20 | 0.22 | 0.24 | 0.29 |
|---|---|---|---|---|---|---|---|---|---|---|
| tasks of 58 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 17 |
| teams | 271 | 331 | 446 | 707 | 874 | 768 | 337 | 66 | 9 | 1 |

**The board shows best draws.** Median 9 tasks over all teams (2,762 above our 8, 446 level, 1,641 below).

| submissions | teams | mean tasks | median | share at 13 or more |
|---|---|---|---|---|
| 1 | 964 | 5.45 | 6 | 0.1% |
| 2 | 658 | 6.60 | 7 | 0.3% |
| 3–4 | 876 | 7.67 | 9 | 0.1% |
| 5–8 | 905 | 8.69 | 9 | 0.8% |
| 9–16 | 892 | 9.79 | 10 | 2.1% |
| 17–40 | 552 | 10.91 | 11 | 8.3% |

The 76 teams at 13 or more have a median of 18 submissions against 4 overall; the top six are (17 tasks, 18 submissions) and (14; 37, 26, 25, 22, 22). The trend mixes selection with effort (half of the one-submission teams are at 5 or below, broken bundles), so it bounds selection and does not isolate it. Among one-submission teams our 8 has 623 below, 104 level and 237 above.

**Identical files, repeated** (public notebooks pulled to `data/kernels_20261105/` and read as text, nothing run). Afshar's analysis of 2026-10-22 traced 141 scored public versions to the files they submit: 6 file sets scored at least twice, 29 draws, pooled sd of one draw 1.56 tasks (0.56 of the equal-rate binomial's 2.79, the same heterogeneity our replicate draws show). The most-run set, "Budget-Fit" (Henao), averages 10.00 of 58 over ten draws (95% interval 8.98–11.02); the four sets with three or more draws average 9.25–10.00. The files behind a 0.22 badge ("budget-aware v2", 13 on its first run) re-drew 7, 7, 8, 9, 10, 10, 10. Gluzdov's summary of 2026-11-03 adds a 31-task local screen (four configurations at 9, 9, 8, 8) and that the 12B with the same limits overran twelve hours and received no score.

**What this does to our numbers.**
1. Like for like, the best-documented public agent is worth 10.0 tasks and ours drew 8 once: two tasks, 1.3 sd of a single draw. The entries at 13 and 14 are consistent with top draws of agents worth about 10 (best of 20 draws at sd 1.56 adds about 3); the leader's 17 stands three tasks clear and is unexplained.
2. The proxy comparison of 6.40 loosens: title-only 26 of 152 task-draws (17.1%, se 3.1) against 8 of 58 (13.8%, binomial se 4.5) is 3.3 points, 0.6 standard errors of the difference, where 6.40 had 1.34 against an unknown task count. With α = 0.138 the iteration ends the same way (raw iterates −0.20 from the half-body draw and −0.17 at the bound).
3. Our base has never been submitted, so the format direction's effect on the hidden set is unmeasured.

**The public frame.** The Budget-Fit bundle has no adapter; it differs from ours in the budget frame and the prompt:

| | Budget-Fit (public) | ours (v0_long) |
|---|---|---|
| `max_time_minutes` | 8 | 5 |
| `max_tool_calls` / `max_turns` | 28 / 80 | 100 / 250 |
| thinking budget / `max_output_tokens` | 4,096 / 8,192 | 512 / 2,048 |
| `timeout_seconds` | 240 | 120 |
| prompt | call plan: about 10 calls to locate, an edit by call 12, no new exploration after call 20 | none |

Its author's reasons, from 129 local runs: half of the fixes the model eventually finds are not in the file after four minutes, runs with no source edit by call 12 fail 87% of the time, and the call cap pays for the longer time cap (116 tasks at the full cap would need 15.7 hours; a scored run finished inside twelve). Our five minutes was the worst-case-safe reading of the twelve-hour limit (6.9), a choice and not a rule, and every comparison since (thinking sweeps, directions, controllers, scaffold changes, distillation) ran inside it: thinking 1024 at five minutes is the largest budget we tried (13 of 76, 6.41). The notebook's remark that grading reuses `timeout_seconds` does not hold in harness 0.2.11, where verification has its own 300-second default (`budget.py`, `config.py`) and falls back to the command timeout only when unset; the scorer's version is not visible to us.

**Test** (`scripts/exp_budget_frame_chain.sh`, `logs/exp_budget_frame_chain.log`, started 18:50 on 11-05; `scripts/extract_public_bundle.py` writes the notebook's four files to `submissions/public_budget_fit` without running it, files hash c9829b6ac7093a02). On the 76 sound title-only hard tasks at concurrency 8, with the cap that buys eight scorer-minutes (34 × 8/5 = 54.4 local minutes): the public bundle at its own frame, our base with only the time cap raised (the control that separates time from the rest), and the public bundle again. Reference: the base at five minutes, 13 and 13.

Results (three runs; the replicate of the public bundle finished on 2026-11-06 and resolved 15 of 76, `runs/public_bf_terse_c8_r2`; the detailed figures below are for the first two runs). The control, our base with only the cap raised to eight scorer-minutes, also resolved 14 of 76 (`runs/v0_long_8min_terse_c8`, 5.7 hours): 19 tasks ran to the longer clock (36 and 38 at five minutes), 3 of the 14 were resolved after the point where the five-minute clock would have stopped them, the mean task took 2,007 local seconds (4.9 scorer-minutes), source edits rose to 0.829 and wrong edits to 0.645, empty patches fell to 15.8%. The public bundle at its own frame resolved 14 of 76 (`runs/public_bf_terse_c8_r1`, 5.1 hours), against the base's 13 and 13 at five minutes: 12 tasks in common with either base draw, 2 gained and 1 lost against each, where the two base draws differ from each other by 3 and 3. Its behaviour is different while its score is not: 2.5% malformed calls (base 4.7% and 5.9%), 3.9% of tasks in a repeat loop (14.5%, 13.2%), 13.2% empty patches (19.7%, 22.4%), an edit attempt on 93.4% of tasks (78.9%), a scratch file written on 71.1% (13.2%), the first edit later (347 s against 241 s and 273 s on the vector's scale), and the same source-edit and wrong-edit rates (0.763 and 0.579 against 0.789/0.737 and 0.618/0.566). The call cap, not the clock, ends its tasks: 30 of 76 stop at 28 calls and 5 at the time cap (base: 36 and 38 of 76 end on the five-minute clock), and the mean task takes 1,814 local seconds, 4.4 scorer-minutes, against the base's 1,642 and 1,686 (4.0 and 4.1).

Paper: the budget paragraph now says the five-minute cap is our choice and names the public frame; the leaderboard paragraph gives 58 tasks, our 8, the best-draw structure and the repeated-draw evidence, with three new references; the statement-calibration paragraph compares against 8 of 58 (0.6 standard errors); a limitation notes the single 58-task draw. Writeup: the same three corrections inside 3,000 words.

## 7. The Kaggle timing notebook

Local speed is not Kaggle speed. The scorer uses four L4 GPUs with the model split across them, and nobody has published how fast that is. The notebook measures it.

It is private, at `bojackkk/gemma4-timing-probe`, and mirrors the official starter notebook. It records:

1. wheel install time,
2. server startup time (this counts against the 12 hours),
3. the same speed probe as section 6.1, for the base model and through a LoRA route,
4. the sample agent on six of the dev tasks with a 5-minute cap: wall time, tool calls, model calls per task.

Results go to `timing_summary.json`. It produces no submission. It is limited to 100 minutes.

It waited in the queue for 8 hours 42 minutes (06:32 to 15:14), then ran for 39 minutes. So the wait was demand for L4 machines, not a fault. Plan any future Kaggle run a day ahead. Results are in `runs/kaggle_timing_v1/`.

### 7.1 What it measured

| Quantity | Kaggle (4x L4) | This machine | Ratio |
|---|---|---|---|
| Decode, one request | 37 tokens/s | 10.5 tokens/s | 3.5x |
| Decode through a LoRA adapter | 33 tokens/s | not measured | |
| Prefill, unseen prompt | about 1,400 tokens/s | about 700 tokens/s | 2x |
| One agent turn, fitted from traces | 0.46 s + tokens / 31 | 1.09 s + tokens / 9.5 | about 3.2x |
| Sample agent, turns in a 5-minute task | 19 to 34 | 9 to 18 | |
| Sample agent, generated tokens per 5-minute task | about 5,600 | about 2,400 | 2.3x |
| KV cache | 27,712 tokens | 47,664 tokens | |
| Server startup | 454 s | about 270 s | |
| Wheel install | 55 s | | |

The "one agent turn" row is the one that matters: the same turn takes about 3.2 times longer here than on Kaggle.

### 7.2 What it changes

**Every dev-set comparison in section 6 ran at roughly a third of the real time budget.** A 5-minute cap here gives the agent what about 1.6 minutes would give it on Kaggle. The finding that the thinking-on agent "cannot reach the edit in time" was true of that harsh budget and has to be re-tested at the real one. On Kaggle, the sample agent produced a patch in all six probe tasks.

To give a local run the same budget as a 5-minute Kaggle task, scale the cap:

| Local concurrency | Tokens/s per request | Cap that matches 5 Kaggle minutes |
|---|---|---|
| 1 | 10.6 | 16 minutes |
| 4 | 6.8 | 25 minutes |
| 8 | 5.0 | 34 minutes |

By coincidence, the long-budget run in 6.9 (30 minutes, eight in parallel) is close to this: it is worth about 4.4 Kaggle minutes per task. So it doubles as the first Kaggle-equivalent measurement of v0.

**The 12-hour budget.** Startup costs about 8.5 minutes (wheels plus server). Each task costs its cap plus roughly 10 seconds of setup and teardown. With a 5-minute cap, 120 tasks take at most 10.5 hours, which leaves room for up to about 137 tasks. A 5.5-minute cap would take 11.5 hours for 120 tasks and has almost no margin if the task count is higher than "about 120". So 5 minutes is the safe setting.

**LoRA cost.** Routing through an adapter (even a zero one) costs about 10% of decode speed.

**None of the six probe tasks was marked resolved on Kaggle.** The same six also failed for the sample agent here, and I have not checked whether the Kaggle notebook environment grades fastapi and rich correctly, so read nothing into that.

---

## 8. Rebuilding from scratch

These are the steps actually taken, in order.

```bash
# harness environment
python3 -m venv .venv
.venv/bin/pip install kaggle
export KAGGLE_API_TOKEN="$(cat kaggle.json)"

# data
.venv/bin/kaggle competitions download gemma-4-developer-agent -p data/zip
unzip -q data/zip/gemma-4-developer-agent.zip -d data/competition
.venv/bin/kaggle models instances versions download \
  google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2 \
  -p models/gemma-4-31b-it-qat-w4a16-ct --untar

# harness: official wheels from the dataset metric/gemma-4-developer-agent-wheelhouse
W=data/wheelhouse
.venv/bin/pip install $W/google_adk-1.36.1-py3-none-any.whl litellm "docker>=7" \
  "networkx>=3" "numpy>=2" "pandas>=2" "pydantic>=2" pyyaml rich cachetools \
  python-dotenv requests pyarrow
.venv/bin/pip install --no-deps $W/swegemma-0.2.7-py3-none-any.whl \
  $W/adk_submission-0.2.12-py3-none-any.whl $W/adk_eval_core-0.1.0-py3-none-any.whl

# what the sandbox inherits: test runner, build backends, test libraries
.venv/bin/pip install pytest "pytest-timeout==2.1.0" typer pdm-backend setuptools wheel \
  poetry-core hatchling flit-core editables \
  inline-snapshot dirty-equals ujson orjson pytest-httpbin typing-inspection

# serving environment
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install vllm==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu130
.venv-vllm/bin/pip install --force-reinstall --no-deps "torch==2.10.0+cu128" \
  "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128
.venv-vllm/bin/pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
.venv-vllm/bin/pip install "transformers==5.13.1" "tokenizers==0.22.2"
```

Then three manual steps: copy the three patched vLLM files over the stock ones (section 3.2), build `data/wheels_aarch64/` (the pure-Python wheels plus ARM builds of `charset_normalizer`, `markupsafe`, `pydantic_core`, `sqlalchemy`), and unpack the Python headers into `tools/pydev/` (section 3.3).

Installing vLLM against the CUDA 12.8 index in the first place would probably avoid the reinstall, but I have not tested that path.

---

## 9. What is not built yet

| Piece | Status (2026-10-13) |
|---|---|
| Our own agent | v0_long: 36/100 at the Kaggle-equivalent budget (6.9); 10/24 and 9/24 on dev24 at the calibrated cap (6.11) |
| Stable evaluator | `run_guarded.py` (4.7) with per-command sandbox limits (6.12); every run since 2026-10-11 has been guarded |
| iE controller over knobs (memo Usage 1) | Pilot complete (6.14 to 6.17): Jacobian, rank diagnostic, four RM steps; the end point is worse than the base on a fixed set, which is the expected answer for knob-only control and the argument for directions |
| Format direction (DPO LoRA) | `adapters_out/format_r16_v2`: cuts garbled calls and loops on dev24 at no cost (6.16) and eliminates malformed calls and loops on the sound tasks (4–6% → 0.3–0.6%, 10–13% → 1–2%); its resolution is 38 and 38 against the base's 31 and 38, +3.5 points pooled, not established (6.25); the first attempt collapsed (6.13). Current submission candidate (`bundles/format1.zip`) |
| Decisiveness direction | `adapters_out/decisive_r16`, trained on 910 hindsight pairs (6.18); characterised on dev24 (6.22): a tilt that trades empty patches against wrong edits, no resolution change |
| iE over directions (memo Usage 2) | Complete (6.19 to 6.21): well-conditioned Jacobian (condition 5.6), three RM steps, and an end point that holds resolution on dev24 while cutting garbled calls to ~0.003 at temperature 1.0; decisiveness weight being isolated (day 10) |
| Skill scripts for locating code and running tests | Not started |
| Held-out validation set from repos outside the four public ones | Built (6.27): 95 sound tasks from 11 less-known repositories; base and format bundle both 56/95. Easier than the public set, so familiarity does not explain the leaderboard gap; the Kaggle calibration (6.29) shows local results transfer to the scoring hardware |
| Paper track | Revised 2026-10-19 with the leaderboard-gap and scaffold sections (6.31). Draft in `paper/` (LaTeX, `tectonic -X compile paper/main.tex`; figures from run folders by `.venv-paper/bin/python paper/make_figures.py`). The paper track is the separate competition `gemma-4-developer-agent-paper` (deadline 2026-11-12, "Submit a Research Paper advancing the state-of-the-art for agentic software engineering"); its length and format rules are on the JS-rendered overview page and still need checking. Author line and two numbers (second base draw, leaderboard score) are placeholders |
| Any Kaggle submission | **One**: ref 57269654, `bundles/format1.zip` (format direction on v0_long), submitted 2026-10-16 12:53 UTC, scored by 23:54 UTC: **public 0.13**, the level of public baseline bundles (0.10–0.15; the public planner→coder notebook without LoRA also scored 0.13); top of the board 0.24, a group at 0.22. Local 34.5–38% on public repos transfers poorly to the hidden private repos (6.26). `scripts/build_submission.py` packages a submission directory as `submission.zip` and runs the harness's own `validate_directory` and `compile_submission` on it; `bundles/format1.zip` (format direction at weight 1, 183 MiB) and `bundles/gated3_end.zip` (round 3's end point, 369 MiB) both pass. Submitting needs the user's go-ahead |

Chains: days 5 to 9 done (6.17 to 6.21); day 10 finishing on 2026-10-14 (decisiveness weight isolated at temperature 0.2: format 1 with decisiveness +1.0 and with −0.5, each twice on dev24; 6.22); day 11 done (6.23, the gate misfired on noise); day 12 running: **round 3 with the revised gate** (`configs/ie_directions_gated3.yaml`, reference 0.354 over the format-only point's four dev24 draws, allowance two tasks), four steps from the format-only point; day 13 done (6.24); day 14 running: the base and the format point on the 100 sound tasks at the calibrated cap (`exp_day14_chain.sh`, 6.25), the scale at which a two- or three-point score difference becomes visible; day 15 done (6.25: format 38, gated end point 33, base 31); day 16 done (the format point's second draw: 38 again); day 17 running: a second base draw on the same 100 tasks (`exp_day17_chain.sh`). Round 2 was **round 2 of the directions controller, gated** (`configs/ie_directions_gated.yaml`): merit gate on resolution with the noise floor's 9.5/24 as the starting reference and a one-task allowance, targets moved to where round 1's responses say they are attainable (garbled 0.005, first edit 200 s, compactions 1.0, empty 0.10), every draw on the fixed dev24 set so the gate compares like with like, the Jacobian and whitening copied from round 1, four RM steps from the format-only point. Logs: `logs/exp_day5_chain.log`, `logs/exp_day6_chain.log`, `logs/exp_day7_chain.log`, `logs/train_decisive.log`, `logs/guard_*.log`.

Known weaknesses of the rig itself:

- The local sandbox is Python 3.12 subprocesses, not the scorer's exact container. Task soundness is validated, but an environment-specific surprise on the hidden repos is still possible.
- Requests and httpx tasks are unusable locally, so the dev set covers only two repos, and both are public. Scores here will overstate hidden-set performance.
- Model seeds may not be honoured end to end, so repeated runs can differ. Comparisons need repeated runs or more tasks than 24 before a small difference is believed.
