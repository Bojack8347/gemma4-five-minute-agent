# What Limits a Five-Minute Coding Agent?

Tian Qin

Code, task sets, training data, run summaries and the paper for our entry to the Gemma 4 Developer Agent competition and its paper track. The paper (`paper/main.pdf`, 19 pages) and the Kaggle writeup (`paper/writeup.md`) describe the work; `docs/IMPLEMENTATION.md` is the full implementation record, written as the work was done, with every run's configuration and result.

The setting: a fixed, 4-bit quantization-aware-trained Gemma 4 31B, served by vLLM through the competition harness, acting as a repository-level coding agent with five minutes per task, where only the declarative configuration and LoRA adapters may change.

## Contents

| path | what it is |
|---|---|
| `paper/` | LaTeX source, bibliography, figures and the compiled PDF; `writeup.md` is the 3,000-word Kaggle writeup; `make_figures.py` regenerates the figures from run folders |
| `docs/IMPLEMENTATION.md` | the implementation notes: sections 1 to 5 set up the rig, section 6 records every experiment in order (6.1 to 6.42), sections 7 to 9 cover the scoring-hardware timing notebook and the remaining pieces |
| `scripts/` | single-file scripts: the guarded evaluator, pair miners, the DPO trainer, the exact adapter merge, the iE controllers, the held-out task builders, and the `exp_*_chain.sh` scripts that ran each experiment |
| `configs/` | controller configurations (`ie_*.yaml`) |
| `submissions/` | the agent configurations (ADK agent, prompts, sampling) of the submissions compared in the paper, without adapter weights |
| `tasks/` | the three held-out task sets (see `tasks/README.md`): task definitions, soundness lists and the mined candidate commits; snapshots are rebuilt from the public repositories |
| `pairs/` | the preference pairs used to train the format, decisiveness and statement directions |
| `traj_sft/` | the 153 informed trajectories, rendered title-only, used for trajectory imitation |
| `runs/` | `summary.json` and per-task results for 64 evaluation runs, plus the controller states; traces are not included |
| `kaggle_nb/` | the timing and calibration notebooks run on the scoring hardware |
| `assets/` | the format direction adapter (rank-16 LoRA, 490 MB) and the submitted bundle; too large for the repository; attached to the [v1.0 release](https://github.com/Bojack8347/gemma4-five-minute-agent/releases/tag/v1.0) |

## Reproducing

Every number in the paper is tied, in `docs/IMPLEMENTATION.md`, to a run folder under `runs/` and to the command that produced it. The main pieces, in the order the paper uses them:

1. **Serve and evaluate.** `scripts/serve.py` starts vLLM 0.19.1 (with the competition's Gemma 4 LoRA patch) on the `gemma-4-31b-it-qat-w4a16-ct` checkpoint; `scripts/run_guarded.py` wraps `scripts/run_eval_v2.py` with per-command sandbox limits and a server-throughput watchdog (notes 4.7 and 6.12). The local cap that matches five scorer minutes is 34 minutes at eight concurrent tasks (notes 7.2).
2. **Behaviour vector.** `scripts/behaviour_vector.py <run folders>` prints the coordinates of the paper's tables from a run folder.
3. **Mine pairs and train a direction.** `scripts/mine_format_pairs.py` rebuilds the agent's contexts at its failed calls and labels resampled completions; `scripts/train_direction_dpo.py` trains the NLL-anchored DPO adapter on bf16-dequantized weights with a cached prefix (notes 6.13 to 6.16); `scripts/merge_directions.py` builds the exact rank-stacked merge that a submission serves as one adapter.
4. **Control with iE.** `scripts/ie_control.py` with a `configs/ie_*.yaml` runs the probe, the Robbins–Monro steps and the merit gate over sampling knobs or direction weights (notes 6.14 to 6.24); `scripts/ie_distill.py` is the moment-equation calibration at fixed states (6.32); `scripts/reach_check.py` the fixed-state check and its validation (6.33); `scripts/ie_train_loop.py` the outer loop over DPO's training settings (6.36); `scripts/ie_debias.py` the selection-bias correction (6.38); `scripts/ie_proxy.py` the calibration of the proxy's statement fraction (6.40); `configs/ie_informed.yaml` with `scripts/ie_informed_report.py` the run that targets the informed agent's measured behaviour (6.41).
5. **Held-out sets.** `scripts/heldout/mine_commits.py` and `scripts/heldout/build_tasks.py`; see `tasks/README.md`.
6. **Statement distillation.** `scripts/mine_statement_pairs.py` and `scripts/train_direction_dpo.py` (6.31); `scripts/build_traj_sft.py` and `scripts/train_traj_sft.py` (6.34).
7. **The leaderboard and the budget frame.** `scripts/leaderboard_stats.py` reads a downloaded public-leaderboard snapshot and prints the task count its scores imply and how a team's best score grows with its submissions; `scripts/extract_public_bundle.py` writes the files of the public Budget-Fit notebook to a submission folder without running it, and `scripts/exp_budget_frame_chain.sh` runs that bundle and our base at an eight-minute cap on the title-only proxy (6.42). The snapshot and the third-party bundle are not redistributed.

The scripts expect the competition harness (`swegemma`) and its data in the layout the notes describe (section 3); neither is redistributed here. Training ran on one NVIDIA GB10 (121 GiB unified memory); the notes give memory figures for each step.


