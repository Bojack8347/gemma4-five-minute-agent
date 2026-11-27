# What Limits a Five-Minute Coding Agent?

## Preference directions, query-only control, and the cost of terse problem statements

*Full paper (PDF), code and task sets: https://github.com/Bojack8347/gemma4-five-minute-agent.*

### Abstract

We study how far a fixed, 4-bit quantization-aware-trained Gemma 4 31B can be improved as a repository-level coding agent when only its declarative configuration and LoRA adapters may change and each task has five minutes. The resolution rate is too noisy to steer by: two draws of one agent differ on 11 of 100 tasks. We therefore steer by a *behaviour vector* of low-noise coordinates on a budget calibrated to the scoring hardware. The diagnosis is one trade-off: with less thinking the model acts sooner but writes malformed tool calls. A *format direction*, a rank-16 LoRA trained with NLL-anchored DPO on pairs mined at the agent's own failure states, cuts malformed calls from 4–6% to 0.3–0.6% and loops from 10–13% to 1–2% on 100 tasks, replicated; its resolution gain is within noise. We then use iterative Expectation (iE) as a query-only controller over sampling knobs, merges of trained directions and the DPO training settings. Finally we locate the gap between local scores (31–38%) and the leaderboard (13%) in the problem statement, not the scoring environment, repository familiarity or fix size: cutting statements to their titles reproduces the leaderboard's level (37% → 17%, p = 0.0001). On that proxy, better tools, guidance, speed, statement distillation and an iE weight calibration change behaviour but not score: under a five-minute budget the binding constraint is the model's reasoning about an underspecified task.

### Introduction

A harness that freezes the model, tools and time budget leaves only prompts, sampling, an agent hierarchy and low-rank adapters to change. Inside that space the fraction of tasks resolved is a poor guide: a hundred tasks flip by a dozen between repeated draws of one agent, so a score-driven search chases noise.

We make four contributions.

1. **Measurement before tuning.** A budget calibrated to the scoring hardware by per-turn timing, a repeated-draw noise floor, and behaviour coordinates whose run-to-run spread is far below the score's.
2. **On-policy preference directions.** LoRA adapters trained on preference pairs mined at the agent's own failure states and rendered exactly as the harness does; a collapse mode of whole-sequence DPO on near-identical pairs, and the gate that prevents it.
3. **Query-only control of directions.** The configuration space cast as root-finding on the expected behaviour vector, solved with iE over exact adapter merges and over DPO's own training settings; a cheap moment-equation calibration of a distilled direction at fixed states; and an iE correction for selection bias.
4. **Locating the leaderboard gap.** Two held-out task sets mined from 2026 commits of less-known repositories and the finding that statement informativeness, not the environment, familiarity or fix size, accounts for most of the drop from local to leaderboard scores; the resulting proxy rejects five changes aimed at the gap.

### Methods

**Setting.** The model is `gemma-4-31b-it-qat-w4a16-ct`, served by vLLM over four L4 GPUs. A submission is a declarative bundle (ADK configuration, prompts, sampling, up to eight LoRA adapters of rank ≤ 128). Each task is a repository snapshot and a pull-request title and description; the harness grades the working-tree diff against hidden tests. About 120 hidden tasks run sequentially in 12 hours; we cap each at five minutes, our choice (the most-run public bundle [17] uses eight, bounded by 28 tool calls). A notebook on the scoring hardware timed one agent turn at 0.46 s + n/31 for n output tokens, against 1.09 s + n/9.5 locally; local runs use the cap buying the same token budget, 34 minutes at eight concurrent tasks.

**Behaviour vector and noise floor.** Per run we compute the malformed-call rate, the fraction of tasks with four or more identical consecutive calls (loops), the median time to the first edit in scorer-equivalent seconds, the fractions of patches that touch real source, are empty, or edit source but fail the tests (wrong edits), and compactions per task; resolution is tracked, never controlled. On 100 sound public tasks (60 fastapi, 40 rich), two draws of the unmodified agent at identical settings resolved 31 and 38, agreeing on 29. Every comparison is paired by task. Picking the best of several configurations inflates its rate; we de-bias such comparisons with iE in its original role (statistic: the sorted observed rates; simulator: a two-way logistic model reproducing the 11 tasks on which two base draws differ). The format direction's +3.5 is unchanged (model s.d. 2.5 tasks); on the proxy and dev24 the observed spreads are below noise and the configurations de-bias to equal rates.

**Guarded evaluation.** Every sandbox command runs in its own control group with the scorer's limits, under a watchdog that aborts runs when server throughput collapses.

**Diagnosis.** Prompt changes altered what the model avoided, not how decisively it acted. The thinking budget did: with 512 or 128 thinking tokens per turn calls are well formed but the agent explores in small steps; with thinking off it edits in over half the tasks but a quarter of its calls are malformed.

**On-policy preference directions.** A direction is a rank-16 LoRA on all seven projections, trained with DPO on pairs that target one coordinate. Traces keep only parsed arguments, so we rebuild the context the model saw at each failed call and sample there. Matching the harness required undocumented details (ADK's identity line, tool declarations from Python signatures, tool results as `tool` messages), checked against each trace's recorded prompt length. At each context we sample eight completions, label them with a strict parser of the native call syntax, and run `edit_file` calls through the harness's own matcher against the reconstructed file state. A pair is a well-formed and a malformed call of the same tool from the same prefix: 738 pairs from 148 contexts.

Training uses the served checkpoint dequantized to bf16. Each prefix's KV cache is computed once; completions are scored against it and back-propagated in 96-token chunks. The first direction looked perfect in training and was unusable: it emitted the string-delimiter token until the length limit, because every pair differs in that token. The second uses β = 0.1, an NLL anchor (λ = 0.2) on the chosen completion, lr 1e-5, early stopping at margin 1.5, and a generation-time gate that refuses an adapter producing a repeat or an unclosed call. A second direction (decisiveness) takes its chosen side from hindsight: the trajectory's later successful edit.

**Query-only control with iE.** iE solves M(θ*) = α, where α is an observed statistic and M(x) the statistic's expectation when data are generated at x, by the accelerated iteration x_{n+1} = x_n + A_n(α − y_n) with one noisy draw y_n per step and Robbins–Monro gains; where M can be simulated, the analytic iteration x_{k+1} = x_k + (α − M(x_k)) applies. Here x is a vector of submission settings, a draw the behaviour vector of a 24-task batch, α a target vector and M the expected behaviour over tasks; the gain is a whitened pseudo-inverse of a forward-difference Jacobian. A combination of directions is realised exactly as the one adapter the scorer serves, by stacking ranks (BA = Σ a_j (α_j/r_j) B_j A_j), negative weights allowed. A merit gate rejects a draw losing more than two tasks in 24 against the running mean at the accepted point.

**Held-out task sets.** We mined 2026 commits from less-known, maintained, pure-Python repositories that change package source and tests: the source diff is the gold patch, the test diff the hidden tests, the commit message the statement, the parent tree the snapshot. An easy set has 95 sound tasks from 12 repositories; a hard set 76 sound tasks with larger fixes (≥ 40 source lines or ≥ 2 files) from 18. A title-only variant of each cuts statements to their first line.

**Statement distillation and a fixed-state check.** Two forms of teaching the agent to act from a title as from the full statement. *Single-call distillation*: at the title-only agent's own early states on tasks the informed agent solved, the same model samples the next call with and without the full statement; where they differ materially, the teacher's call is the chosen side under the student's thought (307 pairs, 156 states). *Trajectory imitation*: supervised fine-tuning on 153 resolved informed trajectories rendered title-only (1,530 steps), masking any thought that quotes the statement. To calibrate its weight we used iE's moment equation at 168 fixed title-only states from unseen repositories: α is the probability that two teacher samples agree, M(w) that a student sample at weight w agrees with a teacher sample; M(w) = α if the student's next call has the teacher's distribution.

### Experiments and results

**The format direction.** On the 100 public tasks at the calibrated budget, two draws per arm:

| | base (1) | base (2) | format (1) | format (2) |
|---|---|---|---|---|
| Resolved / 100 | 31 | 38 | 38 | 38 |
| Malformed calls | 4.2% | 5.7% | 0.3% | 0.6% |
| Tasks with loops | 10% | 13% | 2% | 1% |
| First edit (scorer s) | 270 | 227 | 230 | 229 |
| Wrong edits | 45% | 43% | 39% | 40% |

Malformed calls and loops separate the arms in every pairing of draws; turn counts, compactions and the edit rate are unchanged. After one base draw (31) and two format draws (38, 38) the gain looked replicated; a second base draw at identical settings resolved 38. Pooled, the direction is 3.5 points ahead with a 95% paired-bootstrap interval of −1.5 to +8.5 (sign test p = 0.10). It removes a real failure mode at no cost in score; any score benefit is below the noise of 100 tasks. The bundle scores 0.13 on the public leaderboard, 8 of its 58 tasks in one draw. The board shows each team's best draw: identical public files re-draw with an s.d. of 1.6 tasks and the most-run bundle averages 10.0 over ten draws [18], so entries at 0.22–0.24 are consistent with top draws.

**Query-only control.** Over sampling knobs alone the prior Jacobian has condition 591: the tool-call cap has no effect and the pseudo-inverse drops it. Ungated, the controller drove the knobs to their bounds, where resolution fell and malformed calls rose to 6–8%. Over the format and decisiveness weights plus temperature the Jacobian is well conditioned (5.6) and the format weight is the strongest lever. A format weight of 2 holds malformed calls at 0.1–0.5% at temperature 1.0, which alone raised them to 6–8% and cost resolution. With attainable targets, a fixed task set and the merit gate, the controller hovered near its start; its end point resolved 33 of 100, keeping the format behaviour. No controlled point beat the single direction; the decisiveness weight trades empty patches against wrong edits with no score change.

**Where the leaderboard gap comes from.** Locally the agent resolves 31–38% of public tasks; on the hidden private repositories, 13%. The scoring environment is not the cause: there a notebook resolved 10 of 30 public tasks against 8–12 locally. Repository familiarity is not: the agent resolves 59% of the easy held-out set. Fix size is not: 37% of the hard set, the public level. The problem statement is: the same 76 hard tasks cut to their titles drop to 17%, 15 tasks lost and none gained (sign test p = 0.0001); wrong edits rise from 51% to 62% and empty patches double. The effect replicates on the public tasks, which drop from 31 and 38 to 21 (p = 0.05 and 0.0009). A twice-as-fast 12B model is worse (6 against 10 of 30 on the scoring hardware). Public statements are titles with descriptions. An iE iteration over the fraction of each statement's body kept, targeting the leaderboard's 13%, converged at the title-only bound in two draws (half the body 21 of 76, title only 13 again): no fraction reproduces the leaderboard's level, title-only is nearest (3 points above its 8 of 58, 0.6 s.e.), and the body's information is not front-loaded (its first quarter adds 2 tasks, the second 6). Two title-only draws share 10 resolved tasks and swap 3 each way: the proxy's noise floor.

**Changes on the leaderboard-matched proxy.** On the 76 title-only hard tasks, paired against the base's 13:

| Agent | resolved | lost / gained | wrong edits | empty patches | loops |
|---|---|---|---|---|---|
| Base (thinking 512) | 13 | – | 62% | 20% | 14% |
| + code-search and test tools | 13 | 3 / 3 | 59% | 20% | 9% |
| Thinking off + format direction | 4 | 10 / 1 | 58% | 34% | 41% |
| + tools + guidance for terse titles | 13 | 3 / 3 | 51% | 30% | 8% |
| Statement-distilled direction | 13 | 4 / 4 | 61% | 20% | 8% |
| Imitation of informed trajectories | 13 | 3 / 3 | 62% | 18% | 8% |

The tools cut loops but delayed the first edit; the guidance cut wrong edits from 62% to 51% but the more careful agent ran out of time before editing more often; without thinking the agent edited at 147 instead of 241 scorer-seconds but looped in 41% of tasks. The distilled direction halved malformed calls and loops and traded four tasks for four; its pairs were weak: at 168 fixed states same-side samples differ 20% of the time against 33% across sides; no cutoff separates the statement's effect (one early state in seven) from the student's stochasticity, but a two-sample consensus per side does (false-pair rate 2–4%). The trajectory-imitation adapter is the cleanest title-only agent (1.1% malformed calls, 8% loops) and resolves as many tasks. Its training loss says why: the base model already assigned the informed agent's calls 0.07 nats per token in title-only context; the statement decides a few steps, such as which file to change.

**iE at fixed states.** The moment equation for the distilled direction's weight has no root. Agreement with the teacher is 0.68 at w = 0, 0.69 at w = −1 and 0.64 at the trained weight (paired −0.041, s.e. 0.020), while α = 0.78. A draw costs 15 minutes against 4.5 hours for an agent run; validated against our agent runs, one of 20 comparisons was distinguishable from noise, and the check would have rejected the format direction, whose effect acts through the trajectory: sound for per-call objectives, not for loops, malformed calls or edit timing.

**iE as the outer loop of DPO.** This puts iE outside training: each draw retrains an adapter at settings x (the DPO primal step), runs it on a fixed 24-task batch of the proxy, and iE moves the settings (the dual step). Knobs: the two pair sources' loss weights and the early-stop margin; targets: ceilings on the five behaviour coordinates. Eight draws; none refused, all inside the gate; Jacobian condition 11.4; residual 3.2, 4.9, 5.5, 4.3, 4.0. A fresh adapter already met the three format ceilings; the first edit and empty patches, which carried the residual, did not respond to the knobs, and one low probe draw gave the first-edit column a wrong sign the iteration followed for four steps. The pairs reach format behaviours, not the planning ones that decide resolution; the outer loop has nothing to steer. A last run targeted the informed agent's measured behaviour, with 76-task Jacobian columns refit every step: 54% of the squared whitened distance lies outside the columns' span, the refit caught a wrong sign after one step and stopped the controller, and the residual never fell below the base's. Source edits equal resolution plus wrong edits, so the target was the score by another name.

### Lessons

Calibrate the budget before believing a comparison: conclusions at a third of the real budget did not survive. Steer by behaviour, decide by paired score at scale. Redraw the control, not only the treatment. Make targets ceilings and re-estimate the Jacobian from every draw. Build a local set that matches the leaderboard before tuning against it: our three other sets overstated the hidden-set score two- to four-fold. Faster is not better if the model needs its thinking. Check per-call targets at fixed states and trajectory-level ones in agent runs.

### Related work

Repository-level benchmarks [1] and agent scaffolds [2, 3] established the setting; training open models with trajectories and verifiers is studied in [4]. We use DPO [5] with an NLL term as in [6]; failure modes of DPO on pairs with small edit distance are analysed in [7, 8]. Supervising a student at its own states [9] by a privileged teacher [10] is the template for our statement distillation. Adapter arithmetic [11–13] motivates treating directions as a basis; our merge is exact. iE [14] combines bootstrap iteration [15] with stochastic approximation [16].

### Verifiability

Every number comes from a kept run folder (traces, patches, test outputs) recorded with its configuration in the implementation notes; the scripts and the held-out task sets are released with the code.

### References

1. Jimenez et al., SWE-bench, ICLR 2024.
2. Yang et al., SWE-agent, NeurIPS 2024.
3. Xia et al., Agentless, 2024.
4. Pan et al., SWE-Gym, 2024.
5. Rafailov et al., Direct preference optimization, NeurIPS 2023.
6. Pang et al., Iterative reasoning preference optimization, 2024.
7. Pal et al., Smaug: DPO-positive, 2024.
8. Razin et al., Likelihood displacement in DPO, 2024.
9. Ross, Gordon, Bagnell, DAgger, AISTATS 2011.
10. Chen et al., Learning by cheating, CoRL 2020.
11. Hu et al., LoRA, ICLR 2022.
12. Ilharco et al., Task arithmetic, ICLR 2023.
13. Huang et al., LoraHub, 2023.
14. Qin, Iterative Expectation, PhD dissertation, 2025.
15. Martin, On double bootstrap confidence intervals, 1992.
16. Robbins and Monro, A stochastic approximation method, 1951.
17. Henao, Budget-Fit single agent, Kaggle notebook, 2026.
18. Afshar, Most-run public config averages 10/58, Kaggle notebook, 2026.
