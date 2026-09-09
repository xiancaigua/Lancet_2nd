# AntMaze WSRL vs Lancet-TD Benchmark Protocol v1

Status: **planned; no debug pilot or formal benchmark has been started**  
Protocol date: 2026-09-08  
Protocol scope: first-stage `antmaze-medium-play-v2` comparison only

## Research question and method names

This protocol asks:

> With offline initialization, base actor/critic architecture, dataset, reward
> transform, online budget, and evaluation protocol held fixed, does TD-residual
> critic correction improve WSRL offline-to-online adaptation?

The two compared methods are:

- **WSRL**: the current rl-garden WSRL implementation.
- **Lancet-TD**: WSRL plus the current shared TD-residual critic correction,
  with `lambda_u_variation: 0.0`.

This is explicitly **Lancet-TD / Lancet w/o U**, not full Lancet. The
U-variation objective is intentionally disabled because its mathematical
definition has not yet been finalized. This protocol must not be relabeled as
a full-Lancet evaluation later.

## Scope

The frozen first environment is `antmaze-medium-play-v2`, state observations,
using the verified D4RL legacy 1,000,000-transition dataset. Kitchen is not a
blocker for this phase.

Possible later protocols, not part of this run matrix, may cover:

- `antmaze-medium-diverse-v2`
- `antmaze-large-play-v2`
- `antmaze-large-diverse-v2`

Adding an environment requires a new protocol revision or a separate protocol;
it must not silently widen this one.

## Comparison contract

Only the Lancet residual correction may be the main algorithmic variable. Each
seed is a paired comparison. The pair uses the same dataset file, one shared
WSRL offline checkpoint, and identical base online settings.

The following must match within every seed pair:

- offline dataset and transition ordering;
- shared offline checkpoint and its actor, base critic, target critic,
  temperature state, base optimizer states, and global counters;
- actor architecture and base critic architecture;
- critic count and REDQ target subsampling;
- observation and reward preprocessing;
- discount, Polyak target update, batch size, UTD, and optimizer family;
- base actor, critic, temperature, and CQL-alpha learning rates;
- warmup, replay switch mode, and offline-data ratio;
- online environment, online step budget, and evaluation schedule;
- seed and seed-to-checkpoint mapping.

Current implementation evidence supports most of this contract, but the
stage-specific frozen configs and lineage metadata do not exist yet. Therefore
the contract is planned, not yet certified.

## Shared offline checkpoint strategy

For each seed `s` in `{0,1,2,3,4}`:

1. Train **one WSRL offline initializer** for 1,000,000 offline updates with
   seed `s` and `num_online_steps: 0`.
2. Preserve its `offline_final.pt`, resolved config, evaluation metrics, and
   SHA-256 digest as the shared initialization artifact for seed `s`.
3. Construct fresh WSRL and Lancet agents with the same base configuration and
   seed `s`.
4. Load the same `offline_final.pt` into both agents with strict policy/space
   validation and without a replay snapshot.
5. Verify hashes/equality of actor, base critic, target critic, alpha state,
   base optimizer state, `global_step`, and `global_update` immediately after
   load and before the online switch.
6. Fork two 500,000-step online runs. Both use `online_replay_mode: empty`,
   `offline_data_ratio: 0.0`, and the same 5,000-step frozen-policy warmup.

The 1,000,000-transition offline dataset may be reloaded in both branches for
the fixed offline probe; the online replay buffer is then cleared identically.
A replay-buffer checkpoint is not required because WSRL intentionally begins
online adaptation with an empty replay buffer.

### Lancet residual initialization

The WSRL checkpoint has no residual parameters. Current `Lancet` behavior is
therefore frozen for v1:

- base actor/critic/target and base optimizers come from the shared WSRL
  checkpoint;
- the residual is newly constructed after the base policy under seed `s`;
- its two hidden layers use PyTorch `nn.Linear` default initialization,
  LayerNorm, and ReLU; the output layer is also PyTorch-default initialized;
- the residual optimizer is a fresh Adam optimizer with no accumulated state;
- the initial residual state must be hashed and recorded before its first
  update.

A no-training compatibility probe on 2026-09-08 verified that a WSRL
checkpoint loads strictly into Lancet, all base policy tensors match, two
Lancet constructions with the same seed produce identical residual weights,
and the residual optimizer begins empty. The current residual is not forced to
zero, so corrected Q need not equal base Q at the fork. Changing to zero-output
initialization would define a new method variant and requires a protocol
revision.

Independent WSRL and Lancet offline pretraining runs are forbidden for the
headline comparison because they would confound online correction with
different offline initializations.

## Seed protocol

Formal seeds are `0, 1, 2, 3, 4`. Seed `s` identifies the shared offline
checkpoint and both online branches in a paired trio of artifacts.

| Source | Current behavior | Protocol requirement |
|---|---|---|
| Python | `random.seed(seed)` | keep |
| NumPy | `numpy.random.seed(seed)` | keep |
| PyTorch CPU | `torch.manual_seed(seed)` | keep |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` | keep |
| Network initialization | base constructor reseeds before model setup | keep |
| Residual initialization | follows deterministic construction order after reseed | record initial state hash |
| Dataset/replay sampling | uses global PyTorch RNG (`randint`/`randperm`) | keep identical seed/config |
| Training environment | first online `reset(seed=seed)`; D4RL adapter forwards it | keep |
| Evaluation environment | currently calls `reset()` without an explicit seed | **small patch required** |

The patch should use a separate deterministic evaluation seed stream, derived
from the run seed and evaluation index, without perturbing training RNG. It
must be identical for both methods in each pair.

Current Lancet construction also consumes global PyTorch RNG while creating
the residual, whereas WSRL has no corresponding draw, and checkpoints do not
store RNG state. A small patch must therefore isolate residual initialization
or reset/restore a documented training RNG stream after initialization and
checkpoint load. Without it, the same nominal seed does not guarantee aligned
replay/action sampling streams across the paired branches.

Strict bitwise reproducibility is not promised. The runner enables cuDNN
benchmark mode and high-precision TF32-oriented matmul settings, and some CUDA
kernels may remain nondeterministic. Formal reports must record this limitation;
they must not sacrifice the paper-scale execution path merely to claim bitwise
determinism.

## Frozen WSRL vs Lancet-TD configuration

Values below come from the two checked-in full-scale YAML files and their
2026-09-08 `--print-config` output. A value marked “protocol override” must be
added to the future stage-specific configs before any pilot or formal run.

| Parameter | WSRL | Lancet-TD | Must match? | Notes |
|---|---:|---:|:---:|---|
| environment | `antmaze-medium-play-v2` | same | yes | D4RL legacy |
| dataset | `antmaze-medium-play-v2` | same | yes | verified 1,000,000 transitions |
| observation mode | `state` | same | yes | no RGB |
| offline updates | 1,000,000 shared initializer | load shared initializer | yes | Lancet does not redo offline training |
| online environment steps | 500,000 | 500,000 | yes | branch budget |
| UTD | 4.0 | 4.0 | yes | online critic-update ratio |
| batch size | 1024 | 1024 | yes | divisible by UTD |
| training frequency | 64 | 64 | yes | env steps per rollout iteration |
| buffer size | 1,000,000 | 1,000,000 | yes | CUDA storage |
| learning starts | 4,000 | 4,000 | yes | checkpoint global step is already larger |
| warmup | 5,000 env steps | 5,000 | yes | actor/critic/encoder frozen |
| replay switch | `empty` | `empty` | yes | offline replay discarded |
| offline data ratio | 0.0 | 0.0 | yes | no mixed offline samples online |
| reward scale | 10.0 | 10.0 | yes | train env only |
| reward bias | -5.0 | -5.0 | yes | train env only |
| sparse reward MC | true | true | yes | negative reward -5.0 |
| discount `gamma` | 0.99 | 0.99 | yes | resolved default |
| target `tau` | 0.005 | 0.005 | yes | resolved default |
| actor network | 2 x 256 MLP | same | yes | LayerNorm, uniform std |
| base critic | 4 x 256 MLP | same | yes | LayerNorm |
| kernel init | orthogonal near-zero output | same | yes | base networks |
| critics | 10 | 10 | yes | REDQ ensemble |
| target critic subsample | 2 | 2 | yes | same RNG policy |
| target entropy | 0.0 | 0.0 | yes | entropy backup false |
| optimizer family | Adam | Adam | yes | `use_adamw: false` |
| actor LR | 1e-4 | 1e-4 | yes | base actor |
| critic LR | 3e-4 | 3e-4 | yes | base critic |
| alpha LR | 1e-4 | 1e-4 | yes | temperature |
| CQL-alpha LR | 3e-4 | 3e-4 | yes | offline initializer state shared |
| policy frequency | 1 | 1 | yes | inherited update schedule |
| target update frequency | 1 | 1 | yes | inherited update schedule |
| offline objective | Cal-QL/CQL enabled | inherited checkpoint only | shared | alpha 5.0, autotune, gap 0.8 |
| online CQL | disabled, alpha 0 | disabled, alpha 0 | yes | online SAC behavior |
| offline eval frequency | 50,000 updates | shared initializer result | shared | 20 scheduled evaluations |
| online eval frequency | 2,000 env steps | 2,000 | yes | actual step may cross by one rollout |
| eval budget | 20,000 vector steps | 20,000 | yes | one eval env; 20 full 1000-step episodes |
| log frequency | 1,000 | 1,000 | yes | resolved default |
| periodic checkpoint | 50,000 steps | 50,000 | yes | **protocol override; current YAML default is 0** |
| final checkpoint | true | true | yes | forced save |
| replay checkpoint | false | false | yes | intentional empty online fork |

### Lancet-TD-only parameters

| Parameter | Frozen value | Current implementation |
|---|---:|---|
| `use_lancet` | true | shared residual enabled |
| residual architecture | 2 x 256 MLP, LayerNorm + ReLU | one scalar `R_phi(s,a)` shared by 10 critics |
| residual LR | 3e-4 | fresh Adam |
| `lambda_td_residual` | 1.0 | MSE to detached `target_q - mean(Q_base)` |
| `lambda_residual_reg` | 1e-4 | mean squared residual magnitude |
| `lambda_u_variation` | **0.0** | nonzero intentionally raises |
| residual clipping | none | no output clipping |
| residual normalization | none | no target/output normalization |
| gradient clipping | none | inherited `grad_clip_norm: null` |

The base critic and target TD calculation remain WSRL. The residual is fitted
after the base critic step, the target critic excludes the residual, and the
actor uses `min(Q_base) + R`.

## Medium-scale stability pilot gate

Before any formal initializer or formal online branch, run one seed-0 paired
debug pilot:

- shared WSRL offline initializer: **20,000 offline updates**, zero online
  steps;
- WSRL online branch: load that initializer, **20,000 online env steps**;
- Lancet-TD online branch: load the same initializer, **20,000 online env
  steps**;
- UTD 4.0, batch 1024, warmup 5,000, online eval frequency 2,000;
- debug periodic checkpoint frequency 5,000;
- all archives/checkpoints under `/data/lancet/{runs,checkpoints}/debug/`.

The comparison matrix counts two online debug runs; the shared offline
initializer is a prerequisite artifact, not a third algorithm variant. Pilot
results are engineering evidence only and must not enter a formal result table.

### Pilot pass gate

Both branches must satisfy all of the following:

- no NaN/Inf in logged losses, Q diagnostics, parameters, or checkpoints;
- critic loss and Lancet residual loss do not show an unbounded upward trend;
- base Q, corrected Q, residual, actor, critic, and residual parameters remain
  finite;
- periodic and final checkpoint save/load succeeds;
- evaluation completes at each scheduled point and at the final endpoint;
- no unexpected OOM or container/environment crash;
- residual updates occur and do not silently stop;
- the residual does not show uncontrolled domination of base Q.

Every Lancet log must include `mean(|R|)`, `mean(|Q_base|)`, and
`mean(|R| / (|Q_base| + 1e-8))`. As an engineering warning only, flag sustained
ratio values above 1.0 or a >10x rise over the first post-warmup diagnostic.
This is not a theoretical acceptance threshold; the time series and its
effect on Q/loss/evaluation must be reviewed.

## Evaluation and reporting contract

### Primary metrics

At every evaluation, record:

1. `eval/normalized_score` in D4RL percentage units; higher is better.
2. `eval/success_at_end` as a fraction in `[0,1]`; higher is better.

For this AntMaze backend, current rl-garden derives normalized score as
`100 * success_at_end` when no backend normalized score is present. These are
two required representations of one underlying success endpoint, not
independent outcomes.

The headline scalar per seed is the **final-window mean of the last 10 online
evaluations, including the forced final evaluation at 500,000 online steps**.
This covers approximately the last 20,000 online steps; use actual logged x
coordinates because 64-step rollout boundaries can cross nominal evaluation
frequencies.

### Secondary metrics

- the full online learning curve;
- normalized online AUC for `eval/normalized_score`, using trapezoidal
  integration over actual online-step coordinates from `x=0` to `x=500,000`,
  divided by 500,000 so the result remains in score units;
- the same AUC for raw success fraction;
- final checkpoint evaluation;
- best observed return/score as a supplementary diagnostic only.

The shared offline checkpoint evaluation supplies the identical `x=0` anchor
for both branches. A final evaluation at the exact online endpoint is required;
the current runner does not force one after the loop and needs a small patch.
Best return must never be the sole or headline comparison.

### Statistical reporting

For each method report the five-seed mean and sample standard deviation
(`ddof=1`) for final-window score and normalized AUC. Also report a 95% t
confidence interval, `mean +/- t(0.975,4) * SD / sqrt(5)`, and show every seed
value.

Because each seed pair shares its offline checkpoint and seed, the main effect
estimate is the paired difference `Lancet-TD - WSRL`, reported as mean, sample
SD, and 95% t interval. If a hypothesis test is included, predeclare a
two-sided paired t-test on the final-window normalized score at alpha 0.05;
do not swap tests after seeing results. With only five seeds, effect sizes,
uncertainty, and individual paired values take precedence over binary
significance claims.

## Mechanism diagnostics

### Already logged

- `losses/critic_loss`, `losses/actor_loss`, `entropy/alpha`, and entropy
  diagnostics;
- base ensemble mean during critic updates as `q/predicted`;
- target mean as `q/target` and `losses/td_loss` / `q/td_rmse`;
- Lancet residual mean, standard deviation, absolute mean, total loss,
  TD-residual loss, magnitude regularizer, and U placeholder;
- Lancet `critic/td_error_mean` and `critic/td_error_abs_mean`;
- `lancet/residual_q_ratio`, currently ratio-of-means
  `mean(|R|) / mean(|Q_base|)`.

### Missing but required before the pilot

- critic TD-error standard deviation for both methods;
- explicit base-Q mean, standard deviation, and absolute mean from the same
  diagnostic batch;
- corrected-Q mean and standard deviation for Lancet-TD;
- pointwise `mean(|R| / (|Q_base| + 1e-8))` (different from the existing
  ratio of means);
- explicit finite-parameter/checkpoint scan summaries;
- selected GPU id and measured peak GPU memory;
- exact final-endpoint evaluation.

These are bounded no-grad diagnostic/logging changes and must not alter losses,
sampling, optimizer steps, or RNG. Their test should assert that enabling
diagnostics does not change a seeded update result.

### Optional phase-2 diagnostic

For the same sampled state, evaluate multiple recorded actions and compare
`Q_base(s,a)`, `R(s,a)`, and `Q_corrected(s,a)`. Report action-wise variation,
rank consistency, and rank changes. This is useful for later U-related theory,
but it is not a v1 formal-launch blocker and must not introduce a U objective.

## Checkpoint and reload protocol

Required artifacts are the shared `offline_final.pt`, periodic online
checkpoints every 50,000 steps, and `final.pt`.

WSRL/Lancet reload validation must cover:

- actor, base critic, target critic, and feature extractor;
- alpha/temperature state and base actor/critic/alpha/CQL-alpha optimizers;
- global step, global update, and off2on phase state;
- for Lancet, residual weights, residual optimizer, enable flag, and all
  Lancet hyperparameters;
- deterministic evaluation after reload, compared with the pre-reload
  checkpoint evaluation within numerical tolerance.

Do not silently overwrite an older checkpoint or archive. Record hashes and
parent-child lineage in metadata.

## Archives and experiment identity

Use the existing authoritative layout; do not invent a second directory
convention:

```text
/data/lancet/runs/{debug,formal}/antmaze-medium-play-v2/{wsrl,lancet}/seed_<s>/<timestamp>/
/data/lancet/checkpoints/{debug,formal}/antmaze-medium-play-v2/{wsrl,lancet}/seed_<s>/<timestamp>/
```

The scientific label for registry algorithm `lancet` is `Lancet-TD`. The
canonical experiment identity is the full archive key
`<run_type>/antmaze-medium-play-v2/<registry_algorithm>/seed_<s>/<timestamp>`;
the existing timestamp remains the leaf `experiment_id`. Future launcher
metadata must add protocol id `antmaze_wsrl_lancet_v1`, method label, shared
offline checkpoint path/hash, and GPU id.

Every archive keeps `README.md`, source config, resolved config, `command.txt`,
`metadata.json`, `train.log`, `analysis.md`, `metrics/`, `checkpoints/`, and
`plots/` through the existing archival workflow.

## Hardware scheduling

- Each process uses exactly one GPU; these algorithms are not DDP runs.
- Record physical GPU index, GPU model, driver, and peak allocated/reserved
  CUDA memory in metadata/analysis.
- The server has six 49,140 MiB RTX 4090 GPUs, but current utilization is
  shared and dynamic. Never infer availability from GPU count.
- The tiny-smoke archives recorded GPU models/totals but not peak memory, so
  they do not justify a formal concurrency limit.
- Run the debug pilots serially on one explicitly selected idle GPU. After
  peak memory and CPU D4RL load are measured, distinct seeds may run in
  parallel only with one explicit GPU assignment per process.
- Add launcher support for `--gpu-id` that executes through
  `env CUDA_VISIBLE_DEVICES=<id>` and records the id. Current archive commands
  otherwise default to the container's first visible GPU and can collide.
- Long jobs must be launched through Host tmux or a scheduler around the
  archive launcher. The current archive launcher itself is foreground.

## Planned matrices

### Debug online comparison (shared seed-0 initializer)

| Environment | Algorithm | Seed | Budget | Status |
|---|---|---:|---|---|
| antmaze-medium-play-v2 | WSRL | 0 | 20k online | planned |
| antmaze-medium-play-v2 | Lancet-TD | 0 | 20k online | planned |

### Formal online comparison

| Environment | Algorithm | Seed | Status |
|---|---|---:|---|
| antmaze-medium-play-v2 | WSRL | 0 | planned |
| antmaze-medium-play-v2 | WSRL | 1 | planned |
| antmaze-medium-play-v2 | WSRL | 2 | planned |
| antmaze-medium-play-v2 | WSRL | 3 | planned |
| antmaze-medium-play-v2 | WSRL | 4 | planned |
| antmaze-medium-play-v2 | Lancet-TD | 0 | planned |
| antmaze-medium-play-v2 | Lancet-TD | 1 | planned |
| antmaze-medium-play-v2 | Lancet-TD | 2 | planned |
| antmaze-medium-play-v2 | Lancet-TD | 3 | planned |
| antmaze-medium-play-v2 | Lancet-TD | 4 | planned |

The comparison matrix is 2 algorithms x 5 seeds = 10 formal online runs. Five
seed-specific shared WSRL offline initializer jobs are prerequisite artifacts,
not extra comparison cells.

## Failure and rerun policy

Allowed rerun causes are OOM, host/container failure, corrupted checkpoint,
environment crash, launcher failure, or a verified code/config bug. The old
archive remains immutable and is marked failed/stopped/invalid with a reason.

Low return, an unfavorable seed, high variance, or an unfavorable Lancet result
never justify a rerun. A code-bug rerun requires a new Git commit, invalidation
notes on every affected archive, and explicit parent/replacement links. Never
overwrite or silently delete a result.

## Git freeze

Formal preparation requires a clean working tree, one pushed commit shared by
both algorithms, frozen configs, and the protocol id in metadata. Dirty formal
runs are not allowed for this protocol even though the generic launcher has an
emergency `--allow-dirty-formal` option.

## Benchmark protocol implementation readiness

Overall status: **NEEDS SMALL PATCH**. No architectural blocker prevents the
Lancet-TD comparison, but formal launch remains blocked until the patches and
debug gates below pass.

| Class | Requirement | Current implementation / evidence | Action required |
|---|---|---|---|
| READY | D4RL AntMaze data | audit parsed 1,000,000 finite transitions | none |
| READY | Matching base configs | both `--print-config` outputs match on frozen base fields | preserve with config-diff check |
| READY | WSRL checkpoint into Lancet | `Lancet._compatible_checkpoint_algorithms` includes WSRL; 2026-09-08 no-train probe passed strict load/base equality | add regression test and archive evidence |
| READY | Lancet checkpoint completeness | residual state/optimizer and reload covered by tests/smoke | extend formal hash manifest |
| READY | Output separation/dirty guard | `archive_run.py` enforces run-type roots and clean formal tree | disallow emergency dirty override in this protocol |
| READY | Train env seed | `OffPolicyAlgorithm.learn()` calls `env.reset(seed=self.seed)` | none |
| NEEDS SMALL PATCH | Shared-fork configs | current full YAMLs independently run 1M offline + 500k online | add offline-init and online-fork configs for debug/formal |
| NEEDS SMALL PATCH | Eval seed | `BaseAlgorithm._evaluate()` calls `eval_env.reset()` without seed | implement isolated deterministic eval seed stream |
| NEEDS SMALL PATCH | Paired RNG alignment | residual construction advances global PyTorch RNG; checkpoint has no RNG state | isolate residual initialization and restore/reset documented training RNG streams after load |
| NEEDS SMALL PATCH | Final endpoint | online loop may exit after crossing the final boundary without a final eval | force and test final evaluation at 500k |
| NEEDS SMALL PATCH | Diagnostics | several Q/TD/residual statistics are absent or use ratio-of-means | add no-grad common and Lancet diagnostic tags |
| NEEDS SMALL PATCH | GPU scheduling | launcher records all GPUs but no selected id; command defaults to first visible GPU | add/require `--gpu-id`, metadata, peak-memory record |
| NEEDS SMALL PATCH | Lineage/variant | resolved config has load path, but metadata lacks protocol, method label, parent checkpoint hash | add explicit metadata fields and archive label separate from registry name |
| NEEDS SMALL PATCH | Formal periodic checkpoints | current full YAMLs resolve `checkpoint_freq: 0` | freeze 50k formal / 5k debug in stage configs |
| BLOCKER | Formal launch authorization | no stability pilots; worktree is dirty/unpushed; matrix has not passed gates | complete review, patches, commit, pilots, and checklist |

## Readiness checklist

- [ ] Current Lancet-TD implementation manually reviewed.
- [ ] Working tree clean; frozen commit pushed.
- [ ] Stage-specific offline-init and online-fork configs reviewed.
- [ ] Shared WSRL offline checkpoint strict-load and equality test passed.
- [ ] Shared checkpoint path/hash and Lancet residual-init hash recorded.
- [ ] Evaluation seed behavior verified.
- [ ] Paired post-load training RNG behavior verified.
- [ ] Final endpoint evaluation verified.
- [ ] Required diagnostics present and finite.
- [ ] GPU id/peak-memory metadata verified.
- [ ] WSRL seed-0 debug pilot passed.
- [ ] Lancet-TD seed-0 debug pilot passed.
- [ ] Checkpoint reload/evaluation parity passed.
- [x] Formal run and checkpoint roots currently contain zero files.
- [x] Evaluation metric definitions frozen in this protocol.
- [x] Ten-cell formal comparison matrix frozen.

Only after every item is checked may the formal launcher be used.

## Open research decisions outside v1

The mathematical definition of U-variation, its target, loss weight, and
action-preference interpretation remain open. They are deliberately outside
this TD-only protocol. Action-dependent analysis is optional and does not
authorize implementing or enabling U-variation.
