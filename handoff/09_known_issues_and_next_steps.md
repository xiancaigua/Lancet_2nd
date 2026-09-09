# Known Issues and Next Steps

## P0 — v3 implementation and validation gates

1. Human-review `docs/design/lancet-v3-implementation.md`, its Theory–Code
   Alignment table, and
   `experiments/protocols/antmaze_wsrl_lancet_v3.md`.
2. Implement a new `lancet_v3` identity: shared residual backbone with
   critic-aligned heads, exact-zero output, K=8 local-action centering,
   per-critic detached TD-residual targets, and optional detached U weighting.
   Preserve the executable `lancet` identity as Legacy Lancet-TD.
3. Add the focused v3 unit tests, then run one archived tiny real-data smoke.
   Existing legacy Lancet tests and smoke evidence do not validate v3.
4. Apply the bounded protocol-enablement patches: shared offline-initializer
   and online-fork configs, deterministic evaluation seed, forced final
   endpoint, required diagnostics, selected-GPU/peak-memory and checkpoint
   lineage metadata, and periodic checkpoint cadence.
5. Review and commit/push a clean frozen revision, then run the archived v3
   seed-0 stability pilot. Formal remains blocked until every gate passes.

## P1 — scientific decisions and empirical checks

1. Confirm after the v3 stability pilot—not before—whether K=8, perturbation
   scale 0.1, U-weight cap 3, residual LR 1e-3, regularizer 1e-4, and the 50k
   linear correction window remain frozen for formal work.
2. Treat observed-action centered TD fitting as a practical projection; its
   relationship to an oracle counterfactual projection is an empirical theory
   question, not a proven implementation property.
3. Keep action-wise Q/residual rank diagnostics optional for phase 2. They are
   useful mechanism evidence but are not a blocker for the first v3 pilot.
4. Choose operational destinations such as WandB/ExpNote before formal runs;
   no secret or workspace identifier should be committed.

Lancet v3 no longer has an undefined `lambda_u_variation` loss. It defines
`U_N(s)` as a detached state weight on residual fitting only. The old
`lambda_u_variation=0` statement applies to the Legacy Lancet-TD scaffold and
must not be carried into the v3 implementation.

## P2 — non-blocking environment issues

1. D4RL warns about optional Flow, CARLA, GymBullet/pkg_resources, X11, and
   legacy Gym deprecations. AntMaze validation passes; do not install unrelated
   backends merely to silence warnings.
2. Kitchen reset works but its dataset is not locally archived/validated.
3. `gh` is absent, while GitHub SSH authentication works. CLI installation is
   unnecessary unless a future workflow explicitly needs it.

## Current recommendation

The runtime, archive workflow, audits, and historical WSRL/Legacy-Lancet tiny
smokes are operational. The code-ready v3 design and benchmark protocol are
now drafted, but v3 itself is not implemented. The next approved action is a
small, reviewable v3 implementation plus unit tests—not any training run.
