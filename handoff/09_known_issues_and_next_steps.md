# Known Issues and Next Steps

## P0 — formal-readiness gates

1. Review the frozen planned protocol at
   `experiments/protocols/antmaze_wsrl_lancet_v1.md`. It is explicitly WSRL
   versus Lancet-TD (`lambda_u_variation=0`), not full Lancet.
2. Apply the bounded protocol-enablement patches: stage-specific shared-fork
   configs, deterministic evaluation seeds, final-endpoint evaluation,
   paired post-load RNG alignment, required Q/residual diagnostics, GPU
   assignment/peak-memory metadata, lineage metadata, and periodic checkpoint
   cadence.
3. Review and commit/push the current dirty working tree so pilots and formal
   archives can reference a frozen revision.
4. Run the archived seed-0 stability pilot only after those patches: one 20k
   shared WSRL offline initializer, then paired 20k-online WSRL/Lancet-TD
   debug branches. Formal remains blocked until both pass.

## P1 — method/protocol decisions

1. Define and review the U-variation mathematical target before enabling its
   existing hook. Current nonzero values intentionally fail.
2. Decide only protocol-adjacent operational destinations still open (WandB /
   ExpNote workspace); the checkpoint, five-seed matrix, budget, and metric
   definitions are now frozen in the v1 protocol.
3. Keep action-dependent Q/residual rank diagnostics optional for phase 2; they
   are not a v1 blocker and do not define U-variation.

## P2 — non-blocking environment issues

1. D4RL warns about optional Flow, CARLA, GymBullet/pkg_resources, X11, and
   legacy Gym deprecations. AntMaze validation passes; do not install unrelated
   backends merely to silence warnings.
2. Kitchen reset works but its dataset is not locally archived/validated.
3. `gh` is absent, while GitHub SSH authentication works. CLI installation is
   unnecessary unless a future workflow explicitly needs it.

## Current recommendation

The environment, audits, and real-data tiny smokes are operational. The first
formal protocol is drafted and the current implementation is classified
`NEEDS SMALL PATCH`. Human code review may proceed to the bounded patches and
then the stability pilot; formal benchmark execution is not approved.
