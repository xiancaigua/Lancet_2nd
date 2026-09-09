# rl-garden Architecture for Off2On Work

```text
YAML / CLI
   ↓
examples/train_off2on.py          thin dispatcher
   ↓
training/off2on/_registry.py      discovers public algorithm modules
   ↓
training/off2on/{wsrl,lancet}.py  Args + builder + registration
   ↓
training/off2on/_runner.py        env/dataset/offline/switch/online orchestration
   ↓
algorithms/{wsrl,lancet}.py       update math and algorithm state
   ↓
policies + networks + buffers + env backends + logger/checkpoint helpers
```

The D4RL backend resolves `antmaze-medium-play-v2`, while the dataset registry
loads its legacy HDF5 into the replay buffer. `_runner.py` performs offline
updates, saves `offline_final.pt`, records a probe batch, calls
`switch_to_online_mode`, then enters the inherited online rollout loop.

`Off2OnReplayMixin` owns replay transition behavior:

- `empty`: clear offline replay before online collection;
- `append`: retain it and append online transitions;
- `mixed`: preserve offline replay separately and sample a configured mixture.

WSRL uses the Cal-QL/SAC rollout shell and REDQ-style critic ensemble. SAC core
owns critic/target creation, TD targets, critic/actor steps, target Polyak
updates, and optimizer hooks. Checkpoint serialization is centralized in
`BaseAlgorithm`; algorithms extend it through metadata, optimizer, extra-state,
and training-state hooks.
