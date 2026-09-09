# Repository Structure

```text
/home/zhaozihan/Lancet/
├── rl-garden/
│   ├── AGENTS.md
│   ├── .agents/
│   │   ├── rules/ and runbooks/     # upstream authoritative guidance
│   │   ├── local/                   # ignored machine bindings
│   │   └── lancet/                  # current state + incremental memory
│   ├── handoff/                     # this human documentation
│   ├── dev                          # Docker backend passthrough
│   ├── docker/lancet-d4rl/          # reproducible image/start/bootstrap
│   ├── examples/                    # thin registry entrypoints
│   ├── configs/off2on/              # WSRL/Lancet presets
│   ├── rl_garden/
│   │   ├── algorithms/              # learning logic
│   │   ├── networks/ and policies/  # model components
│   │   ├── buffers/                 # replay and dataset backends
│   │   ├── common/                  # config, logging, checkpoint helpers
│   │   ├── envs/                    # backend-neutral env registry
│   │   └── training/off2on/         # registration and phase runner
│   ├── scripts/
│   │   ├── audit/                   # validation-only checks
│   │   └── experiments/             # archived experiment launcher
│   └── tests/                       # focused regression tests
└── data/
    ├── datasets/d4rl/
    ├── runs/{smoke,debug,formal,legacy_unclassified}/
    ├── checkpoints/{smoke,debug,formal,legacy_unclassified}/
    ├── videos/
    └── cache/
```

There is one source checkout. Datasets, checkpoints, logs, videos, WandB data,
and caches stay outside Git under `data/`. The old `.agent/` tree was fully
migrated to `.agents/lancet/`; two continuity sources are not retained.
