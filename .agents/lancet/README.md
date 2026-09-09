# Lancet Agent Continuity

This namespace holds Lancet-specific current state and incremental engineering
memory. It complements, and never replaces, rl-garden's authoritative rules in
`../rules/` and operational runbooks in `../runbooks/`.

- `CURRENT_STATE.md` is the concise description of the project *now*.
- `memory/` is an incremental journal of meaningful engineering steps.
- `templates/` defines the required memory and human-handoff formats.

## Startup order

1. Read `AGENTS.md`.
2. Read the task-relevant files under `.agents/rules/` and `.agents/runbooks/`.
3. Read `.agents/lancet/CURRENT_STATE.md`.
4. Read `.agents/lancet/memory/INDEX.md`.
5. Read only the latest or task-relevant 3–5 memories.
6. Read the source files involved in the current task.

## Working protocol

For each meaningful engineering unit: inspect, decide, implement, validate,
write one memory immediately, update the index, and update current state when
the project changed. Ordinary inspection commands do not receive memories.
Experiments must use the archival protocol in `handoff/07_experiment_workflow.md`.
