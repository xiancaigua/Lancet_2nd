# Agent Memory: Generation-2 offline dynamic queue launched

- Date: 2026-09-15
- Sequence: 006
- Branch: main
- Commit before: 53a899f0b9681a90dbe61ad48ce053b4c38ff682
- Commit after: pending
- Status: partial

## Goal

Use available non-reserved GPU capacity for the remaining generation-2 WSRL
offline initializers without changing formal scientific configuration.

## What changed

- Prepared formal archive contracts for initializer seeds 1--4.
- Started one unified lifecycle controller with GPU 0 and GPU 4 excluded.
- Started seed 1 on GPU 1 and seed 2 on GPU 2; seeds 3--4 remain queued.

## Validation

Queue/started emails were recorded as sent. Initial processes are alive and
their logs completed D4RL dataset loading; no early traceback/OOM signal was
observed. GPU 5 was sampled at 100% utilization and was correctly rejected.

## Decisions

The dynamic gate uses at least 13,756 MiB free, no more than 50% sampled GPU
utilization, and no more than 20,480 MiB foreign memory. GPU 4 remains
excluded. Online preparation remains disabled until the seed-0 baseline gate.

## Resume hint

Inspect `data/runs/formal/.lifecycle/formal_v2_offline_initializers.json` and
the tmux session `lancet-v2-offline-queue`; never create a second controller.
