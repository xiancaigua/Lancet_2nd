# Lancet Implementation and Validation Plan

Last updated: 2026-09-09

1. Freeze naming, implementation contract, and benchmark protocol.
2. Move the confirmed legacy executable to `LancetV1` / `lancet_v1`.
3. Implement current `Lancet` / `lancet` and unified Raw/Centered/Lancet variants.
4. Pass focused unit tests and WSRL/SAC/CQL regression tests.
5. Commit the implementation at a clean, traceable revision.
6. Run an archived AntMaze tiny smoke from a shared WSRL initializer.
7. Run archived seed-0 WSRL/Lancet 20k debug pilots.
8. Review stability evidence and issue the formal-readiness decision.

No five-seed formal run is part of this plan.
