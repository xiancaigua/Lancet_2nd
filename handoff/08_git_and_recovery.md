# Git and Recovery

## Audited Git state

```text
active branch: integration/upstream-sync-20260914
integration HEAD: 824a5de (before final continuity commit)
main: db57e1e9e1ee6e81ffafd99b1547a96e26389df1 (unchanged)
upstream merged SHA: 252d1e0948618a0cd3675b9a05e0bcf4c29b5afb
origin: git@github.com:xiancaigua/Lancet_2nd.git
upstream: https://github.com/JaimeParker/rl-garden.git
backup branch: backup/pre-upstream-sync-20260914
backup tag: pre-upstream-sync-20260914
```

No token or private key is stored in Docker or these documents. The backup
branch/tag are pushed. The migration gate report now passes under the verified
no-new-lint-regressions policy. Promote integration only through an ordinary
fast-forward or merge; never force push.

## Rebuild the runtime

```bash
cd /home/zhaozihan/Lancet/rl-garden
docker build -f docker/lancet-d4rl/Dockerfile -t lancet-d4rl:cu128 .
./dev start d4rl
scripts/audit/run_all.sh
```

`run.sh` refuses to delete/replace an existing named container. If manual
container removal is ever required, first inspect the exact `lancet-d4rl`
target and never prune shared Docker state.

The checkout and all datasets/results live on the Host, so container loss is
recoverable. Source restoration uses Git; runtime restoration uses the pinned
Dockerfile/bootstrap; data recovery depends on Host storage/backups and is not
provided by Git.
