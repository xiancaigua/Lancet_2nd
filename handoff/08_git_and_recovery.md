# Git and Recovery

## Audited Git state

```text
branch: main
commit: 477abdac2f1e297c6ede82aafde20ce90b65de4b
origin: git@github.com:xiancaigua/Lancet_2nd.git
upstream: https://github.com/JaimeParker/rl-garden.git
SSH auth: authenticated as xiancaigua
GitHub CLI: not installed
```

No token or private key is stored in Docker or these documents. Do not change
identity/remotes or push without explicit scope. At handoff, consolidation
changes are intentionally uncommitted and must be reviewed separately from
unrelated work.

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
