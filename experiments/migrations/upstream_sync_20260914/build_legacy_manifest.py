#!/usr/bin/env python3
"""Freeze a read-only manifest of generation-1 Lancet formal archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPERSEDED = "SUPERSEDED_PRE_UPSTREAM_SYNC"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(job: dict[str, Any]) -> Path | None:
    directory = Path(str(job.get("checkpoint_dir", "")))
    if not directory.is_dir():
        return None
    preferred = "offline_final.pt" if job.get("stage") == "initializer" else "final.pt"
    candidate = directory / preferred
    if candidate.is_file():
        return candidate
    checkpoints = sorted(directory.glob("*.pt"), key=lambda item: item.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--formal-identity", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-head", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = load(args.state)
    identity = load(args.formal_identity)
    rows: list[dict[str, Any]] = []
    for job in state.get("jobs", []):
        archive = Path(str(job.get("archive", "")))
        metadata_path = archive / "metadata.json"
        metadata = load(metadata_path) if metadata_path.is_file() else {}
        summary_path = archive / "metrics" / "summary.json"
        summary = load(summary_path) if summary_path.is_file() else {}
        checkpoint = latest_checkpoint(job)
        checkpoint_hash = job.get("checkpoint_sha256")
        if checkpoint is not None and not checkpoint_hash:
            checkpoint_hash = sha256(checkpoint)
        scalar_ranges = summary.get("scalar_ranges", {})
        eval_range = scalar_ranges.get("eval/normalized_score", {})
        rows.append(
            {
                "job_id": job.get("job_id"),
                "stage": job.get("stage"),
                "method": job.get("method"),
                "seed": job.get("seed"),
                "operational_status": job.get("status"),
                "scientific_status": SUPERSEDED,
                "superseded_by": job.get("superseded_by"),
                "archive": str(archive),
                "metadata": str(metadata_path) if metadata_path.is_file() else None,
                "formal_commit": metadata.get(
                    "formal_algorithm_commit", identity.get("git_commit")
                ),
                "resolved_config_sha256": metadata.get("resolved_config_sha256"),
                "protocol_sha256": metadata.get(
                    "protocol_sha256", identity.get("protocol_sha256")
                ),
                "dataset_sha256": metadata.get(
                    "dataset_sha256", identity.get("dataset_sha256")
                ),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_global_step": summary.get("checkpoint_global_step"),
                "checkpoint_online_step": summary.get("checkpoint_online_step"),
                "finite_gate": summary.get("finite_gate"),
                "agent_reload_verified": summary.get("agent_reload_verified"),
                "eval_normalized_score_min": eval_range.get("min"),
                "eval_normalized_score_max": eval_range.get("max"),
                "completed_at": job.get("completed_at"),
            }
        )
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "reason": "upstream WSRL baseline correction + full framework resync",
        "scientific_status": SUPERSEDED,
        "repo_head_at_audit": args.repo_head,
        "formal_identity_path": str(args.formal_identity),
        "formal_identity": identity,
        "lifecycle_state_path": str(args.state),
        "jobs": rows,
        "exclusion_rule": "Generation-1 runs are archival/debug evidence only and must not enter generation-2 statistics.",
    }
    (args.output_dir / "legacy_formal_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    active = [row for row in rows if not row["superseded_by"]]
    counts: dict[tuple[str, str], int] = {}
    for row in active:
        key = (str(row["method"]), str(row["operational_status"]))
        counts[key] = counts.get(key, 0) + 1
    count_lines = "\n".join(
        f"| {method} | {status} | {count} |"
        for (method, status), count in sorted(counts.items())
    )
    job_lines = "\n".join(
        f"| {row['method']} | {row['seed']} | {row['operational_status']} | "
        f"{row['checkpoint_global_step'] or '—'} | {row['finite_gate']} | "
        f"`{row['archive']}` |"
        for row in active
    )
    summary = f"""# Legacy formal generation-1 summary

Generated: `{generated_at}`  
Scientific status: **{SUPERSEDED}**

Generation 1 is superseded because upstream corrected the WSRL AntMaze
CQL/Cal-QL configuration after launch and the full upstream migration changes
observation, policy, and replay infrastructure. The rerun is not selected due
to unfavorable returns. Old/new results must never be mixed statistically.

No run, scalar, metric, or checkpoint was deleted or rewritten. The JSON
manifest is the authoritative supersession overlay.

## Canonical lifecycle counts

| Method | Operational status | Count |
|---|---|---:|
{count_lines}

## Canonical jobs

| Method | Seed | Operational status | Final/global step | Finite | Archive |
|---|---:|---|---:|---|---|
{job_lines}

Raw and Centered generation-1 formal runs: **none**.

## Exclusion contract

- Preserve generation-1 archives for debugging, regression, and migration audit.
- Exclude `{SUPERSEDED}` from default formal analysis.
- Do not use generation-1 checkpoints as generation-2 scientific initializers.
"""
    (args.output_dir / "legacy_formal_summary.md").write_text(summary, encoding="utf-8")
    print(args.output_dir / "legacy_formal_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
