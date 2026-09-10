#!/usr/bin/env python3
"""Prepare, run, and finalize a Lancet experiment archive from the Host.

This script uses only the Python standard library on the Host. Training and
config resolution are delegated to ``./dev d4rl`` so Host Python never becomes
the training runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOST_DATA = Path(os.environ.get("LANCET_HOST_DATA_ROOT", REPO.parent / "data"))
CONTAINER_DATA = Path("/data/lancet")
RUN_TYPES = ("smoke", "debug", "formal")


def _run(command: list[str], *, check: bool = True, capture: bool = True):
    return subprocess.run(
        command,
        cwd=REPO,
        check=check,
        text=True,
        capture_output=capture,
    )


def _git(*args: str) -> str:
    return _run(["git", *args]).stdout.strip()


def _safe_segment(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise SystemExit(
            f"{field} must contain only letters, digits, '.', '_' or '-': {value!r}"
        )
    return value


def _timestamp() -> str:
    return dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _unique_directory(base: Path, experiment_id: str) -> Path:
    candidate = base / experiment_id
    suffix = 1
    while candidate.exists():
        candidate = base / f"{experiment_id}_{suffix:02d}"
        suffix += 1
    return candidate


def _container_path(host_path: Path) -> str:
    try:
        relative = host_path.resolve().relative_to(HOST_DATA.resolve())
    except ValueError as exc:
        raise SystemExit(f"Archive path escaped data root: {host_path}") from exc
    return str(CONTAINER_DATA / relative)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _parse_json_output(output: str) -> dict:
    """Extract one JSON object from stdout polluted by legacy import notices."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\{", output):
        try:
            value, end = decoder.raw_decode(output[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not output[match.start() + end :].strip():
            return value
    raise json.JSONDecodeError("no complete JSON object in stdout", output, 0)


def _hardware() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _gpu_free_mib(gpu_id: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_id),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return int(result.stdout.strip())


def _wait_for_gpu_capacity(
    gpu_id: int, minimum_free_mib: int, metadata: dict, metadata_path: Path
) -> None:
    while True:
        free_mib = _gpu_free_mib(gpu_id)
        metadata["gpu_free_mib_last"] = free_mib
        metadata["gpu_capacity_checked_at"] = _iso_now()
        if free_mib >= minimum_free_mib:
            metadata["gpu_capacity_state"] = "ready"
            _write_json(metadata_path, metadata)
            print(
                f"gpu {gpu_id} ready: {free_mib} MiB free "
                f"(required {minimum_free_mib} MiB)",
                flush=True,
            )
            return
        metadata["gpu_capacity_state"] = "waiting"
        _write_json(metadata_path, metadata)
        print(
            f"gpu {gpu_id} waiting: {free_mib} MiB free "
            f"(required {minimum_free_mib} MiB)",
            flush=True,
        )
        time.sleep(30)


def _validate_intent(args: argparse.Namespace, config: Path, dirty: bool) -> None:
    name = config.name.lower()
    if args.run_type == "smoke" and "smoke" not in name:
        raise SystemExit("Smoke runs require a config whose filename contains 'smoke'.")
    if args.run_type == "formal" and "smoke" in name:
        raise SystemExit("Formal runs cannot use a smoke config.")
    if args.run_type == "formal" and dirty and not args.allow_dirty_formal:
        raise SystemExit(
            "Formal runs require a clean working tree. Use --allow-dirty-formal only "
            "when the archive must capture git.diff."
        )
    if args.run_type == "formal" and (
        getattr(args, "protocol", None) is None
        or getattr(args, "dataset_path", None) is None
    ):
        raise SystemExit("Formal runs require --protocol and --dataset-path.")


def _readme(args: argparse.Namespace, metadata: dict, exact_command: str) -> str:
    success = "\n".join(f"- {item}" for item in args.success_criteria)
    failure_items = args.failure_criteria or [
        "Process exits non-zero or required evidence is missing."
    ]
    failure = "\n".join(f"- {item}" for item in failure_items)
    important = args.important_hyperparameters or [
        "See config.yaml and resolved_config.json."
    ]
    parameters = "\n".join(f"- {item}" for item in important)
    return f"""# Experiment

## Purpose

{args.purpose}

## Hypothesis / Question

{args.hypothesis}

## Run Type

{args.run_type}

## Algorithm

{args.algorithm}

## Method label

{metadata["method_label"]}

## Environment

{args.environment}

## Seed

{args.seed}

## Git

- Branch: `{metadata["git_branch"]}`
- Commit: `{metadata["git_commit"]}`
- Dirty at preparation: `{str(metadata["git_dirty"]).lower()}`

## Configuration

- Config path: `{metadata["config_path"]}`
- Frozen copy: `config.yaml`
- Effective config: `resolved_config.json`

## Frozen identity

- Source config SHA256: `{metadata["source_config_sha256"]}`
- Resolved config SHA256: `{metadata["resolved_config_sha256"]}`
- Protocol: `{metadata["protocol_path"]}`
- Protocol SHA256: `{metadata["protocol_sha256"]}`
- Dataset: `{metadata["dataset_path"]}`
- Dataset SHA256: `{metadata["dataset_sha256"]}`
- Shared initializer lineage: `{metadata["lineage_checkpoint_path"]}`
- Lineage checkpoint SHA256: `{metadata["lineage_checkpoint_sha256"]}`

Important hyperparameters:

{parameters}

## Baseline / Comparison

{args.baseline}

## Exact Command

```bash
{exact_command}
```

## Expected Outputs

- `train.log`
- `analysis.md`
- effective rl-garden config below `train/`
- checkpoints under `{metadata["checkpoint_path"]}`

## Success Criteria

{success}

## Failure Criteria

{failure}
"""


def _analysis(
    metadata: dict, returncode: int, log_text: str, checkpoints: list[Path]
) -> str:
    status = metadata["status"]
    has_traceback = "Traceback (most recent call last)" in log_text
    has_nan = bool(re.search(r"(?i)(?<![A-Za-z])nan(?![A-Za-z])", log_text))
    checkpoint_lines = (
        "\n".join(f"- `{path}`" for path in checkpoints)
        if checkpoints
        else "- not produced"
    )
    return f"""# Experiment Analysis

## Status

{status} (process return code {returncode})

## Main Metrics

Paper-performance metrics were not collected; this archive is evaluated only
against its declared `{metadata["run_type"]}` success criteria.

## Training Behavior

- Traceback detected in `train.log`: `{str(has_traceback).lower()}`
- NaN token detected in `train.log`: `{str(has_nan).lower()}`
- Checkpoints discovered:
{checkpoint_lines}

## Comparison

No performance comparison is inferred from a smoke/debug run.

## Problems / Anomalies

See `train.log`; absence of a recorded metric means not collected, not zero.

## Interpretation

The process evidence above establishes execution status only. Algorithm-specific
claims require the validation artifacts referenced by this archive.

## Limitations

This automatic summary does not infer quality or convergence from configuration.

## Next Recommended Step

Review `train.log`, checkpoint evidence, and the declared success criteria before
promoting any result or starting a formal run.
"""


def _next_memory_sequence(date: str) -> int:
    memory_dir = REPO / ".agents/lancet/memory"
    seqs = []
    for path in memory_dir.glob(f"{date}_*.md"):
        match = re.match(rf"{re.escape(date)}_(\d{{3}})_", path.name)
        if match:
            seqs.append(int(match.group(1)))
    return max(seqs, default=0) + 1


def _update_memory(metadata: dict, returncode: int, checkpoints: list[Path]) -> Path:
    date = dt.datetime.now().astimezone().strftime("%Y-%m-%d")
    seq = _next_memory_sequence(date)
    slug = (
        f"{metadata['method_label']}-{metadata['environment']}-{metadata['run_type']}"
    )
    path = REPO / ".agents/lancet/memory" / f"{date}_{seq:03d}_{slug}.md"
    result = "passed" if returncode == 0 else "failed"
    path.write_text(
        f"""# Agent Memory: {metadata["method_label"]} {metadata["run_type"]}

- Date: {date}
- Sequence: {seq:03d}
- Agent: archived experiment launcher
- Branch: `{metadata["git_branch"]}`
- Commit before: `{metadata["git_commit"]}`
- Commit after: working tree, not committed
- Status: {"completed" if returncode == 0 else "blocked"}

## Goal

{metadata["purpose"]}

## What changed

Created and finalized one `{metadata["run_type"]}` experiment archive.

## Key files

- `{metadata["output_dir"]}`

## Validation

Process result: {result}; return code {returncode}; checkpoints found: {len(checkpoints)}.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

See `analysis.md` and `train.log` in the archive.

## Next dependency

Review algorithm-specific success criteria before the next experiment.

## Resume hint

Open `{metadata["output_dir"]}/analysis.md` first.
""",
        encoding="utf-8",
    )

    index = REPO / ".agents/lancet/memory/INDEX.md"
    text = index.read_text(encoding="utf-8")
    row = (
        f"| {seq:03d} | {date} | {metadata['method_label']} {metadata['run_type']} | "
        f"{'completed' if returncode == 0 else 'blocked'} | [memory]({path.name}) |"
    )
    marker = "|---|---|---|---|---|\n"
    if row not in text:
        text = text.replace(marker, marker + row + "\n", 1)
    topic = "### Dataset / experiments\n"
    link = f"- [{metadata['method_label']} {metadata['run_type']}]({path.name})\n"
    if link not in text:
        text = text.replace(topic, topic + "\n" + link, 1)
    index.write_text(text, encoding="utf-8")
    return path


def _update_current_state(metadata: dict) -> None:
    path = REPO / ".agents/lancet/CURRENT_STATE.md"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"^Last updated:.*$",
        f"Last updated: {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}  ",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    heading = "## Latest archived experiments\n"
    bullet = (
        f"- `{metadata['experiment_id']}`: {metadata['method_label']} "
        f"{metadata['run_type']} -> {metadata['status']} (`{metadata['output_dir']}`)\n"
    )
    if heading not in text:
        text = text.replace(
            "## Known issues\n", heading + "\n" + bullet + "\n## Known issues\n", 1
        )
    elif bullet not in text:
        text = text.replace(heading, heading + "\n" + bullet, 1)
    path.write_text(text, encoding="utf-8")


def _update_handoff_index(metadata: dict) -> None:
    path = REPO / "handoff/experiments/README.md"
    text = path.read_text(encoding="utf-8")
    output = metadata["output_dir"]
    if metadata["run_type"] == "formal":
        marker = "<!-- formal-rows -->"
        row = (
            f"| {metadata['environment']} | {metadata['method_label']} | {metadata['seed']} | "
            f"{metadata['status']} | not collected | `{output}` |"
        )
    else:
        marker = f"<!-- {metadata['run_type']}-rows -->"
        row = (
            f"| {metadata['experiment_id']} | {metadata['status']} | return code "
            f"{metadata['return_code']} | `{output}` |"
        )
    if row not in text:
        text = text.replace(marker, row + "\n" + marker, 1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-type", required=True, choices=RUN_TYPES)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument(
        "--method-label",
        default=None,
        help="Archive label when multiple methods share one registry algorithm.",
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Expose one physical GPU to the container command as logical cuda:0.",
    )
    parser.add_argument(
        "--min-free-gpu-mib",
        type=int,
        default=None,
        help="Wait under the per-GPU lock until this much GPU memory is free.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Frozen protocol file; required for formal runs.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Host dataset file to hash; required for formal runs.",
    )
    parser.add_argument(
        "--lineage-checkpoint",
        type=Path,
        help="Shared initializer checkpoint to hash for an online fork.",
    )
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--success-criteria", required=True, action="append")
    parser.add_argument("--failure-criteria", action="append", default=[])
    parser.add_argument("--baseline", default="None; standalone validation run.")
    parser.add_argument(
        "--important-hyperparameter",
        dest="important_hyperparameters",
        action="append",
        default=[],
    )
    parser.add_argument("--allow-dirty-formal", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    args.algorithm = _safe_segment(args.algorithm, "algorithm")
    method_label = _safe_segment(args.method_label or args.algorithm, "method-label")
    args.environment = _safe_segment(args.environment, "environment")
    config = args.config if args.config.is_absolute() else REPO / args.config
    config = config.resolve()
    if not config.is_file():
        raise SystemExit(f"Config does not exist: {config}")
    try:
        config_rel = config.relative_to(REPO)
    except ValueError as exc:
        raise SystemExit("Config must be inside the repository.") from exc

    protocol = None
    protocol_rel = None
    if args.protocol is not None:
        protocol = (
            args.protocol if args.protocol.is_absolute() else REPO / args.protocol
        )
        protocol = protocol.resolve()
        if not protocol.is_file():
            raise SystemExit(f"Protocol does not exist: {protocol}")
        try:
            protocol_rel = protocol.relative_to(REPO)
        except ValueError as exc:
            raise SystemExit("Protocol must be inside the repository.") from exc
    dataset_path = args.dataset_path.resolve() if args.dataset_path else None
    if dataset_path is not None and not dataset_path.is_file():
        raise SystemExit(f"Dataset does not exist: {dataset_path}")
    lineage_checkpoint = (
        args.lineage_checkpoint.resolve() if args.lineage_checkpoint else None
    )
    if lineage_checkpoint is not None and not lineage_checkpoint.is_file():
        raise SystemExit(f"Lineage checkpoint does not exist: {lineage_checkpoint}")

    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    _validate_intent(args, config, dirty)
    if args.min_free_gpu_mib is not None:
        if args.gpu_id is None:
            raise SystemExit("--min-free-gpu-mib requires --gpu-id.")
        if args.min_free_gpu_mib <= 0:
            raise SystemExit("--min-free-gpu-mib must be positive.")

    experiment_id = _timestamp()
    output_dir = _unique_directory(
        HOST_DATA
        / "runs"
        / args.run_type
        / args.environment
        / method_label
        / f"seed_{args.seed}",
        experiment_id,
    )
    experiment_id = output_dir.name
    checkpoint_dir = (
        HOST_DATA
        / "checkpoints"
        / args.run_type
        / args.environment
        / method_label
        / f"seed_{args.seed}"
        / experiment_id
    )
    output_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    for child in ("metrics", "plots"):
        (output_dir / child).mkdir()

    container_output = _container_path(output_dir)
    container_checkpoint = _container_path(checkpoint_dir)
    training_args = args.training_args
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    command = ["./dev", "d4rl"]
    if args.gpu_id is not None:
        if args.gpu_id < 0:
            raise SystemExit("--gpu-id must be non-negative.")
        command.extend(["env", f"CUDA_VISIBLE_DEVICES={args.gpu_id}"])
    command.extend(
        [
            "python",
            "examples/train_off2on.py",
            args.algorithm,
            "--config",
            str(config_rel),
            "--log_dir",
            container_output,
            "--checkpoint_dir",
            container_checkpoint,
            "--exp_name",
            "train",
            "--seed",
            str(args.seed),
            *training_args,
        ]
    )
    exact_command = shlex.join(command)

    metadata = {
        "experiment_id": experiment_id,
        "run_type": args.run_type,
        "algorithm": args.algorithm,
        "method_label": method_label,
        "environment": args.environment,
        "seed": args.seed,
        "purpose": args.purpose,
        "git_branch": branch,
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_path": "git.diff" if dirty and args.run_type == "formal" else None,
        "hardware": _hardware(),
        "gpu_id": args.gpu_id,
        "gpu_lock_path": (
            str(HOST_DATA / "locks" / f"gpu_{args.gpu_id}.lock")
            if args.gpu_id is not None
            else None
        ),
        "gpu_lock_state": "not_requested",
        "min_free_gpu_mib": args.min_free_gpu_mib,
        "gpu_free_mib_last": None,
        "gpu_capacity_checked_at": None,
        "gpu_capacity_state": (
            "not_requested" if args.min_free_gpu_mib is None else "pending"
        ),
        "start_time": None,
        "end_time": None,
        "status": "prepared",
        "return_code": None,
        "config_path": str(config_rel),
        "source_config_sha256": _sha256(config),
        "resolved_config_sha256": None,
        "protocol_path": str(protocol_rel) if protocol_rel else None,
        "protocol_sha256": _sha256(protocol) if protocol else None,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_sha256": _sha256(dataset_path) if dataset_path else None,
        "lineage_checkpoint_path": (
            str(lineage_checkpoint) if lineage_checkpoint else None
        ),
        "lineage_checkpoint_sha256": (
            _sha256(lineage_checkpoint) if lineage_checkpoint else None
        ),
        "resolved_config_path": str(output_dir / "resolved_config.json"),
        "command_path": str(output_dir / "command.txt"),
        "log_path": str(output_dir / "train.log"),
        "checkpoint_path": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "container_output_dir": container_output,
        "container_checkpoint_path": container_checkpoint,
    }

    shutil.copy2(config, output_dir / "config.yaml")
    (output_dir / "command.txt").write_text(exact_command + "\n", encoding="utf-8")
    resolve_command = [*command, "--dry-run"]
    resolved = _run(resolve_command)
    try:
        resolved_json = _parse_json_output(resolved.stdout)
    except json.JSONDecodeError as exc:
        (output_dir / "config_resolution.stdout").write_text(
            resolved.stdout, encoding="utf-8"
        )
        (output_dir / "config_resolution.stderr").write_text(
            resolved.stderr, encoding="utf-8"
        )
        raise SystemExit(f"Resolved config was not JSON: {exc}") from exc
    inputs = resolved_json.get("inputs", {})
    if (
        inputs.get("log_dir") != container_output
        or inputs.get("checkpoint_dir") != container_checkpoint
    ):
        raise SystemExit(
            "Resolved output paths do not match the selected run type archive."
        )
    _write_json(output_dir / "resolved_config.json", resolved_json)
    metadata["resolved_config_sha256"] = _sha256(output_dir / "resolved_config.json")
    if dirty and args.run_type == "formal":
        (output_dir / "git.diff").write_text(_git("diff"), encoding="utf-8")
    (output_dir / "README.md").write_text(
        _readme(args, metadata, exact_command), encoding="utf-8"
    )
    _write_json(output_dir / "metadata.json", metadata)
    print(f"prepared: {output_dir}", flush=True)
    if args.prepare_only:
        return 0

    lock_context = contextlib.nullcontext()
    if args.gpu_id is not None:
        metadata["gpu_lock_state"] = "waiting"
        _write_json(output_dir / "metadata.json", metadata)
        lock_context = _exclusive_lock(Path(metadata["gpu_lock_path"]))
    with lock_context:
        if args.gpu_id is not None:
            metadata["gpu_lock_state"] = "acquired"
        if args.min_free_gpu_mib is not None:
            _wait_for_gpu_capacity(
                args.gpu_id,
                args.min_free_gpu_mib,
                metadata,
                output_dir / "metadata.json",
            )
        metadata["status"] = "running"
        metadata["start_time"] = _iso_now()
        _write_json(output_dir / "metadata.json", metadata)
        with (output_dir / "train.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    command,
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                returncode = process.returncode
                status = "finished" if returncode == 0 else "failed"
            except KeyboardInterrupt:
                returncode = 130
                status = "stopped"
    if args.gpu_id is not None:
        metadata["gpu_lock_state"] = "released"
    metadata["return_code"] = returncode
    metadata["end_time"] = _iso_now()
    metadata["status"] = status
    checkpoints = sorted(checkpoint_dir.rglob("*.pt"))
    metadata["checkpoint_files"] = [str(path) for path in checkpoints]
    _write_json(output_dir / "metadata.json", metadata)
    log_text = (output_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    (output_dir / "analysis.md").write_text(
        _analysis(metadata, returncode, log_text, checkpoints), encoding="utf-8"
    )
    with _exclusive_lock(HOST_DATA / "locks" / "continuity.lock"):
        memory = _update_memory(metadata, returncode, checkpoints)
        _update_current_state(metadata)
        _update_handoff_index(metadata)
    print(f"{metadata['status']}: {output_dir}")
    print(f"memory: {memory}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
