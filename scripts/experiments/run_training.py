#!/usr/bin/env python3
"""Low-frequency dynamic GPU queue and lifecycle controller for Lancet runs.

This is an external wrapper around the existing archived command. It never
imports or changes algorithm code. One controller owns all waiting jobs and
attached jobs; a lightweight worker exists only while a managed training
command is actually running.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pwd
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import archive_run
import notify_email

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LANCET_HOST_DATA_ROOT", REPO.parent / "data"))
DEFAULT_STATE = DATA / "runs/formal/.lifecycle/formal_pipeline.json"
DEFAULT_POLL_SECONDS = 5 * 60 * 60
DEFAULT_REQUIRED_FREE_MIB = 13_756  # observed 5,564 + 8 GiB margin
DEFAULT_MAX_UTIL = 20
DEFAULT_MAX_FOREIGN_MIB = 2_048
DEFAULT_SETTLE_SAMPLES = 3
DEFAULT_SETTLE_SECONDS = 30
DEFAULT_EXCLUDED_GPU_IDS: tuple[int, ...] = ()
FORMAL_PROTOCOL = REPO / "experiments/protocols/antmaze_wsrl_lancet.md"
DATASET = DATA / "datasets/d4rl/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5"


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def _locked_state(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text(encoding="utf-8"))
        yield state
        _atomic_json(path, state)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _mutate(path: Path, change: Callable[[dict], None]) -> None:
    with _locked_state(path) as state:
        change(state)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return stat.split()[2] != "Z"


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def _owner(pid: int) -> str:
    try:
        return pwd.getpwuid(Path(f"/proc/{pid}").stat().st_uid).pw_name
    except (FileNotFoundError, KeyError, PermissionError):
        return "unknown"


def _gpu_snapshot() -> list[dict]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    devices: list[dict] = []
    by_uuid: dict[str, dict] = {}
    for line in query.stdout.splitlines():
        index, uuid, total, used, free, util = [part.strip() for part in line.split(",")]
        device = {
            "gpu_id": int(index),
            "uuid": uuid,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(util),
            "processes": [],
        }
        devices.append(device)
        by_uuid[uuid] = device
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in processes.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or parts[0] not in by_uuid:
            continue
        uuid, pid_text, name, memory = parts
        pid = int(pid_text)
        command = _cmdline(pid)
        lancet = "train_off2on.py" in command and (
            "/data/lancet" in command or "antmaze_medium_play" in command
        )
        by_uuid[uuid]["processes"].append(
            {
                "pid": pid,
                "user": _owner(pid),
                "name": name,
                "memory_mib": int(memory),
                "lancet": lancet,
                "command": command[:500],
            }
        )
    return devices


def _gpu_lock_available(gpu_id: int) -> bool:
    path = DATA / "locks" / f"gpu_{gpu_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock, fcntl.LOCK_UN)
        return True


def _stable_candidates(settings: dict) -> tuple[list[int], list[dict]]:
    samples: list[list[dict]] = []
    for index in range(settings["settle_samples"]):
        samples.append(_gpu_snapshot())
        if index + 1 < settings["settle_samples"]:
            time.sleep(settings["settle_seconds"])
    candidate_ids: list[int] = []
    audit: list[dict] = []
    excluded_gpu_ids = {int(gpu_id) for gpu_id in settings.get("excluded_gpu_ids", [])}
    for gpu_id in range(len(samples[0])):
        rows = [sample[gpu_id] for sample in samples]
        free_min = min(row["memory_free_mib"] for row in rows)
        util_max = max(row["utilization_percent"] for row in rows)
        processes = rows[-1]["processes"]
        foreign_mib = sum(item["memory_mib"] for item in processes if not item["lancet"])
        has_lancet = any(item["lancet"] for item in processes)
        lock_available = _gpu_lock_available(gpu_id)
        reasons = []
        if gpu_id in excluded_gpu_ids:
            reasons.append("excluded_by_policy")
        if free_min < settings["required_free_mib"]:
            reasons.append("insufficient_free_memory")
        if util_max > settings["max_utilization_percent"]:
            reasons.append("busy_utilization")
        if foreign_mib > settings["max_foreign_memory_mib"]:
            reasons.append("foreign_memory")
        if has_lancet:
            reasons.append("lancet_already_running")
        if not lock_available:
            reasons.append("gpu_lock_held")
        usable = not reasons
        if usable:
            candidate_ids.append(gpu_id)
        audit.append(
            {
                "gpu_id": gpu_id,
                "free_mib_min": free_min,
                "utilization_percent_max": util_max,
                "foreign_memory_mib": foreign_mib,
                "lock_available": lock_available,
                "usable": usable,
                "reasons": reasons,
                "processes": processes,
            }
        )
    candidate_ids.sort(
        key=lambda gpu_id: next(row["free_mib_min"] for row in audit if row["gpu_id"] == gpu_id),
        reverse=True,
    )
    return candidate_ids, audit


def _tail(path: Path, limit: int = 512 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read().decode(errors="replace")


def _progress(archive: Path, job: dict, gpu: dict | None) -> dict:
    log = archive / "train.log"
    text = _tail(log)
    matches = re.findall(r"(?:offline|online):[^\r\n]*?(\d+)/(\d+)", text)
    current, target = (map(int, matches[-1]) if matches else (None, None))
    start = job.get("started_at")
    rate = None
    eta = None
    seconds_per_1k = None
    if current and start:
        elapsed = dt.datetime.now().astimezone() - dt.datetime.fromisoformat(start)
        hours = elapsed.total_seconds() / 3600
        if hours > 0:
            rate = current / hours
            seconds_per_1k = 3_600_000 / rate
            if target and rate > 0:
                eta = (target - current) / rate
    checkpoint_root = Path(job["checkpoint_dir"])
    checkpoints = sorted(checkpoint_root.rglob("*.pt"), key=lambda path: path.stat().st_mtime)
    result = {
        "job_id": job["job_id"],
        "status": job["status"],
        "last_step": current,
        "target_step": target,
        "updates_per_hour": round(rate, 2) if rate else None,
        "seconds_per_1k_updates": round(seconds_per_1k, 2) if seconds_per_1k else None,
        "estimated_remaining_hours": round(eta, 2) if eta is not None else None,
        "gpu_id": job.get("gpu_id"),
        "gpu_memory_used_mib": gpu["memory_used_mib"] if gpu else None,
        "gpu_utilization_percent": gpu["utilization_percent"] if gpu else None,
        "last_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "last_checkpoint_time": (
            dt.datetime.fromtimestamp(checkpoints[-1].stat().st_mtime).astimezone().isoformat(timespec="seconds")
            if checkpoints
            else None
        ),
        "last_update_time": (
            dt.datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            if log.exists()
            else None
        ),
        "checked_at": _now(),
        "alerts": {
            "traceback": "Traceback (most recent call last)" in text,
            "oom": bool(re.search(r"(?i)(cuda )?out of memory", text)),
            "nan_or_inf": bool(re.search(r"(?i)(?<![A-Za-z])(nan|inf)(?![A-Za-z])", text)),
        },
    }
    _atomic_json(archive / "progress.json", result)
    return result


def _notification(path: Path, job_id: str, event: str, subject: str, body: str) -> None:
    should_send = False
    with _locked_state(path) as state:
        job = next(item for item in state["jobs"] if item["job_id"] == job_id)
        notifications = job.setdefault("notifications", {})
        if event not in notifications:
            notifications[event] = {"status": "attempting", "attempted_at": _now()}
            should_send = True
    if not should_send:
        return
    try:
        result = notify_email.send_notification(subject, body)
    except Exception as exc:  # noqa: BLE001 - SMTP must never affect training.
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": _now()}
    with _locked_state(path) as state:
        job = next(item for item in state["jobs"] if item["job_id"] == job_id)
        job["notifications"][event] = result


def _replace_gpu(command: str, gpu_id: int) -> list[str]:
    parts = shlex.split(command)
    replaced = False
    for index, part in enumerate(parts):
        if part.startswith("CUDA_VISIBLE_DEVICES="):
            parts[index] = f"CUDA_VISIBLE_DEVICES={gpu_id}"
            replaced = True
    if not replaced:
        marker = parts.index("d4rl") + 1
        parts[marker:marker] = ["env", f"CUDA_VISIBLE_DEVICES={gpu_id}"]
    return parts


def _job_from_archive(archive: Path, *, status: str, mode: str, pid: int | None = None) -> dict:
    metadata = json.loads((archive / "metadata.json").read_text(encoding="utf-8"))
    checkpoint_dir = metadata["checkpoint_path"]
    return {
        "job_id": f"{metadata['method_label']}-seed-{metadata['seed']}-{metadata['experiment_id']}",
        "stage": "initializer" if metadata["method_label"] == "wsrl-initializer" else "online",
        "method": metadata["method_label"],
        "algorithm": metadata["algorithm"],
        "seed": metadata["seed"],
        "archive": str(archive),
        "checkpoint_dir": checkpoint_dir,
        "mode": mode,
        "status": status,
        "process_pid": pid,
        "worker_pid": None,
        "gpu_id": metadata.get("gpu_id") if mode == "attach" else None,
        "started_at": metadata.get("start_time"),
        "completed_at": None,
        "validation_status": "pending",
        "online_pair_prepared": False,
        "notifications": {},
    }


def _worker_impl(state_path: Path, job_id: str, gpu_id: int) -> int:
    with _locked_state(state_path) as state:
        job = next(item for item in state["jobs"] if item["job_id"] == job_id)
        archive = Path(job["archive"])
        command = _replace_gpu((archive / "command.txt").read_text(encoding="utf-8"), gpu_id)
        job.update(status="starting", gpu_id=gpu_id, worker_pid=os.getpid(), assigned_at=_now())
        metadata = json.loads((archive / "metadata.json").read_text(encoding="utf-8"))
        metadata.setdefault("gpu_assignment_history", []).append(
            {"gpu_id": gpu_id, "assigned_at": _now(), "scheduler": "dynamic-lifecycle"}
        )
        metadata.update(
            gpu_id=gpu_id,
            gpu_lock_path=str(DATA / "locks" / f"gpu_{gpu_id}.lock"),
            gpu_lock_state="acquiring",
            gpu_capacity_state="dynamic_gate_passed",
            infrastructure_commit=state["infrastructure_commit"],
            formal_algorithm_commit=state["formal_algorithm_commit"],
        )
        _atomic_json(archive / "metadata.json", metadata)
        (archive / "resolved_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    lock_path = DATA / "locks" / f"gpu_{gpu_id}.lock"
    with lock_path.open("a+", encoding="utf-8") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        metadata["gpu_lock_state"] = "acquired"
        metadata["status"] = "running"
        metadata["start_time"] = _now()
        _atomic_json(archive / "metadata.json", metadata)
        log_path = archive / "train.log"
        if log_path.exists() and log_path.stat().st_size:
            raise RuntimeError(f"refusing to overwrite non-empty log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with _locked_state(state_path) as state:
                job = next(item for item in state["jobs"] if item["job_id"] == job_id)
                job.update(status="running", process_pid=process.pid, started_at=metadata["start_time"])
            _notification(
                state_path,
                job_id,
                "started",
                f"[Lancet] STARTED - {job['method']} seed {job['seed']}",
                f"method: {job['method']}\nseed: {job['seed']}\nrun type: formal\nGPU: {gpu_id}\nPID: {process.pid}\nstart time: {metadata['start_time']}\nformal commit: {metadata['formal_algorithm_commit']}\ninfrastructure commit: {metadata['infrastructure_commit']}\nconfig: {metadata['config_path']}\narchive: {archive}\n",
            )
            returncode = process.wait()
        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
    status = "finished" if returncode == 0 else "failed"
    metadata.update(status=status, return_code=returncode, end_time=_now(), gpu_lock_state="released")
    checkpoints = sorted(Path(job["checkpoint_dir"]).rglob("*.pt"))
    metadata["checkpoint_files"] = [str(path) for path in checkpoints]
    _atomic_json(archive / "metadata.json", metadata)
    log_text = _tail(log_path, 2 * 1024 * 1024)
    (archive / "analysis.md").write_text(
        archive_run._analysis(metadata, returncode, log_text, checkpoints), encoding="utf-8"
    )
    with archive_run._exclusive_lock(DATA / "locks/continuity.lock"):
        archive_run._update_memory(metadata, returncode, checkpoints)
        archive_run._update_current_state(metadata)
        archive_run._update_handoff_index(metadata)
    lifecycle_status = "completed" if returncode == 0 else "failed"
    with _locked_state(state_path) as state:
        job = next(item for item in state["jobs"] if item["job_id"] == job_id)
        job.update(status=lifecycle_status, completed_at=metadata["end_time"], return_code=returncode)
    if returncode != 0:
        _notification(
            state_path,
            job_id,
            "failed",
            f"[Lancet] FAILED - {job['method']} seed {job['seed']}",
            f"method: {job['method']}\nseed: {job['seed']}\nstatus: failed\nGPU: {gpu_id}\nreturn code: {returncode}\narchive: {archive}\n",
        )
    return returncode


def _worker(state_path: Path, job_id: str, gpu_id: int) -> int:
    """Run one assigned job and convert wrapper failures into lifecycle state."""

    try:
        return _worker_impl(state_path, job_id, gpu_id)
    except Exception as exc:  # noqa: BLE001 - lifecycle boundary must be total.
        error = f"{type(exc).__name__}: {exc}"
        with _locked_state(state_path) as state:
            job = next(item for item in state["jobs"] if item["job_id"] == job_id)
            job.update(status="failed", completed_at=_now(), wrapper_error=error)
            archive = Path(job["archive"])
            metadata_path = archive / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(status="failed", end_time=_now(), lifecycle_wrapper_error=error)
            _atomic_json(metadata_path, metadata)
        _notification(
            state_path,
            job_id,
            "failed",
            f"[Lancet] FAILED - {job['method']} seed {job['seed']}",
            f"Lifecycle wrapper failed.\nmethod: {job['method']}\nseed: {job['seed']}\nGPU: {gpu_id}\nerror: {error}\narchive: {archive}\n",
        )
        return 1


def _validate_initializer(state_path: Path, job_id: str) -> bool:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = next(item for item in state["jobs"] if item["job_id"] == job_id)
    archive = Path(job["archive"])
    checkpoint = Path(job["checkpoint_dir"]) / "offline_final.pt"
    validation_dir = archive / "validation"
    validation_dir.mkdir(exist_ok=True)
    result = {"started_at": _now(), "checkpoint": str(checkpoint), "checks": {}}
    commands: list[tuple[str, list[str]]] = []
    if checkpoint.is_file():
        container_checkpoint = archive_run._container_path(checkpoint)
        commands.append(
            (
                "finite_checkpoint",
                ["./dev", "d4rl", "python", "scripts/experiments/validate_checkpoint_finite.py", "--checkpoint", container_checkpoint, "--expected-updates", "1000000"],
            )
        )
        commands.append(
            (
                "shared_fork_equality",
                ["./dev", "d4rl", "python", "scripts/experiments/validate_shared_fork.py", "--checkpoint", container_checkpoint],
            )
        )
        for algorithm, config in (
            ("wsrl", "configs/off2on/wsrl_antmaze_medium_play_v2_online.yaml"),
            ("lancet", "configs/off2on/lancet_antmaze_medium_play_v2.yaml"),
        ):
            commands.append(
                (
                    f"{algorithm}_reload_dry_run",
                    [
                        "./dev", "d4rl", "python", "examples/train_off2on.py", algorithm,
                        "--config", config,
                        "--log_dir", archive_run._container_path(validation_dir / algorithm),
                        "--checkpoint_dir", archive_run._container_path(validation_dir / f"{algorithm}_checkpoints"),
                        "--seed", str(job["seed"]), "--load_checkpoint", container_checkpoint, "--dry-run",
                    ],
                )
            )
    else:
        result["error"] = "offline_final.pt missing"
    passed = checkpoint.is_file()
    for name, command in commands:
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        result["checks"][name] = {
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        passed = passed and completed.returncode == 0
    if checkpoint.is_file():
        result["checkpoint_sha256"] = _sha256(checkpoint)
    result.update(status="passed" if passed else "failed", finished_at=_now())
    _atomic_json(validation_dir / "initializer_validation.json", result)
    with _locked_state(state_path) as current:
        target = next(item for item in current["jobs"] if item["job_id"] == job_id)
        target["validation_status"] = result["status"]
        target["validation_path"] = str(validation_dir / "initializer_validation.json")
        target["checkpoint_sha256"] = result.get("checkpoint_sha256")
        if not passed:
            target["status"] = "blocked"
    if not passed:
        _notification(
            state_path, job_id, "blocked", f"[Lancet] BLOCKED - initializer seed {job['seed']}",
            f"Initializer validation failed.\nseed: {job['seed']}\narchive: {archive}\nvalidation: {validation_dir / 'initializer_validation.json'}\n",
        )
    else:
        _notification(
            state_path,
            job_id,
            "completed",
            f"[Lancet] COMPLETED - initializer seed {job['seed']}",
            f"Initializer training and validation passed.\nseed: {job['seed']}\ncheckpoint: {checkpoint}\ncheckpoint sha256: {result['checkpoint_sha256']}\narchive: {archive}\n",
        )
    return passed


def _validate_online(state_path: Path, job_id: str) -> bool:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = next(item for item in state["jobs"] if item["job_id"] == job_id)
    archive = Path(job["archive"])
    checkpoint = Path(job["checkpoint_dir"]) / "final.pt"
    validation_dir = archive / "validation"
    validation_dir.mkdir(exist_ok=True)
    result = {"started_at": _now(), "checkpoint": str(checkpoint), "checks": {}}
    passed = checkpoint.is_file()
    if checkpoint.is_file():
        container_checkpoint = archive_run._container_path(checkpoint)
        container_archive = archive_run._container_path(archive)
        commands = (
            (
                "finite_checkpoint",
                ["./dev", "d4rl", "python", "scripts/experiments/validate_checkpoint_finite.py", "--checkpoint", container_checkpoint],
            ),
            (
                "agent_reload_dry_run",
                [
                    "./dev", "d4rl", "python", "examples/train_off2on.py", job["algorithm"],
                    "--config", f"{container_archive}/config.yaml",
                    "--log_dir", f"{container_archive}/validation/reload",
                    "--checkpoint_dir", f"{container_archive}/validation/reload-checkpoints",
                    "--seed", str(job["seed"]), "--load_checkpoint", container_checkpoint, "--dry-run",
                ],
            ),
        )
        for name, command in commands:
            completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
            result["checks"][name] = {
                "return_code": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
            passed = passed and completed.returncode == 0
        if passed:
            analyzed = subprocess.run(
                [
                    "./dev", "d4rl", "python", "scripts/experiments/analyze_run.py",
                    "--run-dir", container_archive, "--checkpoint", container_checkpoint,
                    "--reload-verified",
                ],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )
            result["checks"]["archive_analysis"] = {
                "return_code": analyzed.returncode,
                "stdout_tail": analyzed.stdout[-4000:],
                "stderr_tail": analyzed.stderr[-4000:],
            }
            passed = analyzed.returncode == 0
        result["checkpoint_sha256"] = _sha256(checkpoint)
    else:
        result["error"] = "final.pt missing"
    result.update(status="passed" if passed else "failed", finished_at=_now())
    output = validation_dir / "online_validation.json"
    _atomic_json(output, result)
    with _locked_state(state_path) as current:
        target = next(item for item in current["jobs"] if item["job_id"] == job_id)
        target["validation_status"] = result["status"]
        target["validation_path"] = str(output)
        target["checkpoint_sha256"] = result.get("checkpoint_sha256")
        if not passed:
            target["status"] = "blocked"
    if passed:
        _notification(
            state_path, job_id, "completed", f"[Lancet] COMPLETED - {job['method']} seed {job['seed']}",
            f"Training, final checkpoint reload, and archive analysis passed.\nmethod: {job['method']}\nseed: {job['seed']}\narchive: {archive}\nanalysis: {archive / 'analysis.md'}\n",
        )
    else:
        _notification(
            state_path, job_id, "blocked", f"[Lancet] BLOCKED - {job['method']} seed {job['seed']}",
            f"Training exited zero but completion validation failed.\nmethod: {job['method']}\nseed: {job['seed']}\nvalidation: {output}\narchive: {archive}\n",
        )
    return passed


def _prepare_online(state_path: Path, initializer_id: str) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    initializer = next(item for item in state["jobs"] if item["job_id"] == initializer_id)
    if initializer.get("online_pair_prepared"):
        return
    checkpoint = Path(initializer["checkpoint_dir"]) / "offline_final.pt"
    container_checkpoint = archive_run._container_path(checkpoint)
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip())
    prepared: list[dict] = []
    variants = (
        ("wsrl", "wsrl", "configs/off2on/wsrl_antmaze_medium_play_v2_online.yaml"),
        ("lancet", "lancet", "configs/off2on/lancet_antmaze_medium_play_v2.yaml"),
    )
    for algorithm, method, config in variants:
        command = [
            sys.executable, "scripts/experiments/archive_run.py",
            "--run-type", "formal", "--algorithm", algorithm, "--method-label", method,
            "--environment", "antmaze-medium-play-v2", "--seed", str(initializer["seed"]),
            "--config", config, "--protocol", str(FORMAL_PROTOCOL.relative_to(REPO)),
            "--dataset-path", str(DATASET), "--lineage-checkpoint", str(checkpoint),
            "--purpose", f"Run the formal {method} online branch from the seed-{initializer['seed']} shared initializer.",
            "--hypothesis", "The frozen online branch completes with finite state under the paired formal protocol.",
            "--baseline", "Paired WSRL versus Lancet main comparison using the identical seed-specific initializer.",
            "--success-criteria", "Nominal 500,000 online steps complete with finite training state.",
            "--success-criteria", "Periodic and final checkpoints are saved and reloadable.",
            "--failure-criteria", "Non-zero exit, NaN/Inf, OOM, environment failure, or corrupted checkpoint.",
            "--important-hyperparameter", "online_steps=500000,utd=4,batch_size=1024,warmup=5000",
            "--prepare-only",
        ]
        if dirty:
            command.append("--allow-dirty-formal")
        command.extend(["--", "--load_checkpoint", container_checkpoint])
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"online archive preparation failed for {method}: {completed.stderr[-4000:]}")
        match = re.search(r"^prepared: (.+)$", completed.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError(f"could not parse prepared archive for {method}")
        online_archive = Path(match.group(1))
        metadata_path = online_archive / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["formal_algorithm_commit"] = state["formal_algorithm_commit"]
        metadata["infrastructure_commit"] = state["infrastructure_commit"]
        _atomic_json(metadata_path, metadata)
        prepared.append(_job_from_archive(online_archive, status="queued", mode="managed"))
    with _locked_state(state_path) as current:
        target = next(item for item in current["jobs"] if item["job_id"] == initializer_id)
        if not target.get("online_pair_prepared"):
            current["jobs"].extend(prepared)
            target["online_pair_prepared"] = True
            target["online_job_ids"] = [item["job_id"] for item in prepared]
    for job in prepared:
        _notification(
            state_path, job["job_id"], "queued", f"[Lancet] QUEUED - {job['method']} seed {job['seed']}",
            f"method: {job['method']}\nseed: {job['seed']}\nrun type: formal\narchive: {job['archive']}\nshared initializer: {checkpoint}\n",
        )


def _refresh(state_path: Path) -> None:
    snapshot = _gpu_snapshot()
    by_gpu = {row["gpu_id"]: row for row in snapshot}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for job in state["jobs"]:
        if job["status"] not in {"running", "starting"}:
            continue
        archive = Path(job["archive"])
        progress = _progress(archive, job, by_gpu.get(job.get("gpu_id")))
        if job["mode"] == "attach":
            metadata = json.loads((archive / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("status") in {"finished", "failed", "stopped"}:
                status = "completed" if metadata["status"] == "finished" else "failed"
                with _locked_state(state_path) as current:
                    target = next(item for item in current["jobs"] if item["job_id"] == job["job_id"])
                    target.update(status=status, completed_at=metadata.get("end_time"), return_code=metadata.get("return_code"))
                if status == "failed":
                    _notification(
                        state_path, job["job_id"], "failed",
                        f"[Lancet] FAILED - {job['method']} seed {job['seed']}",
                        f"method: {job['method']}\nseed: {job['seed']}\nstatus: failed\narchive: {archive}\n",
                    )
            elif not _pid_alive(job.get("process_pid")):
                with _locked_state(state_path) as current:
                    target = next(item for item in current["jobs"] if item["job_id"] == job["job_id"])
                    target.setdefault("dead_process_observed_at", _now())
            elif any(progress["alerts"].values()):
                with _locked_state(state_path) as current:
                    target = next(item for item in current["jobs"] if item["job_id"] == job["job_id"])
                    target["last_alert"] = progress["alerts"]


def _postprocess(state_path: Path) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pending = [
        job for job in state["jobs"]
        if job["stage"] == "initializer" and job["status"] == "completed" and job["validation_status"] == "pending"
    ]
    for job in pending:
        if _validate_initializer(state_path, job["job_id"]):
            try:
                _prepare_online(state_path, job["job_id"])
            except Exception as exc:  # noqa: BLE001 - block rather than launch.
                with _locked_state(state_path) as current:
                    target = next(item for item in current["jobs"] if item["job_id"] == job["job_id"])
                    target["status"] = "blocked"
                    target["online_prepare_error"] = f"{type(exc).__name__}: {exc}"
                _notification(
                    state_path, job["job_id"], "blocked",
                    f"[Lancet] BLOCKED - online fork seed {job['seed']}",
                    f"Online pair preparation failed.\nseed: {job['seed']}\nerror: {type(exc).__name__}: {exc}\narchive: {job['archive']}\n",
                )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    online_pending = [
        job for job in state["jobs"]
        if job["stage"] == "online" and job["status"] == "completed" and job["validation_status"] == "pending"
    ]
    for job in online_pending:
        _validate_online(state_path, job["job_id"])


def _launch_queued(state_path: Path) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    queued = [job for job in state["jobs"] if job["status"] == "queued"]
    if not queued:
        return
    candidates, audit = _stable_candidates(state["settings"])
    with _locked_state(state_path) as current:
        current["last_gpu_audit"] = {"checked_at": _now(), "devices": audit}
    for job, gpu_id in zip(queued, candidates):
        with _locked_state(state_path) as current:
            target = next(item for item in current["jobs"] if item["job_id"] == job["job_id"])
            target.update(status="starting", gpu_id=gpu_id, assigned_at=_now())
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", "--state-file", str(state_path), "--job-id", job["job_id"], "--gpu-id", str(gpu_id)],
            cwd=REPO,
            stdout=(Path(job["archive"]) / "lifecycle.log").open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        with _locked_state(state_path) as current:
            target = next(item for item in current["jobs"] if item["job_id"] == job["job_id"])
            target["worker_pid"] = worker.pid


def _batch_notifications(state_path: Path) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    initializers = [
        job
        for job in state["jobs"]
        if job["stage"] == "initializer" and not job.get("superseded_by")
    ]
    if len(initializers) == 5 and all(job.get("validation_status") == "passed" for job in initializers):
        marker = "formal_initializers_complete"
        with _locked_state(state_path) as current:
            already = marker in current.setdefault("batch_notifications", {})
            if not already:
                current["batch_notifications"][marker] = {"status": "attempting", "attempted_at": _now()}
        if not already:
            lines = [f"seed {job['seed']}: {job.get('checkpoint_sha256')} {job['archive']}" for job in sorted(initializers, key=lambda item: item["seed"])]
            try:
                result = notify_email.send_notification("[Lancet] Formal Initializers Complete", "All five formal initializers validated.\n\n" + "\n".join(lines) + "\n")
            except Exception as exc:  # noqa: BLE001 - batch email is best effort.
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": _now()}
            with _locked_state(state_path) as current:
                current["batch_notifications"][marker] = result

    state = json.loads(state_path.read_text(encoding="utf-8"))
    main_jobs = [
        job for job in state["jobs"]
        if job["stage"] == "online"
        and job["method"] in {"wsrl", "lancet"}
        and not job.get("superseded_by")
    ]
    if len(main_jobs) == 10 and all(
        job["status"] == "completed" and job["validation_status"] == "passed"
        for job in main_jobs
    ):
        marker = "formal_main_comparison_complete"
        with _locked_state(state_path) as current:
            already = marker in current.setdefault("batch_notifications", {})
            if not already:
                current["batch_notifications"][marker] = {"status": "attempting", "attempted_at": _now()}
        if not already:
            pairs = []
            for seed in range(5):
                baseline = next(job for job in main_jobs if job["seed"] == seed and job["method"] == "wsrl")
                candidate = next(job for job in main_jobs if job["seed"] == seed and job["method"] == "lancet")
                output = Path(candidate["archive"]) / "metrics/paired_comparison.json"
                if not output.is_file():
                    subprocess.run(
                        [
                            "./dev", "d4rl", "python", "scripts/experiments/compare_runs.py",
                            "--baseline-run", archive_run._container_path(Path(baseline["archive"])),
                            "--candidate-run", archive_run._container_path(Path(candidate["archive"])),
                        ],
                        cwd=REPO,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                comparison = json.loads(output.read_text(encoding="utf-8"))
                primary = comparison.get("primary_adaptation_auc")
                if not primary:
                    raise RuntimeError(f"seed {seed} has no complete 0-50k primary AUC")
                pairs.append({"seed": seed, **primary})
            differences = [row["paired_difference"] for row in pairs]
            mean = statistics.mean(differences)
            sample_sd = statistics.stdev(differences)
            half_width = 2.7764451051977987 * sample_sd / len(differences) ** 0.5
            summary = {
                "metric": "Adaptation AUC 0-50k",
                "pairs": pairs,
                "paired_difference_mean": mean,
                "paired_difference_sample_sd": sample_sd,
                "paired_difference_95_t_ci": [mean - half_width, mean + half_width],
                "created_at": _now(),
            }
            summary_path = state_path.parent / "formal_main_summary.json"
            _atomic_json(summary_path, summary)
            body = (
                "All five paired WSRL/Lancet formal seeds completed and validated.\n\n"
                f"Adaptation AUC 0-50k paired difference mean: {mean:.6g}\n"
                f"sample SD: {sample_sd:.6g}\n"
                f"95% t-CI: [{mean - half_width:.6g}, {mean + half_width:.6g}]\n"
                f"analysis: {summary_path}\n"
            )
            try:
                result = notify_email.send_notification("[Lancet] Formal Main Comparison Complete", body)
            except Exception as exc:  # noqa: BLE001 - batch email is best effort.
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": _now()}
            with _locked_state(state_path) as current:
                current["batch_notifications"][marker] = result


def _controller(state_path: Path, *, once: bool) -> int:
    wake = False

    def child_finished(_signum, _frame):
        nonlocal wake
        wake = True

    signal.signal(signal.SIGCHLD, child_finished)
    while True:
        _refresh(state_path)
        _postprocess(state_path)
        _launch_queued(state_path)
        _batch_notifications(state_path)
        with _locked_state(state_path) as state:
            state["controller_heartbeat"] = _now()
            state["controller_pid"] = os.getpid()
        if once:
            return 0
        state = json.loads(state_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + state["settings"]["poll_seconds"]
        while time.monotonic() < deadline and not wake:
            time.sleep(min(60, deadline - time.monotonic()))
        wake = False


def _parse_attached(value: str) -> tuple[Path, int]:
    archive, pid = value.rsplit(":", 1)
    return Path(archive).resolve(), int(pid)


def _initialize(args) -> int:
    jobs = []
    for value in args.attach:
        archive, pid = _parse_attached(value)
        jobs.append(_job_from_archive(archive, status="running", mode="attach", pid=pid))
    for value in args.queue:
        archive = Path(value).resolve()
        job = _job_from_archive(archive, status="queued", mode="managed")
        jobs.append(job)
        metadata_path = archive / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("gpu_assignment_history", []).append(
            {"gpu_id": metadata.get("gpu_id"), "stopped_at": args.old_waiters_stopped_at, "reason": "obsolete fixed-GPU waiter replaced by dynamic queue"}
        )
        metadata.update(
            status="queued", gpu_id=None, gpu_lock_state="dynamic_queue", gpu_capacity_state="dynamic_queue",
            old_waiter_stopped_at=args.old_waiters_stopped_at,
        )
        _atomic_json(metadata_path, metadata)
    state = {
        "schema_version": 1,
        "created_at": _now(),
        "formal_algorithm_commit": args.formal_algorithm_commit,
        "infrastructure_commit": args.infrastructure_commit,
        "controller_pid": None,
        "controller_heartbeat": None,
        "settings": {
            "poll_seconds": args.poll_seconds,
            "required_free_mib": args.required_free_mib,
            "observed_initializer_mib": args.observed_initializer_mib,
            "safety_margin_mib": args.safety_margin_mib,
            "max_utilization_percent": args.max_utilization_percent,
            "max_foreign_memory_mib": args.max_foreign_memory_mib,
            "settle_samples": args.settle_samples,
            "settle_seconds": args.settle_seconds,
            "max_lancet_jobs_per_gpu": 1,
            "excluded_gpu_ids": sorted(set(args.exclude_gpu)),
        },
        "jobs": jobs,
        "batch_notifications": {},
        "last_gpu_audit": None,
    }
    _atomic_json(args.state_file, state)
    for job in jobs:
        if job["status"] == "queued":
            _notification(
                args.state_file, job["job_id"], "queued",
                f"[Lancet] QUEUED - {job['method']} seed {job['seed']}",
                f"method: {job['method']}\nseed: {job['seed']}\nrun type: formal\narchive: {job['archive']}\nresource gate: dynamic any-GPU\n",
            )
    print(args.state_file)
    return 0


def _add(args) -> int:
    archive = args.archive.resolve()
    status = "running" if args.attach_pid else "queued"
    mode = "attach" if args.attach_pid else "managed"
    job = _job_from_archive(archive, status=status, mode=mode, pid=args.attach_pid)
    with _locked_state(args.state_file) as state:
        if any(item["job_id"] == job["job_id"] for item in state["jobs"]):
            raise SystemExit(f"job already registered: {job['job_id']}")
        state["jobs"].append(job)
    if status == "queued":
        _notification(
            args.state_file, job["job_id"], "queued",
            f"[Lancet] QUEUED - {job['method']} seed {job['seed']}",
            f"method: {job['method']}\nseed: {job['seed']}\narchive: {job['archive']}\nresource gate: dynamic any-GPU\n",
        )
    print(job["job_id"])
    return 0


def _status(path: Path) -> int:
    state = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "controller_pid": state.get("controller_pid"),
        "controller_alive": _pid_alive(state.get("controller_pid")),
        "controller_heartbeat": state.get("controller_heartbeat"),
        "settings": state["settings"],
        "jobs": [
            {key: job.get(key) for key in ("job_id", "stage", "method", "seed", "status", "mode", "gpu_id", "process_pid", "worker_pid", "validation_status", "archive")}
            for job in state["jobs"]
        ],
    }, indent=2))
    return 0


def _exclude_gpu(path: Path, gpu_id: int, reason: str) -> int:
    if gpu_id < 0:
        raise SystemExit("gpu id must be non-negative")
    with _locked_state(path) as state:
        excluded = {int(item) for item in state["settings"].get("excluded_gpu_ids", [])}
        excluded.add(gpu_id)
        state["settings"]["excluded_gpu_ids"] = sorted(excluded)
        state.setdefault("policy_events", []).append(
            {
                "event": "gpu_excluded",
                "gpu_id": gpu_id,
                "reason": reason,
                "recorded_at": _now(),
            }
        )
    print(f"excluded GPU {gpu_id}: {reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    init.add_argument("--formal-algorithm-commit", required=True)
    init.add_argument("--infrastructure-commit", required=True)
    init.add_argument("--attach", action="append", default=[], metavar="ARCHIVE:PID")
    init.add_argument("--queue", action="append", default=[], metavar="ARCHIVE")
    init.add_argument("--old-waiters-stopped-at", required=True)
    init.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    init.add_argument("--observed-initializer-mib", type=int, default=5_564)
    init.add_argument("--safety-margin-mib", type=int, default=8_192)
    init.add_argument("--required-free-mib", type=int, default=DEFAULT_REQUIRED_FREE_MIB)
    init.add_argument("--max-utilization-percent", type=int, default=DEFAULT_MAX_UTIL)
    init.add_argument("--max-foreign-memory-mib", type=int, default=DEFAULT_MAX_FOREIGN_MIB)
    init.add_argument("--settle-samples", type=int, default=DEFAULT_SETTLE_SAMPLES)
    init.add_argument("--settle-seconds", type=int, default=DEFAULT_SETTLE_SECONDS)
    init.add_argument(
        "--exclude-gpu",
        action="append",
        default=list(DEFAULT_EXCLUDED_GPU_IDS),
        type=int,
        help="GPU id that the dynamic scheduler must never select (repeatable).",
    )
    run = sub.add_parser("run")
    run.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    run.add_argument("--once", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("--state-file", required=True, type=Path)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--gpu-id", required=True, type=int)
    status = sub.add_parser("status")
    status.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    exclude_gpu = sub.add_parser("exclude-gpu")
    exclude_gpu.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    exclude_gpu.add_argument("--gpu-id", type=int, required=True)
    exclude_gpu.add_argument("--reason", required=True)
    add = sub.add_parser("add")
    add.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    add.add_argument("--archive", type=Path, required=True)
    add.add_argument("--attach-pid", type=int)
    args = parser.parse_args()
    if args.command == "init":
        return _initialize(args)
    if args.command == "run":
        return _controller(args.state_file, once=args.once)
    if args.command == "worker":
        return _worker(args.state_file, args.job_id, args.gpu_id)
    if args.command == "add":
        return _add(args)
    if args.command == "exclude-gpu":
        return _exclude_gpu(args.state_file, args.gpu_id, args.reason)
    return _status(args.state_file)


if __name__ == "__main__":
    raise SystemExit(main())
