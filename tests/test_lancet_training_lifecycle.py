"""Tests for the external Lancet lifecycle controller."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts/experiments"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("archive_run", "archive_run.py")
_load("notify_email", "notify_email.py")
lifecycle = _load("lancet_run_training", "run_training.py")


def _gpu(gpu_id: int, *, free: int, util: int, processes=None):
    return {
        "gpu_id": gpu_id,
        "memory_free_mib": free,
        "memory_used_mib": 49_140 - free,
        "utilization_percent": util,
        "processes": processes or [],
    }


def test_replace_gpu_preserves_argument_boundaries():
    command = "./dev d4rl env CUDA_VISIBLE_DEVICES=2 python -c 'print(\"a b\")'"
    parts = lifecycle._replace_gpu(command, 5)
    assert "CUDA_VISIBLE_DEVICES=5" in parts
    assert "CUDA_VISIBLE_DEVICES=2" not in parts
    assert parts[-1] == 'print("a b")'


def test_dynamic_gate_requires_stable_capacity_and_rejects_foreign_load(monkeypatch):
    heavy = [{"memory_mib": 3_000, "lancet": False}]
    samples = [
        [_gpu(0, free=20_000, util=5), _gpu(1, free=30_000, util=1, processes=heavy)],
        [_gpu(0, free=19_000, util=7), _gpu(1, free=30_000, util=1, processes=heavy)],
    ]
    monkeypatch.setattr(lifecycle, "_gpu_snapshot", lambda: samples.pop(0))
    monkeypatch.setattr(lifecycle, "_gpu_lock_available", lambda _gpu_id: True)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    settings = {
        "settle_samples": 2,
        "settle_seconds": 1,
        "required_free_mib": 13_756,
        "max_utilization_percent": 20,
        "max_foreign_memory_mib": 2_048,
    }
    candidates, audit = lifecycle._stable_candidates(settings)
    assert candidates == [0]
    assert audit[0]["free_mib_min"] == 19_000
    assert "foreign_memory" in audit[1]["reasons"]


def test_dynamic_gate_honors_policy_exclusion(monkeypatch):
    samples = [[_gpu(0, free=40_000, util=0), _gpu(1, free=40_000, util=0)]]
    monkeypatch.setattr(lifecycle, "_gpu_snapshot", lambda: samples.pop(0))
    monkeypatch.setattr(lifecycle, "_gpu_lock_available", lambda _gpu_id: True)
    settings = {
        "settle_samples": 1,
        "settle_seconds": 0,
        "required_free_mib": 13_756,
        "max_utilization_percent": 20,
        "max_foreign_memory_mib": 2_048,
        "excluded_gpu_ids": [1],
    }

    candidates, audit = lifecycle._stable_candidates(settings)

    assert candidates == [0]
    assert audit[1]["usable"] is False
    assert "excluded_by_policy" in audit[1]["reasons"]


def test_progress_reads_only_log_tail_and_reports_rate(tmp_path):
    archive = tmp_path / "run"
    archive.mkdir()
    (archive / "train.log").write_text(
        "offline:  25%|xx| 250000/1000000 [01:00:00<03:00:00, 69it/s]\r",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    job = {
        "job_id": "initializer-seed-0",
        "status": "running",
        "started_at": "2026-09-11T01:00:00+08:00",
        "checkpoint_dir": str(checkpoint),
        "gpu_id": 2,
    }
    result = lifecycle._progress(archive, job, _gpu(2, free=40_000, util=90))
    assert result["last_step"] == 250_000
    assert result["target_step"] == 1_000_000
    assert result["alerts"] == {"traceback": False, "oom": False, "nan_or_inf": False}
    assert json.loads((archive / "progress.json").read_text())["last_step"] == 250_000


def test_notification_transition_is_idempotent(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    lifecycle._atomic_json(
        state_path,
        {"jobs": [{"job_id": "x", "notifications": {}}]},
    )
    calls = []
    monkeypatch.setattr(
        lifecycle.notify_email,
        "send_notification",
        lambda subject, body: calls.append((subject, body)) or {"status": "sent"},
    )
    lifecycle._notification(state_path, "x", "queued", "subject", "body")
    lifecycle._notification(state_path, "x", "queued", "subject", "body")
    assert calls == [("subject", "body")]
    state = json.loads(state_path.read_text())
    assert state["jobs"][0]["notifications"]["queued"]["status"] == "sent"


def test_exclude_gpu_persists_policy_and_audit_event(tmp_path):
    state_path = tmp_path / "state.json"
    lifecycle._atomic_json(state_path, {"settings": {}, "jobs": []})

    lifecycle._exclude_gpu(state_path, 4, "reserved by another user")
    lifecycle._exclude_gpu(state_path, 4, "reserved by another user")

    state = json.loads(state_path.read_text())
    assert state["settings"]["excluded_gpu_ids"] == [4]
    assert state["policy_events"][-1]["gpu_id"] == 4
