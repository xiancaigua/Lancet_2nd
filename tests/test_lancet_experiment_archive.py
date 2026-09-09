"""Unit checks for the run-type path guards in the Host archive launcher."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/experiments/archive_run.py"
SPEC = importlib.util.spec_from_file_location("lancet_archive_run", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(archive)


def _args(run_type: str, *, allow_dirty_formal: bool = False):
    return argparse.Namespace(run_type=run_type, allow_dirty_formal=allow_dirty_formal)


def test_smoke_rejects_non_smoke_config():
    with pytest.raises(SystemExit, match="filename contains 'smoke'"):
        archive._validate_intent(_args("smoke"), Path("paper.yaml"), False)


def test_formal_rejects_smoke_config_and_dirty_tree():
    with pytest.raises(SystemExit, match="cannot use a smoke config"):
        archive._validate_intent(_args("formal"), Path("x_smoke.yaml"), False)
    with pytest.raises(SystemExit, match="clean working tree"):
        archive._validate_intent(_args("formal"), Path("paper.yaml"), True)


def test_archive_paths_map_only_below_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "HOST_DATA", tmp_path)
    assert (
        archive._container_path(tmp_path / "runs/smoke/x")
        == "/data/lancet/runs/smoke/x"
    )
    with pytest.raises(SystemExit, match="escaped data root"):
        archive._container_path(tmp_path.parent / "outside")
