"""Unit checks for fixed-coordinate Lancet experiment analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/experiments/compare_runs.py"
SPEC = importlib.util.spec_from_file_location("lancet_compare_runs", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def test_window_auc_uses_exact_horizon_not_later_observations():
    curve = {0: 0.0, 25: 1.0, 50: 0.0, 100: 100.0}
    assert comparison._window_auc_mean(curve, start=0, horizon=50) == pytest.approx(
        0.5
    )


def test_window_auc_interpolates_only_fixed_boundaries():
    curve = {0: 0.0, 20: 2.0, 40: 4.0}
    assert comparison._window_auc_mean(curve, start=10, horizon=20) == pytest.approx(
        2.0
    )


def test_window_auc_rejects_incomplete_horizon():
    with pytest.raises(ValueError, match="outside the observed curve"):
        comparison._window_auc_mean({0: 0.0, 10: 1.0}, start=0, horizon=50)
