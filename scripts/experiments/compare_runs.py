#!/usr/bin/env python3
"""Create an evidence-based paired comparison from two analyzed run archives."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any


def _load(path: Path, name: str) -> dict[str, Any]:
    target = path / "metrics" / name
    if not target.is_file():
        raise FileNotFoundError(f"Run must be analyzed first: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _curve(scalars: dict[str, Any], metric: str) -> dict[int, float]:
    if metric not in scalars:
        raise KeyError(f"Metric {metric!r} was not collected")
    return {int(event["step"]): float(event["value"]) for event in scalars[metric]}


def _interpolate(curve: dict[int, float], step: int) -> float:
    if step in curve:
        return curve[step]
    lower = max(candidate for candidate in curve if candidate < step)
    upper = min(candidate for candidate in curve if candidate > step)
    fraction = (step - lower) / (upper - lower)
    return curve[lower] + fraction * (curve[upper] - curve[lower])


def _trapezoid_mean(points: list[tuple[int, float]]) -> float:
    integral = sum(
        (right_step - left_step) * (left_value + right_value) / 2
        for (left_step, left_value), (right_step, right_value) in pairwise(points)
    )
    return integral / (points[-1][0] - points[0][0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--metric", default="eval/normalized_score")
    parser.add_argument("--formal-primary-horizon", type=int, default=50_000)
    args = parser.parse_args()

    baseline_scalars = _load(args.baseline_run, "scalars.json")
    candidate_scalars = _load(args.candidate_run, "scalars.json")
    candidate_summary = _load(args.candidate_run, "summary.json")
    baseline_summary = _load(args.baseline_run, "summary.json")
    baseline_curve = _curve(baseline_scalars, args.metric)
    candidate_curve = _curve(candidate_scalars, args.metric)
    common_steps = sorted(set(baseline_curve) & set(candidate_curve))
    if len(common_steps) < 2:
        raise ValueError("Paired comparison requires at least two common evaluations")

    adaptation_start = candidate_summary.get("adaptation_start_global_step")
    if adaptation_start is None:
        raise ValueError("Candidate summary has no adaptation_start_global_step")
    adaptation_start = int(adaptation_start)
    online_start = candidate_summary.get("online_start_global_step")
    if online_start is None:
        raise ValueError("Candidate summary has no online_start_global_step")
    online_start = int(online_start)
    if baseline_summary.get("online_start_global_step") != online_start:
        raise ValueError("Paired runs have different online-start coordinates")
    final_step = common_steps[-1]
    if not common_steps[0] <= adaptation_start < final_step:
        raise ValueError("Adaptation start is outside the common evaluation range")

    analysis_steps = [adaptation_start]
    analysis_steps.extend(step for step in common_steps if adaptation_start < step)
    rows = [
        {
            "global_step": step,
            "online_step": step - online_start,
            "adaptation_step": step - adaptation_start,
            "baseline": _interpolate(baseline_curve, step),
            "candidate": _interpolate(candidate_curve, step),
        }
        for step in analysis_steps
    ]
    baseline_points = [(row["global_step"], row["baseline"]) for row in rows]
    candidate_points = [(row["global_step"], row["candidate"]) for row in rows]
    observed_horizon = final_step - adaptation_start
    result = {
        "metric": args.metric,
        "baseline_run": str(args.baseline_run.resolve()),
        "candidate_run": str(args.candidate_run.resolve()),
        "online_start_global_step": online_start,
        "adaptation_start_global_step": adaptation_start,
        "common_final_global_step": final_step,
        "common_final_online_step": final_step - online_start,
        "observed_adaptation_horizon": observed_horizon,
        "formal_primary_horizon": args.formal_primary_horizon,
        "formal_primary_complete": observed_horizon >= args.formal_primary_horizon,
        "observed_auc_mean": {
            "baseline": _trapezoid_mean(baseline_points),
            "candidate": _trapezoid_mean(candidate_points),
        },
        "final": {
            "baseline": baseline_points[-1][1],
            "candidate": candidate_points[-1][1],
        },
        "curve": rows,
    }
    result["observed_auc_mean"]["paired_difference"] = (
        result["observed_auc_mean"]["candidate"]
        - result["observed_auc_mean"]["baseline"]
    )

    output_dir = args.candidate_run / "metrics"
    json_path = output_dir / "paired_comparison.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = "\n".join(
        f"| {row['global_step']} | {row['online_step']} | "
        f"{row['adaptation_step']} | "
        f"{row['baseline']:.6g} | {row['candidate']:.6g} |"
        for row in rows
    )
    markdown = (
        f"""# Paired Debug Comparison

This is engineering evidence, not a paper result.

- Metric: `{args.metric}`
- Online start global step: `{online_start}`
- Adaptation start global step: `{adaptation_start}`
- Common final global / online step: `{final_step}` / `{final_step - online_start}`
- Observed adaptation horizon: `{observed_horizon}`
- Formal 0–{args.formal_primary_horizon} AUC complete: `{result["formal_primary_complete"]}`
- Observed mean AUC (baseline / candidate / difference): """
        f"`{result['observed_auc_mean']['baseline']:.6g}` / "
        f"`{result['observed_auc_mean']['candidate']:.6g}` / "
        f"`{result['observed_auc_mean']['paired_difference']:.6g}`\n\n"
        "| Global step | Online step | Adaptation step | WSRL | Lancet |\n"
        "|---:|---:|---:|---:|---:|\n" + table + "\n"
    )
    (output_dir / "paired_comparison.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
