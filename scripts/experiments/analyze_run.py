#!/usr/bin/env python3
"""Validate one archived WSRL/Lancet run from checkpoint and TensorBoard evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensors(child)


def _all_finite(value: Any) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in _tensors(value))


def _scalars(run_dir: Path) -> dict[str, list[dict[str, float]]]:
    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    merged: dict[str, list[dict[str, float]]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            merged.setdefault(tag, []).extend(
                {
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                }
                for event in accumulator.Scalars(tag)
            )
    for events in merged.values():
        events.sort(key=lambda event: (event["step"], event["wall_time"]))
    return merged


def _last_values(scalars: dict[str, list[dict[str, float]]]) -> dict[str, float]:
    return {tag: events[-1]["value"] for tag, events in scalars.items() if events}


def _range_summary(
    scalars: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for tag, events in scalars.items():
        values = [event["value"] for event in events]
        if values:
            summary[tag] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "last": values[-1],
                "finite": all(math.isfinite(value) for value in values),
            }
    return summary


def _format_metrics(values: dict[str, float], prefixes: tuple[str, ...]) -> str:
    selected = [
        f"- `{key}`: {value:.8g}"
        for key, value in sorted(values.items())
        if key.startswith(prefixes)
    ]
    return "\n".join(selected) if selected else "- not collected"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--reload-verified",
        action="store_true",
        help="Record that a separate agent-construction checkpoint reload passed.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state"]
    training_state = state.get("training_state", {})
    extra = state.get("extra", {})
    lancet_state = extra.get("lancet")
    scalars = _scalars(run_dir)
    scalar_ranges = _range_summary(scalars)
    scalar_finite = all(item["finite"] for item in scalar_ranges.values())

    summary: dict[str, Any] = {
        "algorithm_class": checkpoint["metadata"].get("algorithm_class"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(state.get("global_step", -1)),
        "checkpoint_global_update": int(state.get("global_update", -1)),
        "online_start_step": extra.get("online_start_step"),
        "policy_finite": _all_finite(state.get("policy", {})),
        "optimizer_state_finite": _all_finite(state.get("optimizers", {})),
        "optimizer_names": sorted(state.get("optimizers", {})),
        "tensorboard_event_files": [
            str(path) for path in sorted(run_dir.rglob("events.out.tfevents.*"))
        ],
        "scalar_ranges": scalar_ranges,
        "scalar_values_finite": scalar_finite,
        "last_scalars": _last_values(scalars),
    }
    if lancet_state is not None:
        residual_state = lancet_state.get("residual_network", {})
        residual_tensors = list(_tensors(residual_state))
        summary.update(
            lancet_schema_version=lancet_state.get("schema_version"),
            lancet_variant=lancet_state.get("variant"),
            residual_parameters_finite=_all_finite(residual_state),
            residual_parameter_abs_sum=sum(
                float(tensor.abs().sum().item()) for tensor in residual_tensors
            ),
            residual_update_count=int(
                training_state.get("lancet_residual_update_count", 0)
            ),
            adaptation_start_step=training_state.get("lancet_adaptation_start_step"),
            u_ema=float(training_state.get("lancet_u_ema", 0.0)),
            u_ema_initialized=bool(
                training_state.get("lancet_u_ema_initialized", False)
            ),
        )

    required_finite = [
        summary["policy_finite"],
        summary["optimizer_state_finite"],
        summary["scalar_values_finite"],
    ]
    if lancet_state is not None:
        required_finite.append(summary["residual_parameters_finite"])
    summary["finite_gate"] = all(required_finite)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (metrics_dir / "scalars.json").write_text(
        json.dumps(scalars, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["agent_reload_verified"] = args.reload_verified
    summary["scalars_path"] = str(metrics_dir / "scalars.json")
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["validation_summary_path"] = str(metrics_dir / "summary.json")
    metadata["checkpoint_validation"] = {
        "path": str(checkpoint_path),
        "finite_gate": summary["finite_gate"],
        "policy_finite": summary["policy_finite"],
        "optimizer_state_finite": summary["optimizer_state_finite"],
        "scalar_values_finite": summary["scalar_values_finite"],
        "agent_reload_verified": args.reload_verified,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    last_scalars = summary["last_scalars"]
    lancet_lines = "- not applicable (WSRL)"
    if lancet_state is not None:
        lancet_lines = "\n".join(
            [
                f"- variant: `{summary['lancet_variant']}`",
                f"- residual updates: {summary['residual_update_count']}",
                f"- residual parameter |sum|: {summary['residual_parameter_abs_sum']:.8g}",
                f"- residual parameters finite: {summary['residual_parameters_finite']}",
                f"- U EMA initialized/value: {summary['u_ema_initialized']} / {summary['u_ema']:.8g}",
                f"- adaptation start step: {summary['adaptation_start_step']}",
            ]
        )

    analysis = f"""# Experiment Analysis

## Status

`{metadata["status"]}` with return code `{metadata["return_code"]}`. Evidence
below was read from the archived checkpoint and TensorBoard events; absent
metrics are reported as not collected.

## Main Metrics

{_format_metrics(last_scalars, ("eval/",))}

## Training Behavior

- checkpoint global step/update: {summary["checkpoint_global_step"]} / {summary["checkpoint_global_update"]}
- policy tensors finite: {summary["policy_finite"]}
- optimizer tensors finite: {summary["optimizer_state_finite"]}
- all collected scalar values finite: {summary["scalar_values_finite"]}
- overall finite gate: {summary["finite_gate"]}
- agent-construction checkpoint reload verified: {summary["agent_reload_verified"]}

Selected final training scalars:

{_format_metrics(last_scalars, ("losses/", "entropy/", "lancet/", "q/"))}

## Lancet Evidence

{lancet_lines}

## Comparison

No performance conclusion is drawn from this smoke/debug archive. Paired-run
comparison, when applicable, must use the separately archived peer run.

## Problems / Anomalies

- See `train.log` for legacy D4RL optional-backend warnings.
- Metrics absent from `metrics/summary.json` were not collected.

## Interpretation

This archive {"passes" if summary["finite_gate"] else "fails"} the structural
finite-state checkpoint/scalar gate. It does not establish sample-efficiency
or final-return superiority.

## Limitations

Checkpoint deserialization and structural state were verified here.
{"A separate agent-construction dry-run also passed." if summary["agent_reload_verified"] else "A separate agent-construction dry-run has not been recorded."}

## Next Recommended Step

{"Proceed to the next declared engineering gate." if summary["finite_gate"] else "Diagnose non-finite evidence before any further experiment."}
"""
    (run_dir / "analysis.md").write_text(analysis, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["finite_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
