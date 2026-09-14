#!/usr/bin/env python3
"""Resolve and compare generation-2 WSRL/Lancet online configurations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CONFIGS = {
    "wsrl": ("wsrl", "configs/off2on/wsrl_antmaze_medium_play_v2_online.yaml"),
    "raw": ("lancet", "configs/off2on/lancet_raw_antmaze_medium_play_v2.yaml"),
    "centered": (
        "lancet",
        "configs/off2on/lancet_centered_antmaze_medium_play_v2.yaml",
    ),
    "lancet": ("lancet", "configs/off2on/lancet_antmaze_medium_play_v2.yaml"),
}
LANCET_ONLY = {
    "handoff_window_steps",
    "lancet_variant",
    "local_action_count",
    "local_action_noise_scale",
    "local_action_seed_offset",
    "residual_fit_coef",
    "residual_hidden_dim",
    "residual_hidden_layers",
    "residual_init_seed_offset",
    "residual_lr",
    "residual_small_coef",
    "uncertainty_beta",
    "uncertainty_ema_decay",
    "uncertainty_eps",
    "uncertainty_weight_max",
}


def resolved(algorithm: str, config: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            algorithm,
            "--config",
            config,
            "--print-config",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    resolved_dir = OUT / "resolved_configs"
    resolved_dir.mkdir(exist_ok=True)
    documents: dict[str, dict[str, Any]] = {}
    for method, (algorithm, config) in CONFIGS.items():
        document = resolved(algorithm, config)
        documents[method] = document
        (resolved_dir / f"{method}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    baseline = documents["wsrl"]["inputs"]
    base_keys = sorted(set(baseline) - LANCET_ONLY)
    mismatches: list[dict[str, Any]] = []
    for method in ("raw", "centered", "lancet"):
        candidate = documents[method]["inputs"]
        for key in base_keys:
            if candidate.get(key) != baseline.get(key):
                mismatches.append(
                    {
                        "method": method,
                        "parameter": key,
                        "wsrl": baseline.get(key),
                        "candidate": candidate.get(key),
                    }
                )

    report = {
        "schema_version": 1,
        "comparison": "generation-2 online base configuration parity",
        "reference": CONFIGS["wsrl"][1],
        "configs": {method: config for method, (_, config) in CONFIGS.items()},
        "lancet_only_parameters": sorted(LANCET_ONLY),
        "base_parameter_count": len(base_keys),
        "base_parameters": {key: baseline[key] for key in base_keys},
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    (OUT / "base_config_parity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Generation-2 base configuration parity",
        "",
        f"Status: **{report['status']}**",
        "",
        "Resolved WSRL, Raw, Centered, and Lancet online configurations were",
        "compared after excluding only the declared Lancet residual parameters.",
        f"Compared base parameters: {len(base_keys)}.",
        "",
    ]
    if mismatches:
        lines.extend(
            [
                "| Method | Parameter | WSRL | Candidate |",
                "|---|---|---|---|",
                *[
                    f"| {item['method']} | `{item['parameter']}` | `{item['wsrl']}` | `{item['candidate']}` |"
                    for item in mismatches
                ],
            ]
        )
    else:
        lines.append("All base inputs are resolved-identical.")
    (OUT / "base_config_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report["status"])
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
