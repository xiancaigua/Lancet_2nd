#!/usr/bin/env python3
"""Fingerprint raw and transformed AntMaze dataset semantics for migration."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rl_garden.buffers.d4rl_legacy_dataset import _official_calql_antmaze_dataset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    numeric = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
        "min": float(numeric.min()) if numeric.size else None,
        "max": float(numeric.max()) if numeric.size else None,
        "mean": float(numeric.mean()) if numeric.size else None,
        "first_values": numeric.reshape(-1)[:8].tolist(),
    }


class RawDatasetEnv:
    def __init__(self, raw: dict[str, np.ndarray]) -> None:
        self.raw = raw
        self._max_episode_steps = 1000

    def get_dataset(self) -> dict[str, np.ndarray]:
        return self.raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--reward-bias", type=float, default=-5.0)
    parser.add_argument("--clip-action", type=float, default=0.99999)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as handle:
        raw = {
            key: np.asarray(handle[key])
            for key in ("observations", "actions", "rewards", "terminals", "timeouts")
        }
    transformed = _official_calql_antmaze_dataset(
        RawDatasetEnv(raw),
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        clip_action=args.clip_action,
        gamma=args.gamma,
    )
    n = min(args.count, len(transformed["rewards"]))
    semantic_arrays = {
        "obs/state": transformed["observations"][:n],
        "action": transformed["actions"][:n],
        "reward": transformed["rewards"][:n],
        "next_obs/state": transformed["next_observations"][:n],
        "termination": transformed["terminals"][:n],
        "episode_boundary": transformed["episode_end"][:n],
        "truncation": np.logical_and(
            transformed["episode_end"][:n] > 0,
            transformed["terminals"][:n] == 0,
        ).astype(np.float32),
        "mc_return": transformed["mc_returns"][:n],
        "success": (transformed["rewards"][:n] > args.reward_bias).astype(np.float32),
    }
    records = {key: array_record(value) for key, value in semantic_arrays.items()}
    semantic_digest = hashlib.sha256()
    for key in sorted(records):
        semantic_digest.update(key.encode())
        semantic_digest.update(records[key]["sha256"].encode())
    raw_prefix = {
        key: array_record(value[: min(args.count, len(value))])
        for key, value in raw.items()
    }
    payload = {
        "schema_version": 1,
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(args.dataset),
        "dataset_size_bytes": args.dataset.stat().st_size,
        "raw_dataset_sha256": file_sha256(args.dataset),
        "raw_transition_count": int(len(raw["rewards"])),
        "transformed_transition_count": int(len(transformed["rewards"])),
        "prefix_count": n,
        "transform": {
            "implementation": "_official_calql_antmaze_dataset",
            "reward_scale": args.reward_scale,
            "reward_bias": args.reward_bias,
            "clip_action": args.clip_action,
            "gamma": args.gamma,
            "timeout_rule": "drop timeout row; episode_boundary on last retained row",
        },
        "raw_prefix": raw_prefix,
        "semantic_prefix": records,
        "semantic_sha256": semantic_digest.hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
