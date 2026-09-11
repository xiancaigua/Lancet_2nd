#!/usr/bin/env python3
"""Validate a rl-garden checkpoint without constructing an environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(value, prefix: str = "root"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")


def validate(path: Path, *, expected_updates: int | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state" not in checkpoint:
        raise ValueError("not an rl-garden state checkpoint")
    state = checkpoint["state"]
    tensors = list(_walk(state))
    nonfinite = [name for name, value in tensors if not torch.isfinite(value).all()]
    metadata = checkpoint.get("metadata", {})
    update = int(metadata.get("global_update", state.get("global_update", -1)))
    if expected_updates is not None and update != expected_updates:
        raise ValueError(f"global_update={update}, expected {expected_updates}")
    required_optimizers = {
        "q_optimizer",
        "actor_optimizer",
        "alpha_optimizer",
        "cql_alpha_optimizer",
    }
    optimizers = set(state.get("optimizers", {}))
    missing = sorted(required_optimizers - optimizers)
    if missing:
        raise ValueError("missing optimizer state: " + ", ".join(missing))
    if nonfinite:
        raise ValueError("non-finite tensors: " + ", ".join(nonfinite[:10]))
    return {
        "checkpoint": str(path),
        "sha256": _sha256(path),
        "algorithm_class": metadata.get("algorithm_class"),
        "global_step": metadata.get("global_step", state.get("global_step")),
        "global_update": update,
        "tensor_count": len(tensors),
        "all_tensors_finite": True,
        "optimizer_names": sorted(optimizers),
        "optimizer_state_finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.checkpoint, expected_updates=args.expected_updates)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
