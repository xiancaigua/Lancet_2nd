#!/usr/bin/env python3
"""Compare pre/post-resync AntMaze semantic fingerprint evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    old = json.loads((ROOT / "old_dataset_fingerprint.json").read_text())
    new = json.loads((ROOT / "new_dataset_fingerprint.json").read_text())
    fields = {
        key: {
            "old": old["semantic_prefix"][key]["sha256"],
            "new": new["semantic_prefix"][key]["sha256"],
            "match": (
                old["semantic_prefix"][key]["sha256"]
                == new["semantic_prefix"][key]["sha256"]
            ),
        }
        for key in sorted(old["semantic_prefix"])
    }
    payload = {
        "schema_version": 1,
        "raw_dataset_sha256": old["raw_dataset_sha256"],
        "raw_dataset_match": old["raw_dataset_sha256"] == new["raw_dataset_sha256"],
        "raw_transition_count": {
            "old": old["raw_transition_count"],
            "new": new["raw_transition_count"],
        },
        "transformed_transition_count": {
            "old": old["transformed_transition_count"],
            "new": new["transformed_transition_count"],
        },
        "semantic_sha256": {
            "old": old["semantic_sha256"],
            "new": new["semantic_sha256"],
        },
        "fields": fields,
    }
    payload["status"] = (
        "PASS"
        if payload["raw_dataset_match"]
        and payload["raw_transition_count"]["old"]
        == payload["raw_transition_count"]["new"]
        and payload["transformed_transition_count"]["old"]
        == payload["transformed_transition_count"]["new"]
        and payload["semantic_sha256"]["old"] == payload["semantic_sha256"]["new"]
        and all(record["match"] for record in fields.values())
        else "FAIL"
    )
    (ROOT / "dataset_semantic_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(payload["status"])
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
