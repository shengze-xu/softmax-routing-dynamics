#!/usr/bin/env python3
"""Command-line adapter for the recorded no-drop recovery study."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
from typing import Any

from experiments.models import agnews_no_drop_recovery as core


PRIMARY_PROTOCOL_ID = "agnews_no_drop_clamp_release_v1"
PRIMARY_SEEDS = set(range(4501, 4509))
PRIMARY_WRAPPER_VERSION = 1


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol_id != PRIMARY_PROTOCOL_ID:
        raise ValueError("Unexpected no-drop recovery protocol")
    if args.role != "primary":
        raise ValueError("Recorded adapter requires role=primary")
    if args.seed not in PRIMARY_SEEDS:
        raise ValueError(f"Seed is outside the recorded inventory: {args.seed}")

    # The reusable core excludes the recorded seed set by construction. The
    # protocol validator and this adapter replace only that launch guard; the
    # numerical implementation and configuration remain unchanged.
    frozen_guard = core.PRIMARY_SEEDS
    try:
        core.PRIMARY_SEEDS = set()
        result = core.run(args)
    finally:
        core.PRIMARY_SEEDS = frozen_guard

    result.update(
        {
            "study": "no_drop_recovery_independent_recovery_primary",
            "role": "primary",
            "primary_wrapper_version": PRIMARY_WRAPPER_VERSION,
            "no_drop_release_confirmatory": True,
            "primary_protocol_sha256": args.protocol_sha256,
        }
    )
    destination = args.output_dir.resolve() / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": args.seed,
                "no_drop_release_confirmatory": True,
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--seed", type=int, required=True)
    configuration = parser.add_mutually_exclusive_group(required=True)
    configuration.add_argument("--config", type=Path)
    configuration.add_argument("--config-json")
    parser.add_argument("--device", default="cuda")
    values = parser.parse_args()
    values.config = json.loads(values.config.read_text() if values.config else values.config_json)
    run(values)


if __name__ == "__main__":
    main()
