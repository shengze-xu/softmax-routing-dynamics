#!/usr/bin/env python3
"""Fail-closed integrity audit for router-update mechanism-validation runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.studies.router_update_no_drop import BRANCHES
from experiments.common.router_logit_decomposition import PROBE_FIELDS


REQUIRED_LOG_FIELDS = {
    "protocol_id",
    "seed",
    "cohort",
    "capacity_factor",
    "branch",
    "optimizer_step",
    "layer_zero_based",
    "target_expert",
    "comparator_expert",
    "initial_target_mass",
    "raw_target_logit_pre_update",
    "raw_comparator_logit_pre_update",
    "target_logit_gradient",
    "comparator_logit_gradient",
    "gradient_descent_logodds_proxy",
    "realized_target_logit_update",
    "realized_comparator_logit_update",
    "realized_logodds_update_advantage",
    "cumulative_realized_logodds_update_advantage",
    "heldout_pre_topk_target_probability",
    "heldout_target_traffic",
    "total_assignments",
    "kept_assignments",
    "dropped_assignments",
    "persistent_recovery_indicator",
    "model_state_sha256",
    "optimizer_state_sha256",
    "future_batch_plan_sha256",
}


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_probe_sidecar(
    branch: dict[str, Any],
    *,
    result_root: Path | None,
    expected_rows: int,
) -> tuple[list[str], dict[str, float | int]]:
    failures: list[str] = []
    summary = branch.get("probe_decomposition")
    metrics: dict[str, float | int] = {
        "rows": 0,
        "maximum_logit_reconstruction_error": 0.0,
        "maximum_probability_reconstruction_error": 0.0,
        "threshold_crossing_mismatches": 0,
    }
    if not isinstance(summary, dict):
        return ["missing probe decomposition"], metrics
    if int(summary.get("rows", -1)) != expected_rows:
        failures.append("probe row inventory")
    if int(summary.get("threshold_crossing_mismatches", -1)) != 0:
        failures.append("probe threshold-crossing reconstruction")
    if result_root is None:
        return failures, metrics
    raw_name = summary.get("path")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        return failures + ["probe sidecar path"], metrics
    path = result_root / raw_name
    if not path.is_file():
        return failures + ["missing probe sidecar"], metrics
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PROBE_FIELDS:
            failures.append("probe sidecar schema")
        for row in reader:
            metrics["rows"] = int(metrics["rows"]) + 1
            logit_error = abs(float(row["logit_reconstruction_max_abs_error"]))
            probability_error = abs(
                float(row["probability_reconstruction_abs_error"])
            )
            metrics["maximum_logit_reconstruction_error"] = max(
                float(metrics["maximum_logit_reconstruction_error"]),
                logit_error,
            )
            metrics["maximum_probability_reconstruction_error"] = max(
                float(metrics["maximum_probability_reconstruction_error"]),
                probability_error,
            )
            metrics["threshold_crossing_mismatches"] = int(
                metrics["threshold_crossing_mismatches"]
            ) + (
                int(row["crossed_threshold_observed"])
                != int(row["crossed_threshold_reconstructed"])
            )
    if int(metrics["rows"]) != expected_rows:
        failures.append("probe sidecar row inventory")
    if float(metrics["maximum_logit_reconstruction_error"]) > 1e-8:
        failures.append("probe logit reconstruction")
    if float(metrics["maximum_probability_reconstruction_error"]) > 1e-10:
        failures.append("probe probability reconstruction")
    if int(metrics["threshold_crossing_mismatches"]) != 0:
        failures.append("probe sidecar threshold crossings")
    return failures, metrics


def audit_result(
    result: dict[str, Any],
    *,
    expected_role: str,
    result_root: Path | None = None,
    expected_gpu: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if result.get("schema_version") != 2:
        failures.append("schema")
    implementation = result.get("implementation_version")
    if implementation not in {2, 3} or (
        expected_role == "additional" and implementation != 3
    ):
        failures.append("implementation")
    if result.get("role") != expected_role:
        failures.append("role")
    if result.get("cohort") != "no_drop":
        failures.append("cohort")
    execution = result.get("execution", {})
    if execution.get("device") != "cuda":
        failures.append("CUDA execution")
    if not isinstance(execution.get("gpu"), str) or not execution["gpu"].strip():
        failures.append("GPU identity is missing")
    if expected_gpu and expected_gpu not in str(execution.get("gpu")):
        failures.append(f"GPU identity does not match {expected_gpu}")
    if not finite(execution.get("elapsed_seconds")):
        failures.append("elapsed time")
    expected_masses = {"0.02"} if expected_role == "smoke" else {
        "0.005", "0.01", "0.02", "0.04"
    }
    cells = result.get("cells", {})
    if set(cells) != expected_masses:
        failures.append("initial-mass inventory")
    identity_error = 0.0
    projection_error = 0.0
    log_rows = 0
    probe_rows = 0
    probe_logit_error = 0.0
    probe_probability_error = 0.0
    probe_crossing_mismatches = 0
    for mass, cell in cells.items():
        pairing = cell.get("pairing", {})
        if set(pairing) != {"model", "optimizer", "future_batch_plan"} or not all(
            pairing.values()
        ):
            failures.append(f"{mass}: branch pairing")
        errors = cell.get("calibration", {}).get("absolute_error_by_layer", {})
        if not errors or any(float(value) > 0.001 for value in errors.values()):
            failures.append(f"{mass}: mass entry")
        branches = cell.get("branches", {})
        if set(branches) != set(BRANCHES):
            failures.append(f"{mass}: branch inventory")
            continue
        starts = {
            (
                branch.get("start_model_fingerprint"),
                branch.get("start_optimizer_fingerprint"),
                branch.get("future_batch_plan_fingerprint"),
            )
            for branch in branches.values()
        }
        if len(starts) != 1:
            failures.append(f"{mass}: start fingerprints")
        for name, branch in branches.items():
            if not finite(branch.get("training_loss_first")) or not finite(
                branch.get("training_loss_last")
            ):
                failures.append(f"{mass}/{name}: finite losses")
            logs = branch.get("router_update_log", [])
            if not logs:
                failures.append(f"{mass}/{name}: update log")
                continue
            log_rows += len(logs)
            for row in logs:
                if set(row) != REQUIRED_LOG_FIELDS | {
                    "realized_update_identity_error"
                }:
                    failures.append(f"{mass}/{name}: log schema")
                    break
                identity_error = max(
                    identity_error,
                    abs(float(row["realized_update_identity_error"])),
                )
                for field in (
                    "raw_target_logit_pre_update",
                    "raw_comparator_logit_pre_update",
                    "target_logit_gradient",
                    "comparator_logit_gradient",
                    "realized_target_logit_update",
                    "realized_comparator_logit_update",
                    "realized_logodds_update_advantage",
                    "cumulative_realized_logodds_update_advantage",
                ):
                    if not finite(row[field]):
                        failures.append(f"{mass}/{name}: nonfinite {field}")
            forecasts = branch.get("forecast_rows", [])
            if not forecasts:
                failures.append(f"{mass}/{name}: frozen forecast")
            if name == "zero_advantage_gradient_projection_reference":
                errors = branch.get("projection_max_errors", {})
                projection_error = max(
                    projection_error,
                    *[abs(float(value)) for value in errors.values()],
                )
            if int(result.get("config", {}).get("probe_tokens_per_layer", 0)) > 0:
                expected_probe_rows = (
                    int(result.get("config", {}).get("horizon_steps", -1))
                    // int(result.get("config", {}).get("eval_interval", -1))
                    * len(result.get("config", {}).get("middle_layers", []))
                    * int(
                        result.get("config", {}).get(
                            "probe_tokens_per_layer", -1
                        )
                    )
                )
                probe_failures, probe_metrics = audit_probe_sidecar(
                    branch,
                    result_root=result_root,
                    expected_rows=expected_probe_rows,
                )
                failures.extend(
                    f"{mass}/{name}: {failure}"
                    for failure in probe_failures
                )
                probe_rows += int(probe_metrics["rows"])
                probe_logit_error = max(
                    probe_logit_error,
                    float(probe_metrics["maximum_logit_reconstruction_error"]),
                )
                probe_probability_error = max(
                    probe_probability_error,
                    float(
                        probe_metrics[
                            "maximum_probability_reconstruction_error"
                        ]
                    ),
                )
                probe_crossing_mismatches += int(
                    probe_metrics["threshold_crossing_mismatches"]
                )
    if identity_error > 2e-6:
        failures.append("realized-update identity")
    if projection_error > 1e-7:
        failures.append("gradient projection")
    return {
        "status": "passed" if not failures else "failed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "role": expected_role,
        "cohort": result.get("cohort"),
        "seed": result.get("seed"),
        "failures": failures,
        "router_update_log_rows": log_rows,
        "maximum_realized_update_identity_error": identity_error,
        "maximum_projection_invariant_error": projection_error,
        "probe_decomposition_rows": probe_rows,
        "maximum_probe_logit_reconstruction_error": probe_logit_error,
        "maximum_probe_probability_reconstruction_error": probe_probability_error,
        "probe_threshold_crossing_mismatches": probe_crossing_mismatches,
        "integrity_checks_passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-gpu", help="Optionally require a substring in the recorded GPU name")
    parser.add_argument(
        "--role", choices=("smoke", "primary", "additional"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    audit = audit_result(
        result,
        expected_role=args.role,
        result_root=args.result.resolve().parent,
        expected_gpu=args.expected_gpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
