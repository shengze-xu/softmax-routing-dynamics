#!/usr/bin/env python3
"""Create the protocol-conformant no-drop mechanism evidence intake."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SEEDS = tuple(range(5201, 5209))
LAYERS = (1, 2, 3, 4)
MASSES = (0.005, 0.01, 0.02, 0.04)
BRANCHES = (
    "free_release",
    "continued_suppression_context_reference",
    "zero_advantage_gradient_projection_reference",
)
CENSORING_STEP = 1088
EXPECTED_ROWS_PER_SEED = len(MASSES) * len(BRANCHES) * 1024 * len(LAYERS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_two_sided_sign_p(values: list[float]) -> float:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    count = positive + negative
    if count == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(count, k) for k in range(tail + 1)) / 2**count
    return min(1.0, 2.0 * probability)


def bootstrap_interval(
    values: list[float], *, draws: int, seed: int
) -> list[float]:
    sample = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(sample), size=(draws, len(sample)))
    means = sample[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def censored(value: object) -> float:
    return float(CENSORING_STEP if value is None else value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-name", default="audit.json")
    parser.add_argument("--expected-gpu", help="Optionally require a substring in the recorded GPU name")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(
        protocol["protocol_id"] == "router_update_recovery_v1",
        "Protocol id changed",
    )
    protocol_seeds = next(
        item["seeds"]
        for item in protocol["immutable_design"]["cohorts"]
        if item["cohort"] == "no_drop"
    )
    require(protocol_seeds == list(SEEDS), "No-drop seed inventory changed")
    analysis_design_sha256 = protocol["immutable_design_sha256"]
    accepted_design_hashes = {
        analysis_design_sha256,
        protocol["immutable_design_sha256"],
    }

    shutil.copyfile(protocol_path, output / "protocol_snapshot.json")
    integrity_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    result_hashes: dict[str, str] = {}
    audit_hashes: dict[str, str] = {}
    parquet_writer: pq.ParquetWriter | None = None
    parquet_rows = 0

    for seed in SEEDS:
        seed_root = run_root / str(seed)
        result_path = seed_root / "result.json"
        audit_path = seed_root / args.audit_name
        require(result_path.is_file(), f"Missing result for seed {seed}")
        require(audit_path.is_file(), f"Missing audit for seed {seed}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        result_hashes[str(seed)] = sha256(result_path)
        audit_hashes[str(seed)] = sha256(audit_path)
        require(audit["status"] == "passed" and not audit["failures"], f"Seed {seed} audit failed")
        require(result["seed"] == seed, f"Seed {seed} identity changed")
        require(
            result["immutable_design_sha256"] in accepted_design_hashes,
            f"Seed {seed} design hash changed",
        )
        require(result["execution"]["device"] == "cuda", f"Seed {seed} did not use CUDA")
        if args.expected_gpu:
            require(args.expected_gpu in result["execution"]["gpu"], f"Seed {seed} GPU does not match {args.expected_gpu}")
        require(set(result["cells"]) == {str(value) for value in MASSES}, f"Seed {seed} mass inventory changed")

        seed_log_rows: list[dict[str, Any]] = []
        branch_metrics = {
            branch: {"time": [], "mae": [], "correct": [], "advantage": []}
            for branch in BRANCHES
        }
        max_calibration_error = 0.0
        assignment_checks = 0
        dropped_assignments = 0
        for mass in MASSES:
            cell = result["cells"][str(mass)]
            require(all(cell["pairing"].values()), f"Seed {seed}, mass {mass}: pairing failed")
            max_calibration_error = max(
                max_calibration_error,
                *map(float, cell["calibration"]["absolute_error_by_layer"].values()),
            )
            require(max_calibration_error <= 0.001, f"Seed {seed}: mass entry failed")
            require(set(cell["branches"]) == set(BRANCHES), f"Seed {seed}, mass {mass}: branch inventory changed")
            for branch in BRANCHES:
                record = cell["branches"][branch]
                seed_log_rows.extend(record["router_update_log"])
                forecasts = record["forecast_rows"]
                require(len(forecasts) == len(LAYERS), f"Seed {seed}, mass {mass}, {branch}: forecast inventory changed")
                for item in forecasts:
                    require(item["freeze_step"] == 256, "Forecast freeze step changed")
                    observed = censored(item["observed_recovery_step"])
                    predicted = censored(item["predicted_recovery_step"])
                    recovered = item["observed_recovery_step"] is not None
                    predicted_recovery = predicted < CENSORING_STEP
                    row = {
                        "seed": seed,
                        "layer_zero_based": item["layer_zero_based"],
                        "initial_target_mass": mass,
                        "branch": branch,
                        "freeze_step": item["freeze_step"],
                        "cumulative_advantage_at_freeze": item["cumulative_advantage_at_freeze"],
                        "estimated_slope": item["estimated_slope"],
                        "logodds_budget": item["logodds_budget"],
                        "predicted_recovery_step": predicted,
                        "observed_recovery_step": observed,
                        "absolute_error": abs(predicted - observed),
                        "predicted_recovery": predicted_recovery,
                        "observed_recovery": recovered,
                        "classification_correct": predicted_recovery == recovered,
                    }
                    forecast_rows.append(row)
                    recovery_rows.append(
                        {
                            "seed": seed,
                            "layer_zero_based": item["layer_zero_based"],
                            "initial_target_mass": mass,
                            "branch": branch,
                            "restricted_recovery_step": observed,
                            "persistent_recovery": recovered,
                        }
                    )
                    branch_metrics[branch]["time"].append(observed)
                    branch_metrics[branch]["mae"].append(abs(predicted - observed))
                    branch_metrics[branch]["correct"].append(predicted_recovery == recovered)
                    branch_metrics[branch]["advantage"].append(float(item["cumulative_advantage_at_freeze"]))

        require(len(seed_log_rows) == EXPECTED_ROWS_PER_SEED, f"Seed {seed}: log row count changed")
        for row in seed_log_rows:
            require(row["capacity_factor"] is None, f"Seed {seed}: no-drop capacity changed")
            if row["total_assignments"] is not None:
                assignment_checks += 1
                require(
                    int(row["total_assignments"])
                    == int(row["kept_assignments"]) + int(row["dropped_assignments"]),
                    f"Seed {seed}: assignment conservation failed",
                )
            dropped_assignments += int(row["dropped_assignments"] or 0)
        require(dropped_assignments == 0, f"Seed {seed}: no-drop assignments were dropped")
        table = pa.Table.from_pylist(seed_log_rows)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                output / "router_update_log.parquet",
                table.schema,
                compression="zstd",
            )
        parquet_writer.write_table(table)
        parquet_rows += table.num_rows

        free = branch_metrics["free_release"]
        projection = branch_metrics[
            "zero_advantage_gradient_projection_reference"
        ]
        row = {
            "seed": seed,
            "free_mean_restricted_recovery_step": float(np.mean(free["time"])),
            "projection_mean_restricted_recovery_step": float(np.mean(projection["time"])),
            "projection_minus_free_recovery_step": float(np.mean(projection["time"]) - np.mean(free["time"])),
            "free_forecast_mae": float(np.mean(free["mae"])),
            "projection_forecast_mae": float(np.mean(projection["mae"])),
            "projection_minus_free_forecast_mae": float(np.mean(projection["mae"]) - np.mean(free["mae"])),
            "free_recovery_fraction": float(np.mean([value < CENSORING_STEP for value in free["time"]])),
            "projection_recovery_fraction": float(np.mean([value < CENSORING_STEP for value in projection["time"]])),
            "free_forecast_classification_accuracy": float(np.mean(free["correct"])),
            "projection_forecast_classification_accuracy": float(np.mean(projection["correct"])),
            "free_mean_cumulative_advantage_step_256": float(np.mean(free["advantage"])),
            "projection_mean_cumulative_advantage_step_256": float(np.mean(projection["advantage"])),
        }
        seed_rows.append(row)
        integrity_rows.append(
            {
                "seed": seed,
                "audit_status": audit["status"],
                "result_sha256": result_hashes[str(seed)],
                "audit_sha256": audit_hashes[str(seed)],
                "router_update_log_rows": len(seed_log_rows),
                "assignment_conservation_checks": assignment_checks,
                "dropped_assignments": dropped_assignments,
                "maximum_calibration_absolute_error": max_calibration_error,
                "maximum_projection_invariant_error": audit["maximum_projection_invariant_error"],
                "maximum_realized_update_identity_error": audit["maximum_realized_update_identity_error"],
            }
        )
        environment_rows.append(
            {
                "seed": seed,
                **result["execution"],
            }
        )
    if parquet_writer is not None:
        parquet_writer.close()
    require(parquet_rows == EXPECTED_ROWS_PER_SEED * len(SEEDS), "Parquet row count changed")

    write_csv(output / "integrity_per_seed.csv", integrity_rows)
    write_csv(output / "forecast_frozen_at_step_256.csv", forecast_rows)
    write_csv(output / "recovery_per_seed_layer_mass_branch.csv", recovery_rows)
    write_csv(output / "seed_level_primary_endpoints.csv", seed_rows)
    write_csv(output / "environment.csv", environment_rows)

    recovery_effects = [row["projection_minus_free_recovery_step"] for row in seed_rows]
    forecast_effects = [row["projection_minus_free_forecast_mae"] for row in seed_rows]
    bootstrap_seed = int(protocol["resampling_seed"])
    mass_summary: list[dict[str, Any]] = []
    for mass in MASSES:
        for branch in BRANCHES:
            selected_recovery = [
                row for row in recovery_rows
                if float(row["initial_target_mass"]) == mass and row["branch"] == branch
            ]
            selected_forecast = [
                row for row in forecast_rows
                if float(row["initial_target_mass"]) == mass and row["branch"] == branch
            ]
            mass_summary.append(
                {
                    "initial_target_mass": mass,
                    "branch": branch,
                    "mean_restricted_recovery_step": float(np.mean([row["restricted_recovery_step"] for row in selected_recovery])),
                    "persistent_recoveries": sum(bool(row["persistent_recovery"]) for row in selected_recovery),
                    "cells": len(selected_recovery),
                    "mean_cumulative_advantage_step_256": float(np.mean([row["cumulative_advantage_at_freeze"] for row in selected_forecast])),
                    "mean_forecast_absolute_error": float(np.mean([row["absolute_error"] for row in selected_forecast])),
                }
            )
    write_csv(output / "mass_branch_summary.csv", mass_summary)

    summary = {
        "schema_version": 2,
        "status": "passed",
        "cohort": "no_drop",
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "protocol_design_sha256": protocol["immutable_design_sha256"],
        "primary_seeds": list(SEEDS),
        "statistical_unit": "paired primary seed",
        "layers_and_mass_cells": "repeated measurements grouped within seed",
        "bootstrap": {
            "draws": 100000,
            "confidence_level": 0.95,
            "interval": "percentile",
            "seed": bootstrap_seed,
            "seed_source": "protocol resampling_seed",
        },
        "integrity": {
            "all_seed_audits_passed": True,
            "router_update_log_rows": parquet_rows,
            "forecast_rows": len(forecast_rows),
            "recovery_rows": len(recovery_rows),
            "maximum_calibration_absolute_error": max(row["maximum_calibration_absolute_error"] for row in integrity_rows),
            "maximum_projection_invariant_error": max(row["maximum_projection_invariant_error"] for row in integrity_rows),
            "maximum_realized_update_identity_error": max(row["maximum_realized_update_identity_error"] for row in integrity_rows),
            "assignment_conservation_passed": True,
            "no_drop_passed": True,
            "forecast_frozen_at_step_256": True,
        },
        "primary_recovery_contrast": {
            "estimand": "projection minus free-release restricted recovery step; positive values favor free release",
            "effects_by_seed": recovery_effects,
            "mean": float(np.mean(recovery_effects)),
            "bootstrap_95_interval": bootstrap_interval(recovery_effects, draws=100000, seed=bootstrap_seed),
            "positive_seed_pairs": sum(value > 0 for value in recovery_effects),
            "exact_two_sided_sign_p": exact_two_sided_sign_p(recovery_effects),
        },
        "primary_forecast_contrast": {
            "estimand": "projection minus free-release forecast MAE; positive values favor free release",
            "effects_by_seed": forecast_effects,
            "mean": float(np.mean(forecast_effects)),
            "bootstrap_95_interval": bootstrap_interval(forecast_effects, draws=100000, seed=bootstrap_seed + 1),
            "positive_seed_pairs": sum(value > 0 for value in forecast_effects),
            "exact_two_sided_sign_p": exact_two_sided_sign_p(forecast_effects),
        },
        "capacity_cohort_gate": {
            "integrity": True,
            "entry": True,
            "logging_completeness": True,
            "forecast_freeze": True,
            "passed": True,
        },
        "result_sha256": result_hashes,
        "audit_sha256": audit_hashes,
    }
    summary_path = output / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_names = (
        "protocol_snapshot.json",
        "environment.csv",
        "integrity_per_seed.csv",
        "router_update_log.parquet",
        "forecast_frozen_at_step_256.csv",
        "recovery_per_seed_layer_mass_branch.csv",
        "seed_level_primary_endpoints.csv",
        "mass_branch_summary.csv",
        "analysis_summary.json",
    )
    manifest = "\n".join(f"{sha256(output / name)}  {name}" for name in manifest_names) + "\n"
    (output / "evidence_manifest.txt").write_text(manifest, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
