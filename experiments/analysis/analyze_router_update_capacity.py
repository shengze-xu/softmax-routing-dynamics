#!/usr/bin/env python3
"""Consolidate and independently audit the eight capacity-limited runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_router_update_no_drop import (
    BRANCHES,
    CENSORING_STEP,
    LAYERS,
    MASSES,
    bootstrap_interval,
    censored,
    exact_two_sided_sign_p,
    require,
    sha256,
    write_csv,
)


SEEDS = tuple(range(5401, 5409))
EXPECTED_ROWS_PER_SEED = len(MASSES) * len(BRANCHES) * 1024 * len(LAYERS)
EXPECTED_FORECASTS_PER_SEED = len(MASSES) * len(BRANCHES) * len(LAYERS)


def locate_seed(run_roots: list[Path], seed: int, *, audit_name: str = "audit.json") -> tuple[int, Path]:
    matches = [
        (index, root / str(seed))
        for index, root in enumerate(run_roots)
        if (root / str(seed) / "result.json").is_file()
        and (root / str(seed) / audit_name).is_file()
    ]
    require(len(matches) == 1, f"Seed {seed} has {len(matches)} complete result roots")
    return matches[0]


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    recovery = summary["primary_recovery_contrast"]
    forecast = summary["primary_forecast_contrast"]
    rows = [
        {
            "endpoint": "projection_minus_free_restricted_recovery_step",
            "estimate": recovery["mean"],
            "bootstrap_95_ci_low": recovery["bootstrap_95_interval"][0],
            "bootstrap_95_ci_high": recovery["bootstrap_95_interval"][1],
            "positive_seed_pairs": recovery["positive_seed_pairs"],
            "total_seed_pairs": len(SEEDS),
            "exact_two_sided_sign_p": recovery["exact_two_sided_sign_p"],
        },
        {
            "endpoint": "projection_minus_free_forecast_mae",
            "estimate": forecast["mean"],
            "bootstrap_95_ci_low": forecast["bootstrap_95_interval"][0],
            "bootstrap_95_ci_high": forecast["bootstrap_95_interval"][1],
            "positive_seed_pairs": forecast["positive_seed_pairs"],
            "total_seed_pairs": len(SEEDS),
            "exact_two_sided_sign_p": forecast["exact_two_sided_sign_p"],
        },
    ]
    for endpoint, estimate in summary["descriptive_endpoints"].items():
        rows.append(
            {
                "endpoint": endpoint,
                "estimate": estimate,
                "bootstrap_95_ci_low": "",
                "bootstrap_95_ci_high": "",
                "positive_seed_pairs": "",
                "total_seed_pairs": "",
                "exact_two_sided_sign_p": "",
            }
        )
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-name", default="audit.json")
    parser.add_argument("--expected-gpu", help="Optionally require a substring in the recorded GPU name")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_roots = [path.resolve() for path in args.run_root]
    protocol_path = args.protocol.resolve()
    output = args.output_dir.resolve()
    require(len(set(run_roots)) == len(run_roots), "Run roots must be unique")
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
        if item["cohort"] == "capacity_1_25"
    )
    require(protocol_seeds == list(SEEDS), "Capacity seed inventory changed")
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
    mass_branch_accumulator: dict[tuple[float, str], dict[str, list[Any] | int]] = {}
    result_hashes: dict[str, str] = {}
    audit_hashes: dict[str, str] = {}
    seed_sources: dict[str, dict[str, Any]] = {}
    parquet_writer: pq.ParquetWriter | None = None
    parquet_rows = 0

    for seed in SEEDS:
        root_index, seed_root = locate_seed(run_roots, seed, audit_name=args.audit_name)
        result_path = seed_root / "result.json"
        audit_path = seed_root / args.audit_name
        result = json.loads(result_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        result_hashes[str(seed)] = sha256(result_path)
        audit_hashes[str(seed)] = sha256(audit_path)
        seed_sources[str(seed)] = {
            "run_root": str(run_roots[root_index]),
        }

        require(
            audit["status"] == "passed"
            and audit["integrity_checks_passed"] is True
            and not audit["failures"],
            f"Seed {seed} audit failed",
        )
        require(
            result["seed"] == seed
            and result["role"] == "primary",
            f"Seed {seed} identity changed",
        )
        require(
            result["immutable_design_sha256"] in accepted_design_hashes,
            f"Seed {seed} design hash changed",
        )
        require(
            result["execution"]["device"] == "cuda",
            f"Seed {seed} did not use CUDA",
        )
        if args.expected_gpu:
            require(args.expected_gpu in result["execution"]["gpu"], f"Seed {seed} GPU does not match {args.expected_gpu}")
        config = result["config"]
        require(
            config["capacity_factor"] == 1.25
            and config["capacity_policy"] == "drop_no_rerouting"
            and config["horizon_steps"] == 1024
            and config["forecast_freeze_step"] == 256
            and tuple(config["middle_layers"]) == LAYERS,
            f"Seed {seed} capacity-limited design changed",
        )
        require(
            set(result["cells"]) == {str(value) for value in MASSES},
            f"Seed {seed} mass inventory changed",
        )

        seed_log_rows: list[dict[str, Any]] = []
        branch_metrics = {
            branch: {"time": [], "mae": [], "correct": [], "advantage": []}
            for branch in BRANCHES
        }
        max_calibration_error = 0.0
        assignment_checks = 0
        seed_total_assignments = 0
        seed_kept_assignments = 0
        seed_dropped_assignments = 0
        active_branch_cells = 0

        for mass in MASSES:
            cell = result["cells"][str(mass)]
            require(all(cell["pairing"].values()), f"Seed {seed}, mass {mass}: pairing failed")
            errors = list(map(float, cell["calibration"]["absolute_error_by_layer"].values()))
            require(len(errors) == len(LAYERS), f"Seed {seed}, mass {mass}: calibration inventory changed")
            max_calibration_error = max(max_calibration_error, *errors)
            require(max(errors) <= 0.001, f"Seed {seed}, mass {mass}: mass entry failed")
            require(
                set(cell["branches"]) == set(BRANCHES),
                f"Seed {seed}, mass {mass}: branch inventory changed",
            )
            starts = {
                (
                    branch["start_model_fingerprint"],
                    branch["start_optimizer_fingerprint"],
                    branch["future_batch_plan_fingerprint"],
                )
                for branch in cell["branches"].values()
            }
            require(len(starts) == 1, f"Seed {seed}, mass {mass}: branch starts differ")

            for branch in BRANCHES:
                record = cell["branches"][branch]
                logs = record["router_update_log"]
                require(
                    len(logs) == 1024 * len(LAYERS),
                    f"Seed {seed}, mass {mass}, {branch}: log inventory changed",
                )
                seed_log_rows.extend(logs)
                forecasts = record["forecast_rows"]
                require(
                    len(forecasts) == len(LAYERS),
                    f"Seed {seed}, mass {mass}, {branch}: forecast inventory changed",
                )
                branch_total = 0
                branch_dropped = 0
                for item in forecasts:
                    require(item["freeze_step"] == 256, "Forecast freeze step changed")
                    observed = censored(item["observed_recovery_step"])
                    predicted = censored(item["predicted_recovery_step"])
                    recovered = item["observed_recovery_step"] is not None
                    predicted_recovery = predicted < CENSORING_STEP
                    forecast_rows.append(
                        {
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
                    )
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

                for row in logs:
                    require(
                        row["capacity_factor"] == 1.25,
                        f"Seed {seed}: log design changed",
                    )
                    total = int(row["total_assignments"])
                    kept = int(row["kept_assignments"])
                    dropped = int(row["dropped_assignments"])
                    assignment_checks += 1
                    require(total == kept + dropped, f"Seed {seed}: assignment conservation failed")
                    require(0 <= kept <= total and dropped >= 0, f"Seed {seed}: assignment bounds failed")
                    branch_total += total
                    branch_dropped += dropped
                    seed_total_assignments += total
                    seed_kept_assignments += kept
                    seed_dropped_assignments += dropped
                require(branch_dropped > 0, f"Seed {seed}, mass {mass}, {branch}: capacity inactive")
                require(
                    int(record["total_dropped_assignments"]) == branch_dropped,
                    f"Seed {seed}, mass {mass}, {branch}: dropped total changed",
                )
                active_branch_cells += 1
                accumulator = mass_branch_accumulator.setdefault(
                    (mass, branch),
                    {
                        "time": [],
                        "recovered": [],
                        "advantage": [],
                        "forecast_error": [],
                        "total_assignments": 0,
                        "dropped_assignments": 0,
                    },
                )
                accumulator["time"].extend(branch_metrics[branch]["time"][-len(LAYERS):])
                accumulator["recovered"].extend(
                    [value < CENSORING_STEP for value in branch_metrics[branch]["time"][-len(LAYERS):]]
                )
                accumulator["advantage"].extend(branch_metrics[branch]["advantage"][-len(LAYERS):])
                accumulator["forecast_error"].extend(branch_metrics[branch]["mae"][-len(LAYERS):])
                accumulator["total_assignments"] += branch_total
                accumulator["dropped_assignments"] += branch_dropped

        require(len(seed_log_rows) == EXPECTED_ROWS_PER_SEED, f"Seed {seed}: log row count changed")
        require(active_branch_cells == len(MASSES) * len(BRANCHES), f"Seed {seed}: inactive branch cell")
        require(
            seed_total_assignments
            == seed_kept_assignments + seed_dropped_assignments,
            f"Seed {seed}: aggregate assignment conservation failed",
        )
        require(
            int(audit["router_update_log_rows"]) == len(seed_log_rows)
            and int(audit["total_assignments"]) == seed_total_assignments
            and int(audit["kept_assignments"]) == seed_kept_assignments
            and int(audit["dropped_assignments"]) == seed_dropped_assignments
            and int(audit["branches_with_active_dropping"]) == active_branch_cells,
            f"Seed {seed}: independent totals disagree with primary audit",
        )

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
        projection = branch_metrics["zero_advantage_gradient_projection_reference"]
        seed_rows.append(
            {
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
                "dropped_assignments": seed_dropped_assignments,
                "total_assignments": seed_total_assignments,
                "drop_fraction": seed_dropped_assignments / seed_total_assignments,
            }
        )
        integrity_rows.append(
            {
                "seed": seed,
                "audit_status": audit["status"],
                "result_sha256": result_hashes[str(seed)],
                "audit_sha256": audit_hashes[str(seed)],
                "router_update_log_rows": len(seed_log_rows),
                "assignment_conservation_checks": assignment_checks,
                "active_branch_cells": active_branch_cells,
                "dropped_assignments": seed_dropped_assignments,
                "total_assignments": seed_total_assignments,
                "drop_fraction": seed_dropped_assignments / seed_total_assignments,
                "maximum_calibration_absolute_error": max_calibration_error,
                "maximum_projection_invariant_error": audit["maximum_projection_invariant_error"],
                "maximum_realized_update_identity_error": audit["maximum_realized_update_identity_error"],
            }
        )
        environment_rows.append({"seed": seed, **result["execution"]})

    if parquet_writer is not None:
        parquet_writer.close()
    require(
        parquet_rows == EXPECTED_ROWS_PER_SEED * len(SEEDS),
        "Combined Parquet row count changed",
    )
    require(
        len(forecast_rows) == EXPECTED_FORECASTS_PER_SEED * len(SEEDS)
        and len(recovery_rows) == EXPECTED_FORECASTS_PER_SEED * len(SEEDS),
        "Combined endpoint row count changed",
    )

    mass_summary: list[dict[str, Any]] = []
    for mass in MASSES:
        for branch in BRANCHES:
            values = mass_branch_accumulator[(mass, branch)]
            total = int(values["total_assignments"])
            dropped = int(values["dropped_assignments"])
            mass_summary.append(
                {
                    "initial_target_mass": mass,
                    "branch": branch,
                    "mean_restricted_recovery_step": float(np.mean(values["time"])),
                    "persistent_recoveries": int(sum(values["recovered"])),
                    "cells": len(values["time"]),
                    "mean_cumulative_advantage_step_256": float(np.mean(values["advantage"])),
                    "mean_forecast_absolute_error": float(np.mean(values["forecast_error"])),
                    "dropped_assignments": dropped,
                    "total_assignments": total,
                    "drop_fraction": dropped / total,
                }
            )

    write_csv(output / "integrity_per_seed.csv", integrity_rows)
    write_csv(output / "forecast_frozen_at_step_256.csv", forecast_rows)
    write_csv(output / "recovery_per_seed_layer_mass_branch.csv", recovery_rows)
    write_csv(output / "seed_level_primary_endpoints.csv", seed_rows)
    write_csv(output / "mass_branch_summary.csv", mass_summary)
    write_csv(output / "environment.csv", environment_rows)

    recovery_effects = [row["projection_minus_free_recovery_step"] for row in seed_rows]
    forecast_effects = [row["projection_minus_free_forecast_mae"] for row in seed_rows]
    bootstrap_seed = int(protocol["resampling_seed"])
    total_assignments = sum(int(row["total_assignments"]) for row in integrity_rows)
    dropped_assignments = sum(int(row["dropped_assignments"]) for row in integrity_rows)
    summary = {
        "schema_version": 2,
        "status": "passed",
        "cohort": "capacity_1_25",
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
            "independent_totals_match_seed_audits": True,
            "router_update_log_rows": parquet_rows,
            "forecast_rows": len(forecast_rows),
            "recovery_rows": len(recovery_rows),
            "assignment_conservation_checks": sum(int(row["assignment_conservation_checks"]) for row in integrity_rows),
            "active_branch_cells": sum(int(row["active_branch_cells"]) for row in integrity_rows),
            "total_assignments": total_assignments,
            "dropped_assignments": dropped_assignments,
            "drop_fraction": dropped_assignments / total_assignments,
            "maximum_calibration_absolute_error": max(float(row["maximum_calibration_absolute_error"]) for row in integrity_rows),
            "maximum_projection_invariant_error": max(float(row["maximum_projection_invariant_error"]) for row in integrity_rows),
            "maximum_realized_update_identity_error": max(float(row["maximum_realized_update_identity_error"]) for row in integrity_rows),
            "assignment_conservation_passed": True,
            "active_dropping_passed": True,
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
        "descriptive_endpoints": {
            "free_mean_restricted_recovery_step": float(np.mean([row["free_mean_restricted_recovery_step"] for row in seed_rows])),
            "projection_mean_restricted_recovery_step": float(np.mean([row["projection_mean_restricted_recovery_step"] for row in seed_rows])),
            "free_mean_cumulative_advantage_step_256": float(np.mean([row["free_mean_cumulative_advantage_step_256"] for row in seed_rows])),
            "projection_mean_cumulative_advantage_step_256": float(np.mean([row["projection_mean_cumulative_advantage_step_256"] for row in seed_rows])),
            "free_persistent_recoveries": sum(row["free_recovery_fraction"] * len(MASSES) * len(LAYERS) for row in seed_rows),
            "projection_persistent_recoveries": sum(row["projection_recovery_fraction"] * len(MASSES) * len(LAYERS) for row in seed_rows),
            "continued_suppression_persistent_recoveries": sum(
                bool(row["persistent_recovery"])
                for row in recovery_rows
                if row["branch"] == "continued_suppression_context_reference"
            ),
            "drop_fraction": dropped_assignments / total_assignments,
        },
        "evidence_basis": "independent eight-seed consolidation with frozen protocol, hashes, pairing, calibration, conservation, active-dropping, and endpoint audits",
        "seed_sources": seed_sources,
        "result_sha256": result_hashes,
        "audit_sha256": audit_hashes,
    }
    summary_path = output / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(output / "summary.csv", summary)

    manifest_names = (
        "protocol_snapshot.json",
        "environment.csv",
        "integrity_per_seed.csv",
        "router_update_log.parquet",
        "forecast_frozen_at_step_256.csv",
        "recovery_per_seed_layer_mass_branch.csv",
        "seed_level_primary_endpoints.csv",
        "mass_branch_summary.csv",
        "summary.csv",
        "analysis_summary.json",
    )
    manifest = "\n".join(
        f"{sha256(output / name)}  {name}" for name in manifest_names
    ) + "\n"
    (output / "evidence_manifest.txt").write_text(manifest, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
