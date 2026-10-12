#!/usr/bin/env python3
"""Historical summaries of encoded persistent router recovery.

The 1088 non-recovery code exceeds the 1024-step observation window.
Outputs retain historical names for provenance and do not identify survival
beyond follow-up; see analyze_recovery_endpoint_sensitivity.py.
Kaplan--Meier curves are descriptive at the seed-layer-mass cell level. The
paired restricted-mean contrasts and bootstrap intervals use the seed as the
resampling and inferential unit, matching the experiment protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FREE = "free_release"
DEFAULT_REFERENCE = "zero_advantage_gradient_projection_reference"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_rows(specifications: list[str], tau: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("Each --input must have the form COHORT=CSV_PATH")
        cohort, raw_path = specification.split("=", 1)
        path = Path(raw_path).resolve()
        require(bool(cohort), f"Missing cohort label for {path}")
        require(path.is_file(), f"Missing recovery table: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                event = source["persistent_recovery"].strip().lower() in {
                    "1", "true", "yes"
                }
                time = min(float(source["restricted_recovery_step"]), tau)
                rows.append(
                    {
                        "cohort": cohort,
                        "seed": int(source["seed"]),
                        "layer_zero_based": int(source["layer_zero_based"]),
                        "initial_target_mass": float(source["initial_target_mass"]),
                        "branch": source["branch"],
                        "time": time,
                        "event": bool(event and time < tau),
                    }
                )
    require(bool(rows), "No recovery rows were read")
    return rows


def kaplan_meier(records: list[dict[str, Any]], tau: float) -> list[dict[str, Any]]:
    require(bool(records), "Kaplan--Meier input is empty")
    at_risk = len(records)
    survival = 1.0
    curve = [
        {
            "time": 0.0,
            "at_risk": at_risk,
            "events": 0,
            "censored": 0,
            "survival": survival,
            "recovery_probability": 0.0,
        }
    ]
    times = sorted({float(row["time"]) for row in records if row["time"] <= tau})
    for time in times:
        events = sum(row["time"] == time and row["event"] for row in records)
        censored = sum(
            row["time"] == time and not row["event"] for row in records
        )
        if at_risk <= 0:
            break
        if events:
            survival *= 1.0 - events / at_risk
        curve.append(
            {
                "time": time,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival": survival,
                "recovery_probability": 1.0 - survival,
            }
        )
        at_risk -= events + censored
    return curve


def exact_two_sided_sign_p(values: list[float]) -> float:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    count = positive + negative
    if count == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(count, k) for k in range(tail + 1)) / 2**count
    return min(1.0, 2.0 * probability)


def paired_seed_rows(
    records: list[dict[str, Any]], *, free: str, reference: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault((row["cohort"], row["seed"], row["branch"]), []).append(row)
    output: list[dict[str, Any]] = []
    cohort_seeds = sorted({(row["cohort"], row["seed"]) for row in records})
    for cohort, seed in cohort_seeds:
        free_rows = grouped.get((cohort, seed, free), [])
        reference_rows = grouped.get((cohort, seed, reference), [])
        require(bool(free_rows), f"Missing {free} cells for {cohort} seed {seed}")
        require(
            bool(reference_rows),
            f"Missing {reference} cells for {cohort} seed {seed}",
        )
        free_keys = {
            (row["layer_zero_based"], row["initial_target_mass"])
            for row in free_rows
        }
        reference_keys = {
            (row["layer_zero_based"], row["initial_target_mass"])
            for row in reference_rows
        }
        require(
            free_keys == reference_keys,
            f"Paired cell inventory differs for {cohort} seed {seed}",
        )
        free_rmst = float(np.mean([row["time"] for row in free_rows]))
        reference_rmst = float(np.mean([row["time"] for row in reference_rows]))
        free_recovery = float(np.mean([row["event"] for row in free_rows]))
        reference_recovery = float(
            np.mean([row["event"] for row in reference_rows])
        )
        output.append(
            {
                "cohort": cohort,
                "seed": seed,
                "cells_per_branch": len(free_rows),
                "free_restricted_mean_recovery_time": free_rmst,
                "reference_restricted_mean_recovery_time": reference_rmst,
                "reference_minus_free_restricted_mean_recovery_time": (
                    reference_rmst - free_rmst
                ),
                "free_recovery_probability_by_tau": free_recovery,
                "reference_recovery_probability_by_tau": reference_recovery,
                "reference_minus_free_recovery_probability_by_tau": (
                    reference_recovery - free_recovery
                ),
            }
        )
    return output


def bootstrap_mean_interval(
    values: list[float], *, draws: int, seed: int
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(sample), size=(draws, len(sample)))
    means = sample[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    records: list[dict[str, Any]],
    *,
    tau: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
    free: str = DEFAULT_FREE,
    reference: str = DEFAULT_REFERENCE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    branches = {row["branch"] for row in records}
    require({free, reference} <= branches, "Required paired branches are absent")
    curve_rows: list[dict[str, Any]] = []
    cohorts = sorted({row["cohort"] for row in records})
    for cohort in cohorts:
        cohort_rows = [row for row in records if row["cohort"] == cohort]
        masses: list[str | float] = ["all"] + sorted(
            {row["initial_target_mass"] for row in cohort_rows}
        )
        for branch in sorted({row["branch"] for row in cohort_rows}):
            for mass in masses:
                group = [
                    row for row in cohort_rows
                    if row["branch"] == branch
                    and (mass == "all" or row["initial_target_mass"] == mass)
                ]
                for point in kaplan_meier(group, tau):
                    curve_rows.append(
                        {
                            "cohort": cohort,
                            "branch": branch,
                            "initial_target_mass": mass,
                            **point,
                        }
                    )
    seed_rows = paired_seed_rows(records, free=free, reference=reference)
    summary_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        selected = [row for row in seed_rows if row["cohort"] == cohort]
        rmst_values = [
            row["reference_minus_free_restricted_mean_recovery_time"]
            for row in selected
        ]
        recovery_values = [
            row["reference_minus_free_recovery_probability_by_tau"]
            for row in selected
        ]
        rmst_low, rmst_high = bootstrap_mean_interval(
            rmst_values, draws=bootstrap_draws, seed=bootstrap_seed
        )
        recovery_low, recovery_high = bootstrap_mean_interval(
            recovery_values, draws=bootstrap_draws, seed=bootstrap_seed + 1
        )
        summary_rows.append(
            {
                "cohort": cohort,
                "seed_count": len(selected),
                "tau": tau,
                "mean_reference_minus_free_restricted_mean_recovery_time": float(
                    np.mean(rmst_values)
                ),
                "rmst_difference_bootstrap_95_low": rmst_low,
                "rmst_difference_bootstrap_95_high": rmst_high,
                "rmst_difference_exact_two_sided_sign_p": exact_two_sided_sign_p(
                    rmst_values
                ),
                "positive_zero_negative_rmst_seed_counts": "/".join(
                    str(value)
                    for value in (
                        sum(item > 0 for item in rmst_values),
                        sum(item == 0 for item in rmst_values),
                        sum(item < 0 for item in rmst_values),
                    )
                ),
                "mean_reference_minus_free_recovery_probability_by_tau": float(
                    np.mean(recovery_values)
                ),
                "recovery_probability_difference_bootstrap_95_low": recovery_low,
                "recovery_probability_difference_bootstrap_95_high": recovery_high,
            }
        )
    summary = {
        "estimand": (
            "Paired seed-level difference between the projection reference and "
            "free release in restricted mean persistent-recovery time."
        ),
        "tau": tau,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "inferential_unit": "seed",
        "survival_curve_unit": "seed-layer-mass cell (descriptive; repeated within seed)",
        "free_branch": free,
        "reference_branch": reference,
        "cohorts": summary_rows,
    }
    return curve_rows, seed_rows, summary_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="COHORT=/path/to/recovery_per_seed_layer_mass_branch.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=1088.0)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    require(args.tau > 0, "--tau must be positive")
    require(args.bootstrap_draws > 0, "--bootstrap-draws must be positive")
    output = args.output_dir.resolve()
    require(not output.exists(), f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    records = read_rows(args.input, args.tau)
    curves, seeds, summaries, summary = analyze(
        records,
        tau=args.tau,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv(output / "survival_curve.csv", curves)
    write_csv(output / "seed_level_restricted_mean.csv", seeds)
    write_csv(output / "survival_summary.csv", summaries)
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
