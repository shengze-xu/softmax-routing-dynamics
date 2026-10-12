#!/usr/bin/env python3
"""Consolidate the no-drop seed additional and held-out logit decomposition."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis import analyze_recovery_survival as survival
from experiments.common.router_logit_decomposition import PROBE_FIELDS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(
    root: Path,
    *,
    seed: int,
    role: str,
    cohort: str,
    protocol_id: str,
    audit_name: str = "audit.json",
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    seed_root = root / str(seed)
    result_path = seed_root / "result.json"
    audit_path = seed_root / audit_name
    require(result_path.is_file(), f"Missing result for {cohort} seed {seed}")
    require(audit_path.is_file(), f"Missing audit for {cohort} seed {seed}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(result.get("seed") == seed, f"Seed identity changed for {seed}")
    require(result.get("role") == role, f"Role changed for {cohort} seed {seed}")
    require(result.get("cohort") == cohort, f"Cohort changed for seed {seed}")
    require(
        result.get("protocol_id") == protocol_id,
        f"Protocol changed for {cohort} seed {seed}",
    )
    require(
        audit.get("status") == "passed" and not audit.get("failures"),
        f"Audit failed for {cohort} seed {seed}",
    )
    require(
        int(audit.get("probe_threshold_crossing_mismatches", -1)) == 0,
        f"Probe crossing reconstruction failed for {cohort} seed {seed}",
    )
    return result, audit, seed_root


def recovery_rows_from_result(
    result: dict[str, Any], *, tau: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mass, cell in result["cells"].items():
        for branch, record in cell["branches"].items():
            for forecast in record["forecast_rows"]:
                observed = forecast["observed_recovery_step"]
                rows.append(
                    {
                        "cohort": result["cohort"],
                        "seed": int(result["seed"]),
                        "layer_zero_based": int(forecast["layer_zero_based"]),
                        "initial_target_mass": float(mass),
                        "branch": branch,
                        "time": float(tau if observed is None else observed),
                        "event": observed is not None,
                    }
                )
    return rows


def public_recovery_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": row["seed"],
        "layer_zero_based": row["layer_zero_based"],
        "initial_target_mass": row["initial_target_mass"],
        "branch": row["branch"],
        "restricted_recovery_step": row["time"],
        "persistent_recovery": row["event"],
    }


def merge_probe_sidecars(
    runs: list[tuple[dict[str, Any], Path]], output: Path
) -> tuple[list[dict[str, Any]], int]:
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0
    with gzip.open(output, "wt", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=PROBE_FIELDS)
        writer.writeheader()
        for result, seed_root in runs:
            for cell in result["cells"].values():
                for branch, record in cell["branches"].items():
                    summary = record.get("probe_decomposition")
                    require(isinstance(summary, dict), "Missing probe summary")
                    raw_name = summary.get("path")
                    require(
                        isinstance(raw_name, str) and Path(raw_name).name == raw_name,
                        "Unsafe probe sidecar path",
                    )
                    source = seed_root / raw_name
                    require(source.is_file(), f"Missing probe sidecar: {source}")
                    with gzip.open(
                        source, "rt", encoding="utf-8", newline=""
                    ) as handle:
                        reader = csv.DictReader(handle)
                        require(
                            tuple(reader.fieldnames or ()) == PROBE_FIELDS,
                            f"Probe schema changed: {source}",
                        )
                        for row in reader:
                            writer.writerow(row)
                            row_count += 1
                            key = (row["cohort"], row["branch"])
                            aggregate = aggregates.setdefault(
                                key,
                                {
                                    "cohort": row["cohort"],
                                    "branch": row["branch"],
                                    "rows": 0,
                                    "observed_threshold_crossings": 0,
                                    "threshold_crossing_mismatches": 0,
                                    "parameter_only_threshold_crossings": 0,
                                    "feature_only_threshold_crossings": 0,
                                    "sum_absolute_parameter_gap_displacement": 0.0,
                                    "sum_absolute_feature_gap_displacement": 0.0,
                                    "maximum_logit_reconstruction_error": 0.0,
                                    "maximum_probability_reconstruction_error": 0.0,
                                },
                            )
                            aggregate["rows"] += 1
                            aggregate["observed_threshold_crossings"] += int(
                                row["crossed_threshold_observed"]
                            )
                            aggregate["threshold_crossing_mismatches"] += (
                                int(row["crossed_threshold_observed"])
                                != int(row["crossed_threshold_reconstructed"])
                            )
                            aggregate["parameter_only_threshold_crossings"] += int(
                                row["crossed_threshold_parameter_only"]
                            )
                            aggregate["feature_only_threshold_crossings"] += int(
                                row["crossed_threshold_feature_only"]
                            )
                            aggregate[
                                "sum_absolute_parameter_gap_displacement"
                            ] += abs(float(row["parameter_gap_displacement"]))
                            aggregate[
                                "sum_absolute_feature_gap_displacement"
                            ] += abs(float(row["feature_gap_displacement"]))
                            aggregate["maximum_logit_reconstruction_error"] = max(
                                aggregate["maximum_logit_reconstruction_error"],
                                abs(
                                    float(
                                        row[
                                            "logit_reconstruction_max_abs_error"
                                        ]
                                    )
                                ),
                            )
                            aggregate[
                                "maximum_probability_reconstruction_error"
                            ] = max(
                                aggregate[
                                    "maximum_probability_reconstruction_error"
                                ],
                                abs(
                                    float(
                                        row[
                                            "probability_reconstruction_abs_error"
                                        ]
                                    )
                                ),
                            )
    rows: list[dict[str, Any]] = []
    for aggregate in aggregates.values():
        count = int(aggregate.pop("rows"))
        parameter_sum = float(
            aggregate.pop("sum_absolute_parameter_gap_displacement")
        )
        feature_sum = float(aggregate.pop("sum_absolute_feature_gap_displacement"))
        rows.append(
            {
                "cohort": aggregate.pop("cohort"),
                "branch": aggregate.pop("branch"),
                "rows": count,
                "mean_absolute_parameter_gap_displacement": parameter_sum / count,
                "mean_absolute_feature_gap_displacement": feature_sum / count,
                **aggregate,
            }
        )
    rows.sort(key=lambda row: (row["cohort"], row["branch"]))
    return rows, row_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-protocol", type=Path, required=True)
    parser.add_argument("--audit-name", default="audit.json")
    parser.add_argument("--no-drop-run-root", type=Path, required=True)
    parser.add_argument("--capacity-run-root", type=Path, required=True)
    parser.add_argument("--original-no-drop-recovery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=1088.0)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    protocol_path = args.probe_protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_id = str(protocol["protocol_id"])
    output = args.output_dir.resolve()
    require(not output.exists(), f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    shutil.copyfile(protocol_path, output / "protocol_snapshot.json")

    run_records: list[tuple[dict[str, Any], Path]] = []
    integrity_rows: list[dict[str, Any]] = []
    extension_recovery: list[dict[str, Any]] = []
    inventories = (
        (
            "no_drop",
            "additional",
            protocol["additional_no_drop_seeds"]["seeds"],
            args.no_drop_run_root.resolve(),
        ),
        (
            "capacity_1_25",
            "probe_rerun",
            protocol["logit_decomposition"]["cohorts"]["capacity_1_25"]["seeds"],
            args.capacity_run_root.resolve(),
        ),
    )
    for cohort, role, seeds, root in inventories:
        for seed in seeds:
            result, audit, seed_root = load_run(
                root,
                seed=int(seed),
                role=role,
                cohort=cohort,
                protocol_id=protocol_id,
                audit_name=args.audit_name,
            )
            run_records.append((result, seed_root))
            integrity_rows.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "audit_status": audit["status"],
                    "result_sha256": sha256(seed_root / "result.json"),
                    "audit_sha256": sha256(seed_root / args.audit_name),
                    "probe_decomposition_rows": audit[
                        "probe_decomposition_rows"
                    ],
                    "maximum_probe_logit_reconstruction_error": audit[
                        "maximum_probe_logit_reconstruction_error"
                    ],
                    "maximum_probe_probability_reconstruction_error": audit[
                        "maximum_probe_probability_reconstruction_error"
                    ],
                    "probe_threshold_crossing_mismatches": audit[
                        "probe_threshold_crossing_mismatches"
                    ],
                }
            )
            if cohort == "no_drop":
                extension_recovery.extend(
                    recovery_rows_from_result(result, tau=args.tau)
                )

    write_csv(output / "integrity_per_seed.csv", integrity_rows)
    write_csv(
        output / "no_drop_additional_recovery.csv",
        [public_recovery_row(row) for row in extension_recovery],
    )
    probe_summaries, probe_rows = merge_probe_sidecars(
        run_records, output / "probe_decomposition.csv.gz"
    )
    write_csv(output / "probe_decomposition_summary.csv", probe_summaries)
    require(
        all(row["threshold_crossing_mismatches"] == 0 for row in probe_summaries),
        "Probe threshold-crossing reconstruction failed",
    )

    original = survival.read_rows(
        [f"no_drop={args.original_no_drop_recovery.resolve()}"], args.tau
    )
    pooled = original + extension_recovery
    curves, seed_rows, survival_rows, survival_summary = survival.analyze(
        pooled,
        tau=args.tau,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv(output / "no_drop_pooled_survival_curve.csv", curves)
    write_csv(output / "no_drop_pooled_seed_level_restricted_mean.csv", seed_rows)
    write_csv(output / "no_drop_pooled_survival_summary.csv", survival_rows)
    summary = {
        "protocol_id": protocol_id,
        "status": "analysis_complete_pending_manuscript_review",
        "run_count": len(run_records),
        "probe_decomposition_rows": probe_rows,
        "probe_decomposition": probe_summaries,
        "no_drop_pooled_survival": survival_summary,
        "integrity": {
            "all_seed_audits_passed": True,
            "probe_threshold_crossing_mismatches": 0,
        },
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
