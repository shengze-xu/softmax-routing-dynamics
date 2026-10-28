#!/usr/bin/env python3
"""Compare recovery summaries under alternative non-recovery codes.

Recode the historical 1,088 marker within the 1,024-step observation window
and summarize paired seed contrasts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs/tables"
CODES = (1088, 1024, 960)
SPECS = (
    ("no_drop", "router_update_no_drop_pooled_per_seed.csv",
     tuple(range(5201, 5209)) + tuple(range(5601, 5609)),
     "free_restricted_mean_recovery_time", "projection_restricted_mean_recovery_time",
     "free_recovery_probability_by_step_1088", "projection_recovery_probability_by_step_1088"),
    ("capacity_1_25", "router_update_capacity_per_seed.csv", tuple(range(5401, 5409)),
     "free_mean_restricted_recovery_step", "projection_mean_restricted_recovery_step",
     "free_recovery_fraction", "projection_recovery_fraction"),
)


def recode_mean(mean: float, recovery_fraction: float, code: int) -> float:
    if not math.isfinite(mean) or not 0 <= mean <= 1088:
        raise ValueError("Mean coded time must be finite and in [0,1088]")
    if not math.isfinite(recovery_fraction) or not 0 <= recovery_fraction <= 1:
        raise ValueError("Recovery fraction must be in [0,1]")
    if code not in CODES:
        raise ValueError("Unsupported non-recovery code")
    return mean - (1088 - code) * (1 - recovery_fraction)


def sign_p(differences: list[float]) -> float:
    positive = sum(x > 0 for x in differences)
    negative = sum(x < 0 for x in differences)
    n = positive + negative
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n)


def analyze(tables: Path) -> tuple[list[dict], list[dict]]:
    per_seed, summary = [], []
    for cohort, filename, seeds, free_key, proj_key, free_fraction, proj_fraction in SPECS:
        with (tables / filename).open(newline="") as stream:
            source = list(csv.DictReader(stream))
        if sorted(int(row["seed"]) for row in source) != sorted(seeds):
            raise ValueError(f"Unexpected or duplicate seed inventory in {filename}")
        for code in CODES:
            values, free_means, projection_means = [], [], []
            for row in source:
                free = recode_mean(float(row[free_key]), float(row[free_fraction]), code)
                projection = recode_mean(float(row[proj_key]), float(row[proj_fraction]), code)
                difference = projection - free
                values.append(difference)
                free_means.append(free)
                projection_means.append(projection)
                per_seed.append(dict(
                    cohort=cohort, seed=int(row["seed"]), non_recovery_code=code,
                    observation_steps=1024, latest_confirmable_first_step=960,
                    analysis_role="prespecified" if code == 1088 else "post_hoc_sensitivity",
                    free_mean_coded_time=free, projection_mean_coded_time=projection,
                    projection_minus_free_mean_coded_time=difference,
                ))
            n = len(values)
            summary.append(dict(
                cohort=cohort, non_recovery_code=code, seed_count=n,
                analysis_role="prespecified" if code == 1088 else "post_hoc_sensitivity",
                free_mean_coded_time=sum(free_means) / n,
                projection_mean_coded_time=sum(projection_means) / n,
                projection_minus_free_mean_coded_time=sum(values) / n,
                positive_pairs=sum(x > 0 for x in values),
                negative_pairs=sum(x < 0 for x in values), ties=sum(x == 0 for x in values),
                exact_two_sided_sign_p=sign_p(values),
            ))
    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    per_seed, summary = analyze(args.input_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("per_seed", per_seed), ("summary", summary)):
        with (args.output_dir / f"recovery_endpoint_sensitivity_{suffix}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in summary:
        print(f"{row['cohort']} code={row['non_recovery_code']}: "
              f"difference={row['projection_minus_free_mean_coded_time']}, "
              f"sign_p={row['exact_two_sided_sign_p']}")


if __name__ == "__main__":
    main()
