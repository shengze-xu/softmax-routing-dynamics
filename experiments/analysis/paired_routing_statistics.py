from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_subdir, write_csv


@dataclass(frozen=True)
class Outcome:
    name: str
    direction: float
    favorable_direction: str


@dataclass(frozen=True)
class Comparison:
    name: str
    treatment_source: str
    control_source: str
    treatment: str
    control: str
    outcomes: tuple[Outcome, ...]


ROUTING_OUTCOMES = (
    Outcome("unique_dominant_experts", 1.0, "higher"),
    Outcome("specialization_score", 1.0, "higher"),
)

STUDY_SWITCH_OUTCOMES = ROUTING_OUTCOMES + (
    Outcome("post_shift_final_new_regime_mse", -1.0, "lower"),
)

AGNEWS_OUTCOMES = ROUTING_OUTCOMES + (
    Outcome("test_accuracy", 1.0, "higher"),
)

PREDICTIVE_OUTCOMES = (
    Outcome("test_accuracy", 1.0, "higher"),
)


COMPARISONS = (
    Comparison(
        "reduced_neural_conditioning",
        "input_dependent_break_freeze_per_seed.csv",
        "input_dependent_break_freeze_per_seed.csv",
        "input_conditioned_tau0",
        "shared_global_tau0",
        STUDY_SWITCH_OUTCOMES,
    ),
    Comparison(
        "conditioned_capacity_0p1",
        "neural_capacity_break_freeze_per_seed.csv",
        "neural_capacity_break_freeze_per_seed.csv",
        "input_conditioned_capacity_lambda0p1",
        "input_conditioned_softmax",
        STUDY_SWITCH_OUTCOMES,
    ),
    Comparison(
        "architecture_conditioned_softmax",
        "architecture_level_moe_check_per_seed.csv",
        "architecture_level_moe_check_per_seed.csv",
        "arch_input_conditioned_softmax",
        "arch_shared_global_softmax",
        STUDY_SWITCH_OUTCOMES,
    ),
    Comparison(
        "architecture_conditioned_top2",
        "architecture_level_moe_check_per_seed.csv",
        "architecture_level_moe_check_per_seed.csv",
        "arch_input_conditioned_top2",
        "arch_shared_global_top2",
        STUDY_SWITCH_OUTCOMES,
    ),
    Comparison(
        "agnews_conditioned_top2",
        "transformer_moe_agnews_per_seed.csv",
        "transformer_moe_agnews_per_seed.csv",
        "agnews_conditioned_top2",
        "agnews_global_top2",
        AGNEWS_OUTCOMES,
    ),
    Comparison(
        "agnews_conditioned_top2_balanced",
        "transformer_moe_agnews_per_seed.csv",
        "transformer_moe_agnews_per_seed.csv",
        "agnews_conditioned_top2_balanced",
        "agnews_global_top2",
        AGNEWS_OUTCOMES,
    ),
    Comparison(
        "agnews_conditioned_vs_dense_matched",
        "transformer_moe_agnews_per_seed.csv",
        "agnews_compute_matched_dense_per_seed.csv",
        "agnews_conditioned_top2",
        "agnews_dense_active_matched",
        PREDICTIVE_OUTCOMES,
    ),
    Comparison(
        "agnews_global_vs_dense_matched",
        "transformer_moe_agnews_per_seed.csv",
        "agnews_compute_matched_dense_per_seed.csv",
        "agnews_global_top2",
        "agnews_dense_active_matched",
        PREDICTIVE_OUTCOMES,
    ),
    Comparison(
        "digits_conditioned_top2",
        "digits_moe_validation_per_seed.csv",
        "digits_moe_validation_per_seed.csv",
        "digits_conditioned_top2",
        "digits_global_top2",
        AGNEWS_OUTCOMES,
    ),
)


def read_mode_rows(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    result: dict[str, dict[int, dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            mode = row["mode"]
            seed = int(row["seed"])
            if seed in result.setdefault(mode, {}):
                raise ValueError(f"Duplicate seed {seed} for {mode} in {path}")
            result[mode][seed] = row
    return result


def resolve_mode_source(filename: str, mode: str, minimum_pairs: int = 10, *, input_dir: Path) -> tuple[Path, dict[int, dict[str, str]]]:
    candidates = (input_dir / filename,)
    reports: list[str] = []
    for path in candidates:
        if not path.exists():
            reports.append(f"{path}: missing")
            continue
        modes = read_mode_rows(path)
        rows = modes.get(mode, {})
        if len(rows) >= minimum_pairs:
            return path, rows
        reports.append(f"{path}: {mode} has {len(rows)} rows")
    raise ValueError(
        f"No complete source for {mode} in {filename}; " + "; ".join(reports)
    )


def exact_two_sided_sign_p(favorable_differences: np.ndarray, tol: float = 1e-12) -> tuple[int, int, int, float]:
    wins = int(np.sum(favorable_differences > tol))
    losses = int(np.sum(favorable_differences < -tol))
    ties = int(favorable_differences.size - wins - losses)
    non_ties = wins + losses
    if non_ties == 0:
        return wins, ties, losses, 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(non_ties, index) for index in range(tail + 1)) / (2**non_ties)
    return wins, ties, losses, min(1.0, 2.0 * probability)


def bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    indices = rng.integers(0, differences.size, size=(replicates, differences.size))
    means = np.mean(differences[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired statistical audit.")
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()

    rng = np.random.default_rng(20260717)
    bootstrap_replicates = 100_000
    records: list[dict[str, object]] = []

    for comparison in COMPARISONS:
        treatment_path, treatment = resolve_mode_source(
            comparison.treatment_source,
            comparison.treatment,
            input_dir=args.input_dir.resolve(),
        )
        control_path, control = resolve_mode_source(
            comparison.control_source,
            comparison.control,
            input_dir=args.input_dir.resolve(),
        )
        seeds = sorted(set(treatment) & set(control))
        if set(seeds) != set(treatment) or set(seeds) != set(control):
            raise ValueError(f"Unpaired seed sets for {comparison.name}")

        for outcome in comparison.outcomes:
            treatment_values = np.asarray([float(treatment[seed][outcome.name]) for seed in seeds])
            control_values = np.asarray([float(control[seed][outcome.name]) for seed in seeds])
            differences = treatment_values - control_values
            favorable_differences = outcome.direction * differences
            ci_low, ci_high = bootstrap_mean_ci(
                differences,
                rng=rng,
                replicates=bootstrap_replicates,
            )
            wins, ties, losses, sign_p = exact_two_sided_sign_p(favorable_differences)
            difference_sd = float(np.std(differences, ddof=1))
            paired_dz = float(np.mean(differences) / difference_sd) if difference_sd > 0.0 else float("nan")
            records.append(
                {
                    "comparison": comparison.name,
                    "treatment_source": str(treatment_path),
                    "control_source": str(control_path),
                    "treatment": comparison.treatment,
                    "control": comparison.control,
                    "outcome": outcome.name,
                    "favorable_direction": outcome.favorable_direction,
                    "pair_count": len(seeds),
                    "treatment_mean": float(np.mean(treatment_values)),
                    "control_mean": float(np.mean(control_values)),
                    "paired_mean_difference": float(np.mean(differences)),
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "paired_dz": paired_dz,
                    "favorable_wins": wins,
                    "ties": ties,
                    "favorable_losses": losses,
                    "exact_sign_p": sign_p,
                }
            )

    adjusted = holm_adjust([float(record["exact_sign_p"]) for record in records])
    for record, value in zip(records, adjusted):
        record["holm_adjusted_sign_p"] = value

    header = list(records[0])
    rows = [[record[key] for key in header] for record in records]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output = args.output_dir / "paired_routing_statistics.csv"
    write_csv(output, header, rows)

    print(f"wrote {len(records)} paired tests to {output}")
    for record in records:
        if record["outcome"] == "specialization_score":
            print(
                f"{record['comparison']}: mean difference "
                f"{float(record['paired_mean_difference']):.4f}, "
                f"95% CI [{float(record['bootstrap_95_ci_low']):.4f}, "
                f"{float(record['bootstrap_95_ci_high']):.4f}], "
                f"wins/ties/losses {record['favorable_wins']}/"
                f"{record['ties']}/{record['favorable_losses']}, "
                f"Holm p={float(record['holm_adjusted_sign_p']):.4f}"
            )


if __name__ == "__main__":
    main()
