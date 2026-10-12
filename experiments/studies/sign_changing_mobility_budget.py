#!/usr/bin/env python3
"""Direct integration test of cumulative recovery budgets with sign-changing gaps."""

from __future__ import annotations

import argparse
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
class GapSpec:
    name: str
    label: str

    def value(self, time: float) -> float:
        if self.name == "positive_drift_oscillation":
            return 0.35 + 0.75 * math.sin(1.2 * time - 0.8)
        if self.name == "zero_mean_oscillation":
            return 0.75 * math.sin(time)
        if self.name == "subthreshold_oscillation":
            return 0.15 * math.sin(0.9 * time)
        raise ValueError(f"Unknown gap specification: {self.name}")

    def cumulative(self, time: float) -> float:
        if self.name == "positive_drift_oscillation":
            return 0.35 * time + (0.75 / 1.2) * (
                math.cos(0.8) - math.cos(1.2 * time - 0.8)
            )
        if self.name == "zero_mean_oscillation":
            return 0.75 * (1.0 - math.cos(time))
        if self.name == "subthreshold_oscillation":
            return (0.15 / 0.9) * (1.0 - math.cos(0.9 * time))
        raise ValueError(f"Unknown gap specification: {self.name}")


GAPS = (
    GapSpec("positive_drift_oscillation", "oscillation with positive drift"),
    GapSpec("zero_mean_oscillation", "zero-mean oscillation"),
    GapSpec("subthreshold_oscillation", "subthreshold oscillation"),
)

MOBILITIES = ("j2", "j", "projected")


def budget(mobility: str, start: float, end: float) -> float:
    if mobility == "j2":
        def antiderivative(value: float) -> float:
            return -1.0 / value + 1.0 / (1.0 - value) + 2.0 * math.log(value / (1.0 - value))

        return 0.5 * (antiderivative(end) - antiderivative(start))
    if mobility == "j":
        return math.log(end * (1.0 - start) / ((1.0 - end) * start))
    if mobility == "projected":
        return 2.0 * (end - start)
    raise ValueError(f"Unknown mobility: {mobility}")


def velocity(mobility: str, probability: float, gap: float) -> float:
    if mobility == "j2":
        return 2.0 * probability**2 * (1.0 - probability) ** 2 * gap
    if mobility == "j":
        return probability * (1.0 - probability) * gap
    if mobility == "projected":
        return 0.5 * gap
    raise ValueError(f"Unknown mobility: {mobility}")


def rk4_step(probability: float, time: float, dt: float, mobility: str, gap: GapSpec) -> float:
    k1 = velocity(mobility, probability, gap.value(time))
    k2 = velocity(mobility, probability + 0.5 * dt * k1, gap.value(time + 0.5 * dt))
    k3 = velocity(mobility, probability + 0.5 * dt * k2, gap.value(time + 0.5 * dt))
    k4 = velocity(mobility, probability + dt * k3, gap.value(time + dt))
    return probability + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    gap: GapSpec,
    mobility: str,
    *,
    epsilon: float,
    theta: float,
    dt: float,
    horizon: float,
) -> tuple[dict[str, object], list[list[object]]]:
    threshold_budget = budget(mobility, epsilon, theta)
    grid = np.arange(int(round(horizon / dt)) + 1, dtype=float) * dt
    cumulative_grid = np.array([gap.cumulative(float(time)) for time in grid])
    gap_grid = np.array([gap.value(float(time)) for time in grid])
    sup_cumulative = float(np.max(cumulative_grid))
    predicted = bool(sup_cumulative >= threshold_budget)
    predicted_indices = np.flatnonzero(cumulative_grid >= threshold_budget)
    predicted_time = float(grid[predicted_indices[0]]) if predicted_indices.size else float("nan")

    probability = epsilon
    max_identity_error = 0.0
    min_probability = probability
    max_probability = probability
    observed_time = float("nan")
    sign_positive = int(np.sum(gap_grid > 0.0))
    sign_negative = int(np.sum(gap_grid < 0.0))
    sign_change_before_recovery = False
    stride = max(1, len(grid) // 500)
    trajectory: list[list[object]] = []
    for index, time in enumerate(grid):
        current_gap = gap.value(float(time))
        cumulative = gap.cumulative(float(time))
        identity_error = abs(budget(mobility, epsilon, probability) - cumulative)
        max_identity_error = max(max_identity_error, identity_error)
        min_probability = min(min_probability, probability)
        max_probability = max(max_probability, probability)
        if index % stride == 0 or index == len(grid) - 1 or probability >= theta:
            trajectory.append(
                [
                    gap.name,
                    mobility,
                    float(time),
                    current_gap,
                    cumulative,
                    threshold_budget,
                    probability,
                    identity_error,
                ]
            )
        if probability >= theta:
            observed_time = float(time)
            prefix = gap_grid[: index + 1]
            sign_change_before_recovery = bool(np.any(prefix > 0.0) and np.any(prefix < 0.0))
            break
        if index == len(grid) - 1:
            break
        probability = rk4_step(probability, float(time), dt, mobility, gap)
        if not 0.0 < probability < 1.0:
            raise RuntimeError(f"{gap.name}/{mobility} left the simplex interior before crossing.")

    observed = bool(np.isfinite(observed_time))
    result = {
        "gap_case": gap.name,
        "mobility": mobility,
        "epsilon": epsilon,
        "theta": theta,
        "dt": dt,
        "horizon": horizon,
        "budget": threshold_budget,
        "sup_cumulative_gap": sup_cumulative,
        "predicted_recovery": predicted,
        "observed_recovery": observed,
        "predicted_crossing_time": predicted_time,
        "observed_crossing_time": observed_time,
        "crossing_time_error": abs(predicted_time - observed_time) if predicted and observed else float("nan"),
        "max_budget_identity_error": max_identity_error,
        "minimum_probability": min_probability,
        "maximum_probability": max_probability,
        "positive_gap_samples": sign_positive,
        "negative_gap_samples": sign_negative,
        "sign_changed": bool(sign_positive > 0 and sign_negative > 0),
        "sign_change_before_recovery": sign_change_before_recovery,
        "crossing_gate": bool(predicted == observed and (not observed or abs(predicted_time - observed_time) <= 2.0 * dt)),
    }
    return result, trajectory




def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate sign-changing recovery gaps for three simplex mobilities.")
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--theta", type=float, default=0.30)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--horizon", type=float, default=40.0)
    args = parser.parse_args()
    if not 0.0 < args.epsilon < args.theta < 1.0:
        raise SystemExit("Require 0 < epsilon < theta < 1.")
    if args.dt <= 0.0 or args.horizon <= 0.0:
        raise SystemExit("Time parameters must be positive.")

    rows: list[dict[str, object]] = []
    trajectories: list[list[object]] = []
    for gap in GAPS:
        for mobility in MOBILITIES:
            result, path = simulate(
                gap,
                mobility,
                epsilon=args.epsilon,
                theta=args.theta,
                dt=args.dt,
                horizon=args.horizon,
            )
            rows.append(result)
            trajectories.extend(path)

    header = [
        "gap_case",
        "mobility",
        "epsilon",
        "theta",
        "dt",
        "horizon",
        "budget",
        "sup_cumulative_gap",
        "predicted_recovery",
        "observed_recovery",
        "predicted_crossing_time",
        "observed_crossing_time",
        "crossing_time_error",
        "max_budget_identity_error",
        "minimum_probability",
        "maximum_probability",
        "positive_gap_samples",
        "negative_gap_samples",
        "sign_changed",
        "sign_change_before_recovery",
        "crossing_gate",
    ]
    write_csv(
        ensure_subdir("tables") / "sign_changing_mobility_budget_summary.csv",
        header,
        [[row[key] for key in header] for row in rows],
    )
    write_csv(
        ensure_subdir("tables") / "sign_changing_mobility_budget_trajectories.csv",
        ["gap_case", "mobility", "time", "instantaneous_gap", "cumulative_gap", "budget", "probability", "identity_error"],
        trajectories,
    )
    if not all(bool(row["sign_changed"]) and bool(row["crossing_gate"]) for row in rows):
        raise RuntimeError("At least one sign-changing budget gate failed.")
    print(f"sign-changing crossing gates = {len(rows)}/{len(rows)}")
    print(f"maximum budget identity error = {max(float(row['max_budget_identity_error']) for row in rows):.6e}")
    print(f"recovery outcomes = {sum(bool(row['observed_recovery']) for row in rows)}/{len(rows)}")



if __name__ == "__main__":
    main()
