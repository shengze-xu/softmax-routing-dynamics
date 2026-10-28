#!/usr/bin/env python3
"""Evaluate the normal-velocity decomposition for arbitrary expert counts."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_subdir, write_csv


Array = np.ndarray
ScoreField = Callable[[Array], Array]


def softmax_jacobian(probability: Array) -> Array:
    return np.diag(probability) - np.outer(probability, probability)


def h1(start: float, end: float) -> float:
    def primitive(value: float) -> float:
        return math.log(value / (1.0 - value)) + 1.0 / (1.0 - value)

    return primitive(end) - primitive(start)


def h2(start: float, end: float) -> float:
    def primitive(value: float) -> float:
        return (
            -1.0 / value
            + 1.0 / (1.0 - value)
            + 2.0 * math.log(value / (1.0 - value))
        )

    return primitive(end) - primitive(start)


def normal_quantities(
    probability: Array,
    score: Array,
    *,
    target_idx: int = 0,
) -> dict[str, float]:
    x = float(probability[target_idx])
    active = np.arange(probability.size) != target_idx
    conditional = probability[active] / (1.0 - x)
    active_score = score[active]
    mean_score = float(conditional @ active_score)
    delta = mean_score - float(score[target_idx])
    leakage = float((conditional * conditional) @ (active_score - mean_score))
    conditional_norm_sq = float(conditional @ conditional)
    gamma = leakage + x * (1.0 + conditional_norm_sq) * delta
    jacobian = softmax_jacobian(probability)
    velocity = -(jacobian @ jacobian @ score)
    predicted = x * (1.0 - x) ** 2 * gamma
    return {
        "x": x,
        "leakage": leakage,
        "delta": delta,
        "gamma": gamma,
        "conditional_norm_sq": conditional_norm_sq,
        "actual_normal_velocity": float(velocity[target_idx]),
        "predicted_normal_velocity": predicted,
        "normal_identity_error": abs(float(velocity[target_idx]) - predicted),
    }


def leakage_score_field(rate: float) -> ScoreField:
    """Return a feedback field with Delta=0 and Lambda=rate.

    For conditional active weights r, v=r-||r||_2^2 1 satisfies
    <r,v>=0 and <r^2,v>=Var_{k~r}(r_k). Rescaling v therefore fixes the
    within-face covariance term without prescribing the probability trajectory.
    """

    def score(probability: Array) -> Array:
        x = float(probability[0])
        conditional = probability[1:] / (1.0 - x)
        second_moment = float(conditional @ conditional)
        centered = conditional - second_moment
        weighted_variance = float((conditional * conditional) @ centered)
        if weighted_variance <= 1e-13:
            raise RuntimeError("Covariance-field construction approached a balanced face.")
        active_score = rate * centered / weighted_variance
        return np.concatenate(([0.0], active_score))

    return score


def balanced_score_field(gap: float) -> ScoreField:
    def score(probability: Array) -> Array:
        result = np.zeros(probability.size, dtype=float)
        result[0] = -gap
        return result

    return score


def probability_rhs(probability: Array, score_field: ScoreField) -> tuple[Array, Array]:
    score = score_field(probability)
    jacobian = softmax_jacobian(probability)
    return -(jacobian @ jacobian @ score), score


def rk4_step(probability: Array, dt: float, score_field: ScoreField) -> Array:
    k1, _ = probability_rhs(probability, score_field)
    k2, _ = probability_rhs(probability + 0.5 * dt * k1, score_field)
    k3, _ = probability_rhs(probability + 0.5 * dt * k2, score_field)
    k4, _ = probability_rhs(probability + dt * k3, score_field)
    return probability + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def initial_probability(n: int, epsilon: float, regime: str) -> Array:
    if regime == "leakage":
        conditional = np.arange(1, n, dtype=float) ** 8
        conditional /= np.sum(conditional)
    elif regime == "balanced":
        conditional = np.full(n - 1, 1.0 / (n - 1), dtype=float)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return np.concatenate(([epsilon], (1.0 - epsilon) * conditional))


def simulate_recovery(
    n: int,
    epsilon: float,
    theta: float,
    *,
    regime: str,
    leakage_rate: float,
    balanced_gap: float,
    dt: float,
) -> dict[str, object]:
    probability = initial_probability(n, epsilon, regime)
    if regime == "leakage":
        score_field = leakage_score_field(leakage_rate)
        predicted_time = h1(epsilon, theta) / leakage_rate
    else:
        score_field = balanced_score_field(balanced_gap)
        coefficient = (1.0 + 1.0 / (n - 1)) * balanced_gap
        predicted_time = h2(epsilon, theta) / coefficient

    time = 0.0
    max_time = 1.05 * predicted_time + 2.0 * dt
    maximum_steps = int(math.ceil(max_time / dt)) + 1
    min_leakage = math.inf
    max_leakage = -math.inf
    min_delta = math.inf
    max_delta = -math.inf
    min_gamma = math.inf
    max_gamma = -math.inf
    max_identity_error = 0.0
    max_simplex_error = 0.0
    min_probability = float(np.min(probability))
    max_score_norm = 0.0
    observed_time = math.nan

    for _ in range(maximum_steps):
        velocity, score = probability_rhs(probability, score_field)
        quantities = normal_quantities(probability, score)
        min_leakage = min(min_leakage, quantities["leakage"])
        max_leakage = max(max_leakage, quantities["leakage"])
        min_delta = min(min_delta, quantities["delta"])
        max_delta = max(max_delta, quantities["delta"])
        min_gamma = min(min_gamma, quantities["gamma"])
        max_gamma = max(max_gamma, quantities["gamma"])
        max_identity_error = max(max_identity_error, quantities["normal_identity_error"])
        max_simplex_error = max(max_simplex_error, abs(float(np.sum(probability)) - 1.0))
        min_probability = min(min_probability, float(np.min(probability)))
        max_score_norm = max(max_score_norm, float(np.linalg.norm(score)))
        if probability[0] >= theta:
            observed_time = time
            break

        updated = rk4_step(probability, dt, score_field)
        if not np.all(np.isfinite(updated)) or float(np.min(updated)) <= 0.0:
            raise RuntimeError(f"{regime}/n={n}/epsilon={epsilon} left the simplex interior.")
        if updated[0] >= theta:
            fraction = (theta - probability[0]) / (updated[0] - probability[0])
            observed_time = time + dt * float(fraction)
            probability = updated
            break
        probability = updated
        time += dt

    if not math.isfinite(observed_time):
        raise RuntimeError(f"{regime}/n={n}/epsilon={epsilon} did not recover.")

    relative_time_error = abs(observed_time - predicted_time) / predicted_time
    return {
        "regime": regime,
        "n_experts": n,
        "epsilon": epsilon,
        "theta": theta,
        "dt": dt,
        "observed_time": observed_time,
        "predicted_time": predicted_time,
        "relative_time_error": relative_time_error,
        "minimum_leakage": min_leakage,
        "maximum_leakage": max_leakage,
        "minimum_delta": min_delta,
        "maximum_delta": max_delta,
        "minimum_gamma": min_gamma,
        "maximum_gamma": max_gamma,
        "maximum_normal_identity_error": max_identity_error,
        "maximum_simplex_error": max_simplex_error,
        "minimum_probability": min_probability,
        "maximum_score_norm": max_score_norm,
        "terminal_target_mass": float(probability[0]),
    }


def pointwise_evaluation(n_values: list[int], samples: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for n in n_values:
        for sample in range(samples):
            probability = rng.dirichlet(np.full(n, 0.7, dtype=float))
            score = rng.normal(size=n)
            target_idx = sample % n
            quantities = normal_quantities(probability, score, target_idx=target_idx)
            scale = max(
                abs(quantities["actual_normal_velocity"]),
                abs(quantities["predicted_normal_velocity"]),
                1e-14,
            )
            rows.append(
                {
                    "n_experts": n,
                    "sample": sample,
                    "target_idx": target_idx,
                    "target_mass": quantities["x"],
                    "minimum_probability": float(np.min(probability)),
                    "leakage": quantities["leakage"],
                    "delta": quantities["delta"],
                    "gamma": quantities["gamma"],
                    "actual_normal_velocity": quantities["actual_normal_velocity"],
                    "predicted_normal_velocity": quantities["predicted_normal_velocity"],
                    "absolute_error": quantities["normal_identity_error"],
                    "scaled_error": quantities["normal_identity_error"] / scale,
                }
            )
    return rows


def linear_fit(x_values: Array, y_values: Array) -> tuple[float, float, float]:
    design = np.column_stack((np.ones_like(x_values), x_values))
    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    predictions = design @ coefficients
    residual = float(np.sum((y_values - predictions) ** 2))
    total = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else math.nan
    return float(coefficients[0]), float(coefficients[1]), r_squared


def summarize_recovery(rows: list[dict[str, object]], n_values: list[int]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for regime in ("leakage", "balanced"):
        for n in n_values:
            selected = [row for row in rows if row["regime"] == regime and row["n_experts"] == n]
            epsilon = np.asarray([row["epsilon"] for row in selected], dtype=float)
            observed = np.asarray([row["observed_time"] for row in selected], dtype=float)
            if regime == "leakage":
                intercept, slope, r_squared = linear_fit(np.log(1.0 / epsilon), observed)
                fit_name = "time_vs_log_inverse_mass"
            else:
                intercept, slope, r_squared = linear_fit(np.log(epsilon), np.log(observed))
                fit_name = "log_time_vs_log_mass"
            summaries.append(
                {
                    "regime": regime,
                    "n_experts": n,
                    "fit": fit_name,
                    "intercept": intercept,
                    "slope": slope,
                    "r_squared": r_squared,
                    "maximum_relative_time_error": max(float(row["relative_time_error"]) for row in selected),
                    "maximum_normal_identity_error": max(
                        float(row["maximum_normal_identity_error"]) for row in selected
                    ),
                    "minimum_leakage": min(float(row["minimum_leakage"]) for row in selected),
                    "maximum_absolute_delta": max(
                        max(abs(float(row["minimum_delta"])), abs(float(row["maximum_delta"])))
                        for row in selected
                    ),
                }
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--leakage-epsilons",
        type=float,
        nargs="+",
        default=[0.005, 0.0075, 0.01, 0.02, 0.03],
    )
    parser.add_argument(
        "--balanced-epsilons",
        type=float,
        nargs="+",
        default=[0.00025, 0.0005, 0.001, 0.002, 0.004],
    )
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--leakage-rate", type=float, default=0.001)
    parser.add_argument("--balanced-gap", type=float, default=1.0)
    parser.add_argument("--leakage-dt", type=float, default=0.25)
    parser.add_argument("--balanced-dt", type=float, default=0.1)
    parser.add_argument("--pointwise-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    n_values = sorted(set(args.n_values))
    leakage_epsilons = sorted(set(args.leakage_epsilons))
    balanced_epsilons = sorted(set(args.balanced_epsilons))
    if not n_values or min(n_values) < 3:
        raise SystemExit("The multi-expert experiment requires n >= 3.")
    if not leakage_epsilons or not balanced_epsilons:
        raise SystemExit("Each recovery regime requires at least one initial mass.")
    if not all(
        0.0 < value < args.theta < 1.0
        for value in leakage_epsilons + balanced_epsilons
    ):
        raise SystemExit("Require 0 < epsilon < theta < 1.")
    if min(args.leakage_rate, args.balanced_gap, args.leakage_dt, args.balanced_dt) <= 0.0:
        raise SystemExit("Rates, gaps, and time steps must be positive.")
    if args.pointwise_samples <= 0:
        raise SystemExit("pointwise-samples must be positive.")

    pointwise_rows = pointwise_evaluation(n_values, args.pointwise_samples, args.seed)
    recovery_rows: list[dict[str, object]] = []
    for n in n_values:
        for epsilon in leakage_epsilons:
            recovery_rows.append(
                simulate_recovery(
                    n,
                    epsilon,
                    args.theta,
                    regime="leakage",
                    leakage_rate=args.leakage_rate,
                    balanced_gap=args.balanced_gap,
                    dt=args.leakage_dt,
                )
            )
        for epsilon in balanced_epsilons:
            recovery_rows.append(
                simulate_recovery(
                    n,
                    epsilon,
                    args.theta,
                    regime="balanced",
                    leakage_rate=args.leakage_rate,
                    balanced_gap=args.balanced_gap,
                    dt=args.balanced_dt,
                )
            )
    summary_rows = summarize_recovery(recovery_rows, n_values)

    tables = ensure_subdir("tables")
    write_csv(
        tables / "multiexpert_face_mobility_pointwise.csv",
        list(pointwise_rows[0]),
        [[row[key] for key in pointwise_rows[0]] for row in pointwise_rows],
    )
    write_csv(
        tables / "multiexpert_face_mobility_recovery.csv",
        list(recovery_rows[0]),
        [[row[key] for key in recovery_rows[0]] for row in recovery_rows],
    )
    write_csv(
        tables / "multiexpert_face_mobility_summary.csv",
        list(summary_rows[0]),
        [[row[key] for key in summary_rows[0]] for row in summary_rows],
    )

    maximum_pointwise_error = max(float(row["absolute_error"]) for row in pointwise_rows)
    maximum_time_error = max(float(row["relative_time_error"]) for row in recovery_rows)
    leakage_summaries = [row for row in summary_rows if row["regime"] == "leakage"]
    balanced_summaries = [row for row in summary_rows if row["regime"] == "balanced"]
    gates = {
        "pointwise_identity": maximum_pointwise_error <= 1e-12,
        "clock_time": maximum_time_error <= 5e-4,
        "leakage_regime": all(
            float(row["minimum_leakage"]) >= args.leakage_rate * (1.0 - 1e-8)
            and float(row["maximum_absolute_delta"]) <= 1e-8
            for row in leakage_summaries
        ),
        "balanced_regime": all(
            abs(float(row["minimum_leakage"])) <= 1e-11
            and -1.08 <= float(row["slope"]) <= -0.95
            for row in balanced_summaries
        ),
        "logarithmic_scaling": all(
            0.9 / args.leakage_rate <= float(row["slope"]) <= 1.1 / args.leakage_rate
            for row in leakage_summaries
        ),
    }
    print(
        {
            "n_values": n_values,
            "pointwise_cases": len(pointwise_rows),
            "recovery_cases": len(recovery_rows),
            "maximum_pointwise_error": maximum_pointwise_error,
            "maximum_relative_time_error": maximum_time_error,
            "gates": gates,
        }
    )
    if not all(gates.values()):
        raise RuntimeError("Multi-expert normal-velocity experiment failed one or more checks.")


if __name__ == "__main__":
    main()
