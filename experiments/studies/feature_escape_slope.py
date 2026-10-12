#!/usr/bin/env python3
"""Direct check of the shared-parameter feature-escape slope criterion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_subdir, softmax, write_csv  # noqa: E402


Array = np.ndarray


def probabilities(w_router: Array, bias: Array, features: Array) -> Array:
    return np.stack([softmax(w_router @ feature + bias) for feature in features])


def rhs(
    w_router: Array,
    bias: Array,
    *,
    features: Array,
    experts: Array,
    targets: Array,
    weights: Array,
) -> tuple[Array, Array]:
    grad_w = np.zeros_like(w_router)
    grad_b = np.zeros_like(bias)
    for weight, feature, target in zip(weights, features, targets):
        probability = softmax(w_router @ feature + bias)
        jacobian = np.diag(probability) - np.outer(probability, probability)
        residual = experts.T @ probability - target
        grad_logit = weight * jacobian @ (experts @ residual)
        grad_w += np.outer(grad_logit, feature)
        grad_b += grad_logit
    return -grad_w, -grad_b


def rk4_step(
    w_router: Array,
    bias: Array,
    *,
    dt: float,
    features: Array,
    experts: Array,
    targets: Array,
    weights: Array,
) -> tuple[Array, Array]:
    kwargs = {
        "features": features,
        "experts": experts,
        "targets": targets,
        "weights": weights,
    }
    k1w, k1b = rhs(w_router, bias, **kwargs)
    k2w, k2b = rhs(w_router + 0.5 * dt * k1w, bias + 0.5 * dt * k1b, **kwargs)
    k3w, k3b = rhs(w_router + 0.5 * dt * k2w, bias + 0.5 * dt * k2b, **kwargs)
    k4w, k4b = rhs(w_router + dt * k3w, bias + dt * k3b, **kwargs)
    return (
        w_router + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w),
        bias + (dt / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b),
    )


def pairwise_maximum(rows: Array) -> float:
    return max(
        float(np.linalg.norm(rows[index] - rows[other]))
        for index in range(rows.shape[0])
        for other in range(index + 1, rows.shape[0])
    )


def predicted_slope(
    features: Array,
    experts: Array,
    targets: Array,
    weights: Array,
) -> float:
    probability = np.full(experts.shape[0], 1.0 / experts.shape[0])
    jacobian = np.diag(probability) - np.outer(probability, probability)
    mobility = jacobian @ jacobian
    coupling = np.zeros((experts.shape[0], features.shape[1]))
    for weight, feature, target in zip(weights, features, targets):
        residual = experts.T @ probability - target
        coupling += weight * np.outer(mobility @ (experts @ residual), feature)
    return max(
        float(np.linalg.norm(coupling @ (features[index] - features[other])))
        for index in range(features.shape[0])
        for other in range(index + 1, features.shape[0])
    )


def instance(seed: int, *, null_control: bool, dt: float) -> list[object]:
    rng = np.random.default_rng(seed)
    regimes, experts_count, output_dim, feature_dim = 4, 6, 3, 5
    features = rng.standard_normal((regimes, feature_dim))
    features -= np.mean(features, axis=0, keepdims=True)
    experts = rng.standard_normal((experts_count, output_dim))
    if null_control:
        shared_target = rng.standard_normal(output_dim)
        targets = np.repeat(shared_target[None, :], regimes, axis=0)
    else:
        targets = rng.standard_normal((regimes, output_dim))
    weights = np.full(regimes, 1.0 / regimes)
    w_router = np.zeros((experts_count, feature_dim))
    bias = np.zeros(experts_count)
    slope = predicted_slope(features, experts, targets, weights)
    initial_rows = probabilities(w_router, bias, features)
    w_next, b_next = rk4_step(
        w_router,
        bias,
        dt=dt,
        features=features,
        experts=experts,
        targets=targets,
        weights=weights,
    )
    final_rows = probabilities(w_next, b_next, features)
    observed = pairwise_maximum(final_rows) / dt
    initial_difference = pairwise_maximum(initial_rows)
    absolute_error = abs(observed - slope)
    relative_error = absolute_error / max(slope, 1e-15)
    return [
        "nullspace_control" if null_control else "random_geometry",
        seed,
        regimes,
        experts_count,
        output_dim,
        feature_dim,
        dt,
        slope,
        observed,
        absolute_error,
        relative_error,
        initial_difference,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seeds", type=int, default=64)
    parser.add_argument("--control-seeds", type=int, default=16)
    parser.add_argument("--dt", type=float, default=1e-4)
    args = parser.parse_args()
    if args.random_seeds <= 0 or args.control_seeds <= 0 or args.dt <= 0.0:
        raise ValueError("seed counts and dt must be positive")

    rows = [
        instance(seed, null_control=False, dt=args.dt)
        for seed in range(args.random_seeds)
    ]
    rows.extend(
        instance(10_000 + seed, null_control=True, dt=args.dt)
        for seed in range(args.control_seeds)
    )
    header = [
        "geometry",
        "seed",
        "regimes",
        "experts",
        "output_dimension",
        "feature_dimension",
        "dt",
        "chi_feat",
        "observed_early_escape_slope",
        "absolute_error",
        "relative_error",
        "initial_row_difference",
    ]
    write_csv(ensure_subdir("tables") / "feature_escape_slope.csv", header, rows)

    random_rows = [row for row in rows if row[0] == "random_geometry"]
    control_rows = [row for row in rows if row[0] == "nullspace_control"]
    print(
        "random-geometry median relative error = "
        f"{np.median([float(row[10]) for row in random_rows]):.3e}"
    )
    print(
        "nullspace-control maximum observed slope = "
        f"{max(float(row[8]) for row in control_rows):.3e}"
    )


if __name__ == "__main__":
    main()
