#!/usr/bin/env python3
"""Finite-time fast--slow approximation and mobility-dependent recovery scaling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_output_dir, ensure_subdir, softmax, write_csv


Array = np.ndarray


def canonical_state() -> tuple[Array, Array, Array]:
    q0 = np.zeros(3, dtype=float)
    e0 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0.35, -0.20], dtype=float)
    return q0, e0, y


def rhs(q: Array, e: Array, y: Array, eta: float) -> tuple[Array, Array]:
    p = softmax(q)
    residual = e.T @ p - y
    jac = np.diag(p) - np.outer(p, p)
    return -jac @ (e @ residual), -eta * p[:, None] * residual[None, :]


def rk4_step(q: Array, e: Array, y: Array, eta: float, dt: float) -> tuple[Array, Array]:
    k1q, k1e = rhs(q, e, y, eta)
    k2q, k2e = rhs(q + 0.5 * dt * k1q, e + 0.5 * dt * k1e, y, eta)
    k3q, k3e = rhs(q + 0.5 * dt * k2q, e + 0.5 * dt * k2e, y, eta)
    k4q, k4e = rhs(q + dt * k3q, e + dt * k3e, y, eta)
    return (
        q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q),
        e + (dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
    )


def simulate(eta: float, *, dt: float, horizon: float) -> dict[str, Array]:
    q, e, y = canonical_state()
    e_initial = e.copy()
    steps = int(round(horizon / dt))
    times = np.arange(steps + 1, dtype=float) * dt
    probabilities = np.empty((steps + 1, q.size), dtype=float)
    expert_drift = np.empty(steps + 1, dtype=float)
    residual_norm = np.empty(steps + 1, dtype=float)
    probabilities[0] = softmax(q)
    expert_drift[0] = 0.0
    residual_norm[0] = np.linalg.norm(e.T @ probabilities[0] - y)
    for step in range(1, steps + 1):
        q, e = rk4_step(q, e, y, eta, dt)
        probabilities[step] = softmax(q)
        expert_drift[step] = np.linalg.norm(e - e_initial)
        residual_norm[step] = np.linalg.norm(e.T @ probabilities[step] - y)
    return {
        "times": times,
        "probabilities": probabilities,
        "expert_drift": expert_drift,
        "residual_norm": residual_norm,
    }


def first_hit(times: Array, values: Array, threshold: float) -> float:
    indices = np.flatnonzero(values <= threshold)
    return float(times[indices[0]]) if indices.size else float("nan")


def recovery_integral_j2(epsilon: Array, theta: float) -> Array:
    def antiderivative(p: Array) -> Array:
        return -1.0 / p + 1.0 / (1.0 - p) + 2.0 * np.log(p / (1.0 - p))

    return 0.5 * (antiderivative(np.full_like(epsilon, theta)) - antiderivative(epsilon))


def recovery_integral_j(epsilon: Array, theta: float) -> Array:
    return np.log(theta / (1.0 - theta)) - np.log(epsilon / (1.0 - epsilon))


def recovery_integral_projected(epsilon: Array, theta: float) -> Array:
    return 2.0 * (theta - epsilon)


def fit_loglog_slope(x: Array, y: Array, count: int = 6) -> float:
    selected_x = np.log(x[:count])
    selected_y = np.log(y[:count])
    slope, _ = np.polyfit(selected_x, selected_y, 1)
    return float(slope)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--horizon", type=float, default=2000.0)
    parser.add_argument("--bridge-horizon", type=float, default=40.0)
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.08, 0.16],
    )
    parser.add_argument("--theta", type=float, default=0.25)
    args = parser.parse_args()

    if args.dt <= 0.0 or args.horizon <= 0.0 or args.bridge_horizon <= 0.0:
        raise SystemExit("Time parameters must be positive.")
    etas = sorted(set(float(value) for value in args.etas))
    if not etas or any(value <= 0.0 for value in etas):
        raise SystemExit("Expert rates must be positive.")
    if not 0.0 < args.theta < 1.0:
        raise SystemExit("theta must lie in (0,1).")

    fast_slow = {0.0: simulate(0.0, dt=args.dt, horizon=args.horizon)}
    fast_slow.update(
        {
            eta: simulate(eta, dt=args.dt, horizon=args.horizon)
            for eta in etas
        }
    )
    frozen = fast_slow[0.0]
    bridge_stop = int(round(args.bridge_horizon / args.dt)) + 1
    thresholds = [0.10, 0.05, 0.02, 0.01]
    bridge_errors: dict[float, float] = {}
    fast_rows: list[list[object]] = []
    for eta in etas:
        history = fast_slow[eta]
        error = np.linalg.norm(
            history["probabilities"][:bridge_stop] - frozen["probabilities"][:bridge_stop],
            axis=1,
        )
        bridge_errors[eta] = float(np.max(error))
        p3 = history["probabilities"][:, 2]
        fast_rows.append(
            [
                eta,
                args.dt,
                args.horizon,
                args.bridge_horizon,
                bridge_errors[eta],
                float(np.max(history["expert_drift"][:bridge_stop])),
                float(np.min(p3)),
                float(p3[-1]),
                float(history["residual_norm"][-1]),
                *[
                    "" if np.isnan(value) else value
                    for value in (
                        first_hit(history["times"], p3, threshold)
                        for threshold in thresholds
                    )
                ],
            ]
        )

    tables = ensure_subdir("tables")
    write_csv(
        tables / "fast_slow_bridge.csv",
        [
            "eta",
            "dt",
            "horizon",
            "bridge_horizon",
            "sup_probability_error_vs_frozen",
            "max_expert_drift_bridge",
            "minimum_outside_face_mass",
            "terminal_outside_face_mass",
            "terminal_residual_norm",
            "hit_time_0p10",
            "hit_time_0p05",
            "hit_time_0p02",
            "hit_time_0p01",
        ],
        fast_rows,
    )

    trajectory_rows: list[list[object]] = []
    selected_etas = [0.0, 0.01, 0.04, 0.16]
    stride = max(1, int(round(2.0 / args.dt)))
    for eta in selected_etas:
        history = fast_slow[eta]
        for index in range(0, len(history["times"]), stride):
            trajectory_rows.append(
                [
                    eta,
                    float(history["times"][index]),
                    *history["probabilities"][index].tolist(),
                    float(history["expert_drift"][index]),
                    float(history["residual_norm"][index]),
                ]
            )
    write_csv(
        tables / "fast_slow_trajectories.csv",
        [
            "eta",
            "time",
            "p1",
            "p2",
            "p3",
            "expert_drift",
            "residual_norm",
        ],
        trajectory_rows,
    )

    epsilons = np.logspace(-4.0, -1.0, 13)
    epsilons = epsilons[epsilons < args.theta]
    recovery_times = {
        "j2": recovery_integral_j2(epsilons, args.theta),
        "j": recovery_integral_j(epsilons, args.theta),
        "projected": recovery_integral_projected(epsilons, args.theta),
    }
    mobility_rows = [
        [
            float(epsilon),
            args.theta,
            float(recovery_times["j2"][index]),
            float(recovery_times["j"][index]),
            float(recovery_times["projected"][index]),
            float(epsilon * recovery_times["j2"][index]),
            float(recovery_times["j"][index] / np.log(1.0 / epsilon)),
        ]
        for index, epsilon in enumerate(epsilons)
    ]
    write_csv(
        tables / "mobility_recovery_scaling.csv",
        [
            "epsilon",
            "theta",
            "time_j2",
            "time_j",
            "time_projected",
            "epsilon_time_j2",
            "time_j_over_log_inverse_epsilon",
        ],
        mobility_rows,
    )

    summary_rows = [
        ["fast_slow_loglog_slope", fit_loglog_slope(np.asarray(etas), np.asarray([bridge_errors[x] for x in etas]))],
        ["j2_recovery_loglog_slope", fit_loglog_slope(epsilons, recovery_times["j2"])],
        ["j_recovery_loglog_slope", fit_loglog_slope(epsilons, recovery_times["j"])],
        ["projected_recovery_loglog_slope", fit_loglog_slope(epsilons, recovery_times["projected"])],
        ["smallest_epsilon_time_j2_product", float(epsilons[0] * recovery_times["j2"][0])],
        ["smallest_epsilon_j_log_ratio", float(recovery_times["j"][0] / np.log(1.0 / epsilons[0]))],
        ["smallest_epsilon_projected_time", float(recovery_times["projected"][0])],
    ]
    write_csv(
        tables / "timescale_closure_summary.csv",
        ["metric", "value"],
        summary_rows,
    )

    print(
        "fast_slow_slope="
        f"{summary_rows[0][1]:.6f} "
        "j2_slope="
        f"{summary_rows[1][1]:.6f} "
        "j_log_ratio="
        f"{summary_rows[5][1]:.6f} "
        "projected_limit="
        f"{summary_rows[6][1]:.6f}"
    )


    np.savez(
        ensure_output_dir() / "timescale_closure.npz",
        etas=np.asarray(etas),
        bridge_errors=np.asarray([bridge_errors[value] for value in etas]),
        epsilons=epsilons,
        recovery_time_j2=recovery_times["j2"],
        recovery_time_j=recovery_times["j"],
        recovery_time_projected=recovery_times["projected"],
    )


if __name__ == "__main__":
    main()
