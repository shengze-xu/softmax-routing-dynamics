#!/usr/bin/env python3
"""Nonquadratic PL residual-loss experiment for exact joint router/expert flow."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_subdir, softmax, write_csv
from experiments.studies.exact_joint_gradient_interior import SCENARIOS, Scenario, initial_state  # noqa: E402


Array = np.ndarray


def phi(residual: Array, alpha: float) -> float:
    return float(0.5 * residual @ residual + alpha * np.sum(1.0 - np.cos(residual)))


def grad_phi(residual: Array, alpha: float) -> Array:
    return residual + alpha * np.sin(residual)


def rhs(q: Array, experts: Array, target: Array, eta: float, alpha: float) -> tuple[Array, Array]:
    probability = softmax(q)
    residual = experts.T @ probability - target
    gradient = grad_phi(residual, alpha)
    jacobian = np.diag(probability) - np.outer(probability, probability)
    return -jacobian @ (experts @ gradient), -eta * probability[:, None] * gradient[None, :]


def rk4_step(
    q: Array,
    experts: Array,
    *,
    target: Array,
    eta: float,
    alpha: float,
    dt: float,
) -> tuple[Array, Array]:
    k1q, k1e = rhs(q, experts, target, eta, alpha)
    k2q, k2e = rhs(q + 0.5 * dt * k1q, experts + 0.5 * dt * k1e, target, eta, alpha)
    k3q, k3e = rhs(q + 0.5 * dt * k2q, experts + 0.5 * dt * k2e, target, eta, alpha)
    k4q, k4e = rhs(q + dt * k3q, experts + dt * k3e, target, eta, alpha)
    return (
        q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q),
        experts + (dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
    )


def log_probability_floor(initial_probability: Array, constant: float) -> Array:
    log_numerator = np.log(initial_probability)
    log_denominator = np.logaddexp(
        log_numerator,
        2.0 * constant + np.log1p(-initial_probability),
    )
    return log_numerator - log_denominator


def simulate(
    scenario: Scenario,
    *,
    seed: int,
    dt: float,
    horizon: float,
    alpha: float,
    keep_history: bool,
) -> tuple[dict[str, object], dict[str, Array]]:
    q, experts, target = initial_state(scenario, seed)
    initial_experts = experts.copy()
    probability = softmax(q)
    initial_probability = probability.copy()
    residual = experts.T @ probability - target
    initial_phi = phi(residual, alpha)
    mu = 1.0 - alpha
    smoothness = 1.0 + alpha
    gradient_area_bound = (
        scenario.n * math.sqrt(2.0 * smoothness * initial_phi) / (scenario.eta * mu)
    )
    expert_bound = np.linalg.norm(initial_experts, ord=2) + scenario.eta * gradient_area_bound
    q_constant = expert_bound * gradient_area_bound
    log_floor = log_probability_floor(initial_probability, q_constant)

    steps = int(round(horizon / dt))
    gradient_area = 0.0
    previous_gradient_norm = float(np.linalg.norm(grad_phi(residual, alpha)))
    max_normalized_violation = 0.0
    coordinate_minima = probability.copy()
    tail_start = int(0.9 * steps)
    q_tail = q.copy()
    experts_tail = experts.copy()

    if keep_history:
        stride = max(1, steps // 600)
        times = [0.0]
        loss_history = [initial_phi]
        envelope_history = [initial_phi]
        probability_history = [probability.copy()]

    for step in range(1, steps + 1):
        q, experts = rk4_step(
            q,
            experts,
            target=target,
            eta=scenario.eta,
            alpha=alpha,
            dt=dt,
        )
        probability = softmax(q)
        residual = experts.T @ probability - target
        current_phi = phi(residual, alpha)
        gradient_norm = float(np.linalg.norm(grad_phi(residual, alpha)))
        gradient_area += 0.5 * dt * (previous_gradient_norm + gradient_norm)
        previous_gradient_norm = gradient_norm
        coordinate_minima = np.minimum(coordinate_minima, probability)
        time = step * dt
        envelope = initial_phi * math.exp(-2.0 * scenario.eta * mu * time / scenario.n)
        max_normalized_violation = max(
            max_normalized_violation,
            (current_phi - envelope) / max(initial_phi, 1e-300),
        )
        if step == tail_start:
            q_tail = q.copy()
            experts_tail = experts.copy()
        if keep_history and (step % stride == 0 or step == steps):
            times.append(time)
            loss_history.append(current_phi)
            envelope_history.append(envelope)
            probability_history.append(probability.copy())

    log_floor_slack = float(np.min(np.log(coordinate_minima) - log_floor))
    result: dict[str, object] = {
        "seed": seed,
        "dt": dt,
        "horizon": steps * dt,
        "alpha": alpha,
        "mu": mu,
        "smoothness_L": smoothness,
        "initial_phi": initial_phi,
        "final_phi": current_phi,
        "gradient_area": gradient_area,
        "gradient_area_bound": gradient_area_bound,
        "gradient_area_fraction": gradient_area / max(gradient_area_bound, 1e-300),
        "max_normalized_envelope_violation": max_normalized_violation,
        "envelope_bound_satisfied": bool(max_normalized_violation <= 2e-9),
        "minimum_probability": float(np.min(coordinate_minima)),
        "final_min_probability": float(np.min(probability)),
        "minimum_log10_floor_bound": float(np.min(log_floor) / math.log(10.0)),
        "minimum_log_floor_slack": log_floor_slack,
        "floor_bound_satisfied": bool(log_floor_slack >= -2e-9),
        "tail_q_drift": float(np.linalg.norm(q - q_tail)),
        "tail_e_drift": float(np.linalg.norm(experts - experts_tail)),
    }
    state = {
        "q": q.copy(),
        "E": experts.copy(),
        "p": probability.copy(),
        "y": target.copy(),
    }
    if keep_history:
        state.update(
            {
                "times": np.asarray(times),
                "loss": np.asarray(loss_history),
                "envelope": np.asarray(envelope_history),
                "probabilities": np.asarray(probability_history),
            }
        )
    return result, state


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the PL residual theorem with a smooth nonquadratic loss.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--dts", type=float, nargs="+", default=[0.04, 0.02, 0.01])
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("alpha must lie in (0,1) so the loss is strongly convex.")
    dts = sorted(set(float(value) for value in args.dts), reverse=True)
    if any(value <= 0.0 for value in dts) or args.horizon <= 0.0:
        raise SystemExit("Time parameters must be positive.")

    rows: list[dict[str, object]] = []
    representative: dict[str, Array] | None = None
    finest_dt = min(dts)
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in args.seeds:
            by_dt: dict[float, tuple[dict[str, object], dict[str, Array]]] = {}
            for dt in dts:
                keep_history = scenario_index == 1 and seed == args.seeds[0] and dt == finest_dt
                result, state = simulate(
                    scenario,
                    seed=seed,
                    dt=dt,
                    horizon=args.horizon,
                    alpha=args.alpha,
                    keep_history=keep_history,
                )
                by_dt[dt] = (result, state)
                if keep_history:
                    representative = state
            reference_p = by_dt[finest_dt][1]["p"]
            reference_e = by_dt[finest_dt][1]["E"]
            for dt in dts:
                result, state = by_dt[dt]
                result["final_p_refinement_error"] = float(np.linalg.norm(state["p"] - reference_p))
                result["final_e_refinement_error"] = float(np.linalg.norm(state["E"] - reference_e))
                rows.append({"scenario": scenario.name, "n": scenario.n, "d": scenario.d, "eta": scenario.eta, **result})

    header = [
        "scenario",
        "n",
        "d",
        "eta",
        "seed",
        "dt",
        "horizon",
        "alpha",
        "mu",
        "smoothness_L",
        "initial_phi",
        "final_phi",
        "gradient_area",
        "gradient_area_bound",
        "gradient_area_fraction",
        "max_normalized_envelope_violation",
        "envelope_bound_satisfied",
        "minimum_probability",
        "final_min_probability",
        "minimum_log10_floor_bound",
        "minimum_log_floor_slack",
        "floor_bound_satisfied",
        "tail_q_drift",
        "tail_e_drift",
        "final_p_refinement_error",
        "final_e_refinement_error",
    ]
    write_csv(
        ensure_subdir("tables") / "nonquadratic_pl_interior.csv",
        header,
        [[row[key] for key in header] for row in rows],
    )
    if representative is None:
        raise RuntimeError("Representative PL trajectory was not retained.")
    write_csv(
        ensure_subdir("tables") / "nonquadratic_pl_trajectory.csv",
        ["time", "loss", "envelope", *[f"p{index + 1}" for index in range(representative["probabilities"].shape[1])]],
        [
            [
                representative["times"][index],
                representative["loss"][index],
                representative["envelope"][index],
                *representative["probabilities"][index].tolist(),
            ]
            for index in range(representative["times"].size)
        ],
    )
    finest = [row for row in rows if float(row["dt"]) == finest_dt]
    if not all(bool(row["envelope_bound_satisfied"]) and bool(row["floor_bound_satisfied"]) for row in finest):
        raise RuntimeError("At least one finest-step PL theorem gate failed.")
    print(f"finest-step PL envelope/floor gates = {len(finest)}/{len(finest)}")
    print(f"maximum gradient-area fraction = {max(float(row['gradient_area_fraction']) for row in finest):.6f}")
    print(f"minimum routing probability = {min(float(row['minimum_probability']) for row in finest):.6e}")
    print(f"maximum coarse-to-fine routing error = {max(float(row['final_p_refinement_error']) for row in rows):.6e}")


if __name__ == "__main__":
    main()
