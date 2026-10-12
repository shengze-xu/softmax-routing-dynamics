#!/usr/bin/env python3
"""Finite face-energy barrier audit for fixed-expert router flows."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_subdir, softmax, write_csv


Array = np.ndarray


@dataclass(frozen=True)
class Scenario:
    name: str
    experts: Array
    target: Array
    initial_probability: Array
    minimal_face: tuple[int, ...]


SCENARIOS = (
    Scenario(
        "canonical_triangle",
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([0.35, -0.20]),
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        (0, 1),
    ),
    Scenario(
        "pentagon_edge",
        np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [1.3, 1.0],
                [0.0, 1.5],
                [-1.2, 1.0],
            ]
        ),
        np.array([0.25, -0.50]),
        np.array([0.04, 0.05, 0.25, 0.38, 0.28]),
        (0, 1),
    ),
    Scenario(
        "redundant_hexagon_face",
        np.array(
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.2, 0.9],
                [0.0, 1.5],
                [-1.2, 0.9],
            ]
        ),
        np.array([0.20, -0.40]),
        np.array([0.03, 0.03, 0.04, 0.24, 0.38, 0.28]),
        (0, 1, 2),
    ),
    Scenario(
        "cube_bottom_facet",
        np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        ),
        np.array([0.0, 0.0, -0.40]),
        np.array([0.025, 0.025, 0.025, 0.025, 0.20, 0.22, 0.25, 0.23]),
        (0, 1, 2, 3),
    ),
)


def all_supports(n: int):
    for size in range(1, n + 1):
        yield from itertools.combinations(range(n), size)


def affine_minimizer(experts: Array, target: Array) -> Array:
    """Equality-constrained least-squares minimizer on an affine hull."""
    size = experts.shape[0]
    gram = experts @ experts.T
    system = np.block(
        [
            [gram, np.ones((size, 1))],
            [np.ones((1, size)), np.zeros((1, 1))],
        ]
    )
    rhs = np.concatenate([experts @ target, np.ones(1)])
    solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    return solution[:size]


def project_to_hull(experts: Array, target: Array, *, tolerance: float = 1e-9) -> tuple[Array, float]:
    """Project by enumerating basic convex representations."""
    size, dimension = experts.shape
    best_output: Array | None = None
    best_loss = float("inf")
    for count in range(1, min(size, dimension + 1) + 1):
        for local_support in itertools.combinations(range(size), count):
            weights = affine_minimizer(experts[list(local_support)], target)
            if float(np.min(weights)) < -tolerance:
                continue
            weights = np.maximum(weights, 0.0)
            weights /= np.sum(weights)
            output = experts[list(local_support)].T @ weights
            loss = float(np.sum((output - target) ** 2))
            if loss < best_loss:
                best_output = output
                best_loss = loss
    if best_output is None:
        raise RuntimeError("Convex-hull projection enumeration failed.")
    return best_output, best_loss


def basic_representations(experts: Array, output: Array, *, tolerance: float = 2e-8) -> list[Array]:
    """Return nonnegative basic representations of an output."""
    size, dimension = experts.shape
    result: list[Array] = []
    augmented_target = np.concatenate([output, np.ones(1)])
    for count in range(1, min(size, dimension + 1) + 1):
        for local_support in itertools.combinations(range(size), count):
            matrix = np.vstack([experts[list(local_support)].T, np.ones((1, count))])
            weights, *_ = np.linalg.lstsq(matrix, augmented_target, rcond=None)
            if np.linalg.norm(matrix @ weights - augmented_target) > tolerance:
                continue
            if float(np.min(weights)) < -tolerance:
                continue
            full = np.zeros(size)
            full[list(local_support)] = np.maximum(weights, 0.0)
            if np.sum(full) > 0.0:
                full /= np.sum(full)
                result.append(full)
    return result


def positive_representation(experts: Array, output: Array, *, tolerance: float = 1e-9) -> Array | None:
    """Construct a strictly positive representation when one exists."""
    representations = basic_representations(experts, output)
    selected: list[Array] = []
    for index in range(experts.shape[0]):
        candidates = [weights for weights in representations if weights[index] > tolerance]
        if not candidates:
            return None
        selected.append(max(candidates, key=lambda weights: float(weights[index])))
    weights = np.mean(selected, axis=0)
    weights /= np.sum(weights)
    if float(np.min(weights)) <= tolerance:
        return None
    return weights


def enumerate_stationary_supports(scenario: Scenario) -> tuple[list[dict[str, object]], Array, float, float]:
    experts = scenario.experts
    target = scenario.target
    global_output, global_loss = project_to_hull(experts, target)
    records: list[dict[str, object]] = []
    wrong_losses: list[float] = []
    for support in all_supports(experts.shape[0]):
        local_experts = experts[list(support)]
        output, support_loss = project_to_hull(local_experts, target)
        positive = positive_representation(local_experts, output)
        stationary = positive is not None
        globally_optimal = bool(np.linalg.norm(output - global_output) <= 2e-8)
        wrong = stationary and not globally_optimal
        if wrong:
            wrong_losses.append(support_loss)
        records.append(
            {
                "support": support,
                "output": output,
                "loss": support_loss,
                "stationary": stationary,
                "globally_optimal": globally_optimal,
                "wrong": wrong,
                "positive_weights": positive,
            }
        )
    barrier = min(wrong_losses, default=float("inf"))
    return records, global_output, global_loss, barrier


def rhs_logits(q: Array, experts: Array, target: Array) -> Array:
    probability = softmax(q)
    jacobian = np.diag(probability) - np.outer(probability, probability)
    return -jacobian @ (experts @ (experts.T @ probability - target))


def rk4_step(q: Array, experts: Array, target: Array, dt: float) -> Array:
    k1 = rhs_logits(q, experts, target)
    k2 = rhs_logits(q + 0.5 * dt * k1, experts, target)
    k3 = rhs_logits(q + 0.5 * dt * k2, experts, target)
    k4 = rhs_logits(q + dt * k3, experts, target)
    return q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    scenario: Scenario,
    *,
    dt: float,
    horizon: float,
    active_support: tuple[int, ...] | None = None,
    active_probability: Array | None = None,
) -> tuple[dict[str, object], list[list[object]]]:
    records, global_output, global_loss, barrier = enumerate_stationary_supports(scenario)
    if active_support is None:
        active_support = tuple(range(scenario.experts.shape[0]))
        initial = scenario.initial_probability.copy()
        run_type = "interior"
    else:
        if active_probability is None:
            raise ValueError("Boundary runs require active probabilities.")
        initial = np.asarray(active_probability, dtype=float)
        run_type = "wrong_boundary_control"
    active_experts = scenario.experts[list(active_support)]
    q = np.log(initial)
    q -= np.mean(q)
    steps = int(round(horizon / dt))
    stride = max(1, steps // 400)
    minimal_face = set(scenario.minimal_face)
    previous_loss: float | None = None
    max_increase = 0.0
    entry_time = float("nan")
    minimum_loss = float("inf")
    trajectory: list[list[object]] = []

    for step in range(steps + 1):
        time = step * dt
        active_p = softmax(q)
        full_p = np.zeros(scenario.experts.shape[0])
        full_p[list(active_support)] = active_p
        output = scenario.experts.T @ full_p
        current_loss = float(np.sum((output - scenario.target) ** 2))
        outside_mass = float(sum(full_p[index] for index in range(full_p.size) if index not in minimal_face))
        if previous_loss is not None:
            max_increase = max(max_increase, current_loss - previous_loss)
        previous_loss = current_loss
        minimum_loss = min(minimum_loss, current_loss)
        if np.isnan(entry_time) and current_loss < barrier - 1e-10:
            entry_time = time
        if step % stride == 0 or step == steps:
            trajectory.append(
                [
                    scenario.name,
                    run_type,
                    time,
                    current_loss,
                    barrier,
                    barrier - current_loss,
                    outside_mass,
                    float(np.linalg.norm(output - global_output)),
                ]
            )
        if step < steps:
            q = rk4_step(q, active_experts, scenario.target, dt)

    final_p = np.zeros(scenario.experts.shape[0])
    final_p[list(active_support)] = softmax(q)
    final_output = scenario.experts.T @ final_p
    final_loss = float(np.sum((final_output - scenario.target) ** 2))
    # Reconstruct the initial full probability without coupling the summary to
    # a particular support ordering.
    initial_full = np.zeros(scenario.experts.shape[0])
    initial_full[list(active_support)] = initial
    initial_loss = float(np.sum((scenario.experts.T @ initial_full - scenario.target) ** 2))
    final_outside = float(
        sum(final_p[index] for index in range(final_p.size) if index not in minimal_face)
    )
    summary = {
        "scenario": scenario.name,
        "run_type": run_type,
        "n": scenario.experts.shape[0],
        "d": scenario.experts.shape[1],
        "minimal_face": " ".join(str(index + 1) for index in scenario.minimal_face),
        "active_support": " ".join(str(index + 1) for index in active_support),
        "stationary_support_count": sum(bool(record["stationary"]) for record in records),
        "wrong_stationary_support_count": sum(bool(record["wrong"]) for record in records),
        "global_loss": global_loss,
        "barrier": barrier,
        "initial_loss": initial_loss,
        "entry_time": entry_time,
        "minimum_loss": minimum_loss,
        "terminal_loss": final_loss,
        "terminal_barrier_margin": barrier - final_loss,
        "terminal_outside_mass": final_outside,
        "terminal_projection_error": float(np.linalg.norm(final_output - global_output)),
        "max_loss_increase": max_increase,
        "barrier_crossed": bool(np.isfinite(entry_time)),
        "face_gate": bool(final_outside <= 2e-2 and np.linalg.norm(final_output - global_output) <= 1e-2),
    }
    return summary, trajectory


def boundary_control(scenario: Scenario) -> tuple[tuple[int, ...], Array]:
    records, _, _, _ = enumerate_stationary_supports(scenario)
    minimal_face = set(scenario.minimal_face)
    candidates = [
        record
        for record in records
        if record["wrong"] and any(index not in minimal_face for index in record["support"])
    ]
    if not candidates:
        candidates = [record for record in records if record["wrong"]]
    best_loss = min(float(record["loss"]) for record in candidates)
    tied = [
        record
        for record in candidates
        if abs(float(record["loss"]) - best_loss) <= 1e-12
    ]
    # Symmetric faces can have the same loss up to BLAS-level roundoff.  Use a
    # stable support-order tie break so the named control is platform invariant.
    selected = min(tied, key=lambda record: tuple(record["support"]))
    weights = selected["positive_weights"]
    assert isinstance(weights, np.ndarray)
    return tuple(selected["support"]), weights




def main() -> None:
    parser = argparse.ArgumentParser(description="Audit finite stationary face-energy levels and barrier entry.")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--horizon", type=float, default=2000.0)
    args = parser.parse_args()
    if args.dt <= 0.0 or args.horizon <= 0.0:
        raise SystemExit("Time parameters must be positive.")

    summary: list[dict[str, object]] = []
    trajectories: list[list[object]] = []
    support_rows: list[list[object]] = []
    for scenario in SCENARIOS:
        records, global_output, _, _ = enumerate_stationary_supports(scenario)
        for record in records:
            if not record["stationary"]:
                continue
            support_rows.append(
                [
                    scenario.name,
                    " ".join(str(index + 1) for index in record["support"]),
                    len(record["support"]),
                    record["loss"],
                    record["globally_optimal"],
                    record["wrong"],
                    float(np.linalg.norm(np.asarray(record["output"]) - global_output)),
                ]
            )
        result, path = simulate(scenario, dt=args.dt, horizon=args.horizon)
        summary.append(result)
        trajectories.extend(path)

    control_scenario = SCENARIOS[-1]
    active_support, active_probability = boundary_control(control_scenario)
    control, control_path = simulate(
        control_scenario,
        dt=args.dt,
        horizon=min(args.horizon, 200.0),
        active_support=active_support,
        active_probability=active_probability,
    )
    summary.append(control)
    trajectories.extend(control_path)

    summary_header = [
        "scenario",
        "run_type",
        "n",
        "d",
        "minimal_face",
        "active_support",
        "stationary_support_count",
        "wrong_stationary_support_count",
        "global_loss",
        "barrier",
        "initial_loss",
        "entry_time",
        "minimum_loss",
        "terminal_loss",
        "terminal_barrier_margin",
        "terminal_outside_mass",
        "terminal_projection_error",
        "max_loss_increase",
        "barrier_crossed",
        "face_gate",
    ]
    write_csv(
        ensure_subdir("tables") / "face_energy_barrier_summary.csv",
        summary_header,
        [[record[key] for key in summary_header] for record in summary],
    )
    write_csv(
        ensure_subdir("tables") / "face_energy_stationary_supports.csv",
        ["scenario", "support", "support_size", "loss", "globally_optimal", "wrong", "projection_output_error"],
        support_rows,
    )
    write_csv(
        ensure_subdir("tables") / "face_energy_trajectories.csv",
        ["scenario", "run_type", "time", "loss", "barrier", "barrier_margin", "outside_face_mass", "projection_output_error"],
        trajectories,
    )

    interior = [record for record in summary if record["run_type"] == "interior"]
    if not all(bool(record["barrier_crossed"]) and bool(record["face_gate"]) for record in interior):
        raise RuntimeError("At least one interior face-energy gate failed.")
    if bool(control["barrier_crossed"]) or bool(control["face_gate"]):
        raise RuntimeError("Wrong-boundary control did not remain excluded.")
    print(f"interior barrier/face gates = {len(interior)}/{len(interior)}")
    print(f"minimum terminal barrier margin = {min(float(record['terminal_barrier_margin']) for record in interior):.6e}")
    print(f"maximum terminal outside-face mass = {max(float(record['terminal_outside_mass']) for record in interior):.6e}")
    print(f"wrong-boundary control support = {control['active_support']}")



if __name__ == "__main__":
    main()
