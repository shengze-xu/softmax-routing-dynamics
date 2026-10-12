#!/usr/bin/env python3
"""Many-initialization basin audit for the canonical three-expert geometry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_subdir, write_csv  # noqa: E402
from experiments.studies.face_energy_barrier import SCENARIOS, enumerate_stationary_supports  # noqa: E402


def row_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def rhs(logits: np.ndarray, experts: np.ndarray, target: np.ndarray) -> np.ndarray:
    probability = row_softmax(logits)
    residual = probability @ experts - target[None, :]
    score = residual @ experts.T
    jacobian_score = probability * (
        score - np.sum(probability * score, axis=1, keepdims=True)
    )
    return -jacobian_score


def rk4_step(
    logits: np.ndarray,
    experts: np.ndarray,
    target: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = rhs(logits, experts, target)
    k2 = rhs(logits + 0.5 * dt * k1, experts, target)
    k3 = rhs(logits + 0.5 * dt * k2, experts, target)
    k4 = rhs(logits + dt * k3, experts, target)
    return logits + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initializations", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.initializations <= 0 or args.dt <= 0.0 or args.horizon <= 0.0:
        raise ValueError("initializations, dt, and horizon must be positive")

    scenario = SCENARIOS[0]
    _, global_output, _, barrier = enumerate_stationary_supports(scenario)
    rng = np.random.default_rng(args.seed)
    initial = rng.dirichlet(np.full(3, 0.7), size=args.initializations)
    logits = np.log(initial)
    logits -= np.mean(logits, axis=1, keepdims=True)
    initial_output = initial @ scenario.experts
    initial_loss = np.sum((initial_output - scenario.target[None, :]) ** 2, axis=1)
    entry_time = np.full(args.initializations, np.nan)
    entry_time[initial_loss < barrier] = 0.0

    steps = int(round(args.horizon / args.dt))
    for step in range(1, steps + 1):
        logits = rk4_step(logits, scenario.experts, scenario.target, args.dt)
        probability = row_softmax(logits)
        output = probability @ scenario.experts
        loss = np.sum((output - scenario.target[None, :]) ** 2, axis=1)
        newly_entered = np.isnan(entry_time) & (loss < barrier - 1e-10)
        entry_time[newly_entered] = step * args.dt

    final_probability = row_softmax(logits)
    final_output = final_probability @ scenario.experts
    outside_mass = final_probability[:, 2]
    projection_error = np.linalg.norm(final_output - global_output[None, :], axis=1)
    face_selected = (outside_mass <= 2e-2) & (projection_error <= 1e-2)

    refined_dt = args.dt / 2.0
    refined_logits = np.log(initial)
    refined_logits -= np.mean(refined_logits, axis=1, keepdims=True)
    refined_steps = int(round(args.horizon / refined_dt))
    for _ in range(refined_steps):
        refined_logits = rk4_step(
            refined_logits,
            scenario.experts,
            scenario.target,
            refined_dt,
        )
    refined_probability = row_softmax(refined_logits)
    refined_output = refined_probability @ scenario.experts
    refined_outside_mass = refined_probability[:, 2]
    refined_projection_error = np.linalg.norm(
        refined_output - global_output[None, :], axis=1
    )
    refined_face_selected = (
        (refined_outside_mass <= 2e-2) & (refined_projection_error <= 1e-2)
    )
    refinement_error = np.linalg.norm(
        final_probability - refined_probability,
        axis=1,
    )
    rows = []
    for index in range(args.initializations):
        initially_certified = initial_loss[index] < barrier
        if initially_certified and face_selected[index]:
            outcome = "certified_at_initialization"
        elif np.isfinite(entry_time[index]) and face_selected[index]:
            outcome = "entered_later_and_selected"
        elif face_selected[index]:
            outcome = "selected_without_sampled_entry"
        else:
            outcome = "not_selected_by_horizon"
        rows.append(
            [
                index,
                initial[index, 0],
                initial[index, 1],
                initial[index, 2],
                initial_loss[index],
                barrier,
                entry_time[index],
                outside_mass[index],
                projection_error[index],
                bool(face_selected[index]),
                bool(refined_face_selected[index]),
                refinement_error[index],
                outcome,
                args.dt,
                refined_dt,
                args.horizon,
            ]
        )
    write_csv(
        ensure_subdir("tables") / "face_basin_map.csv",
        [
            "initialization",
            "initial_p1",
            "initial_p2",
            "initial_p3",
            "initial_loss",
            "barrier",
            "entry_time",
            "terminal_outside_mass",
            "terminal_projection_error",
            "face_selected",
            "refined_face_selected",
            "terminal_probability_refinement_error",
            "outcome",
            "dt",
            "refined_dt",
            "horizon",
        ],
        rows,
    )
    print(
        f"initially certified = {sum(initial_loss < barrier)}/{args.initializations}; "
        f"entered by horizon = {sum(np.isfinite(entry_time))}/{args.initializations}; "
        f"selected = {sum(face_selected)}/{args.initializations}; "
        f"classification agreement = {sum(face_selected == refined_face_selected)}/{args.initializations}; "
        f"max refinement error = {np.max(refinement_error):.3e}"
    )


if __name__ == "__main__":
    main()
