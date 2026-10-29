from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_output_dir, ensure_subdir, softmax, write_csv


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    d: int
    eta: float
    q_scale: float
    e_scale: float
    y_scale: float


SCENARIOS = (
    Scenario("slow_three_expert", 3, 2, 0.10, 0.8, 1.0, 0.7),
    Scenario("moderate_five_expert", 5, 4, 0.50, 1.5, 2.0, 1.0),
    Scenario("skewed_eight_expert", 8, 6, 1.00, 2.5, 3.0, 1.5),
)


def rhs(q: np.ndarray, e: np.ndarray, y: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact joint gradient flow for one-half squared mixture loss."""
    p = softmax(q)
    residual = e.T @ p - y
    jac = np.diag(p) - np.outer(p, p)
    dq = -jac @ (e @ residual)
    de = -eta * p[:, None] * residual[None, :]
    return dq, de


def rk4_step(
    q: np.ndarray,
    e: np.ndarray,
    *,
    y: np.ndarray,
    eta: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    k1q, k1e = rhs(q, e, y, eta)
    k2q, k2e = rhs(q + 0.5 * dt * k1q, e + 0.5 * dt * k1e, y, eta)
    k3q, k3e = rhs(q + 0.5 * dt * k2q, e + 0.5 * dt * k2e, y, eta)
    k4q, k4e = rhs(q + dt * k3q, e + dt * k3e, y, eta)
    return (
        q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q),
        e + (dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
    )


def initial_state(scenario: Scenario, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = scenario.q_scale * rng.standard_normal(scenario.n)
    q -= np.mean(q)
    e = scenario.e_scale * rng.standard_normal((scenario.n, scenario.d))
    y = scenario.y_scale * rng.standard_normal(scenario.d)
    return q, e, y


def simulate(
    scenario: Scenario,
    *,
    seed: int,
    dt: float,
    horizon: float,
    keep_history: bool,
) -> tuple[dict[str, float | bool], dict[str, np.ndarray] | None]:
    q, e, y = initial_state(scenario, seed)
    steps = int(round(horizon / dt))
    horizon = steps * dt
    p = softmax(q)
    initial_probability = p.copy()
    initial_expert_operator_norm = float(np.linalg.norm(e, ord=2))
    residual = e.T @ p - y
    residual0 = float(np.linalg.norm(residual))
    residual_sq0 = residual0**2
    area = 0.0
    q_total_variation = 0.0
    e_total_variation = 0.0
    min_probability = float(np.min(p))
    max_normalized_violation = 0.0
    previous_residual_norm = residual0
    tail_start = int(0.9 * steps)
    q_at_tail_start = q.copy()
    e_at_tail_start = e.copy()

    if keep_history:
        stride = max(1, steps // 1200)
        times: list[float] = [0.0]
        residual_sq_history: list[float] = [residual_sq0]
        envelope_history: list[float] = [residual_sq0]
        probability_history: list[np.ndarray] = [p.copy()]

    for step in range(1, steps + 1):
        q_new, e_new = rk4_step(q, e, y=y, eta=scenario.eta, dt=dt)
        q_total_variation += float(np.linalg.norm(q_new - q))
        e_total_variation += float(np.linalg.norm(e_new - e))
        q, e = q_new, e_new
        p = softmax(q)
        residual = e.T @ p - y
        residual_norm = float(np.linalg.norm(residual))
        area += 0.5 * dt * (previous_residual_norm + residual_norm)
        previous_residual_norm = residual_norm
        min_probability = min(min_probability, float(np.min(p)))
        time = step * dt
        envelope = residual_sq0 * np.exp(-2.0 * scenario.eta * time / scenario.n)
        normalized_violation = (residual_norm**2 - envelope) / max(residual_sq0, 1e-300)
        max_normalized_violation = max(max_normalized_violation, normalized_violation)
        if step == tail_start:
            q_at_tail_start = q.copy()
            e_at_tail_start = e.copy()
        if keep_history and (step % stride == 0 or step == steps):
            times.append(time)
            residual_sq_history.append(residual_norm**2)
            envelope_history.append(envelope)
            probability_history.append(p.copy())

    residual_bound = scenario.n * residual0 / scenario.eta
    b0 = initial_expert_operator_norm + scenario.n * residual0
    c0 = float(scenario.n * b0 * residual0 / scenario.eta)
    log_floor = np.log(initial_probability) - np.logaddexp(
        np.log(initial_probability),
        2.0 * c0 + np.log1p(-initial_probability),
    )
    certified_log10_floor = float(np.min(log_floor) / np.log(10.0))
    observed_log10_minimum = float(np.log10(min_probability))
    result: dict[str, float | bool] = {
        "seed": float(seed),
        "dt": float(dt),
        "horizon": float(horizon),
        "initial_residual_norm": residual0,
        "final_residual_norm": previous_residual_norm,
        "residual_area": float(area),
        "residual_area_bound": float(residual_bound),
        "residual_area_fraction": float(area / max(residual_bound, 1e-300)),
        "max_normalized_envelope_violation": float(max_normalized_violation),
        "envelope_bound_satisfied": bool(max_normalized_violation <= 5e-10),
        "min_probability": float(min_probability),
        "final_min_probability": float(np.min(p)),
        "certified_log10_probability_floor": certified_log10_floor,
        "observed_log10_min_probability": observed_log10_minimum,
        "log10_floor_slack": observed_log10_minimum - certified_log10_floor,
        "q_total_variation": float(q_total_variation),
        "e_total_variation": float(e_total_variation),
        "tail_q_drift": float(np.linalg.norm(q - q_at_tail_start)),
        "tail_e_drift": float(np.linalg.norm(e - e_at_tail_start)),
    }
    state = {
        "q": q.copy(),
        "E": e.copy(),
        "p": p.copy(),
        "y": y.copy(),
    }
    if keep_history:
        state.update(
            {
                "times": np.asarray(times),
                "residual_sq": np.asarray(residual_sq_history),
                "envelope": np.asarray(envelope_history),
                "probabilities": np.asarray(probability_history),
            }
        )
    return result, state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate exact joint gradient flow and measure interior convergence."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--dts", type=float, nargs="+", default=[0.04, 0.02, 0.01])
    parser.add_argument("--horizon", type=float, default=60.0)
    args = parser.parse_args()

    dts = sorted(set(float(value) for value in args.dts), reverse=True)
    if any(value <= 0.0 for value in dts):
        raise SystemExit("All step sizes must be positive.")
    finest_dt = min(dts)
    rows: list[list[object]] = []
    records: list[dict[str, object]] = []
    representative_errors: dict[float, float] = {}

    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in args.seeds:
            by_dt: dict[float, tuple[dict[str, float | bool], dict[str, np.ndarray]]] = {}
            for dt in dts:
                keep_history = scenario_index == 1 and seed == args.seeds[0] and dt == finest_dt
                result, state = simulate(
                    scenario,
                    seed=seed,
                    dt=dt,
                    horizon=args.horizon,
                    keep_history=keep_history,
                )
                assert state is not None
                by_dt[dt] = (result, state)

            reference_p = by_dt[finest_dt][1]["p"]
            reference_e = by_dt[finest_dt][1]["E"]
            for dt in dts:
                result, state = by_dt[dt]
                p_error = float(np.linalg.norm(state["p"] - reference_p))
                e_error = float(np.linalg.norm(state["E"] - reference_e))
                result["final_p_refinement_error"] = p_error
                result["final_e_refinement_error"] = e_error
                if scenario_index == 1 and seed == args.seeds[0]:
                    representative_errors[dt] = p_error
                record = {"scenario": scenario.name, **result}
                records.append(record)
                rows.append(
                    [
                        scenario.name,
                        scenario.n,
                        scenario.d,
                        scenario.eta,
                        seed,
                        dt,
                        result["initial_residual_norm"],
                        result["final_residual_norm"],
                        result["residual_area"],
                        result["residual_area_bound"],
                        result["residual_area_fraction"],
                        result["max_normalized_envelope_violation"],
                        result["envelope_bound_satisfied"],
                        result["min_probability"],
                        result["final_min_probability"],
                        result["certified_log10_probability_floor"],
                        result["observed_log10_min_probability"],
                        result["log10_floor_slack"],
                        result["q_total_variation"],
                        result["e_total_variation"],
                        result["tail_q_drift"],
                        result["tail_e_drift"],
                        p_error,
                        e_error,
                    ]
                )

    header = [
        "scenario",
        "n",
        "d",
        "eta",
        "seed",
        "dt",
        "initial_residual_norm",
        "final_residual_norm",
        "residual_area",
        "residual_area_bound",
        "residual_area_fraction",
        "max_normalized_envelope_violation",
        "envelope_bound_satisfied",
        "min_probability",
        "final_min_probability",
        "certified_log10_probability_floor",
        "observed_log10_min_probability",
        "log10_floor_slack",
        "q_total_variation",
        "e_total_variation",
        "tail_q_drift",
        "tail_e_drift",
        "final_p_refinement_error",
        "final_e_refinement_error",
    ]
    write_csv(ensure_subdir("tables") / "exact_joint_gradient_interior.csv", header, rows)

    out_dir = ensure_output_dir()
    np.savez(
        out_dir / "exact_joint_gradient_interior.npz",
        scenario=np.asarray([record["scenario"] for record in records]),
        seed=np.asarray([record["seed"] for record in records], dtype=int),
        dt=np.asarray([record["dt"] for record in records], dtype=float),
        residual_area_fraction=np.asarray(
            [record["residual_area_fraction"] for record in records], dtype=float
        ),
        max_normalized_envelope_violation=np.asarray(
            [record["max_normalized_envelope_violation"] for record in records], dtype=float
        ),
        final_min_probability=np.asarray(
            [record["final_min_probability"] for record in records], dtype=float
        ),
        tail_q_drift=np.asarray([record["tail_q_drift"] for record in records], dtype=float),
        final_p_refinement_error=np.asarray(
            [record["final_p_refinement_error"] for record in records], dtype=float
        ),
    )

    finest_records = [record for record in records if float(record["dt"]) == finest_dt]
    satisfied = sum(bool(record["envelope_bound_satisfied"]) for record in finest_records)
    print(f"finest-step envelope checks = {satisfied}/{len(finest_records)}")
    print(
        "max residual-area fraction = "
        f"{max(float(record['residual_area_fraction']) for record in finest_records):.6f}"
    )
    print(
        "minimum final routing probability = "
        f"{min(float(record['final_min_probability']) for record in finest_records):.6e}"
    )
    print(
        "maximum final coarse-to-fine routing error = "
        f"{max(float(record['final_p_refinement_error']) for record in records):.6e}"
    )


if __name__ == "__main__":
    main()
