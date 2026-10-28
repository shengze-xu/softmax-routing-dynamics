#!/usr/bin/env python3
"""Integrate constructed softmax fields with Gamma=x**alpha.

Use desingularized logit time and compare hitting times with quadrature.
For alpha<1, the constructed scores are unbounded near zero."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import softmax

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPSILONS = tuple(10.0**(-k) for k in range(12, 2, -1))


def scaled_reference_time(epsilon: float, theta: float, alpha: float) -> float:
    """epsilon**alpha times the separated-variable integral, in log mass."""
    return quad(lambda z: np.exp(-alpha*z)/(1-epsilon*np.exp(z))**2,
                0.0, np.log(theta/epsilon), epsabs=1e-12, epsrel=1e-12)[0]


def integrate(n: int, epsilon: float, alpha: float, *, theta: float = 0.05,
              max_step: float = 0.5) -> dict:
    if n < 2 or not 0 < epsilon < theta < 1 or not np.isfinite(alpha) or alpha < 0 or max_step <= 0:
        raise ValueError("Require n>=2, 0<epsilon<theta<1, finite alpha>=0, max_step>0")
    kappa = 1 + 1/(n-1)
    p0 = np.full(n, (1-epsilon)/(n-1))
    p0[0] = epsilon
    initial = np.r_[np.log(p0), 0.0]

    def rhs(s, state):
        p = softmax(state[:-1])
        # -x**(-alpha) J(p)g, evaluated without the cancelling powers of x.
        q_dot = -p/kappa
        q_dot[0] += 1/kappa
        return np.r_[q_dot, (epsilon/p[0])**alpha]

    def event(s, state):
        return softmax(state[:-1])[0] - theta
    event.terminal = True
    event.direction = 1
    # dx/ds=x(1-x)^2 >= x(1-theta)^2 before the event.
    horizon = 1.05*np.log(theta/epsilon)/(1-theta)**2
    sol = solve_ivp(rhs, (0, horizon), initial, method="DOP853", events=event,
                    rtol=1e-11, atol=1e-13, max_step=max_step)
    if not sol.success or len(sol.t_events[0]) != 1:
        raise RuntimeError("The controlled flow failed to reach the threshold")
    tau = float(sol.y_events[0][0, -1])
    reference = scaled_reference_time(epsilon, theta, alpha)
    probs = softmax(sol.y[:-1].T, axis=1)
    return dict(n_experts=n, alpha=alpha, epsilon=epsilon, theta=theta,
                max_step_s=max_step, observed_time=tau/epsilon**alpha,
                quadrature_time=reference/epsilon**alpha,
                relative_time_error=abs(tau-reference)/reference,
                maximum_simplex_error=float(np.max(np.abs(probs.sum(axis=1)-1))),
                maximum_active_weight_spread=float(np.max(np.ptp(probs[:, 1:], axis=1))),
                terminal_mass=float(softmax(sol.y_events[0][0, :-1])[0]))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [integrate(n, eps, alpha, max_step=step)
            for n in (4, 8) for alpha in ALPHAS for eps in EPSILONS for step in (0.5, 0.25)]
    fits = []
    for n in (4, 8):
        for alpha in ALPHAS:
            for epsilon_max in (1e-6, 1e-9):
                selected = [r for r in rows if r['n_experts'] == n and r['alpha'] == alpha
                            and r['max_step_s'] == 0.25 and r['epsilon'] <= epsilon_max]
                x = np.log([r['epsilon'] for r in selected])
                times = np.array([r['observed_time'] for r in selected])
                # alpha=0 is fitted against log(1/epsilon), not as a power law.
                abscissa, ordinate = (-x, times) if alpha == 0 else (x, np.log(times))
                slope, intercept = np.polyfit(abscissa, ordinate, 1)
                fitted = slope*abscissa + intercept
                fits.append(dict(n_experts=n, alpha=alpha, epsilon_min=min(r['epsilon'] for r in selected),
                                 epsilon_max=epsilon_max, points=len(selected),
                                 fit_kind="time_vs_log_inverse_mass" if alpha == 0 else "log_time_vs_log_mass",
                                 fitted_slope=float(slope), fitted_exponent="" if alpha == 0 else float(-slope),
                                 r_squared=float(1-np.sum((ordinate-fitted)**2)/np.sum((ordinate-ordinate.mean())**2))))
    write_rows(args.output_dir/'mobility_exponent_recovery.csv', rows)
    write_rows(args.output_dir/'mobility_exponent_fits.csv', fits)
    keyed = {(r['n_experts'],r['alpha'],r['epsilon'],r['max_step_s']):r for r in rows}
    refinement = max(abs(r['observed_time']-keyed[(r['n_experts'],r['alpha'],r['epsilon'],0.5)]['observed_time'])/r['observed_time']
                     for r in rows if r['max_step_s'] == 0.25)
    summary = dict(experiment="controlled coefficient construction",
                   trajectories=len(rows), maximum_relative_time_error=max(r['relative_time_error'] for r in rows),
                   maximum_relative_refinement_change=refinement,
                   score_scope="g0=-x^(alpha-1)/(1+1/(n-1)); unbounded at zero for alpha<1; not fixed square loss",
                   solver="DOP853 in ds=x^alpha dt; rtol=1e-11, atol=1e-13; max_step=0.5,0.25",
                   fit_window="Both fixed descriptive windows [1e-12,1e-6] and [1e-12,1e-9] are reported; main figure uses the latter.")
    (args.output_dir/'mobility_exponent_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    for fit in fits:
        if fit['n_experts'] == 4 and fit['epsilon_max'] == 1e-9:
            print(fit)


if __name__ == '__main__':
    main()
