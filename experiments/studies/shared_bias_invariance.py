#!/usr/bin/env python3
"""Conditioned-logit control for the shared additive-bias invariant."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import ensure_subdir, write_csv


Array = np.ndarray


def row_softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def specialization(probabilities: Array) -> float:
    mean_row = np.mean(probabilities, axis=0, keepdims=True)
    return float(np.mean(np.sum(np.abs(probabilities - mean_row), axis=1)))


def pairwise_row_distance(probabilities: Array) -> float:
    pairs = list(itertools.combinations(range(probabilities.shape[0]), 2))
    return float(np.mean([np.sum(np.abs(probabilities[first] - probabilities[second])) for first, second in pairs]))


def contrast_tensor(probabilities: Array) -> Array:
    log_probability = np.log(probabilities)
    within_regime = log_probability[:, :, None] - log_probability[:, None, :]
    return within_regime[:, None, :, :] - within_regime[None, :, :, :]


def topk_support(probabilities: Array, k: int = 2) -> Array:
    return np.sort(np.argpartition(probabilities, -k, axis=1)[:, -k:], axis=1)


def run_case(seed: int, bias_scale: float, regimes: int, experts: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    logits = 0.65 * rng.standard_normal((regimes, experts))
    for regime in range(regimes):
        logits[regime, (2 * regime) % experts] += 2.2
        logits[regime, (2 * regime + 1) % experts] += 0.8
    direction = np.linspace(-1.0, 1.0, experts) + 0.25 * rng.standard_normal(experts)
    direction -= np.mean(direction)
    direction /= np.std(direction)
    bias = bias_scale * direction

    baseline = row_softmax(logits)
    corrected = row_softmax(logits + bias[None, :])
    baseline_contrast = contrast_tensor(baseline)
    corrected_contrast = contrast_tensor(corrected)
    contrast_error = float(np.max(np.abs(corrected_contrast - baseline_contrast)))
    baseline_specialization = specialization(baseline)
    corrected_specialization = specialization(corrected)
    baseline_pair_distance = pairwise_row_distance(baseline)
    corrected_pair_distance = pairwise_row_distance(corrected)
    baseline_load = np.mean(baseline, axis=0)
    corrected_load = np.mean(corrected, axis=0)
    support_changes = int(np.sum(np.any(topk_support(baseline) != topk_support(corrected), axis=1)))
    return {
        "seed": seed,
        "regimes": regimes,
        "experts": experts,
        "bias_scale": bias_scale,
        "max_cross_regime_log_odds_contrast_error": contrast_error,
        "baseline_specialization": baseline_specialization,
        "corrected_specialization": corrected_specialization,
        "specialization_change": corrected_specialization - baseline_specialization,
        "baseline_pairwise_row_distance": baseline_pair_distance,
        "corrected_pairwise_row_distance": corrected_pair_distance,
        "pairwise_row_distance_change": corrected_pair_distance - baseline_pair_distance,
        "aggregate_load_l1_change": float(np.sum(np.abs(corrected_load - baseline_load))),
        "top2_changed_regimes": support_changes,
        "invariance_gate": bool(contrast_error <= 2e-14),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check shared-bias log-odds invariance and probability-space changes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--bias-scales", type=float, nargs="+", default=[0.25, 0.75, 1.50])
    parser.add_argument("--regimes", type=int, default=4)
    parser.add_argument("--experts", type=int, default=6)
    args = parser.parse_args()
    if args.regimes < 2 or args.experts < 2:
        raise SystemExit("At least two regimes and experts are required.")
    if any(scale <= 0.0 for scale in args.bias_scales):
        raise SystemExit("Bias scales must be positive.")

    rows = [
        run_case(seed, scale, args.regimes, args.experts)
        for seed in args.seeds
        for scale in sorted(set(args.bias_scales))
    ]
    header = [
        "seed",
        "regimes",
        "experts",
        "bias_scale",
        "max_cross_regime_log_odds_contrast_error",
        "baseline_specialization",
        "corrected_specialization",
        "specialization_change",
        "baseline_pairwise_row_distance",
        "corrected_pairwise_row_distance",
        "pairwise_row_distance_change",
        "aggregate_load_l1_change",
        "top2_changed_regimes",
        "invariance_gate",
    ]
    write_csv(
        ensure_subdir("tables") / "shared_bias_invariance.csv",
        header,
        [[row[key] for key in header] for row in rows],
    )
    if not all(bool(row["invariance_gate"]) for row in rows):
        raise RuntimeError("At least one shared-bias invariant gate failed.")
    if not all(float(row["aggregate_load_l1_change"]) > 1e-3 for row in rows):
        raise RuntimeError("A shared-bias case did not measurably alter aggregate load.")
    print(f"shared-bias invariant gates = {len(rows)}/{len(rows)}")
    print(
        "maximum contrast error = "
        f"{max(float(row['max_cross_regime_log_odds_contrast_error']) for row in rows):.6e}"
    )
    print(
        "aggregate-load L1 change range = "
        f"[{min(float(row['aggregate_load_l1_change']) for row in rows):.6f}, "
        f"{max(float(row['aggregate_load_l1_change']) for row in rows):.6f}]"
    )
    print(f"top-2 support changes = {sum(int(row['top2_changed_regimes']) for row in rows)}")


if __name__ == "__main__":
    main()
