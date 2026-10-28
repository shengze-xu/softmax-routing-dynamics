#!/usr/bin/env python3
"""CPU checks for fixed-token router logit decomposition."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis import analyze_recovery_survival as survival
from experiments.studies import agnews_structure_comparison as agnews_model
from experiments.models import agnews_no_drop_recovery as no_drop_recovery
from experiments.common import router_logit_decomposition as decomposition


def layer_state(
    token_ids: list[int],
    features: list[list[float]],
    weight: list[list[float]],
    bias: list[float],
) -> decomposition.ProbeLayerState:
    h = torch.tensor(features, dtype=torch.float64)
    w = torch.tensor(weight, dtype=torch.float64)
    b = torch.tensor(bias, dtype=torch.float64)
    return decomposition.ProbeLayerState(
        token_ids=torch.tensor(token_ids),
        features=h,
        logits=h @ w.T + b,
        weight=w,
        bias=b,
    )


class LogitDecompositionTests(unittest.TestCase):
    def test_fixed_probe_capture_uses_model_router_features(self) -> None:
        dataset = SimpleNamespace(vocab_size=32, sequence_length=4)
        config = {
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 2,
            "expert_hidden": 16,
            "num_experts": 4,
            "precision": "fp32",
        }
        model = no_drop_recovery.make_model(dataset=dataset, config=config, seed=3)
        no_drop_recovery.configure_intervention(
            "none",
            target_by_layer=[0, 0],
            comparator_by_layer=[1, 1],
            middle_layers=[1],
            clamp_margin=3.0,
            control_margin=-1.0,
        )
        data = (
            torch.randint(0, 32, (4, 5)),
            torch.full((4,), 5, dtype=torch.long),
            torch.arange(4, dtype=torch.long),
        )
        before = decomposition.capture_probe_state(
            model,
            data,
            step=0,
            middle_layers=[1],
            target_domain_by_layer=[0, 0],
            tokens_per_layer=3,
            eval_batch_size=2,
            device=torch.device("cpu"),
            precision="fp32",
            target_mask=agnews_model.target_mask,
            autocast_context=no_drop_recovery.autocast_context,
        )
        with torch.no_grad():
            model.layers[1].moe.router.bias[0] += 0.2
            model.token_embedding.weight.add_(0.01)
        after = decomposition.capture_probe_state(
            model,
            data,
            step=64,
            middle_layers=[1],
            target_domain_by_layer=[0, 0],
            tokens_per_layer=3,
            eval_batch_size=2,
            device=torch.device("cpu"),
            precision="fp32",
            target_mask=agnews_model.target_mask,
            autocast_context=no_drop_recovery.autocast_context,
        )
        self.assertTrue(
            torch.equal(before.layers[1].token_ids, after.layers[1].token_ids)
        )
        trajectory = decomposition.ProbeTrajectory(
            before,
            target_by_layer=[0, 0],
            comparator_by_layer=[1, 1],
            threshold=0.05,
            metadata={
                "protocol_id": "test",
                "seed": 3,
                "cohort": "no_drop",
                "capacity_factor": None,
                "branch": "free_release",
                "initial_target_mass": 0.02,
            },
        )
        trajectory.advance(after)
        self.assertLess(
            trajectory.summary()["maximum_logit_reconstruction_error"], 1e-10
        )

    def test_parameter_and_feature_terms_reconstruct_logits_and_crossing(self) -> None:
        before = decomposition.ProbeState(
            step=0,
            layers={
                1: layer_state(
                    [10, 11],
                    [[-1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                    [0.0, 0.0],
                )
            },
        )
        after = decomposition.ProbeState(
            step=64,
            layers={
                1: layer_state(
                    [10, 11],
                    [[1.0, 0.0], [0.5, 1.0]],
                    [[1.2, 0.1], [-0.2, 0.9]],
                    [0.1, -0.1],
                )
            },
        )
        trajectory = decomposition.ProbeTrajectory(
            before,
            target_by_layer=[0, 0],
            comparator_by_layer=[1, 1],
            threshold=0.5,
            metadata={
                "protocol_id": "test",
                "seed": 1,
                "cohort": "no_drop",
                "capacity_factor": None,
                "branch": "free_release",
                "initial_target_mass": 0.02,
            },
        )
        rows = trajectory.advance(after)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["crossed_threshold_observed"], 1)
        self.assertEqual(rows[0]["crossed_threshold_reconstructed"], 1)
        self.assertLess(rows[0]["logit_reconstruction_max_abs_error"], 1e-12)
        for row in rows:
            self.assertAlmostEqual(
                row["gap_displacement"],
                row["parameter_gap_displacement"]
                + row["feature_gap_displacement"],
                places=12,
            )
        self.assertEqual(trajectory.summary()["threshold_crossing_mismatches"], 0)


class SurvivalAnalysisTests(unittest.TestCase):
    def test_seed_level_pairing_and_censoring_aware_summary(self) -> None:
        records = []
        for seed, free_times, reference_times in (
            (1, [20.0, 100.0], [60.0, 100.0]),
            (2, [30.0, 70.0], [50.0, 90.0]),
        ):
            for branch, times in (
                (survival.DEFAULT_FREE, free_times),
                (survival.DEFAULT_REFERENCE, reference_times),
            ):
                for layer, time in enumerate(times):
                    records.append(
                        {
                            "cohort": "no_drop",
                            "seed": seed,
                            "layer_zero_based": layer,
                            "initial_target_mass": 0.02,
                            "branch": branch,
                            "time": time,
                            "event": time < 100.0,
                        }
                    )
        curves, seed_rows, summaries, summary = survival.analyze(
            records,
            tau=100.0,
            bootstrap_draws=1000,
            bootstrap_seed=7,
        )
        self.assertTrue(curves)
        self.assertEqual(len(seed_rows), 2)
        self.assertAlmostEqual(
            seed_rows[0][
                "reference_minus_free_restricted_mean_recovery_time"
            ],
            20.0,
        )
        self.assertAlmostEqual(
            summaries[0][
                "mean_reference_minus_free_restricted_mean_recovery_time"
            ],
            20.0,
        )
        self.assertEqual(summary["inferential_unit"], "seed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
