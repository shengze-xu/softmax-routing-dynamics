#!/usr/bin/env python3
"""Independent CPU unit tests for mechanism gradient projection and logging."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.models import agnews_no_drop_recovery as no_drop_recovery
from experiments.studies import router_update_no_drop as mechanism


class MechanismValidationTests(unittest.TestCase):
    def make_model(self):
        dataset = SimpleNamespace(vocab_size=32, sequence_length=4)
        config = {
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 2,
            "expert_hidden": 16,
            "num_experts": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "precision": "fp32",
            "middle_layers": [1],
            "clamp_margin": 3.0,
            "control_margin": -1.0,
        }
        return no_drop_recovery.make_model(dataset=dataset, config=config, seed=31), config

    def test_projection_removes_only_pairwise_component(self) -> None:
        model, config = self.make_model()
        optimizer = no_drop_recovery.make_optimizer(model, config)
        tokens = torch.randint(0, 32, (4, 5))
        lengths = torch.full((4,), 5, dtype=torch.long)
        domains = torch.arange(4, dtype=torch.long)
        data = (tokens, lengths, domains)
        no_drop_recovery.configure_intervention(
            "none",
            target_by_layer=[0, 1],
            comparator_by_layer=[1, 3],
            middle_layers=[1],
            clamp_margin=3.0,
            control_margin=-1.0,
        )
        _, _, projection = mechanism.measured_train_step(
            model,
            optimizer,
            data,
            torch.arange(4),
            device=torch.device("cpu"),
            precision="fp32",
            target_by_layer=[0, 1],
            comparator_by_layer=[1, 3],
            target_domain_by_layer=[0, 0],
            middle_layers=[1],
            project_advantage=True,
        )
        record = projection["1"]
        self.assertLessEqual(record["removed_component_weight_error"], 1e-7)
        self.assertLessEqual(record["removed_component_bias_error"], 1e-7)
        self.assertLessEqual(record["common_mode_weight_error"], 1e-7)
        self.assertLessEqual(record["common_mode_bias_error"], 1e-7)
        self.assertEqual(record["other_weight_max_error"], 0.0)
        self.assertEqual(record["other_bias_max_error"], 0.0)

    def test_measured_update_matches_direct_score_change(self) -> None:
        model, config = self.make_model()
        optimizer = no_drop_recovery.make_optimizer(model, config)
        tokens = torch.randint(0, 32, (4, 5))
        lengths = torch.full((4,), 5, dtype=torch.long)
        domains = torch.arange(4, dtype=torch.long)
        data = (tokens, lengths, domains)
        no_drop_recovery.configure_intervention(
            "none",
            target_by_layer=[0, 1],
            comparator_by_layer=[1, 3],
            middle_layers=[1],
            clamp_margin=3.0,
            control_margin=-1.0,
        )
        _, updates, _ = mechanism.measured_train_step(
            model,
            optimizer,
            data,
            torch.arange(4),
            device=torch.device("cpu"),
            precision="fp32",
            target_by_layer=[0, 1],
            comparator_by_layer=[1, 3],
            target_domain_by_layer=[0, 0],
            middle_layers=[1],
            project_advantage=False,
        )
        self.assertLessEqual(updates[0].realized_update_identity_error, 2e-6)

    def test_logit_shift_hits_prescribed_mean_mass(self) -> None:
        values = np.asarray([0.02, 0.1, 0.2, 0.4], dtype=np.float64)
        shift = mechanism.logit_shift_for_mean_mass(values, 0.05)
        scale = np.exp(shift)
        shifted = scale * values / (1.0 - values + scale * values)
        self.assertAlmostEqual(float(shifted.mean()), 0.05, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
