#!/usr/bin/env python3
"""CPU unit tests for the capacity-limited mechanism implementation."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.models import agnews_no_drop_recovery as no_drop_recovery
from experiments.models import agnews_capacity_recovery as capacity_recovery
from experiments.studies import router_update_capacity as mechanism
from experiments.audits.audit_router_update_capacity import audit_result


class MechanismCapacityValidationTests(unittest.TestCase):
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
            "capacity_factor": 1.25,
            "capacity_policy": "drop_no_rerouting",
            "eval_batch_size": 4,
        }
        return capacity_recovery.make_model(dataset=dataset, config=config, seed=41), config

    def force_overflow(self, model) -> None:
        with torch.no_grad():
            for block in model.layers:
                block.moe.router.weight.zero_()
                block.moe.router.bias.copy_(
                    torch.tensor([4.0, 3.0, 0.0, -1.0])
                )

    def data(self):
        tokens = torch.randint(0, 32, (4, 5))
        lengths = torch.full((4,), 5, dtype=torch.long)
        domains = torch.arange(4, dtype=torch.long)
        return tokens, lengths, domains

    def test_capacity_dispatch_drops_without_rerouting(self) -> None:
        model, config = self.make_model()
        self.force_overflow(model)
        no_drop_recovery.configure_intervention(
            "none",
            target_by_layer=[0, 0],
            comparator_by_layer=[1, 1],
            middle_layers=[1],
            clamp_margin=3.0,
            control_margin=-1.0,
        )
        tokens, lengths, _ = self.data()
        mask = torch.ones((4, 4), dtype=torch.bool)
        _, snapshots = model(tokens[:, :-1], mask)
        snapshot = snapshots[1]
        total = int(mask.sum().item()) * 2
        kept = int(snapshot.accepted_assignments.sum().item())
        self.assertEqual(total, 32)
        self.assertEqual(snapshot.capacity_per_expert, 10)
        self.assertGreater(total - kept, 0)
        self.assertTrue(
            torch.equal(
                snapshot.topk_indices,
                torch.tensor([0, 1]).expand_as(snapshot.topk_indices),
            )
        )

    def test_measured_capacity_step_conserves_and_projects(self) -> None:
        model, config = self.make_model()
        self.force_overflow(model)
        optimizer = no_drop_recovery.make_optimizer(model, config)
        data = self.data()
        no_drop_recovery.configure_intervention(
            "none",
            target_by_layer=[0, 0],
            comparator_by_layer=[1, 1],
            middle_layers=[1],
            clamp_margin=3.0,
            control_margin=-1.0,
        )
        record, updates, projection = mechanism.measured_capacity_train_step(
            model,
            optimizer,
            data,
            torch.arange(4),
            device=torch.device("cpu"),
            precision="fp32",
            target_by_layer=[0, 0],
            comparator_by_layer=[1, 1],
            target_domain_by_layer=[0, 0],
            middle_layers=[1],
            project_advantage=True,
        )
        counts = record["assignment_counts"]["1"]
        self.assertEqual(counts["total"], counts["kept"] + counts["dropped"])
        self.assertGreater(counts["dropped"], 0)
        self.assertLessEqual(updates[0].realized_update_identity_error, 2e-6)
        self.assertLessEqual(
            max(
                abs(float(value))
                for key, value in projection["1"].items()
                if key.endswith("error")
            ),
            1e-7,
        )

    def test_mass_calibration_remeasures_after_each_layer(self) -> None:
        model, config = self.make_model()
        config["middle_layers"] = [0, 1]
        target_by_layer = [0, 0]
        comparator_by_layer = [1, 1]
        with mock.patch.object(
            mechanism.no_drop,
            "heldout_target_probabilities",
            wraps=mechanism.no_drop.heldout_target_probabilities,
        ) as measured:
            calibration = mechanism.calibrate_initial_mass(
                model,
                self.data(),
                target_by_layer=target_by_layer,
                comparator_by_layer=comparator_by_layer,
                config=config,
                device=torch.device("cpu"),
                target_mass=0.05,
                tolerance=0.001,
            )
        self.assertEqual(measured.call_count, len(config["middle_layers"]))
        self.assertTrue(
            all(
                error <= 0.001
                for error in calibration["absolute_error_by_layer"].values()
            )
        )

    def test_smoke_audit_uses_configured_forecast_freeze_step(self) -> None:
        log = {
            "protocol_id": "test",
            "seed": 5300,
            "cohort": "capacity_1_25",
            "capacity_factor": 1.25,
            "branch": "",
            "optimizer_step": 1,
            "layer_zero_based": 1,
            "target_expert": 0,
            "comparator_expert": 1,
            "initial_target_mass": 0.02,
            "raw_target_logit_pre_update": 0.0,
            "raw_comparator_logit_pre_update": 0.0,
            "target_logit_gradient": 0.0,
            "comparator_logit_gradient": 0.0,
            "gradient_descent_logodds_proxy": 0.0,
            "realized_target_logit_update": 0.0,
            "realized_comparator_logit_update": 0.0,
            "realized_logodds_update_advantage": 0.0,
            "realized_update_identity_error": 0.0,
            "cumulative_realized_logodds_update_advantage": 0.0,
            "heldout_pre_topk_target_probability": None,
            "heldout_target_traffic": None,
            "total_assignments": 2,
            "kept_assignments": 1,
            "dropped_assignments": 1,
            "persistent_recovery_indicator": None,
            "model_state_sha256": "model",
            "optimizer_state_sha256": "optimizer",
            "future_batch_plan_sha256": "plan",
        }
        branches = {}
        for name in mechanism.BRANCHES:
            row = dict(log, branch=name)
            branches[name] = {
                "start_model_fingerprint": "model",
                "start_optimizer_fingerprint": "optimizer",
                "future_batch_plan_fingerprint": "plan",
                "training_loss_first": 1.0,
                "training_loss_last": 0.9,
                "router_update_log": [row],
                "total_dropped_assignments": 1,
                "forecast_rows": [
                    {"layer_zero_based": 1, "freeze_step": 4}
                ],
                "projection_max_errors": {
                    "removed_component": 0.0,
                    "common_mode": 0.0,
                    "other_rows": 0.0,
                },
            }
        result = {
            "schema_version": 2,
            "implementation_version": 2,
            "role": "smoke",
            "cohort": "capacity_1_25",
            "seed": 5300,
            "config": {
                "capacity_factor": 1.25,
                "capacity_policy": "drop_no_rerouting",
                "horizon_steps": 1,
                "middle_layers": [1],
                "forecast_freeze_step": 4,
            },
            "execution": {
                "device": "cuda",
                "gpu": "NVIDIA GeForce RTX 5090",
                "elapsed_seconds": 1.0,
            },
            "cells": {
                "0.02": {
                    "pairing": {
                        "model": True,
                        "optimizer": True,
                        "future_batch_plan": True,
                    },
                    "calibration": {
                        "absolute_error_by_layer": {"1": 0.0}
                    },
                    "branches": branches,
                }
            },
        }
        audit = audit_result(result, expected_role="smoke")
        self.assertEqual(audit["status"], "passed", audit["failures"])
        result["execution"]["gpu"] = "NVIDIA A100"
        audit = audit_result(result, expected_role="smoke")
        self.assertEqual(audit["status"], "passed", audit["failures"])
        strict = audit_result(result, expected_role="smoke", expected_gpu="RTX 5090")
        self.assertEqual(strict["failures"], ["GPU identity does not match RTX 5090"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
