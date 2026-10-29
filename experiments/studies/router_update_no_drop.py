#!/usr/bin/env python3
"""Direct neural experiments for router-update recovery mechanisms.

A shared checkpoint is suppressed to four target masses and forked into
free-release, maintained-clamp, and zero-advantage-gradient branches.
Router gradients and realized AdamW updates are measured at each step.
Model and intervention settings are read from the supplied JSON design.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.studies import agnews_structure_comparison as agnews_model
from experiments.models import agnews_no_drop_recovery as no_drop_recovery
from experiments.common import router_logit_decomposition as probe_decomposition


SCHEMA_VERSION = 2
IMPLEMENTATION_VERSION = 3
BRANCHES = (
    "free_release",
    "continued_suppression_context_reference",
    "zero_advantage_gradient_projection_reference",
)
PRIMARY_NO_DROP_SEEDS = set(range(5201, 5209))
SMOKE_SEED = 5100


@dataclass
class RouterUpdate:
    layer_zero_based: int
    target_expert: int
    comparator_expert: int
    raw_target_logit_pre_update: float
    raw_comparator_logit_pre_update: float
    target_logit_gradient: float
    comparator_logit_gradient: float
    gradient_descent_logodds_proxy: float
    realized_target_logit_update: float
    realized_comparator_logit_update: float
    realized_logodds_update_advantage: float
    realized_update_identity_error: float


def make_balanced_batch_plan(
    domains: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    domain_values = sorted(set(int(value) for value in domains.tolist()))
    if batch_size % len(domain_values) != 0:
        raise ValueError("Batch size must be divisible by the domain count")
    per_domain = batch_size // len(domain_values)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inventories = {
        domain: torch.where(domains == domain)[0]
        for domain in domain_values
    }
    orders = {
        domain: values[torch.randperm(len(values), generator=generator)]
        for domain, values in inventories.items()
    }
    cursors = {domain: 0 for domain in domain_values}
    plan: list[torch.Tensor] = []
    for _ in range(steps):
        batch: list[torch.Tensor] = []
        for domain in domain_values:
            if cursors[domain] + per_domain > len(orders[domain]):
                values = inventories[domain]
                orders[domain] = values[
                    torch.randperm(len(values), generator=generator)
                ]
                cursors[domain] = 0
            start = cursors[domain]
            batch.append(orders[domain][start : start + per_domain])
            cursors[domain] += per_domain
        plan.append(torch.cat(batch))
    return plan


def heldout_target_probabilities(
    model: agnews_model.IndependentAutoregressiveMoE,
    evaluation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    target_by_layer: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[int, np.ndarray]:
    tokens, lengths, _ = evaluation_data
    middle_layers = list(config["middle_layers"])
    values: dict[int, list[np.ndarray]] = {
        layer: [] for layer in middle_layers
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), int(config["eval_batch_size"])):
            batch_tokens = tokens[
                start : start + int(config["eval_batch_size"])
            ].to(device)
            batch_lengths = lengths[
                start : start + int(config["eval_batch_size"])
            ].to(device)
            inputs = batch_tokens[:, :-1]
            mask = agnews_model.target_mask(batch_lengths, inputs.shape[1])
            with no_drop_recovery.autocast_context(device, str(config["precision"])):
                _, snapshots = model(inputs, mask)
            for layer in middle_layers:
                target = int(target_by_layer[layer])
                values[layer].append(
                    snapshots[layer]
                    .raw_probabilities[..., target][mask]
                    .float()
                    .cpu()
                    .numpy()
                )
    model.train()
    return {
        layer: np.concatenate(chunks)
        for layer, chunks in values.items()
    }


def select_target_domains(
    model: agnews_model.IndependentAutoregressiveMoE,
    evaluation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    target_by_layer: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[int], dict[str, Any]]:
    tokens, lengths, domains = evaluation_data
    middle_layers = list(config["middle_layers"])
    domain_values = sorted(set(int(value) for value in domains.tolist()))
    sums = {
        layer: np.zeros(len(domain_values), dtype=np.float64)
        for layer in middle_layers
    }
    counts = np.zeros(len(domain_values), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), int(config["eval_batch_size"])):
            batch_tokens = tokens[
                start : start + int(config["eval_batch_size"])
            ].to(device)
            batch_lengths = lengths[
                start : start + int(config["eval_batch_size"])
            ].to(device)
            batch_domains = domains[
                start : start + int(config["eval_batch_size"])
            ].to(device)
            inputs = batch_tokens[:, :-1]
            mask = agnews_model.target_mask(batch_lengths, inputs.shape[1])
            with no_drop_recovery.autocast_context(device, str(config["precision"])):
                _, snapshots = model(inputs, mask)
            for domain in domain_values:
                selected = mask & batch_domains.eq(domain).unsqueeze(1)
                count = int(selected.sum().item())
                if count == 0:
                    continue
                counts[domain] += count
                for layer in middle_layers:
                    target = int(target_by_layer[layer])
                    sums[layer][domain] += float(
                        snapshots[layer]
                        .raw_probabilities[..., target][selected]
                        .double()
                        .sum()
                        .item()
                    )
    model.train()
    if np.any(counts == 0):
        raise ValueError("Target-domain selection omitted a regime")
    target_domains = [0] * int(config["num_layers"])
    means: dict[str, list[float]] = {}
    for layer in middle_layers:
        row = sums[layer] / counts
        target_domains[layer] = int(np.argmax(row))
        means[str(layer)] = row.tolist()
    return target_domains, {
        "rule": "Select the held-out regime with maximal pre-fork target-expert mass in each configured middle layer.",
        "target_domain_by_layer": target_domains,
        "heldout_target_mass_by_middle_layer_and_domain": means,
    }


def logit_shift_for_mean_mass(
    probabilities: np.ndarray,
    target_mass: float,
) -> float:
    if not 0.0 < target_mass < 1.0:
        raise ValueError("Initial target mass must lie in (0, 1)")
    values = np.clip(probabilities.astype(np.float64), 1e-12, 1.0 - 1e-12)
    lo, hi = -24.0, 12.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        scale = math.exp(mid)
        shifted = scale * values / (1.0 - values + scale * values)
        if float(shifted.mean()) < target_mass:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def calibrate_initial_mass(
    model: agnews_model.IndependentAutoregressiveMoE,
    evaluation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    target_by_layer: list[int],
    comparator_by_layer: list[int],
    config: dict[str, Any],
    device: torch.device,
    target_mass: float,
    tolerance: float,
) -> dict[str, Any]:
    middle_layers = list(config["middle_layers"])
    routers = [model.layers[layer].moe.router for layer in middle_layers]
    if any(router is None or router.bias is None for router in routers):
        raise ValueError("Mass calibration requires affine router biases")
    no_drop_recovery.configure_intervention(
        "none",
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        middle_layers=middle_layers,
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    for layer, router in zip(middle_layers, routers):
        target = int(target_by_layer[layer])
        samples = heldout_target_probabilities(
            model,
            evaluation_data,
            target_by_layer=target_by_layer,
            config=config,
            device=device,
        )[layer]
        shift = logit_shift_for_mean_mass(samples, target_mass)
        with torch.no_grad():
            router.bias[target].add_(shift)
    no_drop_recovery.configure_intervention(
        "none",
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        middle_layers=middle_layers,
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    diag = no_drop_recovery.routing_diagnostics(
        model,
        evaluation_data,
        batch_size=int(config["eval_batch_size"]),
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        device=device,
        precision=str(config["precision"]),
    )
    errors = {
        str(layer): abs(
            float(diag["effective_target_probability_by_layer"][layer])
            - target_mass
        )
        for layer in middle_layers
    }
    if any(error > tolerance for error in errors.values()):
        raise ValueError(f"Initial-mass calibration failed: {errors}")
    return {"diagnostic": diag, "absolute_error_by_layer": errors}


def router_parameters(
    model: agnews_model.IndependentAutoregressiveMoE,
    *,
    middle_layers: list[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    values: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in middle_layers:
        router = model.layers[layer].moe.router
        if router is None or router.bias is None:
            raise ValueError("Mechanism logging requires affine routers")
        values[layer] = (
            router.weight.detach().clone(),
            router.bias.detach().clone(),
        )
    return values


def project_zero_advantage_gradient(
    model: agnews_model.IndependentAutoregressiveMoE,
    *,
    snapshots: list[no_drop_recovery.RecoverySnapshot],
    active_mask: torch.Tensor,
    batch_domains: torch.Tensor,
    target_domain_by_layer: list[int],
    target_by_layer: list[int],
    comparator_by_layer: list[int],
    middle_layers: list[int],
) -> dict[str, Any]:
    """Remove the target-domain target--comparator gradient component.

    The selected evaluation-domain contribution is reconstructed from raw
    score gradients and router inputs.  Half its target--comparator difference
    is subtracted from the target row and added to the comparator row.  This
    preserves common-mode gradients, nonselected-domain contributions, and
    every other expert row.
    """
    records: dict[str, Any] = {}
    for layer in middle_layers:
        router = model.layers[layer].moe.router
        if router is None or router.weight.grad is None or router.bias.grad is None:
            raise ValueError("Router gradient is missing before projection")
        target = int(target_by_layer[layer])
        comparator = int(comparator_by_layer[layer])
        selected = active_mask & batch_domains.eq(
            int(target_domain_by_layer[layer])
        ).unsqueeze(1)
        if not bool(selected.any()):
            raise ValueError("Training batch omitted the selected domain")
        snapshot = snapshots[layer]
        if snapshot.raw_logits.grad is None:
            raise ValueError("Raw router-logit gradient is missing")
        score_grad = snapshot.raw_logits.grad.float()
        router_input = snapshot.router_inputs.detach().float()
        target_weight = torch.sum(
            score_grad[..., target][selected].unsqueeze(-1)
            * router_input[selected],
            dim=0,
        )
        comparator_weight = torch.sum(
            score_grad[..., comparator][selected].unsqueeze(-1)
            * router_input[selected],
            dim=0,
        )
        target_bias = torch.sum(score_grad[..., target][selected])
        comparator_bias = torch.sum(score_grad[..., comparator][selected])
        weight_difference = target_weight - comparator_weight
        bias_difference = target_bias - comparator_bias
        weight_before = router.weight.grad.detach().clone()
        bias_before = router.bias.grad.detach().clone()
        router.weight.grad[target].sub_(weight_difference / 2.0)
        router.weight.grad[comparator].add_(weight_difference / 2.0)
        router.bias.grad[target].sub_(bias_difference / 2.0)
        router.bias.grad[comparator].add_(bias_difference / 2.0)
        others = [
            index
            for index in range(router.weight.shape[0])
            if index not in {target, comparator}
        ]
        records[str(layer)] = {
            "selected_domain": int(target_domain_by_layer[layer]),
            "selected_token_count": int(selected.sum().item()),
            "removed_component_weight_error": float(
                torch.max(
                    torch.abs(
                        weight_before[target]
                        - weight_before[comparator]
                        - router.weight.grad[target]
                        + router.weight.grad[comparator]
                        - weight_difference
                    )
                ).item()
            ),
            "removed_component_bias_error": float(
                abs(
                    float(
                        bias_before[target].item()
                        - bias_before[comparator].item()
                        - router.bias.grad[target].item()
                        + router.bias.grad[comparator].item()
                        - bias_difference.item()
                    )
                )
            ),
            "common_mode_weight_error": float(
                torch.max(
                    torch.abs(
                        router.weight.grad[target]
                        + router.weight.grad[comparator]
                        - weight_before[target]
                        - weight_before[comparator]
                    )
                ).item()
            ),
            "common_mode_bias_error": float(
                abs(
                    float(
                        router.bias.grad[target].item()
                        + router.bias.grad[comparator].item()
                        - bias_before[target].item()
                        - bias_before[comparator].item()
                    )
                )
            ),
            "other_weight_max_error": float(
                torch.max(
                    torch.abs(router.weight.grad[others] - weight_before[others])
                ).item()
            ),
            "other_bias_max_error": float(
                torch.max(
                    torch.abs(router.bias.grad[others] - bias_before[others])
                ).item()
            ),
        }
    return records


def measured_train_step(
    model: agnews_model.IndependentAutoregressiveMoE,
    optimizer: torch.optim.Optimizer,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    target_by_layer: list[int],
    comparator_by_layer: list[int],
    target_domain_by_layer: list[int],
    middle_layers: list[int],
    project_advantage: bool,
) -> tuple[dict[str, float | int], list[RouterUpdate], dict[str, Any]]:
    tokens, lengths, domains = data
    batch_tokens = tokens[indices].to(device)
    batch_lengths = lengths[indices].to(device)
    batch_domains = domains[indices].to(device)
    inputs = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    mask = agnews_model.target_mask(batch_lengths, inputs.shape[1])
    optimizer.zero_grad(set_to_none=True)
    with no_drop_recovery.autocast_context(device, precision):
        logits, snapshots = model(inputs, mask)
        loss = agnews_model.masked_loss(logits, targets, mask)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite mechanism-validation loss")
    for layer in middle_layers:
        snapshots[layer].raw_logits.retain_grad()
    loss.backward()
    projection = (
        project_zero_advantage_gradient(
            model,
            snapshots=snapshots,
            active_mask=mask,
            batch_domains=batch_domains,
            target_domain_by_layer=target_domain_by_layer,
            target_by_layer=target_by_layer,
            comparator_by_layer=comparator_by_layer,
            middle_layers=middle_layers,
        )
        if project_advantage
        else {}
    )
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Non-finite mechanism-validation gradient")
    before = router_parameters(model, middle_layers=middle_layers)
    optimizer.step()
    after = router_parameters(model, middle_layers=middle_layers)
    updates: list[RouterUpdate] = []
    for layer in middle_layers:
        target = int(target_by_layer[layer])
        comparator = int(comparator_by_layer[layer])
        snapshot = snapshots[layer]
        if snapshot.raw_logits.grad is None:
            raise ValueError("Raw router-logit gradient is missing")
        selected = mask & batch_domains.eq(
            int(target_domain_by_layer[layer])
        ).unsqueeze(1)
        if not bool(selected.any()):
            raise ValueError("Training batch omitted the selected domain")
        raw_target_score = float(
            snapshot.raw_logits[..., target][selected].float().mean().item()
        )
        raw_comparator_score = float(
            snapshot.raw_logits[..., comparator][selected].float().mean().item()
        )
        score_gradient_target = float(
            snapshot.raw_logits.grad[..., target][selected].float().sum().item()
        )
        score_gradient_comparator = float(
            snapshot.raw_logits.grad[..., comparator][selected].float().sum().item()
        )
        weight_before, bias_before = before[layer]
        weight_after, bias_after = after[layer]
        mean_input = snapshot.router_inputs[selected].float().mean(dim=0)
        router = model.layers[layer].moe.router
        if (
            router is None
            or router.weight.grad is None
            or router.bias is None
            or router.bias.grad is None
        ):
            raise ValueError("Router gradient disappeared after clipping")
        target_score_gradient = float(
            (
                router.weight.grad[target].float() @ mean_input
                + router.bias.grad[target].float()
            ).item()
        )
        comparator_score_gradient = float(
            (
                router.weight.grad[comparator].float() @ mean_input
                + router.bias.grad[comparator].float()
            ).item()
        )
        weight_delta = (weight_after - weight_before).float()
        bias_delta = (bias_after - bias_before).float()
        realized_target = float(
            (weight_delta[target] @ mean_input + bias_delta[target]).item()
        )
        realized_comparator = float(
            (
                weight_delta[comparator] @ mean_input
                + bias_delta[comparator]
            ).item()
        )
        with torch.no_grad():
            pre_scores = F.linear(
                snapshot.router_inputs.detach().float(),
                weight_before.float(),
                bias_before.float(),
            )
            post_scores = F.linear(
                snapshot.router_inputs.detach().float(),
                router.weight.detach().float(),
                router.bias.detach().float(),
            )
        direct_target = float(
            (post_scores[..., target] - pre_scores[..., target])[selected]
            .mean()
            .item()
        )
        direct_comparator = float(
            (post_scores[..., comparator] - pre_scores[..., comparator])[selected]
            .mean()
            .item()
        )
        identity_error = max(
            abs(realized_target - direct_target),
            abs(realized_comparator - direct_comparator),
        )
        updates.append(
            RouterUpdate(
                layer_zero_based=layer,
                target_expert=target,
                comparator_expert=comparator,
                raw_target_logit_pre_update=raw_target_score,
                raw_comparator_logit_pre_update=raw_comparator_score,
                target_logit_gradient=score_gradient_target,
                comparator_logit_gradient=score_gradient_comparator,
                gradient_descent_logodds_proxy=-(
                    target_score_gradient - comparator_score_gradient
                ),
                realized_target_logit_update=realized_target,
                realized_comparator_logit_update=realized_comparator,
                realized_logodds_update_advantage=(
                    realized_target - realized_comparator
                ),
                realized_update_identity_error=identity_error,
            )
        )
    return (
        {
            "loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
            "target_tokens": int(mask.sum().item()),
        },
        updates,
        projection,
    )


def context_mode(branch: str) -> str:
    if branch == "continued_suppression_context_reference":
        return "no_positive_advantage_control"
    if branch in {
        "free_release",
        "zero_advantage_gradient_projection_reference",
    }:
        return "none"
    raise ValueError(f"Unknown mechanism branch: {branch}")


def run_mechanism_branch(
    *,
    branch: str,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    dataset: agnews_model.IndependentPreparedDataset,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    evaluation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    future_plan: list[torch.Tensor],
    target_by_layer: list[int],
    comparator_by_layer: list[int],
    target_domain_by_layer: list[int],
    config: dict[str, Any],
    seed: int,
    initial_mass: float,
    cohort: str,
    device: torch.device,
    probe_output_path: Path | None = None,
) -> dict[str, Any]:
    model = no_drop_recovery.make_model(dataset=dataset, config=config, seed=seed)
    model.load_state_dict(copy.deepcopy(model_state))
    model.to(device)
    optimizer = no_drop_recovery.make_optimizer(model, config)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    middle_layers = list(config["middle_layers"])
    no_drop_recovery.configure_intervention(
        context_mode(branch),
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        middle_layers=middle_layers,
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    cumulative = {layer: 0.0 for layer in middle_layers}
    logs: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    evaluation_steps: list[int] = []
    losses: list[float] = []
    projection_max = {
        "removed_component": 0.0,
        "common_mode": 0.0,
        "other_rows": 0.0,
    }
    probe_writer: probe_decomposition.ProbeCSVWriter | None = None
    probe_trajectory: probe_decomposition.ProbeTrajectory | None = None
    probe_tokens = int(config.get("probe_tokens_per_layer", 0))
    if probe_tokens > 0:
        if probe_output_path is None:
            raise ValueError("Probe logging requires an output path")
        probe_writer = probe_decomposition.ProbeCSVWriter(probe_output_path)
        initial_probe = probe_decomposition.capture_probe_state(
            model,
            evaluation_data,
            step=0,
            middle_layers=middle_layers,
            target_domain_by_layer=target_domain_by_layer,
            tokens_per_layer=probe_tokens,
            eval_batch_size=int(config["eval_batch_size"]),
            device=device,
            precision=str(config["precision"]),
            target_mask=agnews_model.target_mask,
            autocast_context=no_drop_recovery.autocast_context,
        )
        probe_trajectory = probe_decomposition.ProbeTrajectory(
            initial_probe,
            target_by_layer=target_by_layer,
            comparator_by_layer=comparator_by_layer,
            threshold=float(config["recovery_probability_threshold"]),
            metadata={
                "protocol_id": config["protocol_id"],
                "seed": seed,
                "cohort": cohort,
                "capacity_factor": None,
                "branch": branch,
                "initial_target_mass": initial_mass,
            },
        )
    for step, indices in enumerate(future_plan, start=1):
        record, updates, projection = measured_train_step(
            model,
            optimizer,
            train_data,
            indices,
            device=device,
            precision=str(config["precision"]),
            target_by_layer=target_by_layer,
            comparator_by_layer=comparator_by_layer,
            target_domain_by_layer=target_domain_by_layer,
            middle_layers=middle_layers,
            project_advantage=(
                branch == "zero_advantage_gradient_projection_reference"
            ),
        )
        losses.append(float(record["loss"]))
        for item in updates:
            cumulative[item.layer_zero_based] += (
                item.realized_logodds_update_advantage
            )
            logs.append(
                {
                    "protocol_id": config["protocol_id"],
                    "seed": seed,
                    "cohort": cohort,
                    "capacity_factor": None,
                    "branch": branch,
                    "optimizer_step": step,
                    "layer_zero_based": item.layer_zero_based,
                    "target_expert": item.target_expert,
                    "comparator_expert": item.comparator_expert,
                    "initial_target_mass": initial_mass,
                    "raw_target_logit_pre_update": (
                        item.raw_target_logit_pre_update
                    ),
                    "raw_comparator_logit_pre_update": (
                        item.raw_comparator_logit_pre_update
                    ),
                    "target_logit_gradient": item.target_logit_gradient,
                    "comparator_logit_gradient": item.comparator_logit_gradient,
                    "gradient_descent_logodds_proxy": (
                        item.gradient_descent_logodds_proxy
                    ),
                    "realized_target_logit_update": (
                        item.realized_target_logit_update
                    ),
                    "realized_comparator_logit_update": (
                        item.realized_comparator_logit_update
                    ),
                    "realized_logodds_update_advantage": (
                        item.realized_logodds_update_advantage
                    ),
                    "realized_update_identity_error": (
                        item.realized_update_identity_error
                    ),
                    "cumulative_realized_logodds_update_advantage": (
                        cumulative[item.layer_zero_based]
                    ),
                    "heldout_pre_topk_target_probability": None,
                    "heldout_target_traffic": None,
                    "total_assignments": None,
                    "kept_assignments": None,
                    "dropped_assignments": 0,
                    "persistent_recovery_indicator": None,
                }
            )
        for metrics in projection.values():
            projection_max["removed_component"] = max(
                projection_max["removed_component"],
                abs(float(metrics["removed_component_weight_error"])),
                abs(float(metrics["removed_component_bias_error"])),
            )
            projection_max["common_mode"] = max(
                projection_max["common_mode"],
                abs(float(metrics["common_mode_weight_error"])),
                abs(float(metrics["common_mode_bias_error"])),
            )
            projection_max["other_rows"] = max(
                projection_max["other_rows"],
                abs(float(metrics["other_weight_max_error"])),
                abs(float(metrics["other_bias_max_error"])),
            )
        if step % int(config["eval_interval"]) == 0:
            diag = no_drop_recovery.routing_diagnostics(
                model,
                evaluation_data,
                batch_size=int(config["eval_batch_size"]),
                target_by_layer=target_by_layer,
                comparator_by_layer=comparator_by_layer,
                device=device,
                precision=str(config["precision"]),
            )
            evaluations.append(diag)
            evaluation_steps.append(step)
            for layer in middle_layers:
                row = next(
                    value
                    for value in reversed(logs)
                    if value["optimizer_step"] == step
                    and value["layer_zero_based"] == layer
                )
                active_tokens = int(
                    diag["active_token_count_by_layer"][layer]
                )
                row["heldout_pre_topk_target_probability"] = float(
                    diag["effective_target_probability_by_layer"][layer]
                )
                row["heldout_target_traffic"] = float(
                    diag["target_traffic_by_layer"][layer]
                )
                row["total_assignments"] = (
                    active_tokens * int(config["top_k"])
                )
                row["kept_assignments"] = row["total_assignments"]
            if probe_trajectory is not None and probe_writer is not None:
                current_probe = probe_decomposition.capture_probe_state(
                    model,
                    evaluation_data,
                    step=step,
                    middle_layers=middle_layers,
                    target_domain_by_layer=target_domain_by_layer,
                    tokens_per_layer=probe_tokens,
                    eval_batch_size=int(config["eval_batch_size"]),
                    device=device,
                    precision=str(config["precision"]),
                    target_mask=agnews_model.target_mask,
                    autocast_context=no_drop_recovery.autocast_context,
                )
                probe_writer.writerows(probe_trajectory.advance(current_probe))
            print(
                json.dumps(
                    {
                        "event": "mechanism_progress",
                        "seed": seed,
                        "initial_mass": initial_mass,
                        "branch": branch,
                        "optimizer_step": step,
                        "loss": losses[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    threshold = float(config["recovery_probability_threshold"])
    persistence = int(config["recovery_persistence"])
    recovery = [None] * int(config["num_layers"])
    for layer in middle_layers:
        series = [
            float(value["effective_target_probability_by_layer"][layer])
            for value in evaluations
        ]
        recovery[layer] = no_drop_recovery.persistent_hit(
            series,
            threshold=threshold,
            persistence=persistence,
            steps=evaluation_steps,
        )
        for row in logs:
            if (
                row["layer_zero_based"] == layer
                and row["optimizer_step"] in evaluation_steps
            ):
                hit = recovery[layer]
                row["persistent_recovery_indicator"] = bool(
                    hit is not None and row["optimizer_step"] >= hit
                )
    forecast_rows: list[dict[str, Any]] = []
    freeze = int(config["forecast_freeze_step"])
    censor = int(config["censoring_step"])
    for layer in middle_layers:
        prefix = [
            row
            for row in logs
            if row["layer_zero_based"] == layer
            and row["optimizer_step"] <= freeze
        ]
        cumulative_freeze = float(
            prefix[-1]["cumulative_realized_logodds_update_advantage"]
        )
        slope = cumulative_freeze / freeze
        budget = math.log(
            threshold * (1.0 - initial_mass)
            / ((1.0 - threshold) * initial_mass)
        )
        predicted = None if slope <= 0.0 else min(censor, budget / slope)
        forecast_rows.append(
            {
                "layer_zero_based": layer,
                "initial_target_mass": initial_mass,
                "freeze_step": freeze,
                "cumulative_advantage_at_freeze": cumulative_freeze,
                "estimated_slope": slope,
                "logodds_budget": budget,
                "predicted_recovery_step": predicted,
                "observed_recovery_step": recovery[layer],
            }
        )
    if probe_writer is not None:
        probe_writer.close()
    result = {
        "branch": branch,
        "initial_target_mass": initial_mass,
        "optimizer_steps": len(future_plan),
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "evaluation_steps": evaluation_steps,
        "evaluations": evaluations,
        "recovery_step_by_layer": recovery,
        "router_update_log": logs,
        "forecast_rows": forecast_rows,
        "projection_max_errors": projection_max,
        "probe_decomposition": (
            None
            if probe_trajectory is None or probe_output_path is None
            else {
                **probe_trajectory.summary(),
                "path": probe_output_path.name,
            }
        ),
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _config_from_design(
    design: dict[str, Any],
    *,
    protocol_id: str,
    smoke: bool,
) -> dict[str, Any]:
    base = {
        "batch_plan_seed_offset": 9500,
        "batch_size": 4,
        "checkpoint_steps": int(design["model"]["checkpoint_steps"]),
        "warmup_steps": 1024,
        "horizon_steps": int(design["model"]["post_fork_steps"]),
        "clamp_margin": 3.0,
        "control_margin": -1.0,
        "d_model": 384,
        "entry_probability_ceiling": 0.025,
        "entry_traffic_ceiling": 0.01,
        "eval_batch_size": 2,
        "eval_interval": int(
            design["model"]["evaluation_interval_steps"]
        ),
        "expected_dataset_status": "prepared_agnews_candidate",
        "expert_hidden": 768,
        "learning_rate": 0.0002,
        "limit_evaluation": 512,
        "limit_train": 119800,
        "middle_layers": list(design["model"]["middle_layers_zero_based"]),
        "num_experts": 8,
        "num_heads": 6,
        "num_layers": 6,
        "precision": "bf16",
        "recovery_persistence": int(
            design["model"]["persistent_evaluations"]
        ),
        "recovery_probability_threshold": float(
            design["model"]["target_probability_threshold"]
        ),
        "target_eligibility_threshold": 0.05,
        "top_k": 2,
        "weight_decay": 0.01,
        "forecast_freeze_step": int(design["forecast_rule"]["forecast_freeze_step"]),
        "censoring_step": int(design["forecast_rule"]["censoring_step"]),
        "protocol_id": protocol_id,
    }
    if smoke:
        base.update(
            {
                "checkpoint_steps": 8,
                "warmup_steps": 4,
                "horizon_steps": 8,
                "eval_interval": 2,
                "forecast_freeze_step": 4,
                "censoring_step": 10,
                "limit_train": 64,
                "limit_evaluation": 16,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 3,
                "expert_hidden": 128,
                "middle_layers": [1],
                "eval_batch_size": 4,
                "precision": "bf16",
            }
        )
    return base


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    design = protocol
    if args.cohort != "no_drop":
        raise ValueError("This implementation launches the no-drop cohort only")
    cohort_design = next(
        item for item in design["cohorts"] if item["cohort"] == args.cohort
    )
    probe_protocol = None
    if args.probe_protocol is not None:
        probe_protocol = json.loads(
            args.probe_protocol.read_text(encoding="utf-8")
        )
    primary = args.role == "primary"
    additional = args.role == "additional"
    if primary and args.seed not in PRIMARY_NO_DROP_SEEDS:
        raise ValueError("Seed is outside the configured no-drop cohort")
    if additional:
        if probe_protocol is None:
            raise ValueError("Additional role requires --probe-protocol")
        allowed = set(probe_protocol["additional_no_drop_seeds"])
        if args.seed not in allowed:
            raise ValueError("Seed is outside the no-drop additional inventory")
    if args.role == "smoke" and args.seed != SMOKE_SEED:
        raise ValueError("Smoke role requires the dedicated smoke seed")
    active_protocol_id = (
        str(probe_protocol["protocol_id"])
        if probe_protocol is not None and additional
        else str(protocol["protocol_id"])
    )
    config = _config_from_design(
        design,
        protocol_id=active_protocol_id,
        smoke=args.role == "smoke",
    )
    if additional:
        probe_design = probe_protocol["logit_decomposition"]
        config["probe_tokens_per_layer"] = int(
            probe_design["tokens_per_layer"]
        )
    elif args.role == "smoke" and probe_protocol is not None:
        config["probe_tokens_per_layer"] = int(
            probe_protocol["logit_decomposition"][
                "smoke_tokens_per_layer"
            ]
        )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Mechanism validation requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not report BF16 support")
    agnews_model.configure_execution(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    dataset = agnews_model.IndependentPreparedDataset(
        args.dataset_dir,
        expected_status=str(config["expected_dataset_status"]),
    )
    train_data = dataset.split("train", limit=int(config["limit_train"]))
    if getattr(args, "independent_recovery_evaluation", False):
        from experiments.common.recovery_evaluation_split import validation_sets
        evaluation_data, recovery_data, sample_design = validation_sets(
            dataset, int(config["limit_evaluation"]))
        # Record the independent evaluation split in the protocol identifier.
        active_protocol_id += "_independent_eval_v1"
        config["protocol_id"] = active_protocol_id
    else:
        evaluation_data = dataset.split(
            "validation", limit=int(config["limit_evaluation"]))
        recovery_data = evaluation_data
        sample_design = {"design": "shared_selection_and_recovery"}
    config["validation_sample_design"] = sample_design
    masses = list(cohort_design["initial_mass_targets"])
    if args.role == "smoke":
        masses = [0.02]
    total_steps = (
        int(config["checkpoint_steps"])
        + int(config["warmup_steps"])
        + int(config["horizon_steps"])
    )
    plan = make_balanced_batch_plan(
        train_data[2],
        steps=total_steps,
        batch_size=int(config["batch_size"]),
        seed=args.seed + int(config["batch_plan_seed_offset"]),
    )
    checkpoint_end = int(config["checkpoint_steps"])
    warmup_end = checkpoint_end + int(config["warmup_steps"])
    checkpoint_plan = plan[:checkpoint_end]
    warmup_plan = plan[checkpoint_end:warmup_end]
    future_plan = plan[warmup_end:]
    model = no_drop_recovery.make_model(dataset=dataset, config=config, seed=args.seed)
    model.to(device)
    optimizer = no_drop_recovery.make_optimizer(model, config)
    defaults = [0] * int(config["num_layers"])
    comparators = [1] * int(config["num_layers"])
    no_drop_recovery.configure_intervention(
        "none",
        target_by_layer=defaults,
        comparator_by_layer=comparators,
        middle_layers=list(config["middle_layers"]),
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    for checkpoint_step, indices in enumerate(checkpoint_plan, start=1):
        no_drop_recovery.train_step(
            model,
            optimizer,
            train_data,
            indices,
            device=device,
            precision=str(config["precision"]),
        )
        if checkpoint_step % 2048 == 0 or checkpoint_step == len(checkpoint_plan):
            print(
                json.dumps(
                    {
                        "event": "checkpoint_progress",
                        "seed": args.seed,
                        "optimizer_step": checkpoint_step,
                        "optimizer_steps_total": len(checkpoint_plan),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    target_by_layer, comparator_by_layer, selection = no_drop_recovery.select_targets(
        model,
        evaluation_data,
        config=config,
        device=device,
    )
    target_domain_by_layer, target_domain_selection = select_target_domains(
        model,
        evaluation_data,
        target_by_layer=target_by_layer,
        config=config,
        device=device,
    )
    no_drop_recovery.configure_intervention(
        "suppressed",
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        middle_layers=list(config["middle_layers"]),
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    for indices in warmup_plan:
        no_drop_recovery.train_step(
            model,
            optimizer,
            train_data,
            indices,
            device=device,
            precision=str(config["precision"]),
        )
    shared_model_state = copy.deepcopy(model.state_dict())
    shared_optimizer_state = copy.deepcopy(optimizer.state_dict())
    del model, optimizer
    cells: dict[str, Any] = {}
    tolerance = float(
        design["entry_absolute_tolerance"]
    )
    for mass in masses:
        calibration_model = no_drop_recovery.make_model(
            dataset=dataset, config=config, seed=args.seed
        )
        calibration_model.load_state_dict(copy.deepcopy(shared_model_state))
        calibration_model.to(device)
        calibration = calibrate_initial_mass(
            calibration_model,
            evaluation_data,
            target_by_layer=target_by_layer,
            comparator_by_layer=comparator_by_layer,
            config=config,
            device=device,
            target_mass=float(mass),
            tolerance=tolerance,
        )
        if getattr(args, "independent_recovery_evaluation", False):
            # Record achieved mass on B without recalibrating or excluding cells.
            calibration["independent_evaluation_at_release"] = no_drop_recovery.routing_diagnostics(
                calibration_model, recovery_data,
                batch_size=int(config["eval_batch_size"]),
                target_by_layer=target_by_layer,
                comparator_by_layer=comparator_by_layer,
                device=device, precision=str(config["precision"]))
        cell_model_state = copy.deepcopy(calibration_model.state_dict())
        del calibration_model
        gc.collect()
        torch.cuda.empty_cache()
        branches = {
            branch: run_mechanism_branch(
                branch=branch,
                model_state=cell_model_state,
                optimizer_state=shared_optimizer_state,
                dataset=dataset,
                train_data=train_data,
                evaluation_data=recovery_data,
                future_plan=future_plan,
                target_by_layer=target_by_layer,
                comparator_by_layer=comparator_by_layer,
                target_domain_by_layer=target_domain_by_layer,
                config=config,
                seed=args.seed,
                initial_mass=float(mass),
                cohort=args.cohort,
                device=device,
                probe_output_path=(
                    output_dir
                    / (
                        f"probe_decomposition_mass_{float(mass):.3f}_"
                        f"{branch}.csv.gz"
                    )
                    if int(config.get("probe_tokens_per_layer", 0)) > 0
                    else None
                ),
            )
            for branch in BRANCHES
        }
        cells[str(mass)] = {
            "calibration": calibration,
            "branches": branches,
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "protocol_id": active_protocol_id,
        "base_protocol_id": protocol["protocol_id"],
        "role": args.role,
        "cohort": args.cohort,
        "seed": args.seed,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "config": config,
        "dataset": {
            "status": dataset.status,
        },
        "target_selection": selection,
        "target_domain_selection": target_domain_selection,
        "cells": cells,
    }
    destination = output_dir / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "role": args.role,
                "cohort": args.cohort,
                "seed": args.seed,
                "output": str(destination),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--probe-protocol", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=("no_drop",), required=True)
    parser.add_argument(
        "--role", choices=("smoke", "primary", "additional"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--independent-recovery-evaluation", action="store_true",
                        help="Use disjoint class-balanced validation sets for selection/calibration and recovery.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
