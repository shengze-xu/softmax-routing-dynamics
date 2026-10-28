#!/usr/bin/env python3
"""Recovery interventions with capacity-limited top-2 dispatch.

Cap expert assignments at capacity factor 1.25 and drop overflow in stable
token order. Track target probabilities before dispatch."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.studies import agnews_structure_comparison as agnews_model
from experiments.models import agnews_no_drop_recovery as no_drop_recovery


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 1
STUDY = "agnews_capacity_recovery"
SMOKE_SEED = 4600
CAPACITY_FACTOR = 1.25


def capacity_recovery_forward(
    self: agnews_model.IndependentSparseMoE,
    hidden: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, no_drop_recovery.RecoverySnapshot]:
    if not self.capacity_limited:
        raise ValueError("capacity recovery requires capacity-limited dispatch")
    if self.router is None:
        raise ValueError("capacity recovery requires input-conditioned routing")

    raw_logits = self.router(hidden)
    layer = int(getattr(self, "_capacity_recovery_layer_index", -1))
    targets = list(no_drop_recovery.INTERVENTION["target_by_layer"])
    comparators = list(no_drop_recovery.INTERVENTION["comparator_by_layer"])
    if not 0 <= layer < len(targets):
        raise ValueError("MoE layer is missing its capacity recovery index")
    target = int(targets[layer])
    comparator = int(comparators[layer])
    mode = str(no_drop_recovery.INTERVENTION["mode"])
    effective_logits = raw_logits
    if (
        layer in set(no_drop_recovery.INTERVENTION["middle_layers"])
        and mode != "none"
    ):
        target_selector = F.one_hot(
            torch.tensor(target, device=raw_logits.device),
            num_classes=self.num_experts,
        ).to(raw_logits.dtype)
        if mode == "suppressed":
            non_target = [
                index
                for index in range(self.num_experts)
                if index != target
            ]
            boundary = (
                torch.topk(
                    raw_logits[..., non_target],
                    k=self.top_k,
                    dim=-1,
                ).values[..., -1]
                - float(no_drop_recovery.INTERVENTION["clamp_margin"])
            )
        elif mode == "no_positive_advantage_control":
            boundary = (
                raw_logits[..., comparator]
                + float(no_drop_recovery.INTERVENTION["control_margin"])
            )
        else:
            raise ValueError(f"Unsupported intervention mode: {mode}")
        effective_target = torch.minimum(raw_logits[..., target], boundary)
        effective_logits = (
            raw_logits
            + (effective_target - raw_logits[..., target]).unsqueeze(-1)
            * target_selector
        )

    raw_probabilities = F.softmax(raw_logits.float(), dim=-1)
    probabilities = F.softmax(effective_logits.float(), dim=-1)
    top_values, top_indices = torch.topk(
        probabilities,
        k=self.top_k,
        dim=-1,
    )
    top_weights = top_values / top_values.sum(dim=-1, keepdim=True)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_indices = top_indices.reshape(-1, self.top_k)
    flat_weights = top_weights.reshape(-1, self.top_k)
    flat_active = active_mask.reshape(-1)
    accepted = flat_active.unsqueeze(-1).expand_as(flat_indices).clone()
    active_assignments = int(accepted.sum().item())
    capacity = math.ceil(
        active_assignments * CAPACITY_FACTOR / self.num_experts
    )
    for expert_index in range(self.num_experts):
        rows, slots = torch.where(
            accepted & (flat_indices == expert_index)
        )
        if rows.numel() > capacity:
            accepted[rows[capacity:], slots[capacity:]] = False

    mixed = torch.zeros_like(flat_hidden)
    for expert_index, expert in enumerate(self.experts):
        rows, slots = torch.where(
            accepted & (flat_indices == expert_index)
        )
        if rows.numel() == 0:
            continue
        expert_output = expert(flat_hidden[rows])
        weighted = expert_output * flat_weights[rows, slots].to(
            expert_output.dtype
        ).unsqueeze(-1)
        mixed.index_add_(0, rows, weighted.to(mixed.dtype))
    snapshot = no_drop_recovery.RecoverySnapshot(
        router_inputs=hidden,
        raw_logits=raw_logits,
        effective_logits=effective_logits,
        raw_probabilities=raw_probabilities,
        pre_topk_probabilities=probabilities,
        topk_indices=top_indices,
        accepted_assignments=accepted.reshape_as(top_indices),
        active_mask=active_mask,
        capacity_per_expert=capacity,
        layer_index=layer,
        intervention_mode=mode,
    )
    return mixed.reshape_as(hidden), snapshot


def patch_model(model: agnews_model.IndependentAutoregressiveMoE) -> None:
    for layer_index, block in enumerate(model.layers):
        block.moe._capacity_recovery_layer_index = layer_index
        block.moe.forward = MethodType(
            capacity_recovery_forward,
            block.moe,
        )


def make_model(
    *,
    dataset: agnews_model.IndependentPreparedDataset,
    config: dict[str, Any],
    seed: int,
) -> agnews_model.IndependentAutoregressiveMoE:
    if float(config["capacity_factor"]) != CAPACITY_FACTOR:
        raise ValueError("capacity recovery capacity factor must remain 1.25")
    if config["capacity_policy"] != "drop_no_rerouting":
        raise ValueError("capacity recovery capacity policy changed")
    condition = agnews_model.Condition(
        "conditioned_top2_capacity_1p25",
        "input_conditioned",
        True,
    )
    model = agnews_model.make_model(
        condition,
        dataset=dataset,
        config=config,
        seed=seed,
    )
    patch_model(model)
    return model


def routing_diagnostics(
    model: agnews_model.IndependentAutoregressiveMoE,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
    target_by_layer: list[int],
    comparator_by_layer: list[int],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    tokens, lengths, _ = data
    layer_count = len(model.layers)
    expert_count = model.layers[0].moe.num_experts
    raw_target = np.zeros(layer_count, dtype=np.float64)
    effective_target = np.zeros(layer_count, dtype=np.float64)
    raw_comparator = np.zeros(layer_count, dtype=np.float64)
    target_logit = np.zeros(layer_count, dtype=np.float64)
    competitor_logsumexp = np.zeros(layer_count, dtype=np.float64)
    selected_target = np.zeros(layer_count, dtype=np.int64)
    accepted_target = np.zeros(layer_count, dtype=np.int64)
    selected_by_expert = np.zeros(
        (layer_count, expert_count),
        dtype=np.int64,
    )
    accepted_by_expert = np.zeros_like(selected_by_expert)
    active_counts = np.zeros(layer_count, dtype=np.int64)
    capacity_min = np.full(layer_count, np.iinfo(np.int64).max)
    capacity_max = np.zeros(layer_count, dtype=np.int64)
    capacity_sum = np.zeros(layer_count, dtype=np.int64)
    diagnostic_batches = np.zeros(layer_count, dtype=np.int64)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch_tokens = tokens[start : start + batch_size].to(device)
            batch_lengths = lengths[start : start + batch_size].to(device)
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]
            mask = agnews_model.target_mask(batch_lengths, inputs.shape[1])
            with no_drop_recovery.autocast_context(device, precision):
                logits, snapshots = model(inputs, mask)
                loss = agnews_model.masked_loss(logits, targets, mask)
            losses.append(float(loss.item()))
            for layer, snapshot in enumerate(snapshots):
                target = int(target_by_layer[layer])
                comparator = int(comparator_by_layer[layer])
                count = int(mask.sum().item())
                raw_target[layer] += float(
                    snapshot.raw_probabilities[..., target][mask]
                    .double()
                    .sum()
                    .item()
                )
                effective_target[layer] += float(
                    snapshot.pre_topk_probabilities[..., target][mask]
                    .double()
                    .sum()
                    .item()
                )
                raw_comparator[layer] += float(
                    snapshot.raw_probabilities[..., comparator][mask]
                    .double()
                    .sum()
                    .item()
                )
                target_logit[layer] += float(
                    snapshot.raw_logits[..., target][mask]
                    .double()
                    .sum()
                    .item()
                )
                non_target = [
                    index
                    for index in range(snapshot.raw_logits.shape[-1])
                    if index != target
                ]
                competitor_logsumexp[layer] += float(
                    torch.logsumexp(
                        snapshot.raw_logits[..., non_target].float(),
                        dim=-1,
                    )[mask]
                    .double()
                    .sum()
                    .item()
                )
                selected = snapshot.topk_indices.eq(target)
                accepted = snapshot.accepted_assignments
                selected_target[layer] += int(
                    selected.any(dim=-1)[mask].sum().item()
                )
                accepted_target[layer] += int(
                    (selected & accepted).any(dim=-1)[mask].sum().item()
                )
                for expert in range(expert_count):
                    selected_by_expert[layer, expert] += int(
                        (
                            snapshot.topk_indices.eq(expert)
                            & mask.unsqueeze(-1)
                        )
                        .sum()
                        .item()
                    )
                    accepted_by_expert[layer, expert] += int(
                        (
                            snapshot.topk_indices.eq(expert)
                            & accepted
                            & mask.unsqueeze(-1)
                        )
                        .sum()
                        .item()
                    )
                active_counts[layer] += count
                capacity = int(snapshot.capacity_per_expert or 0)
                capacity_min[layer] = min(capacity_min[layer], capacity)
                capacity_max[layer] = max(capacity_max[layer], capacity)
                capacity_sum[layer] += capacity
                diagnostic_batches[layer] += 1
    model.train()
    if np.any(active_counts == 0):
        raise ValueError("Routing diagnostic omitted a layer")
    raw_target /= active_counts
    effective_target /= active_counts
    raw_comparator /= active_counts
    target_logit /= active_counts
    competitor_logsumexp /= active_counts
    selected_assignments = selected_by_expert.sum(axis=1)
    accepted_assignments = accepted_by_expert.sum(axis=1)
    dropped_by_expert = selected_by_expert - accepted_by_expert
    dropped_assignments = selected_assignments - accepted_assignments
    eps = np.finfo(np.float64).tiny
    aggregate_log_odds = np.log(
        np.maximum(effective_target, eps)
        / np.maximum(1.0 - effective_target, eps)
    )
    return {
        "loss": float(np.mean(losses)),
        "raw_target_probability_by_layer": raw_target.tolist(),
        "effective_target_probability_by_layer": effective_target.tolist(),
        "raw_comparator_probability_by_layer": raw_comparator.tolist(),
        "target_logit_by_layer": target_logit.tolist(),
        "competitor_logsumexp_by_layer": competitor_logsumexp.tolist(),
        "target_traffic_by_layer": (
            selected_target / active_counts
        ).tolist(),
        "accepted_target_traffic_by_layer": (
            accepted_target / active_counts
        ).tolist(),
        "aggregate_target_log_odds_by_layer": aggregate_log_odds.tolist(),
        "active_token_count_by_layer": active_counts.tolist(),
        "selected_assignment_count_by_layer": selected_assignments.tolist(),
        "accepted_assignment_count_by_layer": accepted_assignments.tolist(),
        "dropped_assignment_count_by_layer": dropped_assignments.tolist(),
        "drop_fraction_by_layer": (
            dropped_assignments / selected_assignments
        ).tolist(),
        "selected_assignment_count_by_expert_by_layer": (
            selected_by_expert.tolist()
        ),
        "accepted_assignment_count_by_expert_by_layer": (
            accepted_by_expert.tolist()
        ),
        "overflow_assignment_count_by_expert_by_layer": (
            dropped_by_expert.tolist()
        ),
        "capacity_per_expert_min_by_layer": capacity_min.tolist(),
        "capacity_per_expert_max_by_layer": capacity_max.tolist(),
        "capacity_per_expert_sum_by_layer": capacity_sum.tolist(),
        "diagnostic_batch_count_by_layer": diagnostic_batches.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    original_make_model = no_drop_recovery.make_model
    original_diagnostics = no_drop_recovery.routing_diagnostics
    original_study = no_drop_recovery.STUDY
    no_drop_recovery.make_model = make_model
    no_drop_recovery.routing_diagnostics = routing_diagnostics
    no_drop_recovery.STUDY = STUDY
    try:
        result = no_drop_recovery.run(args)
    finally:
        no_drop_recovery.make_model = original_make_model
        no_drop_recovery.routing_diagnostics = original_diagnostics
        no_drop_recovery.STUDY = original_study

    result["implementation_version"] = IMPLEMENTATION_VERSION
    result["capacity_recovery_robustness"] = True
    result["independence"].update(
        {
            "primary_path_no_drop": False,
            "capacity_limited": True,
            "capacity_factor": CAPACITY_FACTOR,
            "capacity_policy": "drop_no_rerouting",
        }
    )
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    destination = args.output_dir.resolve() / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
