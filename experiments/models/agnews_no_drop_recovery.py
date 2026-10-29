#!/usr/bin/env python3
"""Train and suppress an AG News MoE, then compare recovery branches.

Branches share model parameters, optimizer state, and future minibatches."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
from types import MethodType
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.studies import agnews_structure_comparison as agnews_model


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 1
STUDY = "agnews_no_drop_recovery"
SMOKE_SEED = 4400

INTERVENTION: dict[str, Any] = {
    "mode": "none",
    "target_by_layer": [],
    "comparator_by_layer": [],
    "middle_layers": (),
    "clamp_margin": 3.0,
    "control_margin": -1.0,
}


@dataclass
class RecoverySnapshot:
    router_inputs: torch.Tensor
    raw_logits: torch.Tensor
    effective_logits: torch.Tensor
    raw_probabilities: torch.Tensor
    pre_topk_probabilities: torch.Tensor
    topk_indices: torch.Tensor
    accepted_assignments: torch.Tensor
    active_mask: torch.Tensor
    capacity_per_expert: int | None
    layer_index: int
    intervention_mode: str


def configure_intervention(
    mode: str,
    *,
    target_by_layer: list[int],
    comparator_by_layer: list[int],
    middle_layers: list[int],
    clamp_margin: float,
    control_margin: float,
) -> None:
    allowed = {"none", "suppressed", "no_positive_advantage_control"}
    if mode not in allowed:
        raise ValueError(f"Unsupported no-drop recovery intervention: {mode}")
    if len(target_by_layer) != len(comparator_by_layer):
        raise ValueError("Target/comparator layer inventories differ")
    if any(
        target == comparator
        for target, comparator in zip(
            target_by_layer,
            comparator_by_layer,
        )
    ):
        raise ValueError("Target and comparator must differ")
    INTERVENTION.update(
        {
            "mode": mode,
            "target_by_layer": list(target_by_layer),
            "comparator_by_layer": list(comparator_by_layer),
            "middle_layers": tuple(int(value) for value in middle_layers),
            "clamp_margin": float(clamp_margin),
            "control_margin": float(control_margin),
        }
    )


def recovery_forward(
    self: agnews_model.IndependentSparseMoE,
    hidden: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, RecoverySnapshot]:
    if self.capacity_limited:
        raise ValueError("no-drop recovery primary recovery path must be no-drop")
    if self.router is None:
        raise ValueError("no-drop recovery requires input-conditioned routing")
    raw_logits = self.router(hidden)
    layer = int(getattr(self, "_no_drop_recovery_layer_index", -1))
    targets = list(INTERVENTION["target_by_layer"])
    comparators = list(INTERVENTION["comparator_by_layer"])
    if not 0 <= layer < len(targets):
        raise ValueError("MoE layer is missing its no-drop recovery index")
    target = int(targets[layer])
    comparator = int(comparators[layer])
    mode = str(INTERVENTION["mode"])
    effective_logits = raw_logits
    if layer in set(INTERVENTION["middle_layers"]) and mode != "none":
        target_selector = F.one_hot(
            torch.tensor(target, device=raw_logits.device),
            num_classes=self.num_experts,
        ).to(raw_logits.dtype)
        if mode == "suppressed":
            non_target = [
                index for index in range(self.num_experts)
                if index != target
            ]
            boundary = (
                torch.topk(
                    raw_logits[..., non_target],
                    k=self.top_k,
                    dim=-1,
                ).values[..., -1]
                - float(INTERVENTION["clamp_margin"])
            )
        elif mode == "no_positive_advantage_control":
            boundary = (
                raw_logits[..., comparator]
                + float(INTERVENTION["control_margin"])
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
    snapshot = RecoverySnapshot(
        router_inputs=hidden,
        raw_logits=raw_logits,
        effective_logits=effective_logits,
        raw_probabilities=raw_probabilities,
        pre_topk_probabilities=probabilities,
        topk_indices=top_indices,
        accepted_assignments=accepted.reshape_as(top_indices),
        active_mask=active_mask,
        capacity_per_expert=None,
        layer_index=layer,
        intervention_mode=mode,
    )
    return mixed.reshape_as(hidden), snapshot


def patch_model(model: agnews_model.IndependentAutoregressiveMoE) -> None:
    for layer_index, block in enumerate(model.layers):
        block.moe._no_drop_recovery_layer_index = layer_index
        block.moe.forward = MethodType(recovery_forward, block.moe)


def make_model(
    *,
    dataset: agnews_model.IndependentPreparedDataset,
    config: dict[str, Any],
    seed: int,
) -> agnews_model.IndependentAutoregressiveMoE:
    condition = agnews_model.Condition(
        "conditioned_top2",
        "input_conditioned",
        False,
    )
    model = agnews_model.make_model(
        condition,
        dataset=dataset,
        config=config,
        seed=seed,
    )
    patch_model(model)
    return model


def make_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def make_batch_plan(
    *,
    row_count: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    if row_count < batch_size:
        raise ValueError("Training split is smaller than one batch")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    plan: list[torch.Tensor] = []
    order = torch.randperm(row_count, generator=generator)
    cursor = 0
    for _ in range(steps):
        if cursor + batch_size > row_count:
            order = torch.randperm(row_count, generator=generator)
            cursor = 0
        plan.append(order[cursor : cursor + batch_size].clone())
        cursor += batch_size
    return plan


def autocast_context(device: torch.device, precision: str) -> Any:
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def train_step(
    model: agnews_model.IndependentAutoregressiveMoE,
    optimizer: torch.optim.Optimizer,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float | int]:
    tokens, lengths, _ = data
    batch_tokens = tokens[indices].to(device)
    batch_lengths = lengths[indices].to(device)
    inputs = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    mask = agnews_model.target_mask(batch_lengths, inputs.shape[1])
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        logits, _ = model(inputs, mask)
        loss = agnews_model.masked_loss(logits, targets, mask)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite no-drop recovery loss")
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Non-finite no-drop recovery gradient")
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "target_tokens": int(mask.sum().item()),
    }


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
    raw_target = np.zeros(layer_count, dtype=np.float64)
    effective_target = np.zeros(layer_count, dtype=np.float64)
    raw_comparator = np.zeros(layer_count, dtype=np.float64)
    target_logit = np.zeros(layer_count, dtype=np.float64)
    competitor_logsumexp = np.zeros(layer_count, dtype=np.float64)
    target_traffic = np.zeros(layer_count, dtype=np.float64)
    counts = np.zeros(layer_count, dtype=np.int64)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch_tokens = tokens[start : start + batch_size].to(device)
            batch_lengths = lengths[start : start + batch_size].to(device)
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]
            mask = agnews_model.target_mask(batch_lengths, inputs.shape[1])
            with autocast_context(device, precision):
                logits, snapshots = model(inputs, mask)
                loss = agnews_model.masked_loss(logits, targets, mask)
            losses.append(float(loss.item()))
            for layer, snapshot in enumerate(snapshots):
                target = int(target_by_layer[layer])
                comparator = int(comparator_by_layer[layer])
                active = mask
                count = int(active.sum().item())
                raw_target[layer] += float(
                    snapshot.raw_probabilities[..., target][active]
                    .double()
                    .sum()
                    .item()
                )
                effective_target[layer] += float(
                    snapshot.pre_topk_probabilities[..., target][active]
                    .double()
                    .sum()
                    .item()
                )
                raw_comparator[layer] += float(
                    snapshot.raw_probabilities[..., comparator][active]
                    .double()
                    .sum()
                    .item()
                )
                target_logit[layer] += float(
                    snapshot.raw_logits[..., target][active]
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
                    )[active]
                    .double()
                    .sum()
                    .item()
                )
                selected = snapshot.topk_indices.eq(target).any(dim=-1)
                target_traffic[layer] += float(
                    selected[active].double().sum().item()
                )
                counts[layer] += count
    model.train()
    if np.any(counts == 0):
        raise ValueError("Routing diagnostic omitted a layer")
    raw_target /= counts
    effective_target /= counts
    raw_comparator /= counts
    target_logit /= counts
    competitor_logsumexp /= counts
    target_traffic /= counts
    eps = np.finfo(np.float64).tiny
    aggregate_log_odds = np.log(
        np.maximum(effective_target, eps)
        / np.maximum(1.0 - effective_target, eps)
    )
    return {
        "loss": float(np.mean(losses)),
        "raw_target_probability_by_layer": raw_target.tolist(),
        "effective_target_probability_by_layer": (
            effective_target.tolist()
        ),
        "raw_comparator_probability_by_layer": raw_comparator.tolist(),
        "target_logit_by_layer": target_logit.tolist(),
        "competitor_logsumexp_by_layer": competitor_logsumexp.tolist(),
        "target_traffic_by_layer": target_traffic.tolist(),
        "aggregate_target_log_odds_by_layer": aggregate_log_odds.tolist(),
        "active_token_count_by_layer": counts.tolist(),
    }


def select_targets(
    model: agnews_model.IndependentAutoregressiveMoE,
    evaluation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[int], list[int], dict[str, Any]]:
    num_layers = int(config["num_layers"])
    num_experts = int(config["num_experts"])
    defaults = list(range(num_layers))
    targets = [0] * num_layers
    comparators = [1] * num_layers
    configure_intervention(
        "none",
        target_by_layer=targets,
        comparator_by_layer=comparators,
        middle_layers=list(config["middle_layers"]),
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    rows = np.zeros((num_layers, num_experts), dtype=np.float64)
    tokens, lengths, _ = evaluation_data
    counts = np.zeros(num_layers, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(
            0,
            len(tokens),
            int(config["eval_batch_size"]),
        ):
            batch_tokens = tokens[
                start : start + int(config["eval_batch_size"])
            ].to(device)
            batch_lengths = lengths[
                start : start + int(config["eval_batch_size"])
            ].to(device)
            inputs = batch_tokens[:, :-1]
            mask = agnews_model.target_mask(batch_lengths, inputs.shape[1])
            with autocast_context(device, str(config["precision"])):
                _, snapshots = model(inputs, mask)
            count = int(mask.sum().item())
            for layer, snapshot in enumerate(snapshots):
                rows[layer] += (
                    snapshot.raw_probabilities[mask]
                    .double()
                    .sum(dim=0)
                    .cpu()
                    .numpy()
                )
                counts[layer] += count
    model.train()
    rows /= counts[:, None]
    eligible: dict[str, list[int]] = {}
    threshold = float(config["target_eligibility_threshold"])
    for layer in list(config["middle_layers"]):
        choices = [
            expert
            for expert, value in enumerate(rows[layer])
            if float(value) >= threshold
        ]
        if not choices:
            raise ValueError(
                f"No eligible target in middle layer {layer}"
            )
        target = min(
            choices,
            key=lambda expert: (float(rows[layer, expert]), expert),
        )
        comparator = max(
            (expert for expert in range(num_experts) if expert != target),
            key=lambda expert: (float(rows[layer, expert]), -expert),
        )
        targets[layer] = target
        comparators[layer] = comparator
        eligible[str(layer)] = choices
    return targets, comparators, {
        "rule": (
            "In each configured middle layer, select the lowest held-out raw "
            "pre-top-k mass at or above the configured eligibility threshold; "
            "use the highest-mass distinct expert as comparator."
        ),
        "held_out_raw_loads_by_layer": rows.tolist(),
        "eligible_by_layer": eligible,
        "target_by_layer": targets,
        "comparator_by_layer": comparators,
        "layer_inventory": defaults,
    }


def persistent_hit(
    values: list[float],
    *,
    threshold: float,
    persistence: int,
    steps: list[int],
) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value >= threshold else 0
        if run >= persistence:
            return int(steps[index - persistence + 1])
    return None


def run_branch(
    *,
    mode: str,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    dataset: agnews_model.IndependentPreparedDataset,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    evaluation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    future_plan: list[torch.Tensor],
    target_by_layer: list[int],
    comparator_by_layer: list[int],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model = make_model(dataset=dataset, config=config, seed=seed)
    model.load_state_dict(copy.deepcopy(model_state))
    model.to(device)
    optimizer = make_optimizer(model, config)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    configure_intervention(
        mode,
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        middle_layers=list(config["middle_layers"]),
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    evaluations: list[dict[str, Any]] = []
    evaluation_steps: list[int] = []
    losses: list[float] = []
    target_tokens = 0
    for step, indices in enumerate(future_plan, start=1):
        record = train_step(
            model,
            optimizer,
            train_data,
            indices,
            device=device,
            precision=str(config["precision"]),
        )
        losses.append(float(record["loss"]))
        target_tokens += int(record["target_tokens"])
        if step % int(config["eval_interval"]) == 0:
            evaluations.append(
                routing_diagnostics(
                    model,
                    evaluation_data,
                    batch_size=int(config["eval_batch_size"]),
                    target_by_layer=target_by_layer,
                    comparator_by_layer=comparator_by_layer,
                    device=device,
                    precision=str(config["precision"]),
                )
            )
            evaluation_steps.append(step)
    threshold = float(config["recovery_probability_threshold"])
    persistence = int(config["recovery_persistence"])
    recovery_by_layer: list[int | None] = [None] * int(
        config["num_layers"]
    )
    budget_surplus_by_layer: list[list[float]] = [
        [] for _ in range(int(config["num_layers"]))
    ]
    for layer in list(config["middle_layers"]):
        series = [
            float(item["effective_target_probability_by_layer"][layer])
            for item in evaluations
        ]
        recovery_by_layer[layer] = persistent_hit(
            series,
            threshold=threshold,
            persistence=persistence,
            steps=evaluation_steps,
        )
        if series:
            initial = max(series[0], 1e-15)
            budget = math.log(
                threshold * (1.0 - initial)
                / ((1.0 - threshold) * initial)
            )
            initial_log_odds = float(
                evaluations[0][
                    "aggregate_target_log_odds_by_layer"
                ][layer]
            )
            budget_surplus_by_layer[layer] = [
                float(
                    item["aggregate_target_log_odds_by_layer"][layer]
                    - initial_log_odds
                    - budget
                )
                for item in evaluations
            ]
    return {
        "mode": mode,
        "optimizer_steps": len(future_plan),
        "target_tokens": target_tokens,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "evaluation_steps": evaluation_steps,
        "evaluations": evaluations,
        "recovery_step_by_layer": recovery_by_layer,
        "budget_surplus_by_layer": budget_surplus_by_layer,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("No-drop recovery run requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not report BF16 support")
    agnews_model.configure_execution(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    config = args.config
    dataset = agnews_model.IndependentPreparedDataset(
        args.dataset_dir,
        expected_status=str(config["expected_dataset_status"]),
    )
    train_data = dataset.split(
        "train",
        limit=int(config["limit_train"]),
    )
    evaluation_data = dataset.split(
        "validation",
        limit=int(config["limit_evaluation"]),
    )
    total_steps = (
        int(config["checkpoint_steps"])
        + int(config["warmup_steps"])
        + int(config["horizon_steps"])
    )
    plan = make_batch_plan(
        row_count=len(train_data[0]),
        steps=total_steps,
        batch_size=int(config["batch_size"]),
        seed=args.seed + int(config["batch_plan_seed_offset"]),
    )
    checkpoint_plan = plan[: int(config["checkpoint_steps"])]
    warmup_start = int(config["checkpoint_steps"])
    warmup_end = warmup_start + int(config["warmup_steps"])
    warmup_plan = plan[warmup_start:warmup_end]
    future_plan = plan[warmup_end:]

    model = make_model(dataset=dataset, config=config, seed=args.seed)
    model.to(device)
    optimizer = make_optimizer(model, config)
    defaults = [0] * int(config["num_layers"])
    comparators = [1] * int(config["num_layers"])
    configure_intervention(
        "none",
        target_by_layer=defaults,
        comparator_by_layer=comparators,
        middle_layers=list(config["middle_layers"]),
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    checkpoint_tokens = 0
    for indices in checkpoint_plan:
        record = train_step(
            model,
            optimizer,
            train_data,
            indices,
            device=device,
            precision=str(config["precision"]),
        )
        checkpoint_tokens += int(record["target_tokens"])
    target_by_layer, comparator_by_layer, target_selection = select_targets(
        model,
        evaluation_data,
        config=config,
        device=device,
    )
    configure_intervention(
        "suppressed",
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        middle_layers=list(config["middle_layers"]),
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    warmup_tokens = 0
    for indices in warmup_plan:
        record = train_step(
            model,
            optimizer,
            train_data,
            indices,
            device=device,
            precision=str(config["precision"]),
        )
        warmup_tokens += int(record["target_tokens"])
    entry_suppressed = routing_diagnostics(
        model,
        evaluation_data,
        batch_size=int(config["eval_batch_size"]),
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        device=device,
        precision=str(config["precision"]),
    )
    configure_intervention(
        "none",
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        middle_layers=list(config["middle_layers"]),
        clamp_margin=float(config["clamp_margin"]),
        control_margin=float(config["control_margin"]),
    )
    entry_raw_release = routing_diagnostics(
        model,
        evaluation_data,
        batch_size=int(config["eval_batch_size"]),
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        device=device,
        precision=str(config["precision"]),
    )
    release_model_state = copy.deepcopy(model.state_dict())
    release_optimizer_state = copy.deepcopy(optimizer.state_dict())
    del model
    del optimizer

    observed = run_branch(
        mode="none",
        model_state=release_model_state,
        optimizer_state=release_optimizer_state,
        dataset=dataset,
        train_data=train_data,
        evaluation_data=evaluation_data,
        future_plan=future_plan,
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        config=config,
        seed=args.seed,
        device=device,
    )
    control = run_branch(
        mode="no_positive_advantage_control",
        model_state=release_model_state,
        optimizer_state=release_optimizer_state,
        dataset=dataset,
        train_data=train_data,
        evaluation_data=evaluation_data,
        future_plan=future_plan,
        target_by_layer=target_by_layer,
        comparator_by_layer=comparator_by_layer,
        config=config,
        seed=args.seed,
        device=device,
    )
    middle_layers = list(config["middle_layers"])
    entry_gate_by_layer = {
        str(layer): (
            float(
                entry_suppressed[
                    "effective_target_probability_by_layer"
                ][layer]
            )
            <= float(config["entry_probability_ceiling"])
            and float(
                entry_suppressed["target_traffic_by_layer"][layer]
            )
            <= float(config["entry_traffic_ceiling"])
        )
        for layer in middle_layers
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "study": STUDY,
        "role": args.role,
        "protocol_id": args.protocol_id,
        "config": {"seed": args.seed, **config},
        "dataset": {
            "path": str(dataset.root),
            "status": dataset.status,
            "domain_names": dataset.domain_names,
            "train_sequences": len(train_data[0]),
            "evaluation_sequences": len(evaluation_data[0]),
        },
        "target_selection": target_selection,
        "checkpoint": {
            "optimizer_steps": int(config["checkpoint_steps"]),
            "target_tokens": checkpoint_tokens,
        },
        "warmup": {
            "optimizer_steps": int(config["warmup_steps"]),
            "target_tokens": warmup_tokens,
            "entry_suppressed": entry_suppressed,
            "entry_raw_release": entry_raw_release,
            "entry_gate_by_layer": entry_gate_by_layer,
        },
        "branches": {
            "observed_release": observed,
            "no_positive_advantage_control": control,
        },
    }
    destination = output_dir / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "entry_gate_by_layer": entry_gate_by_layer,
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return result
