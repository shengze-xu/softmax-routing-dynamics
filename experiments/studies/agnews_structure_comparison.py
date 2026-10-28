#!/usr/bin/env python3
"""Independent AG News routing-structure replication implementation.

This module does not import the earlier multidomain model, routing, logging,
or training implementations and uses a non-overlapping AG News corpus.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 2
DATASET_STATUS = "prepared_agnews_candidate"
CONDITIONS = (
    "shared_global_top2",
    "conditioned_top2",
    "shared_global_top2_capacity_1p25",
    "conditioned_top2_capacity_1p25",
)
CAPACITY_FACTOR = 1.25


def manifest_domain_names(manifest: dict[str, Any]) -> list[str]:
    names = manifest.get("domain_names")
    if names is None:
        names = manifest.get("selection", {}).get("domain_names")
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("Dataset domain names are missing or invalid")
    return list(names)


class IndependentPreparedDataset:
    """Prepared-array loader independent of earlier dataset classes."""

    def __init__(
        self,
        root: Path,
        *,
        expected_status: str = DATASET_STATUS,
    ) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported independent dataset schema")
        if self.manifest.get("status") != expected_status:
            raise ValueError(
                "Dataset status does not match the frozen protocol"
            )
        self.status = str(self.manifest["status"])
        selection = self.manifest["selection"]
        self.domain_names = manifest_domain_names(self.manifest)
        self.sequence_length = int(selection["sequence_length"])
        self.vocab_size = int(self.manifest["tokenizer"]["vocab_size"])
        if len(self.domain_names) != 4:
            raise ValueError("Independent replication requires four regimes")
        self.arrays: dict[str, np.ndarray] = {}
        for name, metadata in self.manifest["arrays"].items():
            path = self.root / metadata["file"]
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != metadata["shape"]:
                raise ValueError(f"Dataset shape mismatch: {name}")
            if str(value.dtype) != metadata["dtype"]:
                raise ValueError(f"Dataset dtype mismatch: {name}")
            self.arrays[name] = value
        for split in ("train", "validation"):
            width = self.arrays[f"{split}_tokens"].shape[1]
            if width != self.sequence_length + 1:
                raise ValueError(f"Unexpected token width for {split}")

    def split(
        self,
        name: str,
        *,
        limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.arrays[f"{name}_tokens"]
        lengths = self.arrays[f"{name}_lengths"]
        domains = self.arrays[f"{name}_domains"]
        if limit < len(self.domain_names):
            raise ValueError("A split limit must retain every regime")
        per_domain, remainder = divmod(limit, len(self.domain_names))
        selected: list[int] = []
        for domain in range(len(self.domain_names)):
            candidates = np.flatnonzero(domains == domain)
            take = per_domain + int(domain < remainder)
            if len(candidates) < take:
                raise ValueError(f"Insufficient rows for domain {domain}")
            selected.extend(candidates[:take].tolist())
        index = np.asarray(sorted(selected), dtype=np.int64)
        return (
            torch.from_numpy(np.asarray(tokens[index], dtype=np.int64)),
            torch.from_numpy(np.asarray(lengths[index], dtype=np.int64)),
            torch.from_numpy(np.asarray(domains[index], dtype=np.int64)),
        )


@dataclass(frozen=True)
class Condition:
    name: str
    router: str
    capacity_limited: bool


def make_conditions() -> tuple[Condition, ...]:
    return (
        Condition("shared_global_top2", "shared_global", False),
        Condition("conditioned_top2", "input_conditioned", False),
        Condition(
            "shared_global_top2_capacity_1p25",
            "shared_global",
            True,
        ),
        Condition(
            "conditioned_top2_capacity_1p25",
            "input_conditioned",
            True,
        ),
    )


@dataclass
class RoutingSnapshot:
    pre_topk_probabilities: torch.Tensor
    topk_indices: torch.Tensor
    accepted_assignments: torch.Tensor
    active_mask: torch.Tensor
    capacity_per_expert: int | None


class IndependentSparseMoE(nn.Module):
    """Second implementation of a sparse top-2 feed-forward layer."""

    def __init__(
        self,
        *,
        d_model: int,
        expert_hidden: int,
        num_experts: int,
        router: str,
        capacity_limited: bool,
    ) -> None:
        super().__init__()
        if router not in {"shared_global", "input_conditioned"}:
            raise ValueError(f"Unknown router: {router}")
        self.num_experts = num_experts
        self.top_k = 2
        self.router_kind = router
        self.capacity_limited = capacity_limited
        if router == "input_conditioned":
            self.router = nn.Linear(d_model, num_experts)
            self.register_parameter("global_logits", None)
        else:
            self.router = None
            self.global_logits = nn.Parameter(torch.empty(num_experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, expert_hidden),
                    nn.GELU(),
                    nn.Linear(expert_hidden, d_model),
                )
                for _ in range(num_experts)
            ]
        )

    def reset_router(self, generator: torch.Generator) -> None:
        if self.router is not None:
            nn.init.normal_(
                self.router.weight,
                mean=0.0,
                std=0.01,
                generator=generator,
            )
            nn.init.zeros_(self.router.bias)
        else:
            nn.init.normal_(
                self.global_logits,
                mean=0.0,
                std=0.01,
                generator=generator,
            )

    def forward(
        self,
        hidden: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, RoutingSnapshot]:
        if self.router is None:
            logits = self.global_logits.view(1, 1, -1).expand(
                hidden.shape[0],
                hidden.shape[1],
                -1,
            )
        else:
            logits = self.router(hidden)
        probabilities = F.softmax(logits.float(), dim=-1)
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
        capacity: int | None = None
        if self.capacity_limited:
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
        snapshot = RoutingSnapshot(
            pre_topk_probabilities=probabilities,
            topk_indices=top_indices,
            accepted_assignments=accepted.reshape_as(top_indices),
            active_mask=active_mask,
            capacity_per_expert=capacity,
        )
        return mixed.reshape_as(hidden), snapshot


class IndependentBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        expert_hidden: int,
        num_experts: int,
        router: str,
        capacity_limited: bool,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_moe = nn.LayerNorm(d_model)
        self.moe = IndependentSparseMoE(
            d_model=d_model,
            expert_hidden=expert_hidden,
            num_experts=num_experts,
            router=router,
            capacity_limited=capacity_limited,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        active_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, RoutingSnapshot]:
        normalized = self.norm_attention(hidden)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = hidden + attention_output
        moe_output, snapshot = self.moe(
            self.norm_moe(hidden),
            active_mask,
        )
        return hidden + moe_output, snapshot


class IndependentAutoregressiveMoE(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        expert_hidden: int,
        num_experts: int,
        router: str,
        capacity_limited: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.layers = nn.ModuleList(
            [
                IndependentBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    expert_hidden=expert_hidden,
                    num_experts=num_experts,
                    router=router,
                    capacity_limited=capacity_limited,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for name, parameter in self.named_parameters():
            if "global_logits" in name or ".router." in name:
                continue
            if parameter.ndim > 1:
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02,
                    generator=generator,
                )
            elif "norm" in name and name.endswith("weight"):
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)
        for layer in self.layers:
            layer.moe.reset_router(generator)

    def forward(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[RoutingSnapshot]]:
        positions = torch.arange(
            tokens.shape[1],
            device=tokens.device,
        ).unsqueeze(0)
        hidden = self.token_embedding(tokens)
        hidden = hidden + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                tokens.shape[1],
                tokens.shape[1],
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        snapshots: list[RoutingSnapshot] = []
        for layer in self.layers:
            hidden, snapshot = layer(
                hidden,
                active_mask,
                causal_mask,
            )
            snapshots.append(snapshot)
        return self.output(self.final_norm(hidden)), snapshots


def target_mask(
    lengths: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    usable = torch.clamp(lengths - 1, min=0, max=sequence_length)
    positions = torch.arange(
        sequence_length,
        device=lengths.device,
    ).unsqueeze(0)
    return positions < usable.unsqueeze(1)


def masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    flat_mask = mask.reshape(-1)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[flat_mask],
        targets.reshape(-1)[flat_mask],
    )


def empty_accounting(num_experts: int) -> dict[str, Any]:
    return {
        "batches": 0,
        "active_tokens": 0,
        "original_assignments": 0,
        "accepted_assignments": 0,
        "overflow_assignments": 0,
        "dropped_assignments": 0,
        "tokens_with_any_drop": 0,
        "fully_dropped_tokens": 0,
        "capacity_slots": 0,
        "capacity_per_expert_min": None,
        "capacity_per_expert_max": None,
        "original_by_expert": [0] * num_experts,
        "accepted_by_expert": [0] * num_experts,
    }


def update_accounting(
    record: dict[str, Any],
    snapshot: RoutingSnapshot,
    num_experts: int,
) -> None:
    active = snapshot.active_mask
    indices = snapshot.topk_indices
    accepted = snapshot.accepted_assignments
    original_mask = active.unsqueeze(-1).expand_as(indices)
    overflow = original_mask & ~accepted
    record["batches"] += 1
    record["active_tokens"] += int(active.sum().item())
    record["original_assignments"] += int(original_mask.sum().item())
    record["accepted_assignments"] += int(accepted.sum().item())
    record["overflow_assignments"] += int(overflow.sum().item())
    record["dropped_assignments"] += int(overflow.sum().item())
    dropped_per_token = overflow.sum(dim=-1)
    record["tokens_with_any_drop"] += int(
        (active & (dropped_per_token > 0)).sum().item()
    )
    record["fully_dropped_tokens"] += int(
        (active & (dropped_per_token == 2)).sum().item()
    )
    for expert in range(num_experts):
        record["original_by_expert"][expert] += int(
            (original_mask & (indices == expert)).sum().item()
        )
        record["accepted_by_expert"][expert] += int(
            (accepted & (indices == expert)).sum().item()
        )
    capacity = snapshot.capacity_per_expert
    if capacity is not None:
        record["capacity_slots"] += capacity * num_experts
        minimum = record["capacity_per_expert_min"]
        maximum = record["capacity_per_expert_max"]
        record["capacity_per_expert_min"] = (
            capacity if minimum is None else min(minimum, capacity)
        )
        record["capacity_per_expert_max"] = (
            capacity if maximum is None else max(maximum, capacity)
        )


def accounting_valid(record: dict[str, Any]) -> bool:
    return (
        record["original_assignments"]
        == record["accepted_assignments"] + record["overflow_assignments"]
        and record["overflow_assignments"] == record["dropped_assignments"]
        and sum(record["original_by_expert"])
        == record["original_assignments"]
        and sum(record["accepted_by_expert"])
        == record["accepted_assignments"]
    )


def structural_score(rows: np.ndarray) -> float:
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError("Structural rows must be regime by expert")
    center = rows.mean(axis=0, keepdims=True)
    return float(np.abs(rows - center).sum(axis=1).mean())


def evaluate(
    model: IndependentAutoregressiveMoE,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
    num_domains: int,
    num_experts: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    tokens, lengths, domains = data
    num_layers = len(model.layers)
    sums = np.zeros(
        (num_layers, num_domains, num_experts),
        dtype=np.float64,
    )
    counts = np.zeros((num_layers, num_domains), dtype=np.int64)
    accounting = [
        empty_accounting(num_experts) for _ in range(num_layers)
    ]
    losses: list[float] = []
    shared_row_error = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch_tokens = tokens[start : start + batch_size].to(device)
            batch_lengths = lengths[start : start + batch_size].to(device)
            batch_domains = domains[start : start + batch_size].to(device)
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]
            mask = target_mask(batch_lengths, inputs.shape[1])
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and precision == "bf16"
                else torch.autocast(device.type, enabled=False)
            )
            with context:
                logits, snapshots = model(inputs, mask)
                loss = masked_loss(logits, targets, mask)
            losses.append(float(loss.item()))
            for layer_index, snapshot in enumerate(snapshots):
                update_accounting(
                    accounting[layer_index],
                    snapshot,
                    num_experts,
                )
                probabilities = snapshot.pre_topk_probabilities
                active_probabilities = probabilities[mask]
                if active_probabilities.shape[0] > 1:
                    reference = active_probabilities[0]
                    shared_row_error = max(
                        shared_row_error,
                        float(
                            torch.max(
                                torch.abs(
                                    active_probabilities - reference
                                )
                            ).item()
                        ),
                    )
                for domain in range(num_domains):
                    domain_mask = mask & batch_domains.eq(domain).unsqueeze(1)
                    count = int(domain_mask.sum().item())
                    if count == 0:
                        continue
                    values = probabilities[domain_mask]
                    sums[layer_index, domain] += (
                        values.sum(dim=0).detach().cpu().numpy()
                    )
                    counts[layer_index, domain] += count
    if np.any(counts == 0):
        raise ValueError("Evaluation omitted a layer-domain cell")
    rows = sums / counts[..., None]
    layer_scores = [
        structural_score(rows[layer]) for layer in range(num_layers)
    ]
    return {
        "loss": float(np.mean(losses)),
        "layer_domain_pre_topk_rows": rows.tolist(),
        "layer_structural_scores": layer_scores,
        "mean_structural_score": float(np.mean(layer_scores)),
        "domain_token_counts": counts[0].tolist(),
        "shared_row_max_error": shared_row_error,
        "capacity_accounting_by_layer": accounting,
        "capacity_accounting_valid": all(
            accounting_valid(record) for record in accounting
        ),
    }


def common_state_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return [
        key
        for key in state
        if ".moe.router." not in key and ".moe.global_logits" not in key
    ]


def make_model(
    condition: Condition,
    *,
    dataset: IndependentPreparedDataset,
    config: dict[str, Any],
    seed: int,
) -> IndependentAutoregressiveMoE:
    return IndependentAutoregressiveMoE(
        vocab_size=dataset.vocab_size,
        sequence_length=dataset.sequence_length,
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        expert_hidden=int(config["expert_hidden"]),
        num_experts=int(config["num_experts"]),
        router=condition.router,
        capacity_limited=condition.capacity_limited,
        seed=seed,
    )


def initial_states(
    *,
    dataset: IndependentPreparedDataset,
    config: dict[str, Any],
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    conditions = make_conditions()
    conditioned = make_model(
        conditions[1],
        dataset=dataset,
        config=config,
        seed=seed,
    )
    shared = make_model(
        conditions[0],
        dataset=dataset,
        config=config,
        seed=seed,
    )
    conditioned_state = {
        key: value.detach().cpu().clone()
        for key, value in conditioned.state_dict().items()
    }
    shared_state = shared.state_dict()
    for key in common_state_keys(shared_state):
        shared_state[key].copy_(conditioned_state[key])
    shared.load_state_dict(shared_state)
    result = {
        "input_conditioned": {
            key: value.detach().cpu().clone()
            for key, value in conditioned.state_dict().items()
        },
        "shared_global": {
            key: value.detach().cpu().clone()
            for key, value in shared.state_dict().items()
        },
    }
    del conditioned
    del shared
    return result


def train_condition(
    condition: Condition,
    *,
    dataset: IndependentPreparedDataset,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    state: dict[str, torch.Tensor],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model = make_model(
        condition,
        dataset=dataset,
        config=config,
        seed=seed,
    )
    model.load_state_dict(state)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    initial = evaluate(
        model,
        validation_data,
        batch_size=int(config["eval_batch_size"]),
        num_domains=len(dataset.domain_names),
        num_experts=int(config["num_experts"]),
        device=device,
        precision=str(config["precision"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_tokens, train_lengths, _ = train_data
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 9001)
    order = torch.randperm(len(train_tokens), generator=generator)
    batch_size = int(config["batch_size"])
    max_steps = int(config["max_steps"])
    training_accounting = [
        empty_accounting(int(config["num_experts"]))
        for _ in range(int(config["num_layers"]))
    ]
    losses: list[float] = []
    gradient_norms: list[float] = []
    training_target_tokens = 0
    training_started = time.monotonic()
    model.train()
    for step in range(max_steps):
        start = (step * batch_size) % len(order)
        index = order[start : start + batch_size]
        if len(index) < batch_size:
            index = torch.cat((index, order[: batch_size - len(index)]))
        batch_tokens = train_tokens[index].to(device)
        batch_lengths = train_lengths[index].to(device)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]
        mask = target_mask(batch_lengths, inputs.shape[1])
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and config["precision"] == "bf16"
            else torch.autocast(device.type, enabled=False)
        )
        optimizer.zero_grad(set_to_none=True)
        with context:
            logits, snapshots = model(inputs, mask)
            loss = masked_loss(logits, targets, mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite AG News smoke loss")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Non-finite AG News gradient")
        optimizer.step()
        losses.append(float(loss.item()))
        gradient_norms.append(float(gradient_norm.item()))
        training_target_tokens += int(mask.sum().item())
        for layer_index, snapshot in enumerate(snapshots):
            update_accounting(
                training_accounting[layer_index],
                snapshot,
                int(config["num_experts"]),
            )
    training_seconds = time.monotonic() - training_started

    final = evaluate(
        model,
        validation_data,
        batch_size=int(config["eval_batch_size"]),
        num_domains=len(dataset.domain_names),
        num_experts=int(config["num_experts"]),
        device=device,
        precision=str(config["precision"]),
    )
    return {
        "condition": {
            "name": condition.name,
            "router": condition.router,
            "capacity_limited": condition.capacity_limited,
            "capacity_factor": (
                CAPACITY_FACTOR if condition.capacity_limited else None
            ),
        },
        "optimizer_steps": max_steps,
        "training_target_tokens": training_target_tokens,
        "training_seconds": training_seconds,
        "training_target_tokens_per_second": (
            training_target_tokens / training_seconds
        ),
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "gradient_norm_last": gradient_norms[-1],
        "training_capacity_accounting_by_layer": training_accounting,
        "training_capacity_accounting_valid": all(
            accounting_valid(record) for record in training_accounting
        ),
        "initial_validation": initial,
        "final_validation": final,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }


def configure_execution(device: torch.device) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Independent-replication run requires CUDA")
    if args.config["precision"] != "bf16":
        raise ValueError("Independent-replication precision must be BF16")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not report BF16 support")
    configure_execution(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    expected_dataset_status = getattr(
        args,
        "expected_dataset_status",
        DATASET_STATUS,
    )
    role = getattr(args, "role", "smoke")
    dataset = IndependentPreparedDataset(
        args.dataset_dir,
        expected_status=expected_dataset_status,
    )
    train_data = dataset.split(
        "train",
        limit=int(args.config["limit_train"]),
    )
    validation_data = dataset.split(
        "validation",
        limit=int(args.config["limit_validation"]),
    )
    states = initial_states(
        dataset=dataset,
        config=args.config,
        seed=args.seed,
    )
    records: list[dict[str, Any]] = []
    for condition in make_conditions():
        print(
            json.dumps(
                {
                    "event": "agnews_model_condition_start",
                    "condition": condition.name,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        record = train_condition(
            condition,
            dataset=dataset,
            train_data=train_data,
            validation_data=validation_data,
            state=states[condition.router],
            config=args.config,
            seed=args.seed,
            device=device,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "agnews_model_condition_complete",
                    "condition": condition.name,
                    "loss": record["training_loss_last"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "study": "agnews_structure_comparison",
        "role": role,
        "protocol_id": args.protocol_id,
        "config": {"seed": args.seed, **args.config},
        "dataset": {
            "path": str(dataset.root),
            "status": dataset.status,
            "domain_names": dataset.domain_names,
            "vocab_size": dataset.vocab_size,
            "sequence_length": dataset.sequence_length,
            "train_sequences": len(train_data[0]),
            "validation_sequences": len(validation_data[0]),
        },
        "conditions": records,
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
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--role", choices=("smoke", "primary"), default="primary")
    parser.add_argument("--seed", type=int, required=True)
    configuration = parser.add_mutually_exclusive_group(required=True)
    configuration.add_argument("--config", type=Path)
    configuration.add_argument("--config-json")
    parser.add_argument("--device", default="cuda")
    values = parser.parse_args()
    values.config = json.loads(values.config.read_text() if values.config else values.config_json)
    values.expected_dataset_status = DATASET_STATUS
    run(values)


if __name__ == "__main__":
    main()
