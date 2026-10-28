#!/usr/bin/env python3
"""Autoregressive MoE models and training utilities for AG News.

Class labels stratify routing summaries and are not inputs to the router."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = 1
DOMAIN_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass(frozen=True)
class Condition:
    name: str
    router: str
    balance_lambda: float
    dense: bool = False


CONDITIONS = (
    Condition("shared_global_top2", "shared_global", 0.0),
    Condition("shared_global_top2_balanced", "shared_global", 0.01),
    Condition("conditioned_top2", "input_conditioned", 0.0),
    Condition("conditioned_top2_balanced", "input_conditioned", 0.01),
    Condition("dense_active_matched", "dense", 0.0, dense=True),
)


@dataclass
class RoutingSnapshot:
    logits: torch.Tensor
    pre_topk_probabilities: torch.Tensor
    topk_indices: torch.Tensor
    post_topk_weights: torch.Tensor


class PreparedDataset:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported prepared-data schema")
        if self.manifest.get("status") != "prepared_agnews_candidate":
            raise ValueError("Prepared-data manifest has an unsupported status")
        self.arrays: dict[str, np.ndarray] = {}
        for name, metadata in self.manifest["arrays"].items():
            path = self.root / metadata["file"]
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
                raise ValueError(f"Prepared-data shape/dtype mismatch: {name}")
            self.arrays[name] = array
        self.vocab_size = int(self.manifest["tokenizer"]["vocab_size"])
        self.sequence_length = int(self.manifest["selection"]["sequence_length"])
        if self.arrays["train_tokens"].shape[1] != self.sequence_length + 1:
            raise ValueError("Prepared token width does not match manifest")

    def split(
        self,
        name: str,
        *,
        limit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.arrays[f"{name}_tokens"]
        lengths = self.arrays[f"{name}_lengths"]
        domains = self.arrays[f"{name}_domains"]
        if not limit or limit <= 0 or limit >= len(tokens):
            index = np.arange(len(tokens))
        else:
            if limit < len(DOMAIN_NAMES):
                raise ValueError(
                    f"{name} limit must retain at least one item per domain"
                )
            selected = []
            per_domain, remainder = divmod(limit, len(DOMAIN_NAMES))
            for domain in range(len(DOMAIN_NAMES)):
                domain_index = np.flatnonzero(domains == domain)
                take = per_domain + int(domain < remainder)
                selected.extend(domain_index[:take].tolist())
            index = np.asarray(sorted(selected), dtype=np.int64)
        return (
            torch.from_numpy(np.asarray(tokens[index], dtype=np.int64)),
            torch.from_numpy(np.asarray(lengths[index], dtype=np.int64)),
            torch.from_numpy(np.asarray(domains[index], dtype=np.int64)),
        )


class SparseTokenMoE(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        expert_hidden: int,
        num_experts: int,
        top_k: int,
        router: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router
        if router == "input_conditioned":
            self.router = nn.Linear(d_model, num_experts)
        elif router == "shared_global":
            self.global_logits = nn.Parameter(torch.zeros(num_experts))
        else:
            raise ValueError(f"Unsupported router: {router}")
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

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, RoutingSnapshot]:
        if self.router_type == "input_conditioned":
            logits = self.router(hidden)
        else:
            logits = self.global_logits.view(1, 1, -1).expand(
                hidden.shape[0],
                hidden.shape[1],
                -1,
            )
        pre_topk = F.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(
            pre_topk,
            k=self.top_k,
            dim=-1,
        )
        topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True)
        post_topk = torch.zeros_like(pre_topk)
        post_topk.scatter_(dim=-1, index=topk_indices, src=topk_weights)

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_weights = topk_weights.reshape(-1, self.top_k)
        mixed = torch.zeros_like(flat_hidden)
        for expert_index, expert in enumerate(self.experts):
            token_rows, slots = torch.where(flat_indices == expert_index)
            if token_rows.numel() == 0:
                continue
            expert_output = expert(flat_hidden[token_rows])
            weighted = expert_output * flat_weights[token_rows, slots].unsqueeze(-1)
            mixed.index_add_(0, token_rows, weighted)
        return mixed.reshape_as(hidden), RoutingSnapshot(
            logits=logits,
            pre_topk_probabilities=pre_topk,
            topk_indices=topk_indices,
            post_topk_weights=post_topk,
        )


class SparseBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        expert_hidden: int,
        num_experts: int,
        top_k: int,
        router: str,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = SparseTokenMoE(
            d_model=d_model,
            expert_hidden=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            router=router,
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, RoutingSnapshot]:
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = self.norm1(hidden + attended)
        update, snapshot = self.moe(hidden)
        return self.norm2(hidden + update), snapshot


class DenseBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dense_hidden: int,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dense_hidden),
            nn.GELU(),
            nn.Linear(dense_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = self.norm1(hidden + attended)
        return self.norm2(hidden + self.ffn(hidden))


class TinyAutoregressiveModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        expert_hidden: int,
        num_experts: int,
        top_k: int,
        condition: Condition,
    ) -> None:
        super().__init__()
        self.condition = condition
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        if condition.dense:
            dense_hidden = top_k * expert_hidden
            self.blocks = nn.ModuleList(
                [
                    DenseBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        dense_hidden=dense_hidden,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    SparseBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        expert_hidden=expert_hidden,
                        num_experts=num_experts,
                        top_k=top_k,
                        router=condition.router,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[RoutingSnapshot]]:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
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
        for block in self.blocks:
            if self.condition.dense:
                hidden = block(hidden, causal_mask)
            else:
                hidden, snapshot = block(hidden, causal_mask)
                snapshots.append(snapshot)
        return self.lm_head(self.final_norm(hidden)), snapshots


def target_mask(lengths: torch.Tensor, sequence_length: int) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=lengths.device).unsqueeze(0)
    return positions < (lengths - 1).unsqueeze(1)


def masked_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    active = mask.reshape(-1)
    return losses[active].mean()


def balance_loss(
    snapshots: list[RoutingSnapshot],
    mask: torch.Tensor,
) -> torch.Tensor:
    penalties = []
    for snapshot in snapshots:
        active_pre = snapshot.pre_topk_probabilities[mask]
        active_post = (snapshot.post_topk_weights[mask] > 0).to(active_pre.dtype)
        importance = active_pre.mean(dim=0)
        dispatch = active_post.mean(dim=0) / snapshot.topk_indices.shape[-1]
        penalties.append(
            importance.numel() * torch.sum(importance * dispatch)
        )
    return torch.stack(penalties).mean()


def batches(
    size: int,
    *,
    batch_size: int,
    orders: list[torch.Tensor] | None = None,
) -> Iterator[torch.Tensor]:
    if orders is None:
        orders = [torch.arange(size)]
    for order in orders:
        for start in range(0, size, batch_size):
            yield order[start : start + batch_size]


def make_orders(size: int, epochs: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 8100)
    return [torch.randperm(size, generator=generator) for _ in range(epochs)]


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_model(
    condition: Condition,
    *,
    dataset: PreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> TinyAutoregressiveModel:
    return TinyAutoregressiveModel(
        vocab_size=dataset.vocab_size,
        max_length=dataset.sequence_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        expert_hidden=args.expert_hidden,
        num_experts=args.num_experts,
        top_k=args.top_k,
        condition=condition,
    ).to(device)


def common_initialization(
    *,
    dataset: PreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    template = build_model(
        next(condition for condition in CONDITIONS if condition.name == "conditioned_top2"),
        dataset=dataset,
        args=args,
        device=torch.device("cpu"),
    )
    state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    keys = [
        key
        for key in state
        if ".moe.router." not in key and ".moe.global_logits" not in key
    ]
    del template
    return state, keys


def apply_common_initialization(
    model: TinyAutoregressiveModel,
    template_state: dict[str, torch.Tensor],
    common_keys: list[str],
) -> None:
    state = model.state_dict()
    copied = []
    for key in common_keys:
        if key in state and state[key].shape == template_state[key].shape:
            state[key].copy_(template_state[key].to(state[key].device))
            copied.append(key)
    model.load_state_dict(state)
    if model.condition.dense:
        return None
    if set(copied) != set(common_keys):
        missing = sorted(set(common_keys) - set(copied))
        raise ValueError(f"Sparse condition missed common initialization keys: {missing}")
    return None


def evaluate(
    model: TinyAutoregressiveModel,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
    dataset: PreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    tokens, lengths, domains = data
    total_loss = 0.0
    total_tokens = 0
    pre_sums = torch.zeros(
        args.num_layers,
        len(DOMAIN_NAMES),
        args.num_experts,
        dtype=torch.float64,
    )
    post_sums = torch.zeros_like(pre_sums)
    route_counts = torch.zeros(
        args.num_layers,
        len(DOMAIN_NAMES),
        dtype=torch.float64,
    )
    model.eval()
    with torch.no_grad():
        for index in batches(len(tokens), batch_size=batch_size):
            batch_tokens = tokens[index].to(device, non_blocking=True)
            batch_lengths = lengths[index].to(device, non_blocking=True)
            batch_domains = domains[index]
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]
            mask = target_mask(batch_lengths, dataset.sequence_length)
            with autocast_context(device, args.precision):
                logits, snapshots = model(inputs)
                loss = masked_lm_loss(logits, targets, mask)
            active_count = int(mask.sum().item())
            total_loss += float(loss.item()) * active_count
            total_tokens += active_count
            for layer, snapshot in enumerate(snapshots):
                pre = snapshot.pre_topk_probabilities.float().cpu()
                post = snapshot.post_topk_weights.float().cpu()
                mask_cpu = mask.cpu()
                for domain in range(len(DOMAIN_NAMES)):
                    selected = batch_domains == domain
                    domain_mask = mask_cpu[selected]
                    count = int(domain_mask.sum().item())
                    if count == 0:
                        continue
                    pre_sums[layer, domain] += pre[selected][domain_mask].double().sum(dim=0)
                    post_sums[layer, domain] += post[selected][domain_mask].double().sum(dim=0)
                    route_counts[layer, domain] += count
    result: dict[str, object] = {
        "loss": total_loss / total_tokens,
        "perplexity": math.exp(min(20.0, total_loss / total_tokens)),
        "target_tokens": total_tokens,
    }
    if not model.condition.dense:
        if bool(torch.any(route_counts == 0)):
            raise ValueError("At least one layer/domain has no validation routing tokens")
        pre_loads = pre_sums / route_counts.unsqueeze(-1)
        post_loads = post_sums / route_counts.unsqueeze(-1)

        def layer_specialization(loads: torch.Tensor) -> list[float]:
            centered = loads - loads.mean(dim=1, keepdim=True)
            return centered.abs().sum(dim=-1).mean(dim=-1).tolist()

        pre_specialization = layer_specialization(pre_loads)
        post_specialization = layer_specialization(post_loads)
        winners = [
            int(torch.unique(torch.argmax(post_loads[layer], dim=-1)).numel())
            for layer in range(args.num_layers)
        ]
        result.update(
            {
                "pre_topk_domain_loads": pre_loads.tolist(),
                "post_topk_domain_loads": post_loads.tolist(),
                "pre_topk_specialization_by_layer": pre_specialization,
                "post_topk_specialization_by_layer": post_specialization,
                "pre_topk_specialization_mean": float(np.mean(pre_specialization)),
                "post_topk_specialization_mean": float(np.mean(post_specialization)),
                "unique_dominant_experts_by_layer": winners,
                "max_post_topk_global_load": float(
                    torch.max(
                        post_sums.sum(dim=1)
                        / route_counts.sum(dim=1, keepdim=True)
                    ).item()
                ),
                "shared_row_max_error": float(
                    torch.max(
                        torch.abs(pre_loads - pre_loads[:, :1, :])
                    ).item()
                )
                if model.condition.router == "shared_global"
                else None,
            }
        )
    return result


def train_condition(
    condition: Condition,
    *,
    dataset: PreparedDataset,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    template_state: dict[str, torch.Tensor],
    common_keys: list[str],
    orders: list[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(args.seed + 100003)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + 100003)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(
        condition,
        dataset=dataset,
        args=args,
        device=device,
    )
    apply_common_initialization(
        model,
        template_state,
        common_keys,
    )
    initial_metrics = evaluate(
        model,
        validation_data,
        batch_size=args.eval_batch_size,
        dataset=dataset,
        args=args,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_tokens, train_lengths, _ = train_data
    model.train()
    losses: list[float] = []
    step = 0
    processed_target_tokens = 0
    started = time.monotonic()
    for index in batches(
        len(train_tokens),
        batch_size=args.batch_size,
        orders=orders,
    ):
        batch_tokens = train_tokens[index].to(device, non_blocking=True)
        batch_lengths = train_lengths[index].to(device, non_blocking=True)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]
        mask = target_mask(batch_lengths, dataset.sequence_length)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            logits, snapshots = model(inputs)
            lm_loss = masked_lm_loss(logits, targets, mask)
            auxiliary = (
                balance_loss(snapshots, mask)
                if condition.balance_lambda > 0.0
                else torch.zeros((), device=device)
            )
            loss = lm_loss + condition.balance_lambda * auxiliary
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss in {condition.name}")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient in {condition.name}")
        optimizer.step()
        losses.append(float(lm_loss.detach().item()))
        processed_target_tokens += int(mask.sum().item())
        step += 1
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_metrics = evaluate(
        model,
        validation_data,
        batch_size=args.eval_batch_size,
        dataset=dataset,
        args=args,
        device=device,
    )
    sparse_madds = 2 * args.top_k * args.d_model * args.expert_hidden
    dense_madds = 2 * args.d_model * (args.top_k * args.expert_hidden)
    result = {
        "condition": asdict(condition),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "activated_ffn_madds_per_token": dense_madds if condition.dense else sparse_madds,
        "actual_sparse_dispatch": not condition.dense,
        "token_dropping": False,
        "optimizer_steps": step,
        "training_target_tokens": processed_target_tokens,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "training_loss_min": min(losses),
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "initial_validation": initial_metrics,
        "final_validation": final_metrics,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 mode requires CUDA")
    if device.type == "cuda" and args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not report native BF16 support")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    dataset = PreparedDataset(args.dataset_dir)
    selected_conditions = [
        condition
        for condition in CONDITIONS
        if not args.conditions or condition.name in args.conditions
    ]
    expected_names = (
        [condition.name for condition in CONDITIONS]
        if not args.conditions
        else args.conditions
    )
    if sorted(condition.name for condition in selected_conditions) != sorted(expected_names):
        raise ValueError("Unknown or duplicate condition requested")
    train_data = dataset.split("train", limit=args.limit_train)
    validation_data = dataset.split("validation", limit=args.limit_validation)
    orders = make_orders(len(train_data[0]), args.epochs, args.seed)
    template_state, common_keys = common_initialization(
        dataset=dataset,
        args=args,
        device=device,
    )
    condition_results = []
    for condition in selected_conditions:
        print(
            f"condition={condition.name} device={device} seed={args.seed}",
            flush=True,
        )
        condition_results.append(
            train_condition(
                condition,
                dataset=dataset,
                train_data=train_data,
                validation_data=validation_data,
                template_state=template_state,
                common_keys=common_keys,
                orders=orders,
                args=args,
                device=device,
            )
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interpretation": (
            "Autoregressive routing comparison with paired model initialization."
        ),
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_steps": args.max_steps,
            "limit_train": args.limit_train,
            "limit_validation": args.limit_validation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "expert_hidden": args.expert_hidden,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "precision": args.precision,
            "conditions": expected_names,
        },
        "dataset": {
            "path": str(dataset.root),
            "vocab_size": dataset.vocab_size,
            "sequence_length": dataset.sequence_length,
            "train_documents": len(train_data[0]),
            "validation_documents": len(validation_data[0]),
        },
        "conditions": condition_results,
    }
    destination = output_dir / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(destination),
        "device": str(device),
        "seed": args.seed,
        "conditions": {
            record["condition"]["name"]: {
                "initial_loss": record["initial_validation"]["loss"],
                "final_loss": record["final_validation"]["loss"],
                "pre_specialization": record["final_validation"].get(
                    "pre_topk_specialization_mean"
                ),
            }
            for record in condition_results
        },
    }, sort_keys=True))
    return result
