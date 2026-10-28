#!/usr/bin/env python3
"""Autoregressive MoE with GPT-style initialization.

Adds differentiable load balancing, router z-loss, and an initialization
quality check to the shared autoregressive model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.models import agnews_autoregressive_moe as base


IMPLEMENTATION_VERSION = 2
ACTIVE_ARGS: argparse.Namespace | None = None


def make_conditions(balance_lambda: float) -> tuple[base.Condition, ...]:
    return (
        base.Condition("shared_global_top2", "shared_global", 0.0),
        base.Condition(
            "shared_global_top2_balanced",
            "shared_global",
            balance_lambda,
        ),
        base.Condition("conditioned_top2", "input_conditioned", 0.0),
        base.Condition(
            "conditioned_top2_balanced",
            "input_conditioned",
            balance_lambda,
        ),
        base.Condition("dense_active_matched", "dense", 0.0, dense=True),
    )


def initialize_model(
    model: base.TinyAutoregressiveModel,
    *,
    init_std: float,
    router_init_std: float,
    num_layers: int,
) -> None:
    def initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    model.apply(initialize)
    model.lm_head.weight = model.token_embedding.weight
    residual_std = init_std / math.sqrt(2.0 * num_layers)
    for block in model.blocks:
        nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
        if model.condition.dense:
            nn.init.normal_(block.ffn[2].weight, mean=0.0, std=residual_std)
        else:
            for expert in block.moe.experts:
                nn.init.normal_(expert[2].weight, mean=0.0, std=residual_std)
            if block.moe.router_type == "input_conditioned":
                nn.init.normal_(
                    block.moe.router.weight,
                    mean=0.0,
                    std=router_init_std,
                )
                nn.init.zeros_(block.moe.router.bias)
            else:
                nn.init.zeros_(block.moe.global_logits)


def build_model(
    condition: base.Condition,
    *,
    dataset: base.PreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> base.TinyAutoregressiveModel:
    model = base.TinyAutoregressiveModel(
        vocab_size=dataset.vocab_size,
        max_length=dataset.sequence_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        expert_hidden=args.expert_hidden,
        num_experts=args.num_experts,
        top_k=args.top_k,
        condition=condition,
    )
    initialize_model(
        model,
        init_std=args.init_std,
        router_init_std=args.router_init_std,
        num_layers=args.num_layers,
    )
    return model.to(device)


def balanced_router_loss(
    snapshots: list[base.RoutingSnapshot],
    mask: torch.Tensor,
) -> torch.Tensor:
    if ACTIVE_ARGS is None:
        raise RuntimeError("Router objective was used before configuration")
    penalties = []
    for snapshot in snapshots:
        active_logits = snapshot.logits[mask]
        active_pre = snapshot.pre_topk_probabilities[mask]
        active_dispatch = (
            snapshot.post_topk_weights[mask] > 0
        ).to(active_pre.dtype)
        active_dispatch = active_dispatch / snapshot.topk_indices.shape[-1]
        importance = active_pre.mean(dim=0)
        dispatch = active_dispatch.mean(dim=0)
        num_experts = importance.numel()
        target = torch.full_like(importance, 1.0 / num_experts)
        switch_term = num_experts * torch.sum(importance * dispatch)
        importance_gap = num_experts * torch.sum((importance - target) ** 2)
        z_loss = torch.mean(torch.logsumexp(active_logits.float(), dim=-1) ** 2)
        z_ratio = (
            ACTIVE_ARGS.router_z_loss / ACTIVE_ARGS.balance_lambda
            if ACTIVE_ARGS.balance_lambda > 0
            else 0.0
        )
        penalties.append(
            switch_term
            + ACTIVE_ARGS.importance_gap_weight * importance_gap
            + z_ratio * z_loss
        )
    return torch.stack(penalties).mean()


def run_gpt(args: argparse.Namespace) -> dict[str, object]:
    global ACTIVE_ARGS
    ACTIVE_ARGS = args
    base.CONDITIONS = make_conditions(args.balance_lambda)
    base.build_model = build_model
    base.balance_loss = balanced_router_loss
    result = base.run(args)
    destination = args.output_dir.resolve() / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
