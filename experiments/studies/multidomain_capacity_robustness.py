#!/usr/bin/env python3
"""Compare shared and input-conditioned routing with capacity limits.

Pair no-drop and capacity-limited runs within each seed. Drop overflow
assignments in row-major order and record dispatch counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.models import agnews_autoregressive_moe as base
from experiments.models import agnews_gpt_moe as gpt_model
from experiments.studies import multidomain_controlled_moe as controlled_model


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 1
CAPACITY_FACTOR = 1.25
SMOKE_SEED = 2400
PRIMARY_SEEDS = set(range(2401, 2405))
MIDDLE_LAYERS = [1, 2, 3, 4]
CONDITION_NAMES = [
    "shared_global_top2",
    "conditioned_top2",
    "shared_global_top2_capacity_1p25",
    "conditioned_top2_capacity_1p25",
]

ACTIVE_MASK: torch.Tensor | None = None
ACTIVE_CONDITION = ""
ACCOUNTING: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}


def make_conditions(_: float) -> tuple[base.Condition, ...]:
    return (
        base.Condition(
            "shared_global_top2",
            "shared_global",
            0.0,
        ),
        base.Condition(
            "conditioned_top2",
            "input_conditioned",
            0.0,
        ),
        base.Condition(
            "shared_global_top2_capacity_1p25",
            "shared_global",
            0.0,
        ),
        base.Condition(
            "conditioned_top2_capacity_1p25",
            "input_conditioned",
            0.0,
        ),
    )


def is_capacity_condition(name: str) -> bool:
    return name.endswith("_capacity_1p25")


def tracked_target_mask(
    lengths: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    global ACTIVE_MASK
    mask = ORIGINAL_TARGET_MASK(lengths, sequence_length)
    ACTIVE_MASK = mask
    return mask


def _zero_vector(reference: torch.Tensor, length: int) -> torch.Tensor:
    return torch.zeros(length, dtype=torch.int64, device=reference.device)


def _record_accounting(
    *,
    module: "CapacitySparseTokenMoE",
    original_indices: torch.Tensor,
    accepted: torch.Tensor,
    active_mask: torch.Tensor,
    capacity: torch.Tensor | None,
) -> None:
    condition = ACTIVE_CONDITION
    mode = "training" if module.training else "evaluation"
    layer = module.capacity_layer_index
    condition_record = ACCOUNTING.setdefault(condition, {})
    mode_record = condition_record.setdefault(mode, {})
    record = mode_record.get(layer)
    if record is None:
        record = {
            "batches": 0,
            "active_tokens": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "original_assignments": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "accepted_original_assignments": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "overflow_assignments": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "rerouted_assignments": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "dropped_assignments": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "accepted_final_assignments": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "tokens_with_overflow": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "tokens_with_reroute": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "tokens_with_any_drop": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "fully_dropped_tokens": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "capacity_slots": torch.zeros(
                (), dtype=torch.int64, device=original_indices.device
            ),
            "capacity_per_expert_min": None,
            "capacity_per_expert_max": None,
            "original_by_expert": _zero_vector(
                original_indices, module.num_experts
            ),
            "accepted_original_by_expert": _zero_vector(
                original_indices, module.num_experts
            ),
            "overflow_by_expert": _zero_vector(
                original_indices, module.num_experts
            ),
            "rerouted_to_by_expert": _zero_vector(
                original_indices, module.num_experts
            ),
            "accepted_final_by_expert": _zero_vector(
                original_indices, module.num_experts
            ),
            "dropped_from_by_expert": _zero_vector(
                original_indices, module.num_experts
            ),
        }
        mode_record[layer] = record

    active_assignments = active_mask.unsqueeze(-1).expand_as(original_indices)
    overflow = active_assignments & ~accepted
    active_indices = original_indices[active_assignments]
    accepted_indices = original_indices[accepted]
    overflow_indices = original_indices[overflow]
    active_tokens = active_mask.sum(dtype=torch.int64)
    original_count = active_assignments.sum(dtype=torch.int64)
    accepted_count = accepted.sum(dtype=torch.int64)
    overflow_count = overflow.sum(dtype=torch.int64)
    overflow_by_token = overflow.sum(dim=-1)

    record["batches"] += 1
    record["active_tokens"] += active_tokens
    record["original_assignments"] += original_count
    record["accepted_original_assignments"] += accepted_count
    record["overflow_assignments"] += overflow_count
    record["dropped_assignments"] += overflow_count
    record["accepted_final_assignments"] += accepted_count
    record["tokens_with_overflow"] += (
        active_mask & (overflow_by_token > 0)
    ).sum(dtype=torch.int64)
    record["tokens_with_any_drop"] += (
        active_mask & (overflow_by_token > 0)
    ).sum(dtype=torch.int64)
    record["fully_dropped_tokens"] += (
        active_mask & (overflow_by_token == module.top_k)
    ).sum(dtype=torch.int64)
    record["original_by_expert"] += torch.bincount(
        active_indices,
        minlength=module.num_experts,
    )
    record["accepted_original_by_expert"] += torch.bincount(
        accepted_indices,
        minlength=module.num_experts,
    )
    record["overflow_by_expert"] += torch.bincount(
        overflow_indices,
        minlength=module.num_experts,
    )
    record["accepted_final_by_expert"] += torch.bincount(
        accepted_indices,
        minlength=module.num_experts,
    )
    record["dropped_from_by_expert"] += torch.bincount(
        overflow_indices,
        minlength=module.num_experts,
    )
    if capacity is not None:
        record["capacity_slots"] += capacity * module.num_experts
        if record["capacity_per_expert_min"] is None:
            record["capacity_per_expert_min"] = capacity.detach().clone()
            record["capacity_per_expert_max"] = capacity.detach().clone()
        else:
            record["capacity_per_expert_min"] = torch.minimum(
                record["capacity_per_expert_min"], capacity
            )
            record["capacity_per_expert_max"] = torch.maximum(
                record["capacity_per_expert_max"], capacity
            )


class CapacitySparseTokenMoE(base.SparseTokenMoE):
    """The controlled multidomain sparse MoE with deterministic assignment capacity."""

    capacity_enabled: bool = False
    capacity_factor: float = CAPACITY_FACTOR
    capacity_layer_index: int = -1

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, base.RoutingSnapshot]:
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
        if ACTIVE_MASK is None or ACTIVE_MASK.shape != hidden.shape[:2]:
            active_mask = torch.ones(
                hidden.shape[:2],
                dtype=torch.bool,
                device=hidden.device,
            )
        else:
            active_mask = ACTIVE_MASK.to(hidden.device)

        active_assignments = active_mask.unsqueeze(-1).expand_as(topk_indices)
        capacity: torch.Tensor | None = None
        if self.capacity_enabled:
            assignment_count = active_assignments.sum(dtype=torch.int64)
            capacity = torch.ceil(
                assignment_count.to(torch.float64)
                * self.capacity_factor
                / self.num_experts
            ).to(torch.int64)
            accepted = torch.zeros_like(active_assignments)
            for expert in range(self.num_experts):
                proposed = active_assignments & (topk_indices == expert)
                ordinal = torch.cumsum(
                    proposed.reshape(-1).to(torch.int64),
                    dim=0,
                ).reshape_as(proposed)
                accepted |= proposed & (ordinal <= capacity)
        else:
            accepted = active_assignments.clone()

        _record_accounting(
            module=self,
            original_indices=topk_indices,
            accepted=accepted,
            active_mask=active_mask,
            capacity=capacity,
        )

        dispatch_weights = topk_weights * (
            accepted | ~active_assignments
        ).to(topk_weights.dtype)
        post_topk = torch.zeros_like(pre_topk)
        post_topk.scatter_add_(
            dim=-1,
            index=topk_indices,
            src=dispatch_weights,
        )
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_weights = dispatch_weights.reshape(-1, self.top_k)
        mixed = torch.zeros_like(flat_hidden)
        for expert_index, expert in enumerate(self.experts):
            token_rows, slots = torch.where(
                (flat_indices == expert_index) & (flat_weights != 0)
            )
            if token_rows.numel() == 0:
                continue
            expert_output = expert(flat_hidden[token_rows])
            weighted = (
                expert_output
                * flat_weights[token_rows, slots].unsqueeze(-1)
            )
            mixed.index_add_(0, token_rows, weighted)
        return mixed.reshape_as(hidden), base.RoutingSnapshot(
            logits=logits,
            pre_topk_probabilities=pre_topk,
            topk_indices=topk_indices,
            post_topk_weights=post_topk,
        )


def build_model(
    condition: base.Condition,
    *,
    dataset: controlled_model.LargeMultidomainPreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> base.TinyAutoregressiveModel:
    original_class = base.SparseTokenMoE
    base.SparseTokenMoE = CapacitySparseTokenMoE
    try:
        model = gpt_model.build_model(
            condition,
            dataset=dataset,
            args=args,
            device=device,
        )
    finally:
        base.SparseTokenMoE = original_class
    capacity_enabled = is_capacity_condition(condition.name)
    for layer, block in enumerate(model.blocks):
        module = block.moe
        if not isinstance(module, CapacitySparseTokenMoE):
            raise TypeError("capacity robustness model did not build capacity-aware MoE")
        module.capacity_enabled = capacity_enabled
        module.capacity_factor = CAPACITY_FACTOR
        module.capacity_layer_index = layer
    return model


def _python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return int(value.detach().cpu().item())
        return [int(item) for item in value.detach().cpu().tolist()]
    return value


def _serialize_accounting(condition: str) -> dict[str, object]:
    modes: dict[str, object] = {}
    for mode, layers in ACCOUNTING.get(condition, {}).items():
        serialized_layers = []
        for layer in sorted(layers):
            record = {
                key: _python_value(value)
                for key, value in layers[layer].items()
            }
            record["layer_zero_based"] = layer
            original = int(record["original_assignments"])
            overflow = int(record["overflow_assignments"])
            active_tokens = int(record["active_tokens"])
            record["overflow_assignment_rate"] = (
                overflow / original if original else 0.0
            )
            record["tokens_with_overflow_rate"] = (
                int(record["tokens_with_overflow"]) / active_tokens
                if active_tokens
                else 0.0
            )
            serialized_layers.append(record)
        modes[mode] = {"layers": serialized_layers}
    return {
        "policy": (
            "capacity_1p25_drop_overflow_no_reroute_no_renormalization"
            if is_capacity_condition(condition)
            else "no_drop_reference"
        ),
        "capacity_enabled": is_capacity_condition(condition),
        "capacity_factor": (
            CAPACITY_FACTOR if is_capacity_condition(condition) else None
        ),
        "unit": (
            "Assignments and token events are counted separately per layer; "
            "cross-layer totals are intentionally not deduplicated."
        ),
        "modes": modes,
    }


ORIGINAL_TARGET_MASK = base.target_mask
ORIGINAL_TRAIN_CONDITION = controlled_model.train_condition


def train_condition(
    condition: base.Condition,
    **kwargs: object,
) -> dict[str, object]:
    global ACTIVE_CONDITION, ACTIVE_MASK
    ACTIVE_CONDITION = condition.name
    ACTIVE_MASK = None
    ACCOUNTING[condition.name] = {}
    record = ORIGINAL_TRAIN_CONDITION(condition, **kwargs)
    record["capacity_accounting"] = _serialize_accounting(condition.name)
    record["token_dropping"] = is_capacity_condition(condition.name)
    record["capacity_truncation"] = is_capacity_condition(condition.name)
    return record


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.role not in ("smoke", "primary"):
        raise ValueError("capacity robustness role must be smoke or primary")
    if args.role == "smoke" and args.seed != SMOKE_SEED:
        raise ValueError("Capacity calibration must use seed 2400")
    if args.role == "primary" and args.seed not in PRIMARY_SEEDS:
        raise ValueError("capacity robustness primary seed must be 2401–2404")
    if float(args.capacity_factor) != CAPACITY_FACTOR:
        raise ValueError("Capacity factor must be 1.25")
    if args.conditions != CONDITION_NAMES:
        raise ValueError("Capacity comparison requires the configured routing conditions")
    if args.retain_checkpoints_for:
        raise ValueError("capacity robustness does not retain checkpoints")

    original = {
        "conditions": controlled_model.CONDITION_NAMES,
        "primary_seeds": controlled_model.PRIMARY_SEEDS,
        "make_conditions": controlled_model.make_conditions,
        "build_model": controlled_model.build_model,
        "train_condition": controlled_model.train_condition,
        "target_mask": base.target_mask,
    }
    controlled_model.CONDITION_NAMES = CONDITION_NAMES
    controlled_model.PRIMARY_SEEDS = PRIMARY_SEEDS
    controlled_model.make_conditions = make_conditions
    controlled_model.build_model = build_model
    controlled_model.train_condition = train_condition
    base.target_mask = tracked_target_mask
    delegated_role = (
        "resource_smoke"
        if args.role == "smoke"
        else "primary"
    )
    delegated_args = argparse.Namespace(**vars(args))
    delegated_args.role = delegated_role
    try:
        result = controlled_model.run_multidomain_large(delegated_args)
    finally:
        controlled_model.CONDITION_NAMES = original["conditions"]
        controlled_model.PRIMARY_SEEDS = original["primary_seeds"]
        controlled_model.make_conditions = original["make_conditions"]
        controlled_model.build_model = original["build_model"]
        controlled_model.train_condition = original["train_condition"]
        base.target_mask = original["target_mask"]

    result["schema_version"] = SCHEMA_VERSION
    result["implementation_version"] = IMPLEMENTATION_VERSION
    result["study"] = "capacity_robustness"
    result["role"] = args.role
    result["primary_multidomain_large_confirmatory"] = False
    result["primary_capacity_robustness"] = args.role == "primary"
    result["capacity_design"] = {
        "capacity_factor": CAPACITY_FACTOR,
        "conditions": CONDITION_NAMES,
        "assignment_order": "active token row-major, then top-k slot",
        "overflow_policy": "drop",
        "reroute_policy": "none",
        "dropped_gate_weight_renormalization": False,
        "primary_endpoint": (
            "conditioned-minus-shared pre-top-k domain specialization, "
            "averaged over middle layers [1,2,3,4], reported separately "
            "for no-drop and capacity-limited pairs"
        ),
    }
    for key in (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "max_steps",
        "limit_train",
        "limit_validation",
        "learning_rate",
        "weight_decay",
        "d_model",
        "num_heads",
        "num_layers",
        "expert_hidden",
        "num_experts",
        "top_k",
        "precision",
        "init_std",
        "router_init_std",
        "balance_lambda",
        "router_z_loss",
        "importance_gap_weight",
        "initial_ce_tolerance",
        "checkpoint_interval",
        "retain_checkpoints_for",
        "capacity_factor",
    ):
        result["config"][key] = getattr(args, key)
    destination = args.output_dir.resolve() / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "study": result["study"],
                "role": result["role"],
                "seed": args.seed,
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
    parser.add_argument(
        "--role",
        choices=("smoke", "primary"),
        required=True,
    )
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", default=CONDITION_NAMES)
    parser.add_argument("--capacity-factor", type=float, default=CAPACITY_FACTOR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-validation", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--expert-hidden", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--router-init-std", type=float, default=0.01)
    parser.add_argument("--balance-lambda", type=float, default=0.0)
    parser.add_argument("--router-z-loss", type=float, default=0.001)
    parser.add_argument("--importance-gap-weight", type=float, default=1.0)
    parser.add_argument("--initial-ce-tolerance", type=float, default=0.75)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--retain-checkpoints-for", nargs="*", default=[])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
