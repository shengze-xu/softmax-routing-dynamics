#!/usr/bin/env python3
"""Controlled Transformer-MoE scale-transfer comparison on four domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.models import agnews_autoregressive_moe as base
from experiments.models import agnews_gpt_moe as gpt_model


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 6
DATASET_ID = "multidomain-64m-2m-2m-v1"
EXPLORATORY_SEEDS = {2101, 2102, 2103}
PRIMARY_SEEDS = set(range(2201, 2209))
MIDDLE_LAYERS = [1, 2, 3, 4]
CONDITION_NAMES = [
    "shared_global_top2",
    "shared_global_top2_balanced",
    "conditioned_top2",
    "conditioned_top2_balanced",
    "dense_active_matched",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_fingerprint(model: nn.Module) -> str:
    state = model.state_dict()
    return base.tensor_fingerprint(state, list(state))


class LargeMultidomainPreparedDataset(base.PreparedDataset):
    """Hash-checked memory-mapped large multidomain corpus."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        audit_path = self.root / "corpus_audit.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported large multidomain dataset schema")
        if self.manifest.get("dataset_id") != DATASET_ID:
            raise ValueError("Unexpected large multidomain dataset ID")
        if self.manifest.get("status") != "prepared_multidomain_large_candidate":
            raise ValueError("Dataset is not a prepared large multidomain candidate")
        self.manifest_sha256 = sha256(manifest_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise ValueError("large multidomain corpus audit did not pass")
        if audit.get("corpus_integrity_checks_passed") is not True:
            raise ValueError("Large multidomain corpus integrity checks did not pass")
        if audit.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("Corpus audit refers to another manifest")
        self.corpus_audit_sha256 = sha256(audit_path)
        self.arrays: dict[str, np.ndarray] = {}
        for name, metadata in self.manifest["arrays"].items():
            path = self.root / metadata["file"]
            if sha256(path) != metadata["sha256"]:
                raise ValueError(f"Prepared-data checksum mismatch: {name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != metadata["shape"]:
                raise ValueError(f"Prepared-data shape mismatch: {name}")
            if str(array.dtype) != metadata["dtype"]:
                raise ValueError(f"Prepared-data dtype mismatch: {name}")
            self.arrays[name] = array
        self.vocab_size = int(self.manifest["tokenizer"]["vocab_size"])
        self.sequence_length = int(self.manifest["selection"]["sequence_length"])
        if self.sequence_length != 512:
            raise ValueError("large multidomain sequence length must be 512")
        base.DOMAIN_NAMES = list(self.manifest["domain_names"])
        if self.arrays["train_tokens"].shape[1] != self.sequence_length + 1:
            raise ValueError("Prepared token width does not match the manifest")


def make_conditions(balance_lambda: float) -> tuple[base.Condition, ...]:
    return gpt_model.make_conditions(balance_lambda)


def build_model(
    condition: base.Condition,
    *,
    dataset: LargeMultidomainPreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> base.TinyAutoregressiveModel:
    return gpt_model.build_model(
        condition,
        dataset=dataset,
        args=args,
        device=device,
    )


def common_initialization(
    *,
    dataset: LargeMultidomainPreparedDataset,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    torch.manual_seed(args.seed)
    template_condition = next(
        condition
        for condition in make_conditions(args.balance_lambda)
        if condition.name == "conditioned_top2"
    )
    template = build_model(
        template_condition,
        dataset=dataset,
        args=args,
        device=torch.device("cpu"),
    )
    state = {
        key: value.detach().clone()
        for key, value in template.state_dict().items()
    }
    keys = [
        key
        for key in state
        if ".moe.router." not in key and ".moe.global_logits" not in key
    ]
    del template
    return state, keys


def apply_common_initialization(
    model: base.TinyAutoregressiveModel,
    template_state: dict[str, torch.Tensor],
    common_keys: list[str],
) -> tuple[str | None, str]:
    state = model.state_dict()
    copied: list[str] = []
    for key in common_keys:
        if key in state and state[key].shape == template_state[key].shape:
            state[key].copy_(template_state[key].to(state[key].device))
            copied.append(key)
    model.load_state_dict(state)
    if not model.condition.dense and set(copied) != set(common_keys):
        missing = sorted(set(common_keys) - set(copied))
        raise ValueError(f"Sparse condition missed common keys: {missing}")
    copied_fingerprint = base.tensor_fingerprint(model.state_dict(), copied)
    sparse_fingerprint = copied_fingerprint if not model.condition.dense else None
    return sparse_fingerprint, copied_fingerprint


def make_orders(size: int, epochs: int, seed: int) -> list[torch.Tensor]:
    return base.make_orders(size, epochs, seed)


def batch_plan_fingerprint(
    orders: list[torch.Tensor],
    *,
    batch_size: int,
    gradient_accumulation: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "batch_size": batch_size,
                "gradient_accumulation": gradient_accumulation,
                "order_count": len(orders),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for order in orders:
        value = order.detach().cpu().contiguous().numpy().astype(
            np.int64,
            copy=False,
        )
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def training_config_fingerprint(args: argparse.Namespace) -> str:
    keys = (
        "seed",
        "epochs",
        "batch_size",
        "gradient_accumulation",
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
    )
    return json_fingerprint({key: getattr(args, key) for key in keys})


def iter_batches(
    size: int,
    *,
    batch_size: int,
    orders: list[torch.Tensor],
) -> Iterator[torch.Tensor]:
    yield from base.batches(size, batch_size=batch_size, orders=orders)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    condition: base.Condition,
    args: argparse.Namespace,
    dataset: LargeMultidomainPreparedDataset,
    config_fingerprint: str,
    batch_plan_hash: str,
    optimizer_step: int,
    microbatch_cursor: int,
    processed_target_tokens: int,
    losses: list[float],
    cumulative_wall_seconds: float,
    initial_validation: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "study": "multidomain_large_controlled_scale_transfer",
        "condition": condition.name,
        "seed": args.seed,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "training_config_fingerprint": config_fingerprint,
        "batch_plan_fingerprint": batch_plan_hash,
        "optimizer_step": optimizer_step,
        "microbatch_cursor": microbatch_cursor,
        "processed_target_tokens": processed_target_tokens,
        "training_losses": losses,
        "cumulative_wall_seconds": cumulative_wall_seconds,
        "initial_validation": initial_validation,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def cuda_rng_states_for_restore(
    states: list[torch.Tensor],
) -> list[torch.Tensor]:
    restored: list[torch.Tensor] = []
    for index, state in enumerate(states):
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"CUDA RNG state {index} is not a tensor")
        if state.dtype != torch.uint8:
            raise TypeError(
                f"CUDA RNG state {index} must have dtype uint8, got {state.dtype}"
            )
        restored.append(state.detach().cpu().contiguous())
    return restored


def configure_deterministic_execution(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def load_resume_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    condition: base.Condition,
    args: argparse.Namespace,
    dataset: LargeMultidomainPreparedDataset,
    config_fingerprint: str,
    batch_plan_hash: str,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    checks = {
        "schema": checkpoint.get("schema_version") == 1,
        "study": checkpoint.get("study") == "multidomain_large_controlled_scale_transfer",
        "condition": checkpoint.get("condition") == condition.name,
        "seed": int(checkpoint.get("seed", -1)) == args.seed,
        "dataset": (
            checkpoint.get("dataset_manifest_sha256")
            == dataset.manifest_sha256
        ),
        "config": (
            checkpoint.get("training_config_fingerprint")
            == config_fingerprint
        ),
        "batch_plan": (
            checkpoint.get("batch_plan_fingerprint") == batch_plan_hash
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Resume checkpoint validation failed: {failures}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_rng_states = checkpoint.get("cuda_rng_state_all") or []
    if device.type == "cuda" and cuda_rng_states:
        if len(cuda_rng_states) != torch.cuda.device_count():
            raise ValueError(
                "Checkpoint CUDA RNG state count does not match visible devices: "
                f"{len(cuda_rng_states)} != {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(
            cuda_rng_states_for_restore(cuda_rng_states)
        )
    return checkpoint


def optimizer_update(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Non-finite gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(gradient_norm.detach().item())


def train_condition(
    condition: base.Condition,
    *,
    dataset: LargeMultidomainPreparedDataset,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    template_state: dict[str, torch.Tensor],
    common_keys: list[str],
    orders: list[torch.Tensor],
    batch_plan_hash: str,
    args: argparse.Namespace,
    device: torch.device,
    condition_root: Path,
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
    sparse_initialization_fingerprint, common_fingerprint = (
        apply_common_initialization(model, template_state, common_keys)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config_fingerprint = training_config_fingerprint(args)
    resume_path = args.resume_checkpoint.resolve() if args.resume_checkpoint else None
    if resume_path is not None and len(args.conditions) != 1:
        raise ValueError("Resume requires exactly one selected condition")

    optimizer_step = 0
    microbatch_cursor = 0
    processed_target_tokens = 0
    losses: list[float] = []
    cumulative_wall_before = 0.0
    if resume_path is None:
        initial_metrics = base.evaluate(
            model,
            validation_data,
            batch_size=args.eval_batch_size,
            dataset=dataset,
            args=args,
            device=device,
        )
        resume_start_metrics = None
    else:
        checkpoint = load_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            condition=condition,
            args=args,
            dataset=dataset,
            config_fingerprint=config_fingerprint,
            batch_plan_hash=batch_plan_hash,
            device=device,
        )
        optimizer_step = int(checkpoint["optimizer_step"])
        microbatch_cursor = int(checkpoint["microbatch_cursor"])
        processed_target_tokens = int(checkpoint["processed_target_tokens"])
        losses = [float(value) for value in checkpoint["training_losses"]]
        cumulative_wall_before = float(
            checkpoint.get("cumulative_wall_seconds", 0.0)
        )
        initial_metrics = checkpoint["initial_validation"]
        resume_start_metrics = base.evaluate(
            model,
            validation_data,
            batch_size=args.eval_batch_size,
            dataset=dataset,
            args=args,
            device=device,
        )
    if args.max_steps > 0 and optimizer_step >= args.max_steps:
        raise ValueError("Resume checkpoint already reached max_steps")

    train_tokens, train_lengths, _ = train_data
    gpt_model.ACTIVE_ARGS = args
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    accumulation_count = 0
    accumulation_losses: list[float] = []
    checkpoint_paths: list[str] = []
    checkpoint_records: list[dict[str, object]] = []
    last_gradient_norm: float | None = None
    stopped_for_max_steps = False

    for batch_number, index in enumerate(
        iter_batches(
            len(train_tokens),
            batch_size=args.batch_size,
            orders=orders,
        )
    ):
        if batch_number < microbatch_cursor:
            continue
        batch_tokens = train_tokens[index].to(device, non_blocking=True)
        batch_lengths = train_lengths[index].to(device, non_blocking=True)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]
        mask = base.target_mask(batch_lengths, dataset.sequence_length)
        with base.autocast_context(device, args.precision):
            logits, snapshots = model(inputs)
            language_model_loss = base.masked_lm_loss(logits, targets, mask)
            auxiliary = (
                gpt_model.balanced_router_loss(snapshots, mask)
                if condition.balance_lambda > 0.0
                else torch.zeros((), device=device)
            )
            loss = language_model_loss + condition.balance_lambda * auxiliary
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss in {condition.name}")
        (loss / args.gradient_accumulation).backward()
        accumulation_losses.append(float(language_model_loss.detach().item()))
        accumulation_count += 1
        processed_target_tokens += int(mask.sum().item())
        microbatch_cursor = batch_number + 1

        if accumulation_count < args.gradient_accumulation:
            continue
        last_gradient_norm = optimizer_update(model=model, optimizer=optimizer)
        optimizer_step += 1
        losses.append(float(np.mean(accumulation_losses)))
        accumulation_losses = []
        accumulation_count = 0

        if (
            args.checkpoint_interval > 0
            and optimizer_step % args.checkpoint_interval == 0
            and not (
                args.max_steps > 0
                and optimizer_step >= args.max_steps
            )
        ):
            elapsed = cumulative_wall_before + (time.monotonic() - started)
            checkpoint_path = (
                condition_root
                / "checkpoints"
                / f"step_{optimizer_step:08d}.pt"
            )
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                condition=condition,
                args=args,
                dataset=dataset,
                config_fingerprint=config_fingerprint,
                batch_plan_hash=batch_plan_hash,
                optimizer_step=optimizer_step,
                microbatch_cursor=microbatch_cursor,
                processed_target_tokens=processed_target_tokens,
                losses=losses,
                cumulative_wall_seconds=elapsed,
                initial_validation=initial_metrics,
            )
            payload["model_fingerprint"] = state_fingerprint(model)
            atomic_torch_save(
                payload,
                checkpoint_path,
            )
            checkpoint_paths.append(str(checkpoint_path))
            checkpoint_records.append(
                {
                    "path": str(checkpoint_path),
                    "optimizer_step": optimizer_step,
                    "microbatch_cursor": microbatch_cursor,
                    "training_target_tokens": processed_target_tokens,
                    "model_fingerprint": payload["model_fingerprint"],
                }
            )
        if args.max_steps > 0 and optimizer_step >= args.max_steps:
            stopped_for_max_steps = True
            break

    if accumulation_count and not stopped_for_max_steps:
        correction = args.gradient_accumulation / accumulation_count
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        last_gradient_norm = optimizer_update(model=model, optimizer=optimizer)
        optimizer_step += 1
        losses.append(float(np.mean(accumulation_losses)))
        accumulation_count = 0

    if not losses:
        raise RuntimeError("No optimizer steps were completed")
    segment_wall_seconds = time.monotonic() - started
    cumulative_wall_seconds = cumulative_wall_before + segment_wall_seconds
    final_metrics = base.evaluate(
        model,
        validation_data,
        batch_size=args.eval_batch_size,
        dataset=dataset,
        args=args,
        device=device,
    )
    final_checkpoint = None
    if condition.name in args.retain_checkpoints_for:
        final_path = (
            condition_root
            / "checkpoints"
            / f"final_step_{optimizer_step:08d}.pt"
        )
        if not final_path.exists():
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    condition=condition,
                    args=args,
                    dataset=dataset,
                    config_fingerprint=config_fingerprint,
                    batch_plan_hash=batch_plan_hash,
                    optimizer_step=optimizer_step,
                    microbatch_cursor=microbatch_cursor,
                    processed_target_tokens=processed_target_tokens,
                    losses=losses,
                    cumulative_wall_seconds=cumulative_wall_seconds,
                    initial_validation=initial_metrics,
                ),
                final_path,
            )
        final_checkpoint = str(final_path)

    sparse_madds = 2 * args.top_k * args.d_model * args.expert_hidden
    dense_madds = 2 * args.d_model * (args.top_k * args.expert_hidden)
    result = {
        "condition": asdict(condition),
        "sparse_initialization_fingerprint": sparse_initialization_fingerprint,
        "common_initialization_fingerprint": common_fingerprint,
        "training_config_fingerprint": config_fingerprint,
        "batch_plan_fingerprint": batch_plan_hash,
        "final_model_fingerprint": state_fingerprint(model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "activated_ffn_madds_per_token": (
            dense_madds if condition.dense else sparse_madds
        ),
        "actual_sparse_dispatch": not condition.dense,
        "token_dropping": False,
        "capacity_truncation": False,
        "optimizer_steps": optimizer_step,
        "microbatch_cursor": microbatch_cursor,
        "gradient_accumulation": args.gradient_accumulation,
        "training_target_tokens": processed_target_tokens,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "training_loss_min": min(losses),
        "last_gradient_norm": last_gradient_norm,
        "segment_wall_seconds": segment_wall_seconds,
        "cumulative_wall_seconds": cumulative_wall_seconds,
        "training_tokens_per_second": (
            processed_target_tokens / cumulative_wall_seconds
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "initial_validation": initial_metrics,
        "resume_start_validation": resume_start_metrics,
        "final_validation": final_metrics,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "periodic_checkpoints": checkpoint_paths,
        "periodic_checkpoint_records": checkpoint_records,
        "retained_final_checkpoint": final_checkpoint,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def engineering_gate(
    results: list[dict[str, object]],
    *,
    expected_conditions: list[str],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    names = [str(record["condition"]["name"]) for record in results]
    if names != expected_conditions:
        failures.append("condition inventory or order mismatch")
    sparse_fingerprints = {
        record["sparse_initialization_fingerprint"]
        for record in results
        if not record["condition"]["dense"]
    }
    if len(sparse_fingerprints) != 1:
        failures.append("sparse non-router initialization differs")
    batch_plans = {record["batch_plan_fingerprint"] for record in results}
    if len(batch_plans) != 1:
        failures.append("future batch plan differs across conditions")
    activated = [
        int(record["activated_ffn_madds_per_token"])
        for record in results
    ]
    if activated and (max(activated) - min(activated)) / max(activated) > 0.01:
        failures.append("activated FFN MAdds differ by more than one percent")
    target_tokens = {
        int(record["training_target_tokens"]) for record in results
    }
    optimizer_steps = {int(record["optimizer_steps"]) for record in results}
    if len(target_tokens) != 1:
        failures.append("training target tokens differ across conditions")
    if len(optimizer_steps) != 1:
        failures.append("optimizer steps differ across conditions")
    for record in results:
        name = str(record["condition"]["name"])
        if record.get("token_dropping") is not False:
            failures.append(f"{name}: token dropping enabled")
        if record.get("capacity_truncation") is not False:
            failures.append(f"{name}: capacity truncation enabled")
        if not math.isfinite(float(record["training_loss_last"])):
            failures.append(f"{name}: non-finite training loss")
        if not math.isfinite(float(record["final_validation"]["loss"])):
            failures.append(f"{name}: non-finite validation loss")
        if record["condition"]["router"] == "shared_global":
            error = float(record["final_validation"]["shared_row_max_error"])
            if error > 1e-7:
                failures.append(f"{name}: shared row error exceeds 1e-7")
    return not failures, failures


def run_multidomain_large(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 requires CUDA")
    if device.type == "cuda" and args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not report native BF16 support")
    configure_deterministic_execution(device)
    if args.role == "exploratory" and args.seed not in EXPLORATORY_SEEDS:
        raise ValueError("large multidomain exploratory must use one of seeds 2101–2103")
    if args.role == "primary" and args.seed not in PRIMARY_SEEDS:
        raise ValueError("large multidomain primary run must use one of seeds 2201–2208")
    if args.num_layers != 6 or args.d_model != 384:
        raise ValueError("large multidomain v1 requires six layers and d_model=384")
    if args.num_heads != 6 or args.expert_hidden != 768:
        raise ValueError("large multidomain v1 head/expert configuration changed")
    if args.num_experts != 8 or args.top_k != 2:
        raise ValueError("large multidomain v1 requires eight experts and top-k=2")
    if any(layer >= args.num_layers for layer in MIDDLE_LAYERS):
        raise ValueError("Frozen middle layers are invalid")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    dataset = LargeMultidomainPreparedDataset(args.dataset_dir)
    gpt_model.ACTIVE_ARGS = args
    conditions = make_conditions(args.balance_lambda)
    selected_conditions = [
        condition
        for condition in conditions
        if not args.conditions or condition.name in args.conditions
    ]
    expected_names = (
        CONDITION_NAMES if not args.conditions else list(args.conditions)
    )
    if [condition.name for condition in selected_conditions] != expected_names:
        raise ValueError("Unknown, duplicate, or out-of-order condition")
    train_data = dataset.split("train", limit=args.limit_train)
    validation_data = dataset.split(
        "validation",
        limit=args.limit_validation,
    )
    orders = make_orders(len(train_data[0]), args.epochs, args.seed)
    plan_hash = batch_plan_fingerprint(
        orders,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
    )
    template_state, common_keys = common_initialization(
        dataset=dataset,
        args=args,
    )
    condition_results: list[dict[str, object]] = []
    for condition in selected_conditions:
        print(
            json.dumps(
                {
                    "event": "condition_start",
                    "condition": condition.name,
                    "device": str(device),
                    "seed": args.seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        condition_root = output_dir / "conditions" / condition.name
        condition_root.mkdir(parents=True)
        record = train_condition(
            condition,
            dataset=dataset,
            train_data=train_data,
            validation_data=validation_data,
            template_state=template_state,
            common_keys=common_keys,
            orders=orders,
            batch_plan_hash=plan_hash,
            args=args,
            device=device,
            condition_root=condition_root,
        )
        condition_result_path = condition_root / "result.json"
        condition_result_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        condition_results.append(record)
        print(
            json.dumps(
                {
                    "event": "condition_complete",
                    "condition": condition.name,
                    "optimizer_steps": record["optimizer_steps"],
                    "training_target_tokens": record["training_target_tokens"],
                    "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    gate_passed, gate_failures = engineering_gate(
        condition_results,
        expected_conditions=expected_names,
    )
    uniform_cross_entropy = math.log(dataset.vocab_size)
    maximum_initial_excess = max(
        abs(float(record["initial_validation"]["loss"]) - uniform_cross_entropy)
        for record in condition_results
    )
    initialization_gate = {
        "passed": maximum_initial_excess <= args.initial_ce_tolerance,
        "uniform_cross_entropy": uniform_cross_entropy,
        "maximum_absolute_excess": maximum_initial_excess,
        "tolerance": args.initial_ce_tolerance,
    }
    if not initialization_gate["passed"]:
        gate_passed = False
        gate_failures.append("initial validation exceeds uniform CE tolerance")

    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "passed" if gate_passed else "failed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "study": "multidomain_large_controlled_scale_transfer",
        "role": args.role,
        "protocol_id": args.protocol_id,
        "primary_multidomain_large_confirmatory": args.role == "primary",
        "engineering_gate_passed": gate_passed,
        "engineering_gate_failures": gate_failures,
        "initialization_gate": initialization_gate,
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
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
            "middle_layers_zero_based": MIDDLE_LAYERS,
            "balance_lambda": args.balance_lambda,
            "router_z_loss": args.router_z_loss,
            "importance_gap_weight": args.importance_gap_weight,
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "path": str(dataset.root),
            "manifest_sha256": dataset.manifest_sha256,
            "corpus_audit_sha256": dataset.corpus_audit_sha256,
            "vocab_size": dataset.vocab_size,
            "sequence_length": dataset.sequence_length,
            "train_sequences": len(train_data[0]),
            "validation_sequences": len(validation_data[0]),
        },
        "execution": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "cuda_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "flash_sdp_enabled": (
                torch.backends.cuda.flash_sdp_enabled()
                if device.type == "cuda"
                else None
            ),
            "memory_efficient_sdp_enabled": (
                torch.backends.cuda.mem_efficient_sdp_enabled()
                if device.type == "cuda"
                else None
            ),
            "math_sdp_enabled": (
                torch.backends.cuda.math_sdp_enabled()
                if device.type == "cuda"
                else None
            ),
        },
        "conditions": condition_results,
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
                "role": args.role,
                "seed": args.seed,
                "engineering_gate_passed": gate_passed,
                "failures": gate_failures,
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    if not gate_passed:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("resource_smoke", "resume_smoke", "exploratory", "primary"),
        required=True,
    )
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-validation", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--expert-hidden", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--router-init-std", type=float, default=0.01)
    parser.add_argument("--balance-lambda", type=float, default=0.1)
    parser.add_argument("--router-z-loss", type=float, default=1e-3)
    parser.add_argument("--importance-gap-weight", type=float, default=1.0)
    parser.add_argument("--initial-ce-tolerance", type=float, default=0.75)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--retain-checkpoints-for",
        nargs="*",
        default=[],
    )
    args = parser.parse_args()
    if args.d_model % args.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if not 1 <= args.top_k <= args.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if args.gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if args.resume_checkpoint and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(args.resume_checkpoint)
    if any(name not in CONDITION_NAMES for name in args.retain_checkpoints_for):
        raise ValueError("Unknown checkpoint retention condition")
    run_multidomain_large(args)


if __name__ == "__main__":
    main()
