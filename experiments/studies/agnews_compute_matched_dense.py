from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import add_torch_device_argument, ensure_subdir, resolve_torch_device, write_csv
from experiments.studies.transformer_moe_agnews import (
    NUM_CLASSES,
    TinyTransformerMoE,
    build_vocab,
    encode_examples,
    ensure_agnews,
    read_examples,
    stratified_subset,
)


torch.set_num_threads(1)


class TinyTransformerDense(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        d_model: int,
        num_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dense_hidden),
            nn.GELU(),
            nn.Linear(dense_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attention_output, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=~mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attention_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return self.classifier(x[:, 0])


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def train_dense(
    *,
    seed: int,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    vocab_size: int,
    max_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    d_model: int,
    num_heads: int,
    dense_hidden: int,
    dropout: float,
    activated_ffn_parameters: int,
    ffn_multiply_adds: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = TinyTransformerDense(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        dense_hidden=dense_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_tokens, train_mask, train_labels = train_data
    generator = torch.Generator().manual_seed(8100 + seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(train_labels.numel(), generator=generator)
        for start in range(0, order.numel(), batch_size):
            index = order[start : start + batch_size]
            logits = model(
                train_tokens[index].to(device),
                train_mask[index].to(device),
            )
            loss = F.cross_entropy(logits, train_labels[index].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    test_tokens, test_mask, test_labels = test_data
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, test_labels.numel(), batch_size):
            stop = min(start + batch_size, test_labels.numel())
            logits = model(
                test_tokens[start:stop].to(device),
                test_mask[start:stop].to(device),
            )
            labels = test_labels[start:stop].to(device)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_count += int(labels.numel())

    return {
        "mode": "agnews_dense_active_matched",
        "label": "dense FFN (expert-path MAdd matched)",
        "seed": seed,
        "test_accuracy": total_correct / total_count,
        "test_cross_entropy": total_loss / total_count,
        "total_parameters": parameter_count(model),
        "activated_ffn_parameters_per_token": activated_ffn_parameters,
        "ffn_multiply_adds_per_token": ffn_multiply_adds,
    }


def build_compute_profiles(
    *,
    vocab_size: int,
    max_length: int,
    d_model: int,
    num_heads: int,
    num_experts: int,
    expert_hidden: int,
    dense_hidden: int,
    dropout: float,
) -> list[dict[str, object]]:
    global_moe = TinyTransformerMoE(
        vocab_size=vocab_size,
        max_length=max_length,
        conditioned=False,
        d_model=d_model,
        num_heads=num_heads,
        num_experts=num_experts,
        expert_hidden=expert_hidden,
        dropout=dropout,
    )
    conditioned_moe = TinyTransformerMoE(
        vocab_size=vocab_size,
        max_length=max_length,
        conditioned=True,
        d_model=d_model,
        num_heads=num_heads,
        num_experts=num_experts,
        expert_hidden=expert_hidden,
        dropout=dropout,
    )
    dense = TinyTransformerDense(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        dense_hidden=dense_hidden,
        dropout=dropout,
    )
    one_expert_parameters = parameter_count(global_moe.moe.experts[0])
    dense_ffn_parameters = parameter_count(dense.ffn)
    top_k = 2
    profiles = [
        {
            "mode": "agnews_global_top2",
            "label": "global top-2",
            "total_parameters": parameter_count(global_moe),
            "stored_ffn_parameters": num_experts * one_expert_parameters,
            "activated_ffn_parameters_per_token": top_k * one_expert_parameters,
            "ffn_multiply_adds_per_token": 2 * top_k * d_model * expert_hidden,
            "router_parameters": num_experts,
        },
        {
            "mode": "agnews_conditioned_top2",
            "label": "conditioned top-2",
            "total_parameters": parameter_count(conditioned_moe),
            "stored_ffn_parameters": num_experts * one_expert_parameters,
            "activated_ffn_parameters_per_token": top_k * one_expert_parameters,
            "ffn_multiply_adds_per_token": 2 * top_k * d_model * expert_hidden,
            "router_parameters": parameter_count(conditioned_moe.moe.router),
        },
        {
            "mode": "agnews_dense_active_matched",
            "label": "dense FFN (expert-path MAdd matched)",
            "total_parameters": parameter_count(dense),
            "stored_ffn_parameters": dense_ffn_parameters,
            "activated_ffn_parameters_per_token": dense_ffn_parameters,
            "ffn_multiply_adds_per_token": 2 * d_model * dense_hidden,
            "router_parameters": 0,
        },
    ]
    for profile in profiles:
        profile["activated_to_stored_ffn_ratio"] = (
            int(profile["activated_ffn_parameters_per_token"])
            / int(profile["stored_ffn_parameters"])
        )
    return profiles


def read_agnews_summary() -> dict[str, dict[str, str]]:
    import csv

    path = ensure_subdir("tables") / "transformer_moe_agnews_summary.csv"
    with path.open(newline="") as handle:
        return {row["mode"]: row for row in csv.DictReader(handle)}




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expert-path-MAdd-matched dense baseline for the AG News MoE exploratory."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--expert-hidden", type=int, default=96)
    parser.add_argument("--dense-hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    add_torch_device_argument(parser)
    args = parser.parse_args()
    device = resolve_torch_device(args.device)
    print(f"device={device}")

    train_path, test_path = ensure_agnews(ROOT / "data" / "ag_news_csv")
    train_examples = stratified_subset(
        read_examples(train_path),
        per_class=args.train_per_class,
        seed=20260717,
    )
    test_examples = stratified_subset(
        read_examples(test_path),
        per_class=args.test_per_class,
        seed=20260717,
    )
    vocab = build_vocab(train_examples, args.max_vocab)
    train_data = encode_examples(train_examples, vocab=vocab, max_length=args.max_length)
    test_data = encode_examples(test_examples, vocab=vocab, max_length=args.max_length)
    profiles = build_compute_profiles(
        vocab_size=len(vocab),
        max_length=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        expert_hidden=args.expert_hidden,
        dense_hidden=args.dense_hidden,
        dropout=args.dropout,
    )
    profile = next(
        item for item in profiles if item["mode"] == "agnews_dense_active_matched"
    )

    records: list[dict[str, object]] = []
    for seed in args.seeds:
        print(f"training expert-path-MAdd-matched dense baseline, seed {seed}")
        records.append(
            train_dense(
                seed=seed,
                train_data=train_data,
                test_data=test_data,
                vocab_size=len(vocab),
                max_length=args.max_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                d_model=args.d_model,
                num_heads=args.num_heads,
                dense_hidden=args.dense_hidden,
                dropout=args.dropout,
                activated_ffn_parameters=int(profile["activated_ffn_parameters_per_token"]),
                ffn_multiply_adds=int(profile["ffn_multiply_adds_per_token"]),
                device=device,
            )
        )

    per_seed_header = list(records[0])
    write_csv(
        ensure_subdir("tables") / "agnews_compute_matched_dense_per_seed.csv",
        per_seed_header,
        [[record[key] for key in per_seed_header] for record in records],
    )
    summary = {
        "mode": "agnews_dense_active_matched",
        "label": "dense FFN (expert-path MAdd matched)",
        "seed_count": len(records),
    }
    for metric in ("test_accuracy", "test_cross_entropy"):
        values = np.asarray([record[metric] for record in records], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    for metric in (
        "total_parameters",
        "activated_ffn_parameters_per_token",
        "ffn_multiply_adds_per_token",
    ):
        summary[metric] = records[0][metric]
    summary_header = list(summary)
    write_csv(
        ensure_subdir("tables") / "agnews_compute_matched_dense_summary.csv",
        summary_header,
        [[summary[key] for key in summary_header]],
    )
    profile_header = list(profiles[0])
    write_csv(
        ensure_subdir("tables") / "agnews_compute_profile.csv",
        profile_header,
        [[profile[key] for key in profile_header] for profile in profiles],
    )

    print(
        "dense matched: accuracy "
        f"{float(summary['test_accuracy_mean']):.4f} +/- "
        f"{float(summary['test_accuracy_std']):.4f}"
    )


if __name__ == "__main__":
    main()
