from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
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

from experiments.common.numerics import add_torch_device_argument, ensure_output_dir, ensure_subdir, resolve_torch_device, write_csv


torch.set_num_threads(1)

DATA_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}
DATA_MD5 = {
    "train": "b1a00f826fdfbd249f79597b59e1dc12",
    "test": "d52ea96a97a2d943681189a97654912d",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)
NUM_CLASSES = 4


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    conditioned: bool
    balance_lambda: float


MODES = (
    Mode("agnews_global_top2", "global top-2", False, 0.0),
    Mode("agnews_conditioned_top2", "conditioned top-2", True, 0.0),
    Mode("agnews_conditioned_top2_balanced", "conditioned top-2 + balance", True, 0.10),
)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_agnews(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in DATA_URLS.items():
        path = data_dir / f"{split}.csv"
        if not path.exists() or md5(path) != DATA_MD5[split]:
            print(f"downloading AG News {split} split")
            urllib.request.urlretrieve(url, path)
        actual = md5(path)
        if actual != DATA_MD5[split]:
            raise RuntimeError(f"AG News {split} checksum mismatch: {actual}")
        paths[split] = path
    return paths["train"], paths["test"]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def read_examples(path: Path) -> list[tuple[int, str]]:
    examples: list[tuple[int, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            label = int(row[0]) - 1
            examples.append((label, f"{row[1]} {row[2]}"))
    return examples


def stratified_subset(
    examples: list[tuple[int, str]],
    *,
    per_class: int,
    seed: int,
) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    grouped: list[list[tuple[int, str]]] = [[] for _ in range(NUM_CLASSES)]
    for example in examples:
        grouped[example[0]].append(example)
    result: list[tuple[int, str]] = []
    for group in grouped:
        rng.shuffle(group)
        if len(group) < per_class:
            raise ValueError("Requested subset exceeds available examples.")
        result.extend(group[:per_class])
    rng.shuffle(result)
    return result


def build_vocab(examples: list[tuple[int, str]], max_vocab: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, text in examples:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    for token, _ in counts.most_common(max_vocab - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def encode_examples(
    examples: list[tuple[int, str]],
    *,
    vocab: dict[str, int],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((len(examples), max_length), dtype=torch.long)
    mask = torch.zeros((len(examples), max_length), dtype=torch.bool)
    labels = torch.zeros(len(examples), dtype=torch.long)
    for row, (label, text) in enumerate(examples):
        ids = [vocab["<cls>"]]
        ids.extend(vocab.get(token, vocab["<unk>"]) for token in tokenize(text)[: max_length - 1])
        length = len(ids)
        tokens[row, :length] = torch.tensor(ids, dtype=torch.long)
        mask[row, :length] = True
        labels[row] = label
    return tokens, mask, labels


def top2_straight_through(logits: torch.Tensor) -> torch.Tensor:
    soft = F.softmax(logits, dim=-1)
    indices = torch.topk(logits, k=2, dim=-1).indices
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(dim=-1, index=indices, src=logits.gather(dim=-1, index=indices))
    hard = F.softmax(masked, dim=-1)
    return hard.detach() + soft - soft.detach()


class TokenMoE(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden: int,
        num_experts: int,
        conditioned: bool,
    ) -> None:
        super().__init__()
        self.conditioned = conditioned
        if conditioned:
            self.router = nn.Linear(d_model, num_experts)
        else:
            self.global_logits = nn.Parameter(torch.zeros(num_experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, d_model),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conditioned:
            logits = self.router(x)
        else:
            logits = self.global_logits.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
        probabilities = top2_straight_through(logits)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = torch.sum(probabilities.unsqueeze(-1) * expert_outputs, dim=-2)
        return mixed, probabilities


class TinyTransformerMoE(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        conditioned: bool,
        d_model: int,
        num_heads: int,
        num_experts: int,
        expert_hidden: int,
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
        self.moe = TokenMoE(
            d_model=d_model,
            hidden=expert_hidden,
            num_experts=num_experts,
            conditioned=conditioned,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        moe_output, probabilities = self.moe(x)
        x = self.norm2(x + self.dropout(moe_output))
        logits = self.classifier(x[:, 0])
        return logits, probabilities


def balance_penalty(probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = probabilities[mask]
    load = active.mean(dim=0)
    target = torch.full_like(load, 1.0 / load.numel())
    return torch.sum((load - target) ** 2)


def train_mode(
    mode: Mode,
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
    num_experts: int,
    expert_hidden: int,
    dropout: float,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = TinyTransformerMoE(
        vocab_size=vocab_size,
        max_length=max_length,
        conditioned=mode.conditioned,
        d_model=d_model,
        num_heads=num_heads,
        num_experts=num_experts,
        expert_hidden=expert_hidden,
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
            tokens = train_tokens[index].to(device)
            mask = train_mask[index].to(device)
            labels = train_labels[index].to(device)
            logits, probabilities = model(tokens, mask)
            loss = F.cross_entropy(logits, labels)
            if mode.balance_lambda > 0.0:
                loss = loss + mode.balance_lambda * balance_penalty(
                    probabilities,
                    mask,
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    test_tokens, test_mask, test_labels = test_data
    model.eval()
    class_load_sum = np.zeros((NUM_CLASSES, num_experts), dtype=float)
    class_count = np.zeros(NUM_CLASSES, dtype=int)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    global_load_sum = np.zeros(num_experts, dtype=float)
    global_token_count = 0
    with torch.no_grad():
        for start in range(0, test_labels.numel(), batch_size):
            stop = min(start + batch_size, test_labels.numel())
            tokens = test_tokens[start:stop]
            mask = test_mask[start:stop]
            labels = test_labels[start:stop]
            logits, probabilities = model(tokens.to(device), mask.to(device))
            labels_device = labels.to(device)
            total_loss += float(
                F.cross_entropy(logits, labels_device, reduction="sum").item()
            )
            total_correct += int(
                (logits.argmax(dim=-1) == labels_device).sum().item()
            )
            total_count += int(labels.numel())
            active = probabilities.float().cpu() * mask.unsqueeze(-1)
            per_document = active.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            for label in range(NUM_CLASSES):
                selected = labels == label
                if bool(selected.any()):
                    class_load_sum[label] += per_document[selected].sum(dim=0).cpu().numpy()
                    class_count[label] += int(selected.sum().item())
            global_load_sum += active.sum(dim=(0, 1)).cpu().numpy()
            global_token_count += int(mask.sum().item())

    regime_loads = class_load_sum / class_count[:, None]
    global_load = global_load_sum / global_token_count
    row_center = regime_loads.mean(axis=0)
    specialization = float(np.mean(np.abs(regime_loads - row_center[None, :]).sum(axis=1)))
    unique_winners = int(np.unique(np.argmax(regime_loads, axis=1)).size)
    return {
        "mode": mode.name,
        "label": mode.label,
        "balance_lambda": mode.balance_lambda,
        "seed": seed,
        "test_accuracy": total_correct / total_count,
        "test_cross_entropy": total_loss / total_count,
        "unique_dominant_experts": unique_winners,
        "specialization_score": specialization,
        "max_global_load": float(np.max(global_load)),
        "regime_loads": regime_loads,
    }


def aggregate(records: list[dict[str, object]], mode: Mode) -> dict[str, object]:
    selected = [record for record in records if record["mode"] == mode.name]
    result: dict[str, object] = {
        "mode": mode.name,
        "label": mode.label,
        "balance_lambda": mode.balance_lambda,
        "seed_count": len(selected),
    }
    for metric in (
        "test_accuracy",
        "test_cross_entropy",
        "unique_dominant_experts",
        "specialization_score",
        "max_global_load",
    ):
        values = np.asarray([record[metric] for record in selected], dtype=float)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values))
    result["regime_loads_mean"] = np.mean(
        np.stack([record["regime_loads"] for record in selected]),
        axis=0,
    )
    return result




def main() -> None:
    parser = argparse.ArgumentParser(description="Small Transformer-MoE audit on AG News.")
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

    records: list[dict[str, object]] = []
    for mode in MODES:
        for seed in args.seeds:
            print(f"training {mode.name}, seed {seed}")
            records.append(
                train_mode(
                    mode,
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
                    num_experts=args.num_experts,
                    expert_hidden=args.expert_hidden,
                    dropout=args.dropout,
                    device=device,
                )
            )

    per_seed_header = [
        "mode",
        "label",
        "balance_lambda",
        "seed",
        "test_accuracy",
        "test_cross_entropy",
        "unique_dominant_experts",
        "specialization_score",
        "max_global_load",
    ]
    write_csv(
        ensure_subdir("tables") / "transformer_moe_agnews_per_seed.csv",
        per_seed_header,
        [[record[key] for key in per_seed_header] for record in records],
    )

    aggregates = [aggregate(records, mode) for mode in MODES]
    summary_header = [
        "mode",
        "label",
        "balance_lambda",
        "seed_count",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_cross_entropy_mean",
        "test_cross_entropy_std",
        "unique_dominant_experts_mean",
        "unique_dominant_experts_std",
        "specialization_score_mean",
        "specialization_score_std",
        "max_global_load_mean",
        "max_global_load_std",
    ]
    write_csv(
        ensure_subdir("tables") / "transformer_moe_agnews_summary.csv",
        summary_header,
        [[record[key] for key in summary_header] for record in aggregates],
    )

    load_rows: list[list[object]] = []
    for record in records:
        matrix = np.asarray(record["regime_loads"])
        for class_index in range(NUM_CLASSES):
            load_rows.append(
                [
                    record["mode"],
                    record["seed"],
                    class_index,
                    *matrix[class_index].tolist(),
                ]
            )
    write_csv(
        ensure_subdir("tables") / "transformer_moe_agnews_regime_loads.csv",
        ["mode", "seed", "class_index", *[f"expert_{i + 1}" for i in range(args.num_experts)]],
        load_rows,
    )

    manifest = ensure_output_dir() / "transformer_moe_agnews_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"train_url={DATA_URLS['train']}",
                f"train_md5={DATA_MD5['train']}",
                f"test_url={DATA_URLS['test']}",
                f"test_md5={DATA_MD5['test']}",
                "dataset_citation=Zhang_Zhao_LeCun_2015_Character-level_Convolutional_Networks_for_Text_Classification",
                f"train_per_class={args.train_per_class}",
                f"test_per_class={args.test_per_class}",
                f"subset_seed=20260717",
                f"vocab_size={len(vocab)}",
                f"seeds={','.join(str(seed) for seed in args.seeds)}",
                f"epochs={args.epochs}",
                f"batch_size={args.batch_size}",
                f"learning_rate={args.learning_rate}",
                f"weight_decay={args.weight_decay}",
                f"max_length={args.max_length}",
                f"max_vocab={args.max_vocab}",
                f"d_model={args.d_model}",
                f"num_heads={args.num_heads}",
                f"num_experts={args.num_experts}",
                f"expert_hidden={args.expert_hidden}",
                f"dropout={args.dropout}",
                "model=one_block_transformer_token_top2_moe",
                "optimizer=AdamW",
                f"device={device}",
                f"torch_version={torch.__version__}",
                f"numpy_version={np.__version__}",
            ]
        )
        + "\n"
    )

    for record in aggregates:
        print(
            f"{record['label']}: accuracy "
            f"{float(record['test_accuracy_mean']):.4f} +/- "
            f"{float(record['test_accuracy_std']):.4f}, specialization "
            f"{float(record['specialization_score_mean']):.4f} +/- "
            f"{float(record['specialization_score_std']):.4f}"
        )


if __name__ == "__main__":
    main()
