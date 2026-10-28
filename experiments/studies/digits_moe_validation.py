from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.numerics import add_torch_device_argument, ensure_subdir, resolve_torch_device, write_csv


torch.set_num_threads(1)
NUM_CLASSES = 10


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    conditioned: bool


MODES = (
    Mode("digits_global_top2", "global top-2", False),
    Mode("digits_conditioned_top2", "conditioned top-2", True),
)


def top2_straight_through(logits: torch.Tensor) -> torch.Tensor:
    soft = F.softmax(logits, dim=-1)
    indices = torch.topk(logits, k=2, dim=-1).indices
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(dim=-1, index=indices, src=logits.gather(dim=-1, index=indices))
    hard = F.softmax(masked, dim=-1)
    return hard.detach() + soft - soft.detach()


class DigitsMoE(nn.Module):
    def __init__(
        self,
        *,
        conditioned: bool,
        width: int,
        expert_hidden: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.conditioned = conditioned
        self.encoder = nn.Sequential(nn.Linear(64, width), nn.GELU())
        if conditioned:
            self.router = nn.Linear(width, num_experts)
        else:
            self.global_logits = nn.Parameter(torch.zeros(num_experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, expert_hidden),
                    nn.GELU(),
                    nn.Linear(expert_hidden, width),
                )
                for _ in range(num_experts)
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        if self.conditioned:
            logits = self.router(encoded)
        else:
            logits = self.global_logits.unsqueeze(0).expand(x.shape[0], -1)
        probabilities = top2_straight_through(logits)
        expert_outputs = torch.stack([expert(encoded) for expert in self.experts], dim=1)
        mixed = torch.sum(probabilities.unsqueeze(-1) * expert_outputs, dim=1)
        prediction = self.classifier(self.norm(encoded + mixed))
        return prediction, probabilities


def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bundle = load_digits()
    features = np.asarray(bundle.data, dtype=np.float32) / 16.0
    labels = np.asarray(bundle.target, dtype=np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=20260717,
        stratify=labels,
    )
    return (
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(test_x),
        torch.from_numpy(test_y),
    )


def specialization_score(matrix: np.ndarray) -> float:
    """Mean rowwise L1 distance from the unweighted class-load centroid."""
    center = matrix.mean(axis=0)
    return float(np.mean(np.abs(matrix - center[None, :]).sum(axis=1)))


def train_mode(
    mode: Mode,
    *,
    seed: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    width: int,
    expert_hidden: int,
    num_experts: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = DigitsMoE(
        conditioned=mode.conditioned,
        width=width,
        expert_hidden=expert_hidden,
        num_experts=num_experts,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(9200 + seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(train_y.numel(), generator=generator)
        for start in range(0, order.numel(), batch_size):
            index = order[start : start + batch_size]
            logits, _ = model(train_x[index].to(device))
            loss = F.cross_entropy(logits, train_y[index].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    class_load_sum = np.zeros((NUM_CLASSES, num_experts), dtype=float)
    class_count = np.zeros(NUM_CLASSES, dtype=int)
    with torch.no_grad():
        for start in range(0, test_y.numel(), batch_size):
            stop = min(start + batch_size, test_y.numel())
            labels = test_y[start:stop]
            labels_device = labels.to(device)
            logits, probabilities = model(test_x[start:stop].to(device))
            total_loss += float(
                F.cross_entropy(logits, labels_device, reduction="sum").item()
            )
            total_correct += int(
                (logits.argmax(dim=-1) == labels_device).sum().item()
            )
            probabilities = probabilities.float().cpu()
            for label in range(NUM_CLASSES):
                selected = labels == label
                if bool(selected.any()):
                    class_load_sum[label] += probabilities[selected].sum(dim=0).cpu().numpy()
                    class_count[label] += int(selected.sum().item())

    regime_loads = class_load_sum / class_count[:, None]
    return {
        "mode": mode.name,
        "label": mode.label,
        "seed": seed,
        "test_accuracy": total_correct / int(test_y.numel()),
        "test_cross_entropy": total_loss / int(test_y.numel()),
        "unique_dominant_experts": int(np.unique(np.argmax(regime_loads, axis=1)).size),
        "specialization_score": specialization_score(regime_loads),
        "regime_loads": regime_loads,
    }


def aggregate(records: list[dict[str, object]], mode: Mode) -> dict[str, object]:
    selected = [record for record in records if record["mode"] == mode.name]
    result: dict[str, object] = {
        "mode": mode.name,
        "label": mode.label,
        "seed_count": len(selected),
    }
    for metric in (
        "test_accuracy",
        "test_cross_entropy",
        "unique_dominant_experts",
        "specialization_score",
    ):
        values = np.asarray([record[metric] for record in selected], dtype=float)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values))
    result["regime_loads_mean"] = np.mean(
        np.stack([np.asarray(record["regime_loads"]) for record in selected]),
        axis=0,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-scale top-2 MoE routing comparison on sklearn digits."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--expert-hidden", type=int, default=96)
    parser.add_argument("--num-experts", type=int, default=6)
    add_torch_device_argument(parser)
    args = parser.parse_args()
    device = resolve_torch_device(args.device)
    print(f"device={device}")

    train_x, train_y, test_x, test_y = load_data()
    records: list[dict[str, object]] = []
    for mode in MODES:
        for seed in args.seeds:
            print(f"training {mode.name}, seed {seed}")
            records.append(
                train_mode(
                    mode,
                    seed=seed,
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x,
                    test_y=test_y,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    width=args.width,
                    expert_hidden=args.expert_hidden,
                    num_experts=args.num_experts,
                    device=device,
                )
            )

    per_seed_header = [
        "mode",
        "label",
        "seed",
        "test_accuracy",
        "test_cross_entropy",
        "unique_dominant_experts",
        "specialization_score",
    ]
    write_csv(
        ensure_subdir("tables") / "digits_moe_validation_per_seed.csv",
        per_seed_header,
        [[record[key] for key in per_seed_header] for record in records],
    )

    aggregates = [aggregate(records, mode) for mode in MODES]
    summary_header = [
        "mode",
        "label",
        "seed_count",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_cross_entropy_mean",
        "test_cross_entropy_std",
        "unique_dominant_experts_mean",
        "unique_dominant_experts_std",
        "specialization_score_mean",
        "specialization_score_std",
    ]
    write_csv(
        ensure_subdir("tables") / "digits_moe_validation_summary.csv",
        summary_header,
        [[record[key] for key in summary_header] for record in aggregates],
    )

    load_rows: list[list[object]] = []
    for record in records:
        matrix = np.asarray(record["regime_loads"])
        for class_index in range(NUM_CLASSES):
            load_rows.append(
                [record["mode"], record["seed"], class_index, *matrix[class_index].tolist()]
            )
    write_csv(
        ensure_subdir("tables") / "digits_moe_validation_regime_loads.csv",
        ["mode", "seed", "class_index", *[f"expert_{i + 1}" for i in range(args.num_experts)]],
        load_rows,
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
