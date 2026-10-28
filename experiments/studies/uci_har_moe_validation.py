from __future__ import annotations

import argparse
import io
import math
import os
import sys
import urllib.request
import zipfile
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

from experiments.common.numerics import add_torch_device_argument, ensure_subdir, resolve_torch_device, write_csv


torch.set_num_threads(1)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DATASET_DOI = "10.24432/C54S4K"
NUM_FEATURES = 561
NUM_CLASSES = 6
ACTIVITY_NAMES = (
    "walking",
    "upstairs",
    "downstairs",
    "sitting",
    "standing",
    "laying",
)


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    conditioned: bool


MODES = (
    Mode("har_global_top2", "global top-2", False),
    Mode("har_conditioned_top2", "conditioned top-2", True),
)


@dataclass(frozen=True)
class DataBundle:
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_subjects: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_subjects: torch.Tensor


def obtain_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "uci_har_dataset_240.zip"
    if archive_path.exists():
        return archive_path

    temporary_path = archive_path.with_suffix(".download")
    print(f"downloading UCI HAR from {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, temporary_path)
    temporary_path.replace(archive_path)
    return archive_path


def load_text_array(
    archive: zipfile.ZipFile,
    member: str,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    with archive.open(member) as handle:
        return np.loadtxt(handle, dtype=dtype)


def load_data(cache_dir: Path) -> DataBundle:
    archive_path = obtain_archive(cache_dir)
    with zipfile.ZipFile(archive_path) as outer:
        inner_bytes = outer.read("UCI HAR Dataset.zip")

    prefix = "UCI HAR Dataset"
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as archive:
        train_x_raw = load_text_array(
            archive,
            f"{prefix}/train/X_train.txt",
            dtype=np.float32,
        )
        train_y = load_text_array(
            archive,
            f"{prefix}/train/y_train.txt",
            dtype=np.int64,
        )
        train_subjects = load_text_array(
            archive,
            f"{prefix}/train/subject_train.txt",
            dtype=np.int64,
        )
        test_x_raw = load_text_array(
            archive,
            f"{prefix}/test/X_test.txt",
            dtype=np.float32,
        )
        test_y = load_text_array(
            archive,
            f"{prefix}/test/y_test.txt",
            dtype=np.int64,
        )
        test_subjects = load_text_array(
            archive,
            f"{prefix}/test/subject_test.txt",
            dtype=np.int64,
        )

    expected_shapes = {
        "train_x": ((7352, NUM_FEATURES), train_x_raw.shape),
        "train_y": ((7352,), train_y.shape),
        "train_subjects": ((7352,), train_subjects.shape),
        "test_x": ((2947, NUM_FEATURES), test_x_raw.shape),
        "test_y": ((2947,), test_y.shape),
        "test_subjects": ((2947,), test_subjects.shape),
    }
    for name, (expected, actual) in expected_shapes.items():
        if actual != expected:
            raise ValueError(f"Unexpected {name} shape: {actual} != {expected}")

    train_subject_ids = set(np.unique(train_subjects).tolist())
    test_subject_ids = set(np.unique(test_subjects).tolist())
    if len(train_subject_ids) != 21 or len(test_subject_ids) != 9:
        raise ValueError(
            f"Unexpected official subject counts: {len(train_subject_ids)} train, "
            f"{len(test_subject_ids)} test"
        )
    if train_subject_ids & test_subject_ids:
        raise ValueError("Official UCI HAR train/test subject sets are not disjoint")
    if set(np.unique(train_y)) != set(range(1, NUM_CLASSES + 1)):
        raise ValueError("Unexpected UCI HAR training labels")
    if set(np.unique(test_y)) != set(range(1, NUM_CLASSES + 1)):
        raise ValueError("Unexpected UCI HAR test labels")


    train64 = train_x_raw.astype(np.float64)
    test64 = test_x_raw.astype(np.float64)
    feature_mean = train64.mean(axis=0)
    feature_std = train64.std(axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    train_x = ((train64 - feature_mean) / feature_std).astype(np.float32)
    test_x = ((test64 - feature_mean) / feature_std).astype(np.float32)

    return DataBundle(
        train_x=torch.from_numpy(train_x),
        train_y=torch.from_numpy(train_y - 1),
        train_subjects=torch.from_numpy(train_subjects),
        test_x=torch.from_numpy(test_x),
        test_y=torch.from_numpy(test_y - 1),
        test_subjects=torch.from_numpy(test_subjects),
    )


def top2_straight_through(logits: torch.Tensor) -> torch.Tensor:
    soft = F.softmax(logits, dim=-1)
    indices = torch.topk(logits, k=2, dim=-1).indices
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(dim=-1, index=indices, src=logits.gather(dim=-1, index=indices))
    hard = F.softmax(masked, dim=-1)
    return hard.detach() + soft - soft.detach()


class HarMoE(nn.Module):
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
        self.encoder = nn.Sequential(nn.Linear(NUM_FEATURES, width), nn.GELU())
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

        # Router parameters are created last so paired modes share identical
        # encoder/expert/classifier initialization for each seed.
        if conditioned:
            self.router = nn.Linear(width, num_experts)
        else:
            self.global_logits = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        if self.conditioned:
            router_logits = self.router(encoded)
        else:
            router_logits = self.global_logits.unsqueeze(0).expand(x.shape[0], -1)
        probabilities = top2_straight_through(router_logits)
        expert_outputs = torch.stack([expert(encoded) for expert in self.experts], dim=1)
        mixed = torch.sum(probabilities.unsqueeze(-1) * expert_outputs, dim=1)
        prediction = self.classifier(self.norm(encoded + mixed))
        return prediction, probabilities


def specialization_score(matrix: np.ndarray) -> float:
    center = matrix.mean(axis=0)
    return float(np.mean(np.abs(matrix - center[None, :]).sum(axis=1)))


def train_mode(
    mode: Mode,
    *,
    seed: int,
    data: DataBundle,
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
    model = HarMoE(
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
        order = torch.randperm(data.train_y.numel(), generator=generator)
        for start in range(0, order.numel(), batch_size):
            index = order[start : start + batch_size]
            logits, _ = model(data.train_x[index].to(device))
            loss = F.cross_entropy(logits, data.train_y[index].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    activity_load_sum = np.zeros((NUM_CLASSES, num_experts), dtype=float)
    activity_count = np.zeros(NUM_CLASSES, dtype=int)
    subject_correct: dict[int, int] = {}
    subject_count: dict[int, int] = {}
    with torch.no_grad():
        for start in range(0, data.test_y.numel(), batch_size):
            stop = min(start + batch_size, data.test_y.numel())
            labels = data.test_y[start:stop]
            subjects = data.test_subjects[start:stop]
            labels_device = labels.to(device)
            logits, probabilities = model(data.test_x[start:stop].to(device))
            predictions = logits.argmax(dim=-1).cpu()
            total_loss += float(
                F.cross_entropy(logits, labels_device, reduction="sum").item()
            )
            total_correct += int((predictions == labels).sum().item())
            probabilities = probabilities.float().cpu()

            for label in range(NUM_CLASSES):
                selected = labels == label
                if bool(selected.any()):
                    activity_load_sum[label] += probabilities[selected].sum(dim=0).cpu().numpy()
                    activity_count[label] += int(selected.sum().item())
            for subject in torch.unique(subjects).tolist():
                selected = subjects == subject
                subject = int(subject)
                subject_correct[subject] = subject_correct.get(subject, 0) + int(
                    (predictions[selected] == labels[selected]).sum().item()
                )
                subject_count[subject] = subject_count.get(subject, 0) + int(selected.sum().item())

    regime_loads = activity_load_sum / activity_count[:, None]
    subject_accuracies = np.asarray(
        [subject_correct[subject] / subject_count[subject] for subject in sorted(subject_count)]
    )
    return {
        "mode": mode.name,
        "label": mode.label,
        "seed": seed,
        "test_accuracy": total_correct / int(data.test_y.numel()),
        "test_cross_entropy": total_loss / int(data.test_y.numel()),
        "macro_subject_accuracy": float(subject_accuracies.mean()),
        "worst_subject_accuracy": float(subject_accuracies.min()),
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
        "macro_subject_accuracy",
        "worst_subject_accuracy",
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


def exact_two_sided_sign_p(differences: np.ndarray, tol: float = 1e-12) -> tuple[int, int, int, float]:
    wins = int(np.sum(differences > tol))
    losses = int(np.sum(differences < -tol))
    ties = int(differences.size - wins - losses)
    non_ties = wins + losses
    if non_ties == 0:
        return wins, ties, losses, 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(non_ties, index) for index in range(tail + 1)) / (2**non_ties)
    return wins, ties, losses, min(1.0, 2.0 * probability)


def paired_statistics(records: list[dict[str, object]]) -> dict[str, object]:
    by_mode = {
        mode.name: {int(record["seed"]): record for record in records if record["mode"] == mode.name}
        for mode in MODES
    }
    seeds = sorted(set(by_mode[MODES[0].name]) & set(by_mode[MODES[1].name]))
    if len(seeds) < 2:
        raise ValueError("Paired inference requires at least two seeds")
    control = np.asarray(
        [float(by_mode[MODES[0].name][seed]["specialization_score"]) for seed in seeds]
    )
    treatment = np.asarray(
        [float(by_mode[MODES[1].name][seed]["specialization_score"]) for seed in seeds]
    )
    differences = treatment - control
    rng = np.random.default_rng(20260717)
    indices = rng.integers(0, differences.size, size=(100_000, differences.size))
    bootstrap_means = np.mean(differences[indices], axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    wins, ties, losses, sign_p = exact_two_sided_sign_p(differences)
    difference_sd = float(np.std(differences, ddof=1))
    paired_dz = float(np.mean(differences) / difference_sd) if difference_sd > 0 else float("nan")
    return {
        "family": "uci_har_prospective_local_confirmatory_replication",
        "endpoint_role": "single_confirmatory_primary",
        "outcome": "specialization_score",
        "treatment": MODES[1].name,
        "control": MODES[0].name,
        "favorable_direction": "higher",
        "pair_count": len(seeds),
        "family_size": 1,
        "treatment_mean": float(np.mean(treatment)),
        "control_mean": float(np.mean(control)),
        "paired_mean_difference": float(np.mean(differences)),
        "bootstrap_95_ci_low": float(ci_low),
        "bootstrap_95_ci_high": float(ci_high),
        "paired_dz": paired_dz,
        "favorable_wins": wins,
        "ties": ties,
        "favorable_losses": losses,
        "exact_two_sided_sign_p": sign_p,
        "multiplicity_adjusted_p": sign_p,
        "reject_at_alpha_0p05": sign_p < 0.05,
        "externally_preregistered": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subject-disjoint UCI HAR top-2 MoE confirmatory replication."
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "uci_har")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--expert-hidden", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=6)
    add_torch_device_argument(parser)
    args = parser.parse_args()
    device = resolve_torch_device(args.device)
    print(f"device={device}")


    data = load_data(args.cache_dir)
    records: list[dict[str, object]] = []
    for mode in MODES:
        for seed in args.seeds:
            print(f"training {mode.name}, seed {seed}")
            records.append(
                train_mode(
                    mode,
                    seed=seed,
                    data=data,
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
        "macro_subject_accuracy",
        "worst_subject_accuracy",
        "unique_dominant_experts",
        "specialization_score",
    ]
    write_csv(
        ensure_subdir("tables") / "uci_har_moe_validation_per_seed.csv",
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
        "macro_subject_accuracy_mean",
        "macro_subject_accuracy_std",
        "worst_subject_accuracy_mean",
        "worst_subject_accuracy_std",
        "unique_dominant_experts_mean",
        "unique_dominant_experts_std",
        "specialization_score_mean",
        "specialization_score_std",
    ]
    write_csv(
        ensure_subdir("tables") / "uci_har_moe_validation_summary.csv",
        summary_header,
        [[record[key] for key in summary_header] for record in aggregates],
    )

    load_rows: list[list[object]] = []
    for record in records:
        matrix = np.asarray(record["regime_loads"])
        for activity_index, activity_name in enumerate(ACTIVITY_NAMES):
            load_rows.append(
                [
                    record["mode"],
                    record["seed"],
                    activity_index,
                    activity_name,
                    *matrix[activity_index].tolist(),
                ]
            )
    write_csv(
        ensure_subdir("tables") / "uci_har_moe_validation_regime_loads.csv",
        [
            "mode",
            "seed",
            "activity_index",
            "activity_name",
            *[f"expert_{i + 1}" for i in range(args.num_experts)],
        ],
        load_rows,
    )

    statistics = paired_statistics(records)
    statistics_header = list(statistics)
    write_csv(
        ensure_subdir("tables") / "uci_har_confirmatory_test.csv",
        statistics_header,
        [[statistics[key] for key in statistics_header]],
    )


    for record in aggregates:
        print(
            f"{record['label']}: accuracy "
            f"{float(record['test_accuracy_mean']):.4f} +/- "
            f"{float(record['test_accuracy_std']):.4f}, specialization "
            f"{float(record['specialization_score_mean']):.4f} +/- "
            f"{float(record['specialization_score_std']):.4f}"
        )
    print(
        "confirmatory specialization difference "
        f"{float(statistics['paired_mean_difference']):.4f}, "
        f"95% CI [{float(statistics['bootstrap_95_ci_low']):.4f}, "
        f"{float(statistics['bootstrap_95_ci_high']):.4f}], "
        f"wins/ties/losses {statistics['favorable_wins']}/{statistics['ties']}/"
        f"{statistics['favorable_losses']}, exact p={float(statistics['exact_two_sided_sign_p']):.6f}"
    )


if __name__ == "__main__":
    main()
