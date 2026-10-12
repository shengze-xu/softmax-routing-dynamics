from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.common.numerics import add_torch_device_argument, ensure_output_dir, ensure_subdir, resolve_torch_device, write_csv
from experiments.models.synthetic_regime_shift import DatasetBundle, MediumExpert, NUM_EXPERTS, NUM_REGIMES, build_phase_datasets, move_bundle_to_device, parse_seeds, regime_load_matrix, router_entropy, support_size_from_load


torch.set_num_threads(1)


@dataclass(frozen=True)
class ModeSpec:
    name: str
    label: str
    conditioned: bool
    top_k: int | None
    balance_lambda: float


@dataclass
class TrainResult:
    mode: str
    label: str
    seed: int
    top_k: int
    balance_lambda: float
    curve_epochs: np.ndarray
    specialization_curve: np.ndarray
    val_mse_curve: np.ndarray
    new_regime_mse_curve: np.ndarray
    max_load_curve: np.ndarray
    post_shift_final_entropy: float
    post_shift_support_eps_005: int
    post_shift_final_val_mse: float
    post_shift_final_new_regime_mse: float
    unique_dominant_experts: int
    specialization_score: float
    post_shift_final_max_load: float
    final_regime_load_matrix: np.ndarray


@dataclass
class AggregatedResult:
    mode: str
    label: str
    top_k: int
    balance_lambda: float
    seed_count: int
    curve_epochs: np.ndarray
    specialization_curve_mean: np.ndarray
    specialization_curve_std: np.ndarray
    val_mse_curve_mean: np.ndarray
    val_mse_curve_std: np.ndarray
    new_regime_mse_curve_mean: np.ndarray
    new_regime_mse_curve_std: np.ndarray
    max_load_curve_mean: np.ndarray
    max_load_curve_std: np.ndarray
    post_shift_final_entropy_mean: float
    post_shift_final_entropy_std: float
    post_shift_support_mean: float
    post_shift_support_std: float
    post_shift_support_min: int
    post_shift_support_max: int
    post_shift_final_val_mse_mean: float
    post_shift_final_val_mse_std: float
    post_shift_final_new_regime_mse_mean: float
    post_shift_final_new_regime_mse_std: float
    unique_dominant_experts_mean: float
    unique_dominant_experts_std: float
    specialization_score_mean: float
    specialization_score_std: float
    post_shift_final_max_load_mean: float
    post_shift_final_max_load_std: float
    final_regime_load_matrix_mean: np.ndarray
    final_regime_load_matrix_std: np.ndarray


def specialization_score_from_matrix(matrix: np.ndarray) -> float:
    global_load = matrix.mean(axis=0)
    return float(np.mean(np.abs(matrix - global_load[None, :]).sum(axis=1)))


def topk_straight_through(logits: torch.Tensor, k: int) -> torch.Tensor:
    soft = F.softmax(logits, dim=-1)
    if k >= logits.shape[-1]:
        return soft

    topk_indices = torch.topk(logits, k=k, dim=-1).indices
    masked_logits = torch.full_like(logits, float("-inf"))
    masked_logits.scatter_(dim=-1, index=topk_indices, src=logits.gather(dim=-1, index=topk_indices))
    hard = F.softmax(masked_logits, dim=-1)
    return hard.detach() + soft - soft.detach()


class GlobalRouterMoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int = NUM_EXPERTS,
        temperature: float = 4.5,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.global_logits = nn.Parameter(0.01 * torch.randn(num_experts))
        self.experts = nn.ModuleList([MediumExpert() for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.global_logits.unsqueeze(0).expand(x.shape[0], -1) * self.temperature
        if self.top_k is None:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = topk_straight_through(logits, self.top_k)
        expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=-1)
        prediction = (probs * expert_outputs).sum(dim=-1)
        return prediction, probs


class ConditionedRouterMoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int = NUM_EXPERTS,
        temperature: float = 4.5,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.router = nn.Sequential(
            nn.Linear(2, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, num_experts),
        )
        self.experts = nn.ModuleList([MediumExpert() for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x) * self.temperature
        if self.top_k is None:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = topk_straight_through(logits, self.top_k)
        expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=-1)
        prediction = (probs * expert_outputs).sum(dim=-1)
        return prediction, probs


def build_mode_specs(top_k: int, balance_lambda: float) -> list[ModeSpec]:
    return [
        ModeSpec(
            name="arch_shared_global_softmax",
            label="global softmax",
            conditioned=False,
            top_k=None,
            balance_lambda=0.0,
        ),
        ModeSpec(
            name=f"arch_shared_global_top{top_k}",
            label=f"global top-{top_k}",
            conditioned=False,
            top_k=top_k,
            balance_lambda=0.0,
        ),
        ModeSpec(
            name="arch_input_conditioned_softmax",
            label="conditioned softmax",
            conditioned=True,
            top_k=None,
            balance_lambda=0.0,
        ),
        ModeSpec(
            name=f"arch_input_conditioned_top{top_k}",
            label=f"conditioned top-{top_k}",
            conditioned=True,
            top_k=top_k,
            balance_lambda=0.0,
        ),
        ModeSpec(
            name=f"arch_input_conditioned_top{top_k}_balanced",
            label=f"conditioned top-{top_k}+balance",
            conditioned=True,
            top_k=top_k,
            balance_lambda=balance_lambda,
        ),
    ]


def load_balance_penalty(probs: torch.Tensor) -> torch.Tensor:
    mean_load = probs.mean(dim=0)
    target = torch.full_like(mean_load, 1.0 / mean_load.numel())
    return (mean_load - target).pow(2).sum()


def train_epoch(
    model: nn.Module,
    bundle: DatasetBundle,
    optimizer: torch.optim.Optimizer,
    spec: ModeSpec,
    *,
    batch_size: int,
    generator: torch.Generator,
    clip_grad: float,
) -> None:
    permutation = torch.randperm(bundle.x.shape[0], generator=generator)
    for start in range(0, permutation.numel(), batch_size):
        index = permutation[start : start + batch_size].to(bundle.x.device)
        pred, probs = model(bundle.x[index])
        mse = ((pred - bundle.y[index]) ** 2).mean()
        loss = mse
        if spec.balance_lambda > 0.0:
            loss = loss + spec.balance_lambda * load_balance_penalty(probs)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()


def evaluate_checkpoint(
    model: nn.Module,
    reference_eval: DatasetBundle,
) -> tuple[float, float, float, float, int, float, np.ndarray]:
    new_regime_mask = reference_eval.regimes == 3
    with torch.no_grad():
        pred, probs = model(reference_eval.x)
        val_mse = float(((pred - reference_eval.y) ** 2).mean().item())
        new_regime_mse = float(((pred[new_regime_mask] - reference_eval.y[new_regime_mask]) ** 2).mean().item())
        entropy = float(router_entropy(probs).item())
        load = probs.mean(dim=0).detach().cpu().numpy()
        matrix = regime_load_matrix(probs, reference_eval.regimes)
    return (
        val_mse,
        new_regime_mse,
        entropy,
        float(load.max()),
        support_size_from_load(load),
        specialization_score_from_matrix(matrix),
        matrix,
    )


def train_one_setting(
    spec: ModeSpec,
    seed: int,
    *,
    pre_shift_epochs: int,
    post_shift_epochs: int,
    lr: float,
    batch_size: int,
    eval_interval: int,
    clip_grad: float,
    device: torch.device,
) -> TrainResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    pre_shift_train, post_shift_train, reference_eval = build_phase_datasets(seed)
    pre_shift_train = move_bundle_to_device(pre_shift_train, device)
    post_shift_train = move_bundle_to_device(post_shift_train, device)
    reference_eval = move_bundle_to_device(reference_eval, device)
    model: nn.Module
    if spec.conditioned:
        model = ConditionedRouterMoE(top_k=spec.top_k)
    else:
        model = GlobalRouterMoE(top_k=spec.top_k)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(7500 + seed)

    for _ in range(pre_shift_epochs):
        train_epoch(
            model,
            pre_shift_train,
            optimizer,
            spec,
            batch_size=batch_size,
            generator=generator,
            clip_grad=clip_grad,
        )

    curve_epochs: list[int] = []
    specialization_curve: list[float] = []
    val_mse_curve: list[float] = []
    new_regime_mse_curve: list[float] = []
    max_load_curve: list[float] = []

    def append_checkpoint(epoch: int) -> None:
        val_mse, new_regime_mse, _, max_load, _, specialization, _ = evaluate_checkpoint(model, reference_eval)
        curve_epochs.append(epoch)
        specialization_curve.append(specialization)
        val_mse_curve.append(val_mse)
        new_regime_mse_curve.append(new_regime_mse)
        max_load_curve.append(max_load)

    append_checkpoint(0)
    for epoch in range(1, post_shift_epochs + 1):
        train_epoch(
            model,
            post_shift_train,
            optimizer,
            spec,
            batch_size=batch_size,
            generator=generator,
            clip_grad=clip_grad,
        )
        if epoch % eval_interval == 0 or epoch == post_shift_epochs:
            append_checkpoint(epoch)

    (
        final_val_mse,
        final_new_regime_mse,
        final_entropy,
        final_max_load,
        final_support,
        final_specialization,
        final_matrix,
    ) = evaluate_checkpoint(model, reference_eval)

    return TrainResult(
        mode=spec.name,
        label=spec.label,
        seed=seed,
        top_k=spec.top_k or 0,
        balance_lambda=spec.balance_lambda,
        curve_epochs=np.array(curve_epochs, dtype=int),
        specialization_curve=np.array(specialization_curve, dtype=float),
        val_mse_curve=np.array(val_mse_curve, dtype=float),
        new_regime_mse_curve=np.array(new_regime_mse_curve, dtype=float),
        max_load_curve=np.array(max_load_curve, dtype=float),
        post_shift_final_entropy=final_entropy,
        post_shift_support_eps_005=final_support,
        post_shift_final_val_mse=final_val_mse,
        post_shift_final_new_regime_mse=final_new_regime_mse,
        unique_dominant_experts=len({int(np.argmax(row)) for row in final_matrix}),
        specialization_score=final_specialization,
        post_shift_final_max_load=final_max_load,
        final_regime_load_matrix=final_matrix,
    )


def aggregate_results(results: list[TrainResult]) -> AggregatedResult:
    entropy = np.array([result.post_shift_final_entropy for result in results], dtype=float)
    support = np.array([result.post_shift_support_eps_005 for result in results], dtype=float)
    val_mse = np.array([result.post_shift_final_val_mse for result in results], dtype=float)
    new_regime_mse = np.array([result.post_shift_final_new_regime_mse for result in results], dtype=float)
    winners = np.array([result.unique_dominant_experts for result in results], dtype=float)
    specialization = np.array([result.specialization_score for result in results], dtype=float)
    max_load = np.array([result.post_shift_final_max_load for result in results], dtype=float)
    regime_matrices = np.stack([result.final_regime_load_matrix for result in results], axis=0)
    specialization_curves = np.stack([result.specialization_curve for result in results], axis=0)
    val_mse_curves = np.stack([result.val_mse_curve for result in results], axis=0)
    new_regime_mse_curves = np.stack([result.new_regime_mse_curve for result in results], axis=0)
    max_load_curves = np.stack([result.max_load_curve for result in results], axis=0)

    first = results[0]
    return AggregatedResult(
        mode=first.mode,
        label=first.label,
        top_k=first.top_k,
        balance_lambda=first.balance_lambda,
        seed_count=len(results),
        curve_epochs=first.curve_epochs,
        specialization_curve_mean=specialization_curves.mean(axis=0),
        specialization_curve_std=specialization_curves.std(axis=0),
        val_mse_curve_mean=val_mse_curves.mean(axis=0),
        val_mse_curve_std=val_mse_curves.std(axis=0),
        new_regime_mse_curve_mean=new_regime_mse_curves.mean(axis=0),
        new_regime_mse_curve_std=new_regime_mse_curves.std(axis=0),
        max_load_curve_mean=max_load_curves.mean(axis=0),
        max_load_curve_std=max_load_curves.std(axis=0),
        post_shift_final_entropy_mean=float(entropy.mean()),
        post_shift_final_entropy_std=float(entropy.std()),
        post_shift_support_mean=float(support.mean()),
        post_shift_support_std=float(support.std()),
        post_shift_support_min=int(support.min()),
        post_shift_support_max=int(support.max()),
        post_shift_final_val_mse_mean=float(val_mse.mean()),
        post_shift_final_val_mse_std=float(val_mse.std()),
        post_shift_final_new_regime_mse_mean=float(new_regime_mse.mean()),
        post_shift_final_new_regime_mse_std=float(new_regime_mse.std()),
        unique_dominant_experts_mean=float(winners.mean()),
        unique_dominant_experts_std=float(winners.std()),
        specialization_score_mean=float(specialization.mean()),
        specialization_score_std=float(specialization.std()),
        post_shift_final_max_load_mean=float(max_load.mean()),
        post_shift_final_max_load_std=float(max_load.std()),
        final_regime_load_matrix_mean=regime_matrices.mean(axis=0),
        final_regime_load_matrix_std=regime_matrices.std(axis=0),
    )










def write_outputs(results_by_mode: list[list[TrainResult]], aggregated: list[AggregatedResult], seeds: list[int]) -> None:
    out_dir = ensure_output_dir()
    tables_dir = ensure_subdir("tables")

    np.savez(
        out_dir / "architecture_level_moe_check.npz",
        mode_names=np.array([result.mode for result in aggregated], dtype=object),
        mode_labels=np.array([result.label for result in aggregated], dtype=object),
        seeds=np.array(seeds, dtype=int),
        curve_epochs=aggregated[0].curve_epochs,
        specialization_curve_mean=np.stack([result.specialization_curve_mean for result in aggregated], axis=0),
        specialization_curve_std=np.stack([result.specialization_curve_std for result in aggregated], axis=0),
        new_regime_mse_curve_mean=np.stack([result.new_regime_mse_curve_mean for result in aggregated], axis=0),
        new_regime_mse_curve_std=np.stack([result.new_regime_mse_curve_std for result in aggregated], axis=0),
        final_regime_load_matrix_mean=np.stack([result.final_regime_load_matrix_mean for result in aggregated], axis=0),
        final_regime_load_matrix_std=np.stack([result.final_regime_load_matrix_std for result in aggregated], axis=0),
    )

    summary_rows: list[list[object]] = []
    for result in aggregated:
        summary_rows.append(
            [
                result.mode,
                result.label,
                result.top_k,
                round(result.balance_lambda, 8),
                result.seed_count,
                round(result.post_shift_final_entropy_mean, 8),
                round(result.post_shift_final_entropy_std, 8),
                round(result.post_shift_support_mean, 8),
                round(result.post_shift_support_std, 8),
                result.post_shift_support_min,
                result.post_shift_support_max,
                round(result.post_shift_final_val_mse_mean, 8),
                round(result.post_shift_final_val_mse_std, 8),
                round(result.post_shift_final_new_regime_mse_mean, 8),
                round(result.post_shift_final_new_regime_mse_std, 8),
                round(result.unique_dominant_experts_mean, 8),
                round(result.unique_dominant_experts_std, 8),
                round(result.specialization_score_mean, 8),
                round(result.specialization_score_std, 8),
                round(result.post_shift_final_max_load_mean, 8),
                round(result.post_shift_final_max_load_std, 8),
            ]
        )
    write_csv(
        tables_dir / "architecture_level_moe_check_summary.csv",
        [
            "mode",
            "label",
            "top_k",
            "balance_lambda",
            "seed_count",
            "post_shift_final_entropy_mean",
            "post_shift_final_entropy_std",
            "post_shift_support_eps_005_mean",
            "post_shift_support_eps_005_std",
            "post_shift_support_eps_005_min",
            "post_shift_support_eps_005_max",
            "post_shift_final_val_mse_mean",
            "post_shift_final_val_mse_std",
            "post_shift_final_new_regime_mse_mean",
            "post_shift_final_new_regime_mse_std",
            "unique_dominant_experts_mean",
            "unique_dominant_experts_std",
            "specialization_score_mean",
            "specialization_score_std",
            "post_shift_final_max_load_mean",
            "post_shift_final_max_load_std",
        ],
        summary_rows,
    )

    per_seed_rows: list[list[object]] = []
    for mode_results in results_by_mode:
        for result in mode_results:
            per_seed_rows.append(
                [
                    result.mode,
                    result.label,
                    result.top_k,
                    round(result.balance_lambda, 8),
                    result.seed,
                    round(result.post_shift_final_entropy, 8),
                    result.post_shift_support_eps_005,
                    round(result.post_shift_final_val_mse, 8),
                    round(result.post_shift_final_new_regime_mse, 8),
                    result.unique_dominant_experts,
                    round(result.specialization_score, 8),
                    round(result.post_shift_final_max_load, 8),
                ]
            )
    write_csv(
        tables_dir / "architecture_level_moe_check_per_seed.csv",
        [
            "mode",
            "label",
            "top_k",
            "balance_lambda",
            "seed",
            "post_shift_final_entropy",
            "post_shift_support_eps_005",
            "post_shift_final_val_mse",
            "post_shift_final_new_regime_mse",
            "unique_dominant_experts",
            "specialization_score",
            "post_shift_final_max_load",
        ],
        per_seed_rows,
    )

    regime_rows: list[list[object]] = []
    for result in aggregated:
        for regime in range(NUM_REGIMES):
            regime_rows.append(
                [
                    result.mode,
                    result.label,
                    result.top_k,
                    round(result.balance_lambda, 8),
                    regime,
                    *np.round(result.final_regime_load_matrix_mean[regime], 6).tolist(),
                    *np.round(result.final_regime_load_matrix_std[regime], 6).tolist(),
                ]
            )
    write_csv(
        tables_dir / "architecture_level_moe_check_regime_loads.csv",
        [
            "mode",
            "label",
            "top_k",
            "balance_lambda",
            "regime",
            "load_e1_mean",
            "load_e2_mean",
            "load_e3_mean",
            "load_e4_mean",
            "load_e5_mean",
            "load_e6_mean",
            "load_e1_std",
            "load_e2_std",
            "load_e3_std",
            "load_e4_std",
            "load_e5_std",
            "load_e6_std",
        ],
        regime_rows,
    )

    curve_rows: list[list[object]] = []
    for result in aggregated:
        for index, epoch in enumerate(result.curve_epochs):
            curve_rows.append(
                [
                    result.mode,
                    result.label,
                    int(epoch),
                    round(result.specialization_curve_mean[index], 8),
                    round(result.specialization_curve_std[index], 8),
                    round(result.val_mse_curve_mean[index], 8),
                    round(result.val_mse_curve_std[index], 8),
                    round(result.new_regime_mse_curve_mean[index], 8),
                    round(result.new_regime_mse_curve_std[index], 8),
                    round(result.max_load_curve_mean[index], 8),
                    round(result.max_load_curve_std[index], 8),
                ]
            )
    write_csv(
        tables_dir / "architecture_level_moe_check_curves.csv",
        [
            "mode",
            "label",
            "post_shift_epoch",
            "specialization_score_mean",
            "specialization_score_std",
            "val_mse_mean",
            "val_mse_std",
            "new_regime_mse_mean",
            "new_regime_mse_std",
            "max_load_mean",
            "max_load_std",
        ],
        curve_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-shift-epochs", type=int, default=80)
    parser.add_argument("--post-shift-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--balance-lambda", type=float, default=0.10)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    add_torch_device_argument(parser)
    args = parser.parse_args()
    device = resolve_torch_device(args.device)
    print(f"device={device}")

    if args.top_k <= 0 or args.top_k > NUM_EXPERTS:
        raise SystemExit(f"--top-k must be in [1, {NUM_EXPERTS}]")
    if args.eval_interval <= 0:
        raise SystemExit("--eval-interval must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    seeds = parse_seeds(args.seeds)
    specs = build_mode_specs(args.top_k, args.balance_lambda)
    results_by_mode = [
        [
            train_one_setting(
                spec,
                seed,
                pre_shift_epochs=args.pre_shift_epochs,
                post_shift_epochs=args.post_shift_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                eval_interval=args.eval_interval,
                clip_grad=args.clip_grad,
                device=device,
            )
            for seed in seeds
        ]
        for spec in specs
    ]
    aggregated = [aggregate_results(results) for results in results_by_mode]

    for result in aggregated:
        balance_suffix = "" if result.balance_lambda == 0.0 else f" balance={result.balance_lambda:g}"
        topk_suffix = "" if result.top_k == 0 else f" k={result.top_k}"
        print(
            f"[{result.mode}] seeds={result.seed_count}{topk_suffix}{balance_suffix} "
            f"unique_winners={result.unique_dominant_experts_mean:.2f} "
            f"specialization={result.specialization_score_mean:.4f}+/-{result.specialization_score_std:.4f} "
            f"max_load={result.post_shift_final_max_load_mean:.3f}+/-{result.post_shift_final_max_load_std:.3f} "
            f"new_regime_mse={result.post_shift_final_new_regime_mse_mean:.5f}+/-"
            f"{result.post_shift_final_new_regime_mse_std:.5f}"
        )

    write_outputs(results_by_mode, aggregated, seeds)



if __name__ == "__main__":
    main()
