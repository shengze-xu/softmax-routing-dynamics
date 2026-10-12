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
from experiments.models.synthetic_regime_shift import MediumExpert, MediumMoE, NUM_EXPERTS, NUM_REGIMES, build_phase_datasets, move_bundle_to_device, parse_seeds, regime_load_matrix, router_entropy, support_size_from_load


torch.set_num_threads(1)


@dataclass(frozen=True)
class ModeSpec:
    name: str
    label: str
    tau: float
    conditioned: bool


@dataclass
class TrainResult:
    mode: str
    label: str
    tau: float
    seed: int
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
    tau: float
    seed_count: int
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


class GlobalRouterMoE(nn.Module):
    def __init__(self, num_experts: int = NUM_EXPERTS, temperature: float = 4.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.global_logits = nn.Parameter(torch.zeros(num_experts))
        self.experts = nn.ModuleList([MediumExpert() for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.global_logits.unsqueeze(0).expand(x.shape[0], -1) * self.temperature
        probs = F.softmax(logits, dim=-1)
        expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=-1)
        prediction = (probs * expert_outputs).sum(dim=-1)
        return prediction, probs


def build_mode_specs() -> list[ModeSpec]:
    return [
        ModeSpec(
            name="shared_global_tau0",
            label="shared-global",
            tau=0.0,
            conditioned=False,
        ),
        ModeSpec(
            name="input_conditioned_tau0",
            label="input-conditioned",
            tau=0.0,
            conditioned=True,
        ),
        ModeSpec(
            name="input_conditioned_tau0p05",
            label="input-conditioned + entropy",
            tau=0.05,
            conditioned=True,
        ),
    ]


def specialization_score_from_matrix(matrix: np.ndarray) -> float:
    global_load = matrix.mean(axis=0)
    return float(np.mean(np.abs(matrix - global_load[None, :]).sum(axis=1)))


def train_one_setting(
    spec: ModeSpec,
    seed: int,
    *,
    pre_shift_epochs: int,
    post_shift_epochs: int,
    lr: float,
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
    new_regime_mask = reference_eval.regimes == 3
    model: nn.Module
    if spec.conditioned:
        model = MediumMoE()
    else:
        model = GlobalRouterMoE()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(pre_shift_epochs):
        pred, probs = model(pre_shift_train.x)
        mse = ((pred - pre_shift_train.y) ** 2).mean()
        loss = mse - spec.tau * router_entropy(probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(post_shift_epochs):
        pred, probs = model(post_shift_train.x)
        mse = ((pred - post_shift_train.y) ** 2).mean()
        loss = mse - spec.tau * router_entropy(probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        ref_pred, ref_probs = model(reference_eval.x)
        post_shift_final_val_mse = float(((ref_pred - reference_eval.y) ** 2).mean().item())
        post_shift_final_entropy = float(router_entropy(ref_probs).item())
        post_shift_load = ref_probs.mean(dim=0).detach().cpu().numpy()
        post_shift_final_new_regime_mse = float(
            ((ref_pred[new_regime_mask] - reference_eval.y[new_regime_mask]) ** 2).mean().item()
        )
        final_regime_matrix = regime_load_matrix(ref_probs, reference_eval.regimes)

    return TrainResult(
        mode=spec.name,
        label=spec.label,
        tau=spec.tau,
        seed=seed,
        post_shift_final_entropy=post_shift_final_entropy,
        post_shift_support_eps_005=support_size_from_load(post_shift_load),
        post_shift_final_val_mse=post_shift_final_val_mse,
        post_shift_final_new_regime_mse=post_shift_final_new_regime_mse,
        unique_dominant_experts=len({int(np.argmax(row)) for row in final_regime_matrix}),
        specialization_score=specialization_score_from_matrix(final_regime_matrix),
        post_shift_final_max_load=float(post_shift_load.max()),
        final_regime_load_matrix=final_regime_matrix,
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

    first = results[0]
    return AggregatedResult(
        mode=first.mode,
        label=first.label,
        tau=first.tau,
        seed_count=len(results),
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




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-shift-epochs", type=int, default=120)
    parser.add_argument("--post-shift-epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    add_torch_device_argument(parser)
    args = parser.parse_args()
    device = resolve_torch_device(args.device)
    print(f"device={device}")

    out_dir = ensure_output_dir()
    tables_dir = ensure_subdir("tables")
    seeds = parse_seeds(args.seeds)
    specs = build_mode_specs()

    results_by_mode = [
        [
            train_one_setting(
                spec,
                seed,
                pre_shift_epochs=args.pre_shift_epochs,
                post_shift_epochs=args.post_shift_epochs,
                lr=args.lr,
                device=device,
            )
            for seed in seeds
        ]
        for spec in specs
    ]
    aggregated = [aggregate_results(results) for results in results_by_mode]

    for result in aggregated:
        print(
            f"[{result.mode}] seeds={result.seed_count} "
            f"unique_winners={result.unique_dominant_experts_mean:.2f} "
            f"specialization={result.specialization_score_mean:.4f}+/-{result.specialization_score_std:.4f} "
            f"val_mse={result.post_shift_final_val_mse_mean:.5f}+/-{result.post_shift_final_val_mse_std:.5f} "
            f"new_regime_mse={result.post_shift_final_new_regime_mse_mean:.5f}+/-{result.post_shift_final_new_regime_mse_std:.5f}"
        )

    np.savez(
        out_dir / "input_dependent_break_freeze.npz",
        mode_names=np.array([result.mode for result in aggregated], dtype=object),
        taus=np.array([result.tau for result in aggregated], dtype=float),
        seeds=np.array(seeds, dtype=int),
        specialization_score_mean=np.array([result.specialization_score_mean for result in aggregated], dtype=float),
        specialization_score_std=np.array([result.specialization_score_std for result in aggregated], dtype=float),
        unique_dominant_experts_mean=np.array(
            [result.unique_dominant_experts_mean for result in aggregated],
            dtype=float,
        ),
        final_regime_load_matrix_mean=np.stack(
            [result.final_regime_load_matrix_mean for result in aggregated],
            axis=0,
        ),
        final_regime_load_matrix_std=np.stack(
            [result.final_regime_load_matrix_std for result in aggregated],
            axis=0,
        ),
    )

    summary_rows = []
    for result in aggregated:
        summary_rows.append(
            [
                result.mode,
                result.label,
                round(result.tau, 8),
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
        tables_dir / "input_dependent_break_freeze_summary.csv",
        [
            "mode",
            "label",
            "tau",
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

    per_seed_rows = []
    for results in results_by_mode:
        for result in results:
            per_seed_rows.append(
                [
                    result.mode,
                    result.label,
                    round(result.tau, 8),
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
        tables_dir / "input_dependent_break_freeze_per_seed.csv",
        [
            "mode",
            "label",
            "tau",
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

    regime_rows = []
    for result in aggregated:
        for regime in range(NUM_REGIMES):
            regime_rows.append(
                [
                    result.mode,
                    result.label,
                    round(result.tau, 8),
                    regime,
                    *np.round(result.final_regime_load_matrix_mean[regime], 6).tolist(),
                    *np.round(result.final_regime_load_matrix_std[regime], 6).tolist(),
                ]
            )
    write_csv(
        tables_dir / "input_dependent_break_freeze_regime_loads.csv",
        [
            "mode",
            "label",
            "tau",
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



if __name__ == "__main__":
    main()
