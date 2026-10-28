from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_num_threads(1)

NUM_EXPERTS = 6
NUM_REGIMES = 4
RECOVERY_THRESHOLD = 0.15


@dataclass
class DatasetBundle:
    x: torch.Tensor
    y: torch.Tensor
    regimes: torch.Tensor


def move_bundle_to_device(
    bundle: DatasetBundle,
    device: torch.device,
) -> DatasetBundle:
    return DatasetBundle(
        x=bundle.x.to(device),
        y=bundle.y.to(device),
        regimes=bundle.regimes.to(device),
    )


@dataclass
class TrainResult:
    seed: int
    tau: float
    pre_shift_entropy_curve: np.ndarray
    post_shift_entropy_curve: np.ndarray
    post_shift_starved_new_regime_load_curve: np.ndarray
    post_shift_router_diagnostic_curve: np.ndarray
    post_shift_val_mse_curve: np.ndarray
    post_shift_new_regime_mse_curve: np.ndarray
    pre_shift_final_entropy: float
    pre_shift_support_eps_005: int
    pre_shift_starved_expert: int
    pre_shift_starved_load: float
    post_shift_final_entropy: float
    post_shift_support_eps_005: int
    post_shift_peak_starved_new_regime_load: float
    post_shift_recovery_epoch_0p15: int
    post_shift_final_val_mse: float
    post_shift_final_new_regime_mse: float
    post_shift_final_max_load: float
    final_regime_load_matrix: np.ndarray


@dataclass
class AggregatedResult:
    tau: float
    seed_count: int
    pre_shift_entropy_curve_mean: np.ndarray
    pre_shift_entropy_curve_std: np.ndarray
    post_shift_entropy_curve_mean: np.ndarray
    post_shift_entropy_curve_std: np.ndarray
    post_shift_starved_new_regime_load_curve_mean: np.ndarray
    post_shift_starved_new_regime_load_curve_std: np.ndarray
    post_shift_val_mse_curve_mean: np.ndarray
    post_shift_val_mse_curve_std: np.ndarray
    post_shift_new_regime_mse_curve_mean: np.ndarray
    post_shift_new_regime_mse_curve_std: np.ndarray
    pre_shift_final_entropy_mean: float
    pre_shift_final_entropy_std: float
    pre_shift_support_mean: float
    pre_shift_support_std: float
    pre_shift_support_min: int
    pre_shift_support_max: int
    pre_shift_starved_load_mean: float
    pre_shift_starved_load_std: float
    post_shift_final_entropy_mean: float
    post_shift_final_entropy_std: float
    post_shift_support_mean: float
    post_shift_support_std: float
    post_shift_support_min: int
    post_shift_support_max: int
    post_shift_peak_starved_new_regime_load_mean: float
    post_shift_peak_starved_new_regime_load_std: float
    post_shift_recovered_count_0p15: int
    post_shift_recovery_epoch_0p15_mean: float
    post_shift_final_val_mse_mean: float
    post_shift_final_val_mse_std: float
    post_shift_final_new_regime_mse_mean: float
    post_shift_final_new_regime_mse_std: float
    post_shift_final_max_load_mean: float
    post_shift_final_max_load_std: float
    final_regime_load_matrix_mean: np.ndarray
    final_regime_load_matrix_std: np.ndarray


def parse_seeds(values: Iterable[int]) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seeds.append(value)
        seen.add(value)
    return seeds


def regime_target(x: torch.Tensor, regime: int) -> torch.Tensor:
    x0 = x[:, 0]
    x1 = x[:, 1]
    if regime == 0:
        return torch.sin(2.4 * x0) + 0.35 * x1
    if regime == 1:
        return torch.cos(2.0 * x1) - 0.55 * x0
    if regime == 2:
        return 0.65 * x0 * x1 + 0.25 * x0 - 0.15 * x1
    if regime == 3:
        return torch.tanh(1.7 * x0 - 1.4 * x1) + 0.45 * x1 + 0.2
    raise ValueError(f"Unsupported regime: {regime}")


def build_dataset(
    counts: list[int],
    *,
    seed: int,
    noise_std: float,
) -> DatasetBundle:
    if len(counts) != NUM_REGIMES:
        raise ValueError(f"Expected {NUM_REGIMES} regime counts, got {len(counts)}.")

    centers = torch.tensor(
        [
            [-1.35, -1.05],
            [1.30, -0.80],
            [-1.10, 1.15],
            [1.20, 1.05],
        ],
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [
            [0.35, 0.28],
            [0.30, 0.35],
            [0.32, 0.30],
            [0.28, 0.32],
        ],
        dtype=torch.float32,
    )

    generator = torch.Generator().manual_seed(seed)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    regimes: list[torch.Tensor] = []
    for regime, count in enumerate(counts):
        if count <= 0:
            continue
        x = centers[regime] + scales[regime] * torch.randn(count, 2, generator=generator)
        y = regime_target(x, regime)
        if noise_std > 0.0:
            y = y + noise_std * torch.randn(count, generator=generator)
        xs.append(x)
        ys.append(y)
        regimes.append(torch.full((count,), regime, dtype=torch.long))

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    regime_tensor = torch.cat(regimes, dim=0)

    perm = torch.randperm(x.shape[0], generator=generator)
    return DatasetBundle(x=x[perm], y=y[perm], regimes=regime_tensor[perm])


def build_phase_datasets(seed: int) -> tuple[DatasetBundle, DatasetBundle, DatasetBundle]:
    pre_shift_train = build_dataset([384, 384, 384, 0], seed=1000 + seed, noise_std=0.03)
    post_shift_train = build_dataset([192, 192, 192, 768], seed=2000 + seed, noise_std=0.03)
    reference_eval = build_dataset([192, 192, 192, 192], seed=3000 + seed, noise_std=0.0)
    return pre_shift_train, post_shift_train, reference_eval


class MediumExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MediumMoE(nn.Module):
    def __init__(self, num_experts: int = NUM_EXPERTS, temperature: float = 4.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.router = nn.Sequential(
            nn.Linear(2, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, num_experts),
        )
        self.experts = nn.ModuleList([MediumExpert() for _ in range(num_experts)])

    def routing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x) * self.temperature
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, probs = self.routing(x)
        expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=-1)
        prediction = (probs * expert_outputs).sum(dim=-1)
        return prediction, probs


def router_entropy(probs: torch.Tensor) -> torch.Tensor:
    safe = probs.clamp_min(1e-9)
    return -(safe * safe.log()).sum(dim=-1).mean()


def support_size_from_load(load: np.ndarray, eps: float = 5e-2) -> int:
    return int(np.sum(load > eps))


def regime_load_matrix(
    probs: torch.Tensor,
    regimes: torch.Tensor,
) -> np.ndarray:
    regimes = regimes.to(probs.device)
    rows = []
    for regime in range(NUM_REGIMES):
        mask = regimes == regime
        rows.append(probs[mask].mean(dim=0).detach().cpu().numpy())
    return np.stack(rows, axis=0)


def evaluate_model(
    model: MediumMoE,
    bundle: DatasetBundle,
) -> tuple[float, float, np.ndarray, torch.Tensor]:
    with torch.no_grad():
        pred, probs = model(bundle.x)
        mse = float(((pred - bundle.y) ** 2).mean().item())
        entropy = float(router_entropy(probs).item())
        load = probs.mean(dim=0).detach().cpu().numpy()
    return mse, entropy, load, probs


def router_diagnostic(
    model: MediumMoE,
    x: torch.Tensor,
    expert: int,
) -> np.ndarray:
    with torch.no_grad():
        logits, probs = model.routing(x)
        selected_probability = probs[:, expert]
        other_mask = torch.arange(probs.shape[1], device=probs.device) != expert
        competitor_logsumexp = torch.logsumexp(logits[:, other_mask], dim=-1)
        token_log_odds = logits[:, expert] - competitor_logsumexp
        mean_probability = selected_probability.mean()
        load_log_odds = torch.logit(mean_probability.clamp(1e-9, 1.0 - 1e-9))
        return np.array(
            [
                float(mean_probability.item()),
                float(load_log_odds.item()),
                float(token_log_odds.mean().item()),
                float(logits[:, expert].mean().item()),
                float(competitor_logsumexp.mean().item()),
            ],
            dtype=float,
        )


def train_one_setting(
    tau: float,
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

    model = MediumMoE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pre_shift_entropy_curve: list[float] = []
    post_shift_entropy_curve: list[float] = []
    post_shift_starved_new_regime_load_curve: list[float] = []
    post_shift_val_mse_curve: list[float] = []
    post_shift_new_regime_mse_curve: list[float] = []
    post_shift_router_diagnostic_curve: list[np.ndarray] = []

    for _ in range(pre_shift_epochs):
        pred, probs = model(pre_shift_train.x)
        mse = ((pred - pre_shift_train.y) ** 2).mean()
        entropy = router_entropy(probs)
        loss = mse - tau * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, eval_entropy, _, _ = evaluate_model(model, reference_eval)
        pre_shift_entropy_curve.append(eval_entropy)

    with torch.no_grad():
        _, pre_shift_probs = model(pre_shift_train.x)
        pre_shift_load = pre_shift_probs.mean(dim=0).detach().cpu().numpy()
        pre_shift_starved_expert = int(np.argmin(pre_shift_load))
        pre_shift_final_entropy = float(router_entropy(pre_shift_probs).item())
        post_shift_router_diagnostic_curve.append(
            router_diagnostic(
                model,
                reference_eval.x[new_regime_mask],
                pre_shift_starved_expert,
            )
        )

    post_shift_recovery_epoch = -1
    post_shift_peak_starved_new_regime_load = 0.0
    for epoch in range(post_shift_epochs):
        pred, probs = model(post_shift_train.x)
        mse = ((pred - post_shift_train.y) ** 2).mean()
        entropy = router_entropy(probs)
        loss = mse - tau * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ref_pred, ref_probs = model(reference_eval.x)
            ref_mse = float(((ref_pred - reference_eval.y) ** 2).mean().item())
            ref_entropy = float(router_entropy(ref_probs).item())
            new_regime_load = float(ref_probs[new_regime_mask, pre_shift_starved_expert].mean().item())
            new_regime_mse = float(
                ((ref_pred[new_regime_mask] - reference_eval.y[new_regime_mask]) ** 2).mean().item()
            )

        post_shift_entropy_curve.append(ref_entropy)
        post_shift_starved_new_regime_load_curve.append(new_regime_load)
        post_shift_val_mse_curve.append(ref_mse)
        post_shift_new_regime_mse_curve.append(new_regime_mse)
        post_shift_peak_starved_new_regime_load = max(post_shift_peak_starved_new_regime_load, new_regime_load)
        post_shift_router_diagnostic_curve.append(
            router_diagnostic(
                model,
                reference_eval.x[new_regime_mask],
                pre_shift_starved_expert,
            )
        )
        if post_shift_recovery_epoch < 0 and new_regime_load >= RECOVERY_THRESHOLD:
            post_shift_recovery_epoch = epoch + 1

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
        seed=seed,
        tau=tau,
        pre_shift_entropy_curve=np.array(pre_shift_entropy_curve, dtype=float),
        post_shift_entropy_curve=np.array(post_shift_entropy_curve, dtype=float),
        post_shift_starved_new_regime_load_curve=np.array(post_shift_starved_new_regime_load_curve, dtype=float),
        post_shift_router_diagnostic_curve=np.stack(post_shift_router_diagnostic_curve, axis=0),
        post_shift_val_mse_curve=np.array(post_shift_val_mse_curve, dtype=float),
        post_shift_new_regime_mse_curve=np.array(post_shift_new_regime_mse_curve, dtype=float),
        pre_shift_final_entropy=pre_shift_final_entropy,
        pre_shift_support_eps_005=support_size_from_load(pre_shift_load),
        pre_shift_starved_expert=pre_shift_starved_expert,
        pre_shift_starved_load=float(pre_shift_load[pre_shift_starved_expert]),
        post_shift_final_entropy=post_shift_final_entropy,
        post_shift_support_eps_005=support_size_from_load(post_shift_load),
        post_shift_peak_starved_new_regime_load=post_shift_peak_starved_new_regime_load,
        post_shift_recovery_epoch_0p15=post_shift_recovery_epoch,
        post_shift_final_val_mse=post_shift_final_val_mse,
        post_shift_final_new_regime_mse=post_shift_final_new_regime_mse,
        post_shift_final_max_load=float(post_shift_load.max()),
        final_regime_load_matrix=final_regime_matrix,
    )


def aggregate_results(tau_results: list[TrainResult]) -> AggregatedResult:
    pre_shift_entropy_curves = np.stack([result.pre_shift_entropy_curve for result in tau_results], axis=0)
    post_shift_entropy_curves = np.stack([result.post_shift_entropy_curve for result in tau_results], axis=0)
    post_shift_starved_curves = np.stack(
        [result.post_shift_starved_new_regime_load_curve for result in tau_results],
        axis=0,
    )
    post_shift_val_mse_curves = np.stack([result.post_shift_val_mse_curve for result in tau_results], axis=0)
    post_shift_new_regime_mse_curves = np.stack(
        [result.post_shift_new_regime_mse_curve for result in tau_results],
        axis=0,
    )
    regime_load_matrices = np.stack([result.final_regime_load_matrix for result in tau_results], axis=0)

    pre_shift_support = np.array([result.pre_shift_support_eps_005 for result in tau_results], dtype=float)
    post_shift_support = np.array([result.post_shift_support_eps_005 for result in tau_results], dtype=float)
    recovered_epochs = np.array(
        [result.post_shift_recovery_epoch_0p15 for result in tau_results if result.post_shift_recovery_epoch_0p15 >= 0],
        dtype=float,
    )

    return AggregatedResult(
        tau=tau_results[0].tau,
        seed_count=len(tau_results),
        pre_shift_entropy_curve_mean=pre_shift_entropy_curves.mean(axis=0),
        pre_shift_entropy_curve_std=pre_shift_entropy_curves.std(axis=0),
        post_shift_entropy_curve_mean=post_shift_entropy_curves.mean(axis=0),
        post_shift_entropy_curve_std=post_shift_entropy_curves.std(axis=0),
        post_shift_starved_new_regime_load_curve_mean=post_shift_starved_curves.mean(axis=0),
        post_shift_starved_new_regime_load_curve_std=post_shift_starved_curves.std(axis=0),
        post_shift_val_mse_curve_mean=post_shift_val_mse_curves.mean(axis=0),
        post_shift_val_mse_curve_std=post_shift_val_mse_curves.std(axis=0),
        post_shift_new_regime_mse_curve_mean=post_shift_new_regime_mse_curves.mean(axis=0),
        post_shift_new_regime_mse_curve_std=post_shift_new_regime_mse_curves.std(axis=0),
        pre_shift_final_entropy_mean=float(np.mean([result.pre_shift_final_entropy for result in tau_results])),
        pre_shift_final_entropy_std=float(np.std([result.pre_shift_final_entropy for result in tau_results])),
        pre_shift_support_mean=float(pre_shift_support.mean()),
        pre_shift_support_std=float(pre_shift_support.std()),
        pre_shift_support_min=int(pre_shift_support.min()),
        pre_shift_support_max=int(pre_shift_support.max()),
        pre_shift_starved_load_mean=float(np.mean([result.pre_shift_starved_load for result in tau_results])),
        pre_shift_starved_load_std=float(np.std([result.pre_shift_starved_load for result in tau_results])),
        post_shift_final_entropy_mean=float(np.mean([result.post_shift_final_entropy for result in tau_results])),
        post_shift_final_entropy_std=float(np.std([result.post_shift_final_entropy for result in tau_results])),
        post_shift_support_mean=float(post_shift_support.mean()),
        post_shift_support_std=float(post_shift_support.std()),
        post_shift_support_min=int(post_shift_support.min()),
        post_shift_support_max=int(post_shift_support.max()),
        post_shift_peak_starved_new_regime_load_mean=float(
            np.mean([result.post_shift_peak_starved_new_regime_load for result in tau_results])
        ),
        post_shift_peak_starved_new_regime_load_std=float(
            np.std([result.post_shift_peak_starved_new_regime_load for result in tau_results])
        ),
        post_shift_recovered_count_0p15=int(sum(result.post_shift_recovery_epoch_0p15 >= 0 for result in tau_results)),
        post_shift_recovery_epoch_0p15_mean=float(recovered_epochs.mean()) if len(recovered_epochs) else float("nan"),
        post_shift_final_val_mse_mean=float(np.mean([result.post_shift_final_val_mse for result in tau_results])),
        post_shift_final_val_mse_std=float(np.std([result.post_shift_final_val_mse for result in tau_results])),
        post_shift_final_new_regime_mse_mean=float(
            np.mean([result.post_shift_final_new_regime_mse for result in tau_results])
        ),
        post_shift_final_new_regime_mse_std=float(
            np.std([result.post_shift_final_new_regime_mse for result in tau_results])
        ),
        post_shift_final_max_load_mean=float(np.mean([result.post_shift_final_max_load for result in tau_results])),
        post_shift_final_max_load_std=float(np.std([result.post_shift_final_max_load for result in tau_results])),
        final_regime_load_matrix_mean=regime_load_matrices.mean(axis=0),
        final_regime_load_matrix_std=regime_load_matrices.std(axis=0),
    )
