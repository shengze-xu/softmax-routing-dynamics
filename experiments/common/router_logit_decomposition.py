#!/usr/bin/env python3
"""Fixed-probe decomposition of router-logit motion.

For a fixed held-out token with router feature ``h`` and affine router
``q = W h + b``, the change between two evaluation times is decomposed as

    q_1 - q_0 = (W_1 - W_0) h_0 + (b_1 - b_0)
                  + W_1 (h_1 - h_0).

The first two terms describe parameter updates; the last describes feature
changes. Reconstruction uses float64 copies of the captured tensors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class ProbeLayerState:
    token_ids: torch.Tensor
    features: torch.Tensor
    logits: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class ProbeState:
    step: int
    layers: dict[int, ProbeLayerState]


PROBE_FIELDS = (
    "protocol_id",
    "seed",
    "cohort",
    "capacity_factor",
    "branch",
    "initial_target_mass",
    "step_start",
    "step_end",
    "layer_zero_based",
    "probe_token_id",
    "target_expert",
    "comparator_expert",
    "target_probability_before",
    "target_probability_after",
    "target_probability_reconstructed",
    "target_probability_parameter_only_before",
    "target_probability_parameter_only_after",
    "target_probability_feature_only_before",
    "target_probability_feature_only_after",
    "target_logit_before",
    "target_logit_after",
    "target_parameter_displacement",
    "target_feature_displacement",
    "comparator_logit_before",
    "comparator_logit_after",
    "comparator_parameter_displacement",
    "comparator_feature_displacement",
    "gap_displacement",
    "parameter_gap_displacement",
    "feature_gap_displacement",
    "cumulative_gap_displacement",
    "cumulative_parameter_gap_displacement",
    "cumulative_feature_gap_displacement",
    "full_logit_displacement_l2",
    "parameter_displacement_l2",
    "feature_displacement_l2",
    "logit_reconstruction_max_abs_error",
    "probability_reconstruction_abs_error",
    "crossed_threshold_observed",
    "crossed_threshold_reconstructed",
    "crossed_threshold_parameter_only",
    "crossed_threshold_feature_only",
)


class ProbeCSVWriter:
    """Stream probe rows to one compressed sidecar without retaining them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = gzip.open(path, "wt", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=PROBE_FIELDS)
        self._writer.writeheader()
        self.rows = 0

    def writerows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self._writer.writerow(row)
            self.rows += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "ProbeCSVWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _affine(features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return features @ weight.transpose(0, 1) + bias


def _target_probability(logits: torch.Tensor, target: int) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)[:, target]


def capture_probe_state(
    model: torch.nn.Module,
    evaluation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    step: int,
    middle_layers: list[int],
    target_domain_by_layer: list[int],
    tokens_per_layer: int,
    eval_batch_size: int,
    device: torch.device,
    precision: str,
    target_mask: Callable[[torch.Tensor, int], torch.Tensor],
    autocast_context: Callable[[torch.device, str], Any],
) -> ProbeState:
    """Capture the same deterministic held-out token inventory at every time."""
    if tokens_per_layer <= 0:
        raise ValueError("tokens_per_layer must be positive")
    tokens, lengths, domains = evaluation_data
    layer_ids: dict[int, list[torch.Tensor]] = {layer: [] for layer in middle_layers}
    layer_features: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in middle_layers
    }
    counts = {layer: 0 for layer in middle_layers}
    sequence_width = int(tokens.shape[1] - 1)
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), eval_batch_size):
            if all(counts[layer] >= tokens_per_layer for layer in middle_layers):
                break
            batch_tokens = tokens[start : start + eval_batch_size].to(device)
            batch_lengths = lengths[start : start + eval_batch_size].to(device)
            batch_domains = domains[start : start + eval_batch_size].to(device)
            inputs = batch_tokens[:, :-1]
            active = target_mask(batch_lengths, inputs.shape[1])
            with autocast_context(device, precision):
                _, snapshots = model(inputs, active)
            batch_rows = torch.arange(
                start,
                start + len(batch_tokens),
                device=device,
                dtype=torch.long,
            ).unsqueeze(1)
            positions = torch.arange(
                sequence_width, device=device, dtype=torch.long
            ).unsqueeze(0)
            token_ids = batch_rows * sequence_width + positions
            for layer in middle_layers:
                remaining = tokens_per_layer - counts[layer]
                if remaining <= 0:
                    continue
                selected = active & batch_domains.eq(
                    int(target_domain_by_layer[layer])
                ).unsqueeze(1)
                ids = token_ids[selected][:remaining]
                features = snapshots[layer].router_inputs[selected][:remaining]
                if ids.numel() == 0:
                    continue
                layer_ids[layer].append(ids.detach().cpu())
                layer_features[layer].append(
                    features.detach().double().cpu().contiguous()
                )
                counts[layer] += int(ids.numel())
    if was_training:
        model.train()
    layers: dict[int, ProbeLayerState] = {}
    for layer in middle_layers:
        if counts[layer] != tokens_per_layer:
            raise ValueError(
                f"Layer {layer} supplied {counts[layer]} probe tokens; "
                f"expected {tokens_per_layer}"
            )
        router = model.layers[layer].moe.router
        if router is None or router.bias is None:
            raise ValueError(f"Layer {layer} has no affine router")
        ids = torch.cat(layer_ids[layer])
        features = torch.cat(layer_features[layer])
        weight = router.weight.detach().double().cpu().contiguous()
        bias = router.bias.detach().double().cpu().contiguous()
        layers[layer] = ProbeLayerState(
            token_ids=ids,
            features=features,
            logits=_affine(features, weight, bias),
            weight=weight,
            bias=bias,
        )
    return ProbeState(step=step, layers=layers)


class ProbeTrajectory:
    """Accumulate exact interval decompositions for one experimental branch."""

    def __init__(
        self,
        initial: ProbeState,
        *,
        target_by_layer: list[int],
        comparator_by_layer: list[int],
        threshold: float,
        metadata: dict[str, Any],
    ) -> None:
        self.previous = initial
        self.initial_logits = {
            layer: state.logits.clone() for layer, state in initial.layers.items()
        }
        self.cumulative_parameter = {
            layer: torch.zeros_like(state.logits)
            for layer, state in initial.layers.items()
        }
        self.cumulative_feature = {
            layer: torch.zeros_like(state.logits)
            for layer, state in initial.layers.items()
        }
        self.target_by_layer = target_by_layer
        self.comparator_by_layer = comparator_by_layer
        self.threshold = float(threshold)
        self.metadata = dict(metadata)
        self.intervals = 0
        self.rows = 0
        self.observed_crossings = 0
        self.crossing_mismatches = 0
        self.max_logit_error = 0.0
        self.max_probability_error = 0.0

    def advance(self, current: ProbeState) -> list[dict[str, Any]]:
        if current.step <= self.previous.step:
            raise ValueError("Probe steps must be strictly increasing")
        rows: list[dict[str, Any]] = []
        for layer, before in self.previous.layers.items():
            after = current.layers[layer]
            if not torch.equal(before.token_ids, after.token_ids):
                raise ValueError(f"Layer {layer} probe-token inventory changed")
            parameter = _affine(
                before.features,
                after.weight - before.weight,
                after.bias - before.bias,
            )
            feature = (after.features - before.features) @ after.weight.transpose(0, 1)
            displacement = after.logits - before.logits
            reconstructed = before.logits + parameter + feature
            logit_error = torch.max(torch.abs(reconstructed - after.logits), dim=1).values
            probability_before = _target_probability(
                before.logits, int(self.target_by_layer[layer])
            )
            probability_after = _target_probability(
                after.logits, int(self.target_by_layer[layer])
            )
            probability_reconstructed = _target_probability(
                reconstructed, int(self.target_by_layer[layer])
            )
            probability_error = torch.abs(
                probability_reconstructed - probability_after
            )
            parameter_probability_before = _target_probability(
                self.initial_logits[layer] + self.cumulative_parameter[layer],
                int(self.target_by_layer[layer]),
            )
            feature_probability_before = _target_probability(
                self.initial_logits[layer] + self.cumulative_feature[layer],
                int(self.target_by_layer[layer]),
            )
            self.cumulative_parameter[layer] += parameter
            self.cumulative_feature[layer] += feature
            parameter_probability_after = _target_probability(
                self.initial_logits[layer] + self.cumulative_parameter[layer],
                int(self.target_by_layer[layer]),
            )
            feature_probability_after = _target_probability(
                self.initial_logits[layer] + self.cumulative_feature[layer],
                int(self.target_by_layer[layer]),
            )
            cumulative = after.logits - self.initial_logits[layer]
            target = int(self.target_by_layer[layer])
            comparator = int(self.comparator_by_layer[layer])
            observed_cross = (probability_before < self.threshold) & (
                probability_after >= self.threshold
            )
            reconstructed_cross = (probability_before < self.threshold) & (
                probability_reconstructed >= self.threshold
            )
            parameter_cross = (
                parameter_probability_before < self.threshold
            ) & (parameter_probability_after >= self.threshold)
            feature_cross = (feature_probability_before < self.threshold) & (
                feature_probability_after >= self.threshold
            )
            for index, token_id in enumerate(before.token_ids.tolist()):
                row = dict(self.metadata)
                row.update(
                    {
                        "step_start": self.previous.step,
                        "step_end": current.step,
                        "layer_zero_based": layer,
                        "probe_token_id": int(token_id),
                        "target_expert": target,
                        "comparator_expert": comparator,
                        "target_probability_before": float(probability_before[index]),
                        "target_probability_after": float(probability_after[index]),
                        "target_probability_reconstructed": float(
                            probability_reconstructed[index]
                        ),
                        "target_probability_parameter_only_before": float(
                            parameter_probability_before[index]
                        ),
                        "target_probability_parameter_only_after": float(
                            parameter_probability_after[index]
                        ),
                        "target_probability_feature_only_before": float(
                            feature_probability_before[index]
                        ),
                        "target_probability_feature_only_after": float(
                            feature_probability_after[index]
                        ),
                        "target_logit_before": float(before.logits[index, target]),
                        "target_logit_after": float(after.logits[index, target]),
                        "target_parameter_displacement": float(parameter[index, target]),
                        "target_feature_displacement": float(feature[index, target]),
                        "comparator_logit_before": float(before.logits[index, comparator]),
                        "comparator_logit_after": float(after.logits[index, comparator]),
                        "comparator_parameter_displacement": float(
                            parameter[index, comparator]
                        ),
                        "comparator_feature_displacement": float(
                            feature[index, comparator]
                        ),
                        "gap_displacement": float(
                            displacement[index, target] - displacement[index, comparator]
                        ),
                        "parameter_gap_displacement": float(
                            parameter[index, target] - parameter[index, comparator]
                        ),
                        "feature_gap_displacement": float(
                            feature[index, target] - feature[index, comparator]
                        ),
                        "cumulative_gap_displacement": float(
                            cumulative[index, target] - cumulative[index, comparator]
                        ),
                        "cumulative_parameter_gap_displacement": float(
                            self.cumulative_parameter[layer][index, target]
                            - self.cumulative_parameter[layer][index, comparator]
                        ),
                        "cumulative_feature_gap_displacement": float(
                            self.cumulative_feature[layer][index, target]
                            - self.cumulative_feature[layer][index, comparator]
                        ),
                        "full_logit_displacement_l2": float(
                            torch.linalg.vector_norm(displacement[index])
                        ),
                        "parameter_displacement_l2": float(
                            torch.linalg.vector_norm(parameter[index])
                        ),
                        "feature_displacement_l2": float(
                            torch.linalg.vector_norm(feature[index])
                        ),
                        "logit_reconstruction_max_abs_error": float(logit_error[index]),
                        "probability_reconstruction_abs_error": float(
                            probability_error[index]
                        ),
                        "crossed_threshold_observed": int(observed_cross[index]),
                        "crossed_threshold_reconstructed": int(
                            reconstructed_cross[index]
                        ),
                        "crossed_threshold_parameter_only": int(
                            parameter_cross[index]
                        ),
                        "crossed_threshold_feature_only": int(
                            feature_cross[index]
                        ),
                    }
                )
                rows.append(row)
            self.max_logit_error = max(
                self.max_logit_error, float(logit_error.max())
            )
            self.max_probability_error = max(
                self.max_probability_error, float(probability_error.max())
            )
            self.observed_crossings += int(observed_cross.sum())
            self.crossing_mismatches += int(
                (observed_cross != reconstructed_cross).sum()
            )
        self.previous = current
        self.intervals += 1
        self.rows += len(rows)
        return rows

    def summary(self) -> dict[str, Any]:
        return {
            "probe_tokens_per_layer": len(
                next(iter(self.previous.layers.values())).token_ids
            ),
            "evaluation_intervals": self.intervals,
            "rows": self.rows,
            "observed_threshold_crossings": self.observed_crossings,
            "threshold_crossing_mismatches": self.crossing_mismatches,
            "maximum_logit_reconstruction_error": self.max_logit_error,
            "maximum_probability_reconstruction_error": self.max_probability_error,
        }
