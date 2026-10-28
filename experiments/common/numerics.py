from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


Array = np.ndarray


def add_torch_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add an explicit device selector while preserving CPU reproducibility."""
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=os.environ.get("MOE_DEVICE", "cpu"),
        help=(
            "Execution device. Defaults to MOE_DEVICE when set, otherwise cpu "
            "to preserve the frozen experiment protocol."
        ),
    )


def resolve_torch_device(requested: str):
    """Resolve and validate a PyTorch device without importing torch globally."""
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Run inside a GPU allocation and use a CUDA-enabled PyTorch build."
        )
    return torch.device(requested)


def softmax(q: Array) -> Array:
    shifted = q - np.max(q)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def ensure_output_dir() -> Path:
    out = Path(os.environ.get("MOE_OUTPUT_DIR", "outputs")).expanduser()
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_subdir(name: str) -> Path:
    path = ensure_output_dir() / name
    path.mkdir(exist_ok=True)
    return path


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
