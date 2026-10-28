#!/usr/bin/env python3
"""Small-model autoregressive MoE comparison on a four-domain corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.models import agnews_autoregressive_moe as base
from experiments.models import agnews_gpt_moe as gpt_model


class SmallMultidomainPreparedDataset(base.PreparedDataset):
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported small multidomain prepared-data schema")
        if self.manifest.get("status") != "prepared_multidomain_small_candidate":
            raise ValueError("Dataset is not a prepared small multidomain corpus")
        self.arrays: dict[str, np.ndarray] = {}
        for name, metadata in self.manifest["arrays"].items():
            path = self.root / metadata["file"]
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
                raise ValueError(f"Prepared-data shape/dtype mismatch: {name}")
            self.arrays[name] = array
        self.vocab_size = int(self.manifest["tokenizer"]["vocab_size"])
        self.sequence_length = int(self.manifest["selection"]["sequence_length"])
        base.DOMAIN_NAMES = list(self.manifest["domain_names"])
        if self.arrays["train_tokens"].shape[1] != self.sequence_length + 1:
            raise ValueError("Prepared token width does not match manifest")


def run_multidomain_small(args: argparse.Namespace) -> dict[str, object]:
    original_dataset = base.PreparedDataset
    base.PreparedDataset = SmallMultidomainPreparedDataset
    try:
        result = gpt_model.run_gpt(args)
    finally:
        base.PreparedDataset = original_dataset
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    result["multidomain_small"] = {
        "protocol_id": args.protocol_id,
        "role": args.role,
        "primary_multidomain_small_confirmatory": args.role == "confirmatory",
        "domain_names": manifest["domain_names"],
        "source_revisions": {
            source["repository"]: source["revision"]
            for source in manifest["sources"]
        },
    }
    result["interpretation"] = (
        "small multidomain multi-source autoregressive routing "
        f"{args.role}"
    )
    destination = args.output_dir.resolve() / "result.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "multidomain_small_role": args.role,
        "protocol_id": args.protocol_id,
        "domain_names": manifest["domain_names"],
        "output": str(destination),
    }, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("benchmark", "exploratory", "confirmatory"), required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-validation", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--expert-hidden", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--router-init-std", type=float, default=0.01)
    parser.add_argument("--balance-lambda", type=float, default=0.1)
    parser.add_argument("--router-z-loss", type=float, default=1e-3)
    parser.add_argument("--importance-gap-weight", type=float, default=1.0)
    parser.add_argument("--initial-ce-tolerance", type=float, default=0.75)
    args = parser.parse_args()
    if args.d_model % args.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if not 1 <= args.top_k <= args.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    run_multidomain_small(args)


if __name__ == "__main__":
    main()
