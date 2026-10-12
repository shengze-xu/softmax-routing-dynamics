#!/usr/bin/env python3
"""Audit the large multidomain 64M/2M/2M four-domain corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data.paths import resolve_tokenizer_directory

from experiments.data.prepare_multidomain_small_corpus import DOMAINS
from experiments.data.prepare_multidomain_large_corpus import DATASET_ID, SPLITS


EXPECTED_REQUESTED = {
    "train": 16_000_000,
    "validation": 500_000,
    "shift": 500_000,
}
EXPECTED_SEQUENCE_LENGTH = 512


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, help="Override the tokenizer location recorded in the manifest")
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if manifest.get("schema_version") != 1:
        failures.append("unsupported manifest schema")
    if manifest.get("dataset_id") != DATASET_ID:
        failures.append("dataset ID mismatch")
    if manifest.get("status") != "prepared_multidomain_large_candidate":
        failures.append("unexpected dataset status")
    expected_domains = [domain.name for domain in DOMAINS]
    if manifest.get("domain_names") != expected_domains:
        failures.append("domain inventory or ordering mismatch")
    observed_revisions = {
        source["repository"]: source["revision"]
        for source in manifest.get("sources", [])
    }
    expected_revisions = {
        domain.repository: domain.revision for domain in DOMAINS
    }
    if observed_revisions != expected_revisions:
        failures.append("source revisions do not match the frozen sources")

    selection = manifest.get("selection", {})
    if int(selection.get("sequence_length", -1)) != EXPECTED_SEQUENCE_LENGTH:
        failures.append("sequence length is not 512")
    if selection.get("requested_tokens_per_domain") != EXPECTED_REQUESTED:
        failures.append("requested token budget mismatch")
    for name, summary in manifest.get("source_summaries", {}).items():
        if int(summary.get("rejected_license_rows", -1)) != 0:
            failures.append(f"{name}: rejected or missing license rows")
        if not summary.get("observed_licenses"):
            failures.append(f"{name}: observed license inventory is empty")

    tokenizer = manifest.get("tokenizer", {})
    tokenizer_root = resolve_tokenizer_directory(tokenizer, root, override=args.tokenizer_dir)
    if not tokenizer_root.is_dir():
        failures.append("tokenizer directory is missing")
    else:
        for name, expected in tokenizer.get("files_sha256", {}).items():
            path = tokenizer_root / name
            if not path.is_file() or digest(path) != expected:
                failures.append(f"tokenizer checksum mismatch: {name}")

    arrays: dict[str, np.ndarray] = {}
    for name, metadata in manifest.get("arrays", {}).items():
        path = root / metadata["file"]
        if not path.is_file() or digest(path) != metadata["sha256"]:
            failures.append(f"array checksum mismatch: {name}")
            continue
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != metadata["shape"]:
            failures.append(f"array shape mismatch: {name}")
        if str(value.dtype) != metadata["dtype"]:
            failures.append(f"array dtype mismatch: {name}")
        arrays[name] = value

    split_summary: dict[str, object] = {}
    document_sets: dict[str, set[int]] = {}
    for split in SPLITS:
        required = [
            f"{split}_tokens",
            f"{split}_lengths",
            f"{split}_domains",
            f"{split}_document_hashes",
        ]
        if any(name not in arrays for name in required):
            failures.append(f"missing arrays for {split}")
            continue
        tokens = arrays[f"{split}_tokens"]
        lengths = np.asarray(arrays[f"{split}_lengths"], dtype=np.int64)
        domains = np.asarray(arrays[f"{split}_domains"], dtype=np.int64)
        document_hashes = np.asarray(arrays[f"{split}_document_hashes"])
        if not (
            len(tokens) == len(lengths) == len(domains) == len(document_hashes)
        ):
            failures.append(f"row count mismatch in {split}")
        if tokens.ndim != 2 or tokens.shape[1] != EXPECTED_SEQUENCE_LENGTH + 1:
            failures.append(f"sequence width mismatch in {split}")
        if np.any(lengths < 2) or np.any(lengths > EXPECTED_SEQUENCE_LENGTH + 1):
            failures.append(f"invalid sequence lengths in {split}")
        expected_domain_ids = set(range(len(DOMAINS)))
        if set(np.unique(domains).tolist()) != expected_domain_ids:
            failures.append(f"domain inventory incomplete in {split}")
        token_counts: dict[str, int] = {}
        for domain_index, domain in enumerate(DOMAINS):
            selected = domains == domain_index
            token_count = int(np.sum(lengths[selected] - 1))
            token_counts[domain.name] = token_count
            if token_count < EXPECTED_REQUESTED[split]:
                failures.append(
                    f"{split} {domain.name}: below requested token budget"
                )
        document_sets[split] = set(document_hashes.tolist())
        split_summary[split] = {
            "sequences": len(tokens),
            "target_tokens_by_domain": token_counts,
            "target_tokens_total": sum(token_counts.values()),
            "unique_documents": len(document_sets[split]),
        }

    overlaps: dict[str, int] = {}
    if len(document_sets) == len(SPLITS):
        overlaps = {
            "train_validation": len(
                document_sets["train"] & document_sets["validation"]
            ),
            "train_shift": len(
                document_sets["train"] & document_sets["shift"]
            ),
            "validation_shift": len(
                document_sets["validation"] & document_sets["shift"]
            ),
        }
        if any(overlaps.values()):
            failures.append("document leakage across splits")

    result = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "multidomain_large_controlled_scale_transfer_corpus",
        "dataset_id": DATASET_ID,
        "manifest_sha256": digest(manifest_path),
        "source_revisions": observed_revisions,
        "split_summary": split_summary,
        "document_overlap_counts": overlaps,
        "failures": failures,
        "corpus_integrity_checks_passed": not failures,
    }
    destination = root / "corpus_audit.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite corpus audit: {destination}")
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_sha256": result["manifest_sha256"],
                "corpus_integrity_checks_passed": (
                    result["corpus_integrity_checks_passed"]
                ),
                "failures": failures,
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
