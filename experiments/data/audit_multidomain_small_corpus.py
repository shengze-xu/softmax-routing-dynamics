#!/usr/bin/env python3
"""Audit hashes, provenance, balance, and document isolation of small multidomain data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_REVISIONS = {
    "common-pile/arxiv_abstracts": "828e35d1000f94579da8850f5f640c138279bdb5",
    "common-pile/project_gutenberg": "01dc90a5002f8977c7fb03a372c14bca29c65cf1",
    "common-pile/usgpo": "0b314ed37117654d04c563b7d21b6ae3445a45df",
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}
SPLITS = ("train", "validation", "shift")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    if manifest.get("status") != "prepared_multidomain_small_candidate":
        failures.append("unexpected dataset status")
    observed_revisions = {
        source["repository"]: source["revision"]
        for source in manifest["sources"]
    }
    if observed_revisions != EXPECTED_REVISIONS:
        failures.append("source revisions do not match the frozen candidate list")

    arrays = {}
    for name, metadata in manifest["arrays"].items():
        path = root / metadata["file"]
        if not path.is_file() or digest(path) != metadata["sha256"]:
            failures.append(f"array checksum mismatch: {name}")
            continue
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
            failures.append(f"array shape/dtype mismatch: {name}")
        arrays[name] = value

    domain_count = len(manifest["domain_names"])
    split_summary = {}
    document_sets = {}
    requested = manifest["selection"]["requested_tokens_per_domain"]
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
            failures.append(f"row-count mismatch in {split}")
        if tokens.shape[1] != int(manifest["selection"]["sequence_length"]) + 1:
            failures.append(f"sequence width mismatch in {split}")
        if np.any(lengths < 2) or np.any(lengths > tokens.shape[1]):
            failures.append(f"invalid sequence lengths in {split}")
        counts = {}
        for domain in range(domain_count):
            selected = domains == domain
            token_count = int(np.sum(lengths[selected] - 1))
            counts[manifest["domain_names"][domain]] = token_count
            if token_count < int(requested[split]):
                failures.append(
                    f"{split} domain {domain} below requested token budget"
                )
        if set(np.unique(domains).tolist()) != set(range(domain_count)):
            failures.append(f"domain inventory incomplete in {split}")
        document_sets[split] = set(document_hashes.tolist())
        split_summary[split] = {
            "sequences": len(tokens),
            "target_tokens_by_domain": counts,
            "unique_documents": len(document_sets[split]),
        }

    overlap = {}
    if len(document_sets) == len(SPLITS):
        overlap = {
            "train_validation": len(
                document_sets["train"] & document_sets["validation"]
            ),
            "train_shift": len(document_sets["train"] & document_sets["shift"]),
            "validation_shift": len(
                document_sets["validation"] & document_sets["shift"]
            ),
        }
        if any(overlap.values()):
            failures.append("document leakage across splits")

    result = {
        "status": "passed" if not failures else "failed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "multidomain_small_multidomain_corpus_candidate",
        "manifest_sha256": digest(manifest_path),
        "domain_names": manifest["domain_names"],
        "source_revisions": observed_revisions,
        "split_summary": split_summary,
        "document_overlap_counts": overlap,
        "failures": failures,
    }
    destination = root / "corpus_audit.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
