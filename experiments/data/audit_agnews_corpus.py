#!/usr/bin/env python3
"""Independently audit the full AG News AG News corpus candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data.paths import resolve_tokenizer_directory



DATASET_ID = "agnews-full-balanced-v1"
DOMAIN_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
SOURCE_DIGESTS = {
    "train.csv": {
        "md5": "b1a00f826fdfbd249f79597b59e1dc12",
        "sha256": (
            "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e"
        ),
    },
    "test.csv": {
        "md5": "d52ea96a97a2d943681189a97654912d",
        "sha256": (
            "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699"
        ),
    },
}
EXPECTED_PER_DOMAIN = {"train": 29_950, "validation": 1_899}
EXPECTED_ROWS = {"train": 119_800, "validation": 7_596}
SEQUENCE_LENGTH = 64
OUTPUT_SHUFFLE_SEED = 20260729


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def raw_records(path: Path) -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                raise ValueError(f"Malformed source row in {path}")
            domain = int(row[0]) - 1
            title = " ".join(row[1].split())
            description = " ".join(row[2].split())
            text = f"{title} {description}".strip()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            result.append((domain, text, digest))
    return result


def select_expected(
    train_path: Path,
    validation_path: Path,
) -> tuple[
    dict[str, list[tuple[int, str, str]]],
    dict[str, dict[str, int]],
]:
    train_raw = raw_records(train_path)
    validation_raw = raw_records(validation_path)
    train_seen: set[str] = set()
    train_unique: list[tuple[int, str, str]] = []
    train_duplicates: Counter[int] = Counter()
    for record in train_raw:
        if record[2] in train_seen:
            train_duplicates[record[0]] += 1
            continue
        train_seen.add(record[2])
        train_unique.append(record)
    validation_seen: set[str] = set()
    validation_unique: list[tuple[int, str, str]] = []
    validation_duplicates: Counter[int] = Counter()
    validation_overlap: Counter[int] = Counter()
    for record in validation_raw:
        if record[2] in validation_seen:
            validation_duplicates[record[0]] += 1
            continue
        validation_seen.add(record[2])
        if record[2] in train_seen:
            validation_overlap[record[0]] += 1
            continue
        validation_unique.append(record)

    selected: dict[str, list[tuple[int, str, str]]] = {}
    for split, records, per_domain, seed in (
        (
            "train",
            train_unique,
            EXPECTED_PER_DOMAIN["train"],
            OUTPUT_SHUFFLE_SEED,
        ),
        (
            "validation",
            validation_unique,
            EXPECTED_PER_DOMAIN["validation"],
            OUTPUT_SHUFFLE_SEED + 1,
        ),
    ):
        combined: list[tuple[int, str, str]] = []
        for domain in range(len(DOMAIN_NAMES)):
            group = sorted(
                (record for record in records if record[0] == domain),
                key=lambda record: record[2],
            )
            combined.extend(group[:per_domain])
        order = np.random.default_rng(seed).permutation(len(combined))
        selected[split] = [combined[int(index)] for index in order]
    summary = {
        "train_duplicates": {
            name: int(train_duplicates[index])
            for index, name in enumerate(DOMAIN_NAMES)
        },
        "validation_duplicates": {
            name: int(validation_duplicates[index])
            for index, name in enumerate(DOMAIN_NAMES)
        },
        "validation_overlap": {
            name: int(validation_overlap[index])
            for index, name in enumerate(DOMAIN_NAMES)
        },
    }
    return selected, summary


def digest_matrix(
    records: list[tuple[int, str, str]],
) -> np.ndarray:
    value = np.empty((len(records), 4), dtype=np.uint64)
    for row, record in enumerate(records):
        payload = bytes.fromhex(record[2])
        value[row] = [
            int.from_bytes(payload[start : start + 8], "big")
            for start in range(0, 32, 8)
        ]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, help="Override the tokenizer location recorded in the manifest")
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    source_dir = args.source_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if manifest.get("schema_version") != 1:
        failures.append("manifest schema mismatch")
    if manifest.get("dataset_id") != DATASET_ID:
        failures.append("dataset ID mismatch")
    if manifest.get("status") != "prepared_agnews_candidate":
        failures.append("dataset status mismatch")
    if manifest.get("domain_names") != DOMAIN_NAMES:
        failures.append("domain inventory or order mismatch")
    for name, digests in SOURCE_DIGESTS.items():
        path = source_dir / name
        for algorithm, expected in digests.items():
            if not path.is_file() or file_digest(path, algorithm) != expected:
                failures.append(f"{name}: {algorithm} checksum mismatch")

    expected, raw_summary = select_expected(
        source_dir / "train.csv",
        source_dir / "test.csv",
    )
    selection = manifest.get("selection", {})
    if selection.get("sequence_length") != SEQUENCE_LENGTH:
        failures.append("sequence length mismatch")
    if selection.get("train_per_domain") != EXPECTED_PER_DOMAIN["train"]:
        failures.append("train per-domain count mismatch")
    if (
        selection.get("validation_per_domain")
        != EXPECTED_PER_DOMAIN["validation"]
    ):
        failures.append("validation per-domain count mismatch")
    if (
        selection.get("within_split_duplicates_removed", {}).get("train")
        != raw_summary["train_duplicates"]
    ):
        failures.append("train duplicate summary mismatch")
    if (
        selection.get("within_split_duplicates_removed", {}).get("validation")
        != raw_summary["validation_duplicates"]
    ):
        failures.append("validation duplicate summary mismatch")
    if (
        selection.get("cross_split_rows_removed_from_validation")
        != raw_summary["validation_overlap"]
    ):
        failures.append("cross-split exclusion summary mismatch")

    tokenizer = manifest.get("tokenizer", {})
    tokenizer_root = resolve_tokenizer_directory(tokenizer, root, override=args.tokenizer_dir)
    if not tokenizer_root.is_dir():
        failures.append("tokenizer directory missing")
    else:
        for name, expected_digest in tokenizer.get(
            "files_sha256",
            {},
        ).items():
            path = tokenizer_root / name
            if not path.is_file() or file_digest(path) != expected_digest:
                failures.append(f"tokenizer checksum mismatch: {name}")

    arrays: dict[str, np.ndarray] = {}
    for name, metadata in manifest.get("arrays", {}).items():
        path = root / metadata["file"]
        if not path.is_file() or file_digest(path) != metadata["sha256"]:
            failures.append(f"array checksum mismatch: {name}")
            continue
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != metadata["shape"]:
            failures.append(f"array shape mismatch: {name}")
        if str(value.dtype) != metadata["dtype"]:
            failures.append(f"array dtype mismatch: {name}")
        arrays[name] = value

    split_summary: dict[str, Any] = {}
    hash_sets: dict[str, set[tuple[int, ...]]] = {}
    sample_indices_by_split: dict[str, list[int]] = {}
    for split in ("train", "validation"):
        required = [
            f"{split}_tokens",
            f"{split}_lengths",
            f"{split}_domains",
            f"{split}_document_hashes",
        ]
        if any(name not in arrays for name in required):
            failures.append(f"{split}: missing arrays")
            continue
        tokens = arrays[f"{split}_tokens"]
        lengths = np.asarray(arrays[f"{split}_lengths"], dtype=np.int64)
        domains = np.asarray(arrays[f"{split}_domains"], dtype=np.int64)
        hashes = np.asarray(arrays[f"{split}_document_hashes"])
        if len(tokens) != EXPECTED_ROWS[split]:
            failures.append(f"{split}: row count mismatch")
        if tokens.shape != (EXPECTED_ROWS[split], SEQUENCE_LENGTH + 1):
            failures.append(f"{split}: token shape mismatch")
        if np.any(lengths < 2) or np.any(lengths > SEQUENCE_LENGTH + 1):
            failures.append(f"{split}: invalid lengths")
        if np.any(tokens < 0) or np.any(tokens >= tokenizer["vocab_size"]):
            failures.append(f"{split}: token ID outside vocabulary")
        expected_domains = np.asarray(
            [record[0] for record in expected[split]],
            dtype=np.int8,
        )
        if not np.array_equal(domains, expected_domains):
            failures.append(f"{split}: deterministic domain order mismatch")
        expected_hashes = digest_matrix(expected[split])
        if not np.array_equal(hashes, expected_hashes):
            failures.append(f"{split}: deterministic document hashes mismatch")
        eos = int(tokenizer["eos_token_id"])
        for row in range(len(tokens)):
            length = int(lengths[row])
            if int(tokens[row, length - 1]) != eos:
                failures.append(f"{split}: missing EOS at row {row}")
                break
            if length < tokens.shape[1] and np.any(tokens[row, length:] != eos):
                failures.append(f"{split}: invalid padding at row {row}")
                break
        counts = {
            name: int(np.sum(domains == index))
            for index, name in enumerate(DOMAIN_NAMES)
        }
        if any(value != EXPECTED_PER_DOMAIN[split] for value in counts.values()):
            failures.append(f"{split}: class balance mismatch")
        hash_sets[split] = {tuple(row) for row in hashes.tolist()}
        sample_count = min(128, len(tokens))
        sample_indices_by_split[split] = np.linspace(
            0,
            len(tokens) - 1,
            num=sample_count,
            dtype=np.int64,
        ).tolist()
        split_summary[split] = {
            "rows": len(tokens),
            "domain_counts": counts,
            "target_tokens": int(np.sum(lengths - 1)),
            "unique_document_hashes": len(hash_sets[split]),
        }
    overlap = (
        len(hash_sets["train"] & hash_sets["validation"])
        if len(hash_sets) == 2
        else -1
    )
    if overlap != 0:
        failures.append("document leakage across train and validation")

    if not failures and tokenizer_root.is_dir():
        from transformers import AutoTokenizer

        tokenizer_object = AutoTokenizer.from_pretrained(
            tokenizer_root,
            local_files_only=True,
            use_fast=True,
        )
        eos = int(tokenizer_object.eos_token_id)
        for split, indices in sample_indices_by_split.items():
            texts = [expected[split][index][1] for index in indices]
            encoded = tokenizer_object(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=SEQUENCE_LENGTH,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            tokens = arrays[f"{split}_tokens"]
            lengths = arrays[f"{split}_lengths"]
            for index, token_ids in zip(indices, encoded):
                document = list(token_ids[:SEQUENCE_LENGTH]) + [eos]
                length = int(lengths[index])
                if length != len(document):
                    failures.append(f"{split}: sampled length mismatch")
                    break
                if not np.array_equal(
                    tokens[index, :length],
                    np.asarray(document, dtype=np.int32),
                ):
                    failures.append(f"{split}: sampled tokenization mismatch")
                    break

    passed = not failures
    audit = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "study": "agnews_balanced_corpus",
        "dataset_id": DATASET_ID,
        "manifest_sha256": file_digest(manifest_path),
        "split_summary": split_summary,
        "train_validation_document_overlap": overlap,
        "sampled_tokenization_rows_per_split": 128,
        "failures": failures,
        "corpus_integrity_checks_passed": passed,
    }
    destination = root / "corpus_audit.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {destination}")
    destination.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
