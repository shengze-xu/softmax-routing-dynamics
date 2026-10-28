#!/usr/bin/env python3
"""Prepare the full, deduplicated and balanced AG News corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
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

from experiments.data.paths import manifest_relative_path


SCHEMA_VERSION = 1
DATASET_ID = "agnews-full-balanced-v1"
DOMAIN_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
TRAIN_PER_DOMAIN = 29_950
VALIDATION_PER_DOMAIN = 1_899
SEQUENCE_LENGTH = 64
OUTPUT_SHUFFLE_SEED = 20260729


@dataclass(frozen=True)
class Record:
    domain: int
    text: str
    digest: str


def normalize_text(title: str, description: str) -> str:
    normalized_title = " ".join(title.split())
    normalized_description = " ".join(description.split())
    return f"{normalized_title} {normalized_description}".strip()


def read_records(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{row_number}: expected three columns")
            domain = int(row[0]) - 1
            if not 0 <= domain < len(DOMAIN_NAMES):
                raise ValueError(f"{path}:{row_number}: invalid class")
            text = normalize_text(row[1], row[2])
            if not text:
                raise ValueError(f"{path}:{row_number}: empty normalized text")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            records.append(Record(domain, text, digest))
    return records


def deduplicate(
    records: list[Record],
    *,
    excluded_hashes: set[str] | None = None,
) -> tuple[list[Record], Counter[int], Counter[int]]:
    excluded_hashes = excluded_hashes or set()
    seen: set[str] = set()
    retained: list[Record] = []
    duplicates: Counter[int] = Counter()
    excluded: Counter[int] = Counter()
    for record in records:
        if record.digest in seen:
            duplicates[record.domain] += 1
            continue
        seen.add(record.digest)
        if record.digest in excluded_hashes:
            excluded[record.domain] += 1
            continue
        retained.append(record)
    return retained, duplicates, excluded


def balanced_selection(
    records: list[Record],
    *,
    per_domain: int,
) -> list[Record]:
    grouped: list[list[Record]] = [
        [] for _ in range(len(DOMAIN_NAMES))
    ]
    for record in records:
        grouped[record.domain].append(record)
    selected: list[Record] = []
    for domain, group in enumerate(grouped):
        ordered = sorted(group, key=lambda record: record.digest)
        if len(ordered) < per_domain:
            raise ValueError(
                f"Domain {domain} has {len(ordered)} unique rows; "
                f"requested {per_domain}"
            )
        selected.extend(ordered[:per_domain])
    return selected


def shuffled(records: list[Record], *, seed: int) -> list[Record]:
    order = np.random.default_rng(seed).permutation(len(records))
    return [records[int(index)] for index in order]


def digest_matrix(records: list[Record]) -> np.ndarray:
    result = np.empty((len(records), 4), dtype=np.uint64)
    for row, record in enumerate(records):
        payload = bytes.fromhex(record.digest)
        result[row] = [
            int.from_bytes(payload[start : start + 8], "big")
            for start in range(0, 32, 8)
        ]
    return result


def encode_records(
    records: list[Record],
    *,
    tokenizer: Any,
    sequence_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eos = int(tokenizer.eos_token_id)
    tokens = np.full(
        (len(records), sequence_length + 1),
        eos,
        dtype=np.int32,
    )
    lengths = np.empty(len(records), dtype=np.int16)
    domains = np.asarray(
        [record.domain for record in records],
        dtype=np.int8,
    )
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        encoded = tokenizer(
            [record.text for record in records[start:stop]],
            add_special_tokens=False,
            truncation=True,
            max_length=sequence_length,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        for offset, token_ids in enumerate(encoded):
            row = start + offset
            document = list(token_ids[:sequence_length]) + [eos]
            if len(document) < 2:
                raise ValueError(f"Row {row} produced fewer than two LM tokens")
            tokens[row, : len(document)] = np.asarray(
                document,
                dtype=np.int32,
            )
            lengths[row] = len(document)
        print(
            json.dumps(
                {
                    "event": "tokenization_progress",
                    "completed": stop,
                    "total": len(records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tokens, lengths, domains, digest_matrix(records)


def save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, value, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def count_by_domain(records: list[Record]) -> dict[str, int]:
    counts = Counter(record.domain for record in records)
    return {
        name: int(counts[index]) for index, name in enumerate(DOMAIN_NAMES)
    }


def counter_by_domain(value: Counter[int]) -> dict[str, int]:
    return {
        name: int(value[index]) for index, name in enumerate(DOMAIN_NAMES)
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    output_dir = args.output_dir.resolve()
    staging_dir = output_dir.with_name(f"{output_dir.name}.incomplete")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite corpus or staging directory: {output_dir}"
        )

    train_raw = read_records(source_dir / "train.csv")
    validation_raw = read_records(source_dir / "test.csv")
    train_unique, train_duplicates, train_excluded = deduplicate(train_raw)
    train_hashes = {record.digest for record in train_unique}
    validation_unique, validation_duplicates, validation_excluded = (
        deduplicate(validation_raw, excluded_hashes=train_hashes)
    )
    if train_excluded:
        raise ValueError("Training records unexpectedly excluded")
    train_selected = shuffled(
        balanced_selection(
            train_unique,
            per_domain=TRAIN_PER_DOMAIN,
        ),
        seed=OUTPUT_SHUFFLE_SEED,
    )
    validation_selected = shuffled(
        balanced_selection(
            validation_unique,
            per_domain=VALIDATION_PER_DOMAIN,
        ),
        seed=OUTPUT_SHUFFLE_SEED + 1,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    train = encode_records(
        train_selected,
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=args.tokenizer_batch_size,
    )
    validation = encode_records(
        validation_selected,
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=args.tokenizer_batch_size,
    )

    staging_dir.mkdir(parents=True)
    arrays: dict[str, dict[str, Any]] = {}
    for split, payload in (("train", train), ("validation", validation)):
        suffixes = ("tokens", "lengths", "domains", "document_hashes")
        for suffix, value in zip(suffixes, payload):
            name = f"{split}_{suffix}"
            arrays[name] = save_array(staging_dir / f"{name}.npy", value)
    token_counts: dict[str, dict[str, int]] = {}
    for split, records, payload in (
        ("train", train_selected, train),
        ("validation", validation_selected, validation),
    ):
        lengths = payload[1].astype(np.int64)
        domains = payload[2].astype(np.int64)
        token_counts[split] = {
            name: int(np.sum(lengths[domains == index] - 1))
            for index, name in enumerate(DOMAIN_NAMES)
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "status": "prepared_agnews_candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "interpretation": (
            "AG News categories are four non-overlapping routing regimes for "
            "the independent AG News structural replication."
        ),
        "source": {
            name: {
                "path": manifest_relative_path(source_dir / name, output_dir),
            }
            for name in ("train.csv", "test.csv")
        },
        "tokenizer": {
            "path": manifest_relative_path(tokenizer_dir, output_dir),
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "domain_names": DOMAIN_NAMES,
        "selection": {
            "text_normalization": (
                "collapse Unicode whitespace independently in title and "
                "description, join with one ASCII space, then strip"
            ),
            "document_identity": "sha256(normalized UTF-8 text)",
            "duplicate_policy": "retain first hash occurrence within raw split",
            "cross_split_policy": (
                "exclude validation hashes appearing anywhere in raw train"
            ),
            "balanced_selection": (
                "sort retained rows by document hash within class and take "
                "the frozen per-class count"
            ),
            "output_shuffle_seed": OUTPUT_SHUFFLE_SEED,
            "sequence_length": SEQUENCE_LENGTH,
            "train_per_domain": TRAIN_PER_DOMAIN,
            "validation_per_domain": VALIDATION_PER_DOMAIN,
            "raw_counts": {
                "train": count_by_domain(train_raw),
                "validation": count_by_domain(validation_raw),
            },
            "within_split_duplicates_removed": {
                "train": counter_by_domain(train_duplicates),
                "validation": counter_by_domain(validation_duplicates),
            },
            "cross_split_rows_removed_from_validation": (
                counter_by_domain(validation_excluded)
            ),
            "selected_counts": {
                "train": count_by_domain(train_selected),
                "validation": count_by_domain(validation_selected),
            },
            "target_token_counts": token_counts,
        },
        "arrays": arrays,
    }
    manifest_path = staging_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging_dir.rename(output_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset_id": DATASET_ID,
                "output_dir": str(output_dir),
                "train_documents": len(train_selected),
                "validation_documents": len(validation_selected),
            },
            sort_keys=True,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.tokenizer_batch_size <= 0:
        raise ValueError("Tokenizer batch size must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
