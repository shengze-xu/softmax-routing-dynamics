#!/usr/bin/env python3
"""Prepare the 68M-token four-domain corpus from pinned source revisions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data.paths import manifest_relative_path

from experiments.data.prepare_multidomain_small_corpus import (
    DOMAINS, assign_split, license_ok, row_id, save_array, stable_u64,
)


SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "shift")
DEFAULT_SPLIT_KEY = "6d6f652d7468656f72792d7068617365332d7631"
DATASET_ID = "multidomain-64m-2m-2m-v1"


def stream_domain(
    domain,
    *,
    domain_index: int,
    tokenizer,
    sequence_length: int,
    targets: dict[str, int],
    split_key: str,
    stream_seed: int,
    shuffle_buffer: int,
    max_tokens_per_document: int,
    minimum_fragment_tokens: int,
    progress_every_documents: int,
) -> tuple[dict[str, list[tuple[np.ndarray, int, int]]], dict[str, object]]:
    from datasets import load_dataset

    kwargs = {
        "path": domain.repository,
        "split": "train",
        "streaming": True,
        "revision": domain.revision,
    }
    if domain.config is not None:
        kwargs["name"] = domain.config
    stream = load_dataset(**kwargs).shuffle(
        seed=stream_seed + domain_index,
        buffer_size=shuffle_buffer,
    )
    payload: dict[str, list[tuple[np.ndarray, int, int]]] = {
        split: [] for split in SPLITS
    }
    token_counts = {split: 0 for split in SPLITS}
    document_counts = {split: 0 for split in SPLITS}
    observed_licenses: set[str] = set()
    rejected_licenses = 0
    eos = int(tokenizer.eos_token_id)
    scanned = 0
    for index, row in enumerate(stream):
        scanned = index + 1
        if all(token_counts[split] >= targets[split] for split in SPLITS):
            break
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        identifier = row_id(domain, row, index)
        split = assign_split(domain.name, identifier, split_key)
        if token_counts[split] >= targets[split]:
            continue
        accepted, observed_license = license_ok(domain, row)
        observed_licenses.add(observed_license)
        if not accepted:
            rejected_licenses += 1
            continue
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens_per_document,
            return_attention_mask=False,
        )["input_ids"]
        token_ids = list(token_ids) + [eos]
        document_hash = stable_u64(f"{domain.name}:{identifier}")
        added_document = False
        for start in range(0, len(token_ids) - 1, sequence_length):
            fragment = token_ids[start : start + sequence_length + 1]
            if len(fragment) < minimum_fragment_tokens + 1:
                continue
            tokens = np.full(sequence_length + 1, eos, dtype=np.int32)
            tokens[: len(fragment)] = np.asarray(fragment, dtype=np.int32)
            length = len(fragment)
            payload[split].append((tokens, length, document_hash))
            token_counts[split] += length - 1
            added_document = True
            if token_counts[split] >= targets[split]:
                break
        if added_document:
            document_counts[split] += 1
        if progress_every_documents > 0 and scanned % progress_every_documents == 0:
            print(
                json.dumps(
                    {
                        "event": "corpus_progress",
                        "domain": domain.name,
                        "rows_scanned": scanned,
                        "tokens": token_counts,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if any(token_counts[split] < targets[split] for split in SPLITS):
        raise RuntimeError(
            f"{domain.name} stream exhausted before token targets: {token_counts}"
        )
    return payload, {
        "domain_index": domain_index,
        "rows_scanned": scanned,
        "documents": document_counts,
        "target_tokens": token_counts,
        "observed_licenses": sorted(observed_licenses),
        "rejected_license_rows": rejected_licenses,
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    output_dir = args.output_dir.resolve()
    staging_dir = output_dir.with_name(f"{output_dir.name}.incomplete")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite corpus or staging directory: {output_dir}"
        )
    tokenizer_dir = args.tokenizer_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    targets = {
        "train": args.train_tokens_per_domain,
        "validation": args.validation_tokens_per_domain,
        "shift": args.shift_tokens_per_domain,
    }
    combined: dict[str, list[tuple[np.ndarray, int, int, int]]] = {
        split: [] for split in SPLITS
    }
    domain_summaries: dict[str, object] = {}
    for domain_index, domain in enumerate(DOMAINS):
        print(
            json.dumps(
                {
                    "event": "domain_start",
                    "domain": domain.name,
                    "repository": domain.repository,
                    "revision": domain.revision,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        payload, summary = stream_domain(
            domain,
            domain_index=domain_index,
            tokenizer=tokenizer,
            sequence_length=args.sequence_length,
            targets=targets,
            split_key=args.split_key,
            stream_seed=args.stream_seed,
            shuffle_buffer=args.shuffle_buffer,
            max_tokens_per_document=args.max_tokens_per_document,
            minimum_fragment_tokens=args.minimum_fragment_tokens,
            progress_every_documents=args.progress_every_documents,
        )
        domain_summaries[domain.name] = summary
        for split in SPLITS:
            combined[split].extend(
                (tokens, length, document_hash, domain_index)
                for tokens, length, document_hash in payload[split]
            )
        print(
            json.dumps(
                {"event": "domain_complete", "domain": domain.name, **summary},
                sort_keys=True,
            ),
            flush=True,
        )

    staging_dir.mkdir(parents=True)
    try:
        arrays: dict[str, object] = {}
        split_summaries: dict[str, object] = {}
        rng = np.random.default_rng(args.output_shuffle_seed)
        for split in SPLITS:
            records = combined[split]
            order = rng.permutation(len(records))
            tokens = np.stack([records[index][0] for index in order])
            lengths = np.asarray(
                [records[index][1] for index in order],
                dtype=np.int16,
            )
            document_hashes = np.asarray(
                [records[index][2] for index in order],
                dtype=np.uint64,
            )
            domains = np.asarray(
                [records[index][3] for index in order],
                dtype=np.int8,
            )
            for suffix, value in (
                ("tokens", tokens),
                ("lengths", lengths),
                ("domains", domains),
                ("document_hashes", document_hashes),
            ):
                name = f"{split}_{suffix}"
                arrays[name] = save_array(staging_dir / f"{name}.npy", value)
            split_summaries[split] = {
                "sequences": len(records),
                "target_tokens": int(np.sum(lengths.astype(np.int64) - 1)),
                "unique_documents": int(np.unique(document_hashes).size),
                "sequences_by_domain": {
                    domain.name: int(np.sum(domains == index))
                    for index, domain in enumerate(DOMAINS)
                },
            }

        document_sets = {
            split: set(
                np.load(
                    staging_dir / f"{split}_document_hashes.npy",
                    allow_pickle=False,
                ).tolist()
            )
            for split in SPLITS
        }
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
            raise ValueError(f"Document split leakage detected: {overlaps}")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "status": "prepared_multidomain_large_candidate",
            "created_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "study": "multidomain_large_controlled_scale_transfer_corpus",
            "domain_names": [domain.name for domain in DOMAINS],
            "sources": [asdict(domain) for domain in DOMAINS],
            "source_summaries": domain_summaries,
            "selection": {
                "sequence_length": args.sequence_length,
                "split_key": args.split_key,
                "stream_seed": args.stream_seed,
                "shuffle_buffer": args.shuffle_buffer,
                "output_shuffle_seed": args.output_shuffle_seed,
                "max_tokens_per_document": args.max_tokens_per_document,
                "minimum_fragment_tokens": args.minimum_fragment_tokens,
                "requested_tokens_per_domain": targets,
                "split_summaries": split_summaries,
                "document_overlap_counts": overlaps,
            },
            "tokenizer": {
                "path": manifest_relative_path(tokenizer_dir, output_dir),
                "class": type(tokenizer).__name__,
                "vocab_size": len(tokenizer),
                "eos_token_id": int(tokenizer.eos_token_id),
            },
            "arrays": arrays,
        }
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    except BaseException:
        if staging_dir.exists() and not any(staging_dir.iterdir()):
            staging_dir.rmdir()
        raise

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset_id": DATASET_ID,
                "output": str(output_dir),
                "domains": manifest["domain_names"],
                "splits": split_summaries,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--train-tokens-per-domain", type=int, default=16_000_000)
    parser.add_argument(
        "--validation-tokens-per-domain",
        type=int,
        default=500_000,
    )
    parser.add_argument("--shift-tokens-per-domain", type=int, default=500_000)
    parser.add_argument("--max-tokens-per-document", type=int, default=4096)
    parser.add_argument("--minimum-fragment-tokens", type=int, default=32)
    parser.add_argument("--split-key", default=DEFAULT_SPLIT_KEY, help="Hexadecimal key for document-level splits")
    parser.add_argument("--stream-seed", type=int, default=20260731)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--output-shuffle-seed", type=int, default=20260801)
    parser.add_argument("--progress-every-documents", type=int, default=10_000)
    args = parser.parse_args()
    if args.sequence_length != 512:
        raise ValueError("large multidomain v1 requires sequence_length=512")
    for name in (
        "train_tokens_per_domain",
        "validation_tokens_per_domain",
        "shift_tokens_per_domain",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
