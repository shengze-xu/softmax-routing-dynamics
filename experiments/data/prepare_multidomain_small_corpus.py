#!/usr/bin/env python3
"""Prepare the fixed-source four-domain corpus for small-model routing comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data.paths import manifest_relative_path


SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "shift")
DEFAULT_SPLIT_KEY = "6d6f652d7468656f72792d7068617365312d7631"


@dataclass(frozen=True)
class Domain:
    name: str
    repository: str
    revision: str
    config: str | None
    declared_license: str
    expected_license_fragment: str | None


DOMAINS = (
    Domain(
        "scientific_abstracts",
        "common-pile/arxiv_abstracts",
        "828e35d1000f94579da8850f5f640c138279bdb5",
        None,
        "CC0-1.0 metadata/abstract release in Common Pile",
        "Creative Commons Zero",
    ),
    Domain(
        "public_domain_books",
        "common-pile/project_gutenberg",
        "01dc90a5002f8977c7fb03a372c14bca29c65cf1",
        None,
        "Public Domain according to Project Gutenberg metadata",
        "Public Domain",
    ),
    Domain(
        "us_government",
        "common-pile/usgpo",
        "0b314ed37117654d04c563b7d21b6ae3445a45df",
        None,
        "Public Domain U.S. Government works",
        "Public Domain",
    ),
    Domain(
        "encyclopedic",
        "Salesforce/wikitext",
        "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wikitext-103-raw-v1",
        "CC BY-SA 3.0 and GFDL",
        None,
    ),
)


def stable_u64(text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8],
        "big",
        signed=False,
    )


def assign_split(domain: str, document_id: str, key_hex: str) -> str:
    """Assign document-disjoint splits using a reproducible byte key."""
    message = bytes.fromhex(key_hex) + f":{domain}:{document_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % 10_000
    if value < 8_000:
        return "train"
    if value < 9_000:
        return "validation"
    return "shift"


def save_array(path: Path, value: np.ndarray) -> dict[str, object]:
    np.save(path, value, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def row_id(domain: Domain, row: dict[str, object], index: int) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    text = str(row.get("text", ""))
    return f"{index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def license_ok(domain: Domain, row: dict[str, object]) -> tuple[bool, str]:
    if domain.expected_license_fragment is None:
        return True, domain.declared_license
    metadata = row.get("metadata") or {}
    observed = str(metadata.get("license", ""))
    return domain.expected_license_fragment in observed, observed


def stream_domain(
    domain: Domain,
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
    for index, row in enumerate(stream):
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
            target_count = length - 1
            payload[split].append((tokens, length, document_hash))
            token_counts[split] += target_count
            added_document = True
            if token_counts[split] >= targets[split]:
                break
        if added_document:
            document_counts[split] += 1
    if any(token_counts[split] < targets[split] for split in SPLITS):
        raise RuntimeError(
            f"{domain.name} stream exhausted before token targets: {token_counts}"
        )
    return payload, {
        "domain_index": domain_index,
        "documents": document_counts,
        "target_tokens": token_counts,
        "observed_licenses": sorted(observed_licenses),
        "rejected_license_rows": rejected_licenses,
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite corpus: {output_dir}")
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
    domain_summaries = {}
    for domain_index, domain in enumerate(DOMAINS):
        print(f"streaming domain={domain.name}", flush=True)
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
        )
        domain_summaries[domain.name] = summary
        for split in SPLITS:
            combined[split].extend(
                (tokens, length, document_hash, domain_index)
                for tokens, length, document_hash in payload[split]
            )

    output_dir.mkdir(parents=True)
    arrays = {}
    split_summaries = {}
    rng = np.random.default_rng(args.output_shuffle_seed)
    for split in SPLITS:
        records = combined[split]
        order = rng.permutation(len(records))
        tokens = np.stack([records[index][0] for index in order])
        lengths = np.asarray([records[index][1] for index in order], dtype=np.int16)
        document_hashes = np.asarray(
            [records[index][2] for index in order],
            dtype=np.uint64,
        )
        domains = np.asarray([records[index][3] for index in order], dtype=np.int8)
        for suffix, value in (
            ("tokens", tokens),
            ("lengths", lengths),
            ("domains", domains),
            ("document_hashes", document_hashes),
        ):
            name = f"{split}_{suffix}"
            arrays[name] = save_array(output_dir / f"{name}.npy", value)
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
                output_dir / f"{split}_document_hashes.npy",
                allow_pickle=False,
            ).tolist()
        )
        for split in SPLITS
    }
    overlaps = {
        "train_validation": len(document_sets["train"] & document_sets["validation"]),
        "train_shift": len(document_sets["train"] & document_sets["shift"]),
        "validation_shift": len(
            document_sets["validation"] & document_sets["shift"]
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"Document split leakage detected: {overlaps}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_multidomain_small_candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "multidomain_small_corpus",
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
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "output": str(output_dir),
        "domains": manifest["domain_names"],
        "splits": split_summaries,
    }, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-tokens-per-domain", type=int, default=2_000_000)
    parser.add_argument("--validation-tokens-per-domain", type=int, default=100_000)
    parser.add_argument("--shift-tokens-per-domain", type=int, default=100_000)
    parser.add_argument("--max-tokens-per-document", type=int, default=4096)
    parser.add_argument("--minimum-fragment-tokens", type=int, default=32)
    parser.add_argument("--split-key", default=DEFAULT_SPLIT_KEY, help="Hexadecimal key for document-level splits")
    parser.add_argument("--stream-seed", type=int, default=20260727)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--output-shuffle-seed", type=int, default=20260728)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
