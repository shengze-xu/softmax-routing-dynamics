"""Resolve prepared-data references without binding them to one machine."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def manifest_relative_path(path: Path, dataset_dir: Path) -> str:
    """Store a path relative to the final dataset directory, not its staging dir."""
    return Path(os.path.relpath(path.resolve(), dataset_dir.resolve())).as_posix()


def resolve_tokenizer_directory(
    metadata: dict[str, Any],
    dataset_dir: Path,
    *,
    override: Path | None = None,
) -> Path:
    """Read portable references or explicitly relocate a legacy absolute path."""
    if override is not None:
        return override.expanduser().resolve()
    recorded = metadata.get("path")
    if not isinstance(recorded, str) or not recorded.strip():
        raise ValueError("Tokenizer path is missing; provide --tokenizer-dir")
    path = Path(recorded).expanduser()
    if not path.is_absolute():
        path = dataset_dir / path
    return path.resolve()
