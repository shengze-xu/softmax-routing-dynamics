"""Resolve prepared-data references without binding them to one machine."""
from __future__ import annotations

import os
from pathlib import Path


def manifest_relative_path(path: Path, dataset_dir: Path) -> str:
    """Store a path relative to the final dataset directory."""
    return Path(os.path.relpath(path.resolve(), dataset_dir.resolve())).as_posix()
