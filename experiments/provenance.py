"""Git hash, package versions, config I/O — the reproducibility record.

No result is real without these. Experiment scripts write the dict
into the output directory next to the numbers.
"""

from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_VERSIONS = ("numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer", "tunnelvision")


def git_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"{head}-dirty" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _PACKAGE_VERSIONS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping, got {type(loaded)}")
    return loaded
