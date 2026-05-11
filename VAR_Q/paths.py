from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def third_party_root() -> Path:
    return repo_root() / "third_party"


def third_party_repo(repo_name: str) -> Path:
    return third_party_root() / repo_name


def require_third_party_repo(repo_name: str, clone_url: str | None = None) -> Path:
    repo_dir = third_party_repo(repo_name)
    if repo_dir.exists():
        return repo_dir

    message = (
        f"Missing third-party repository: {repo_dir}. "
        f"Clone it into third_party/{repo_name} before running this entrypoint."
    )
    if clone_url:
        message += f" Suggested source: {clone_url}"
    raise FileNotFoundError(message)


def prepend_sys_path(paths: Iterable[Path | str]) -> None:
    for raw_path in paths:
        path = str(raw_path)
        if path not in sys.path:
            sys.path.insert(0, path)
