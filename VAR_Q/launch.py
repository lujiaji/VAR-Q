"""Utilities for running an upstream entrypoint with a VAR-Q hook."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Callable, Sequence


def resolve_path(value: str, *bases: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (bases[0] / candidate).resolve()


def run_upstream_with_hook(
    *,
    repository_root: Path,
    upstream_dir: str,
    entrypoint: str,
    pipeline_module: str,
    pipeline_class: str,
    upstream_args: Sequence[str],
    install_hook: Callable[[Any], Any],
    model_name: str,
) -> None:
    upstream = resolve_path(upstream_dir, repository_root, Path.cwd())
    script = resolve_path(entrypoint, upstream, repository_root, Path.cwd())
    if not upstream.is_dir():
        raise FileNotFoundError(f"upstream directory does not exist: {upstream}")
    if not script.is_file():
        raise FileNotFoundError(f"upstream entrypoint does not exist: {script}")

    sys.path[:0] = [str(repository_root), str(upstream)]
    old_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        module = importlib.import_module(pipeline_module)
        pipeline_type = getattr(module, pipeline_class, None)
        if pipeline_type is None:
            raise AttributeError(f"{pipeline_module} does not export {pipeline_class}")
        original_init = pipeline_type.__init__

        def init_with_varq(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._varq_public_handle = install_hook(self)
            print(f"[VAR-Q] installed {model_name} hooks", flush=True)

        pipeline_type.__init__ = init_with_varq
        old_argv = sys.argv
        try:
            forwarded = list(upstream_args)
            if forwarded[:1] == ["--"]:
                forwarded = forwarded[1:]
            sys.argv = [str(script), *forwarded]
            runpy.run_path(str(script), run_name="__main__")
        finally:
            pipeline_type.__init__ = original_init
            sys.argv = old_argv
    finally:
        os.chdir(old_cwd)


__all__ = ["resolve_path", "run_upstream_with_hook"]
