#!/usr/bin/env python3
"""Run Self-Forcing with VAR-Q hooks installed automatically."""

from __future__ import annotations

import argparse
from pathlib import Path

from VAR_Q.hooks import install_self_forcing_hooks
from VAR_Q.launch import resolve_path, run_upstream_with_hook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--entrypoint", default="inference.py")
    parser.add_argument("--pipeline-module", default="pipeline.causal_inference")
    parser.add_argument("--pipeline-class", default="CausalInferencePipeline")
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = resolve_path(args.config, REPOSITORY_ROOT, Path.cwd())
    if not config.is_file():
        raise FileNotFoundError(f"VAR-Q config does not exist: {config}")
    run_upstream_with_hook(
        repository_root=REPOSITORY_ROOT,
        upstream_dir=args.upstream_dir,
        entrypoint=args.entrypoint,
        pipeline_module=args.pipeline_module,
        pipeline_class=args.pipeline_class,
        upstream_args=args.upstream_args,
        install_hook=lambda pipeline: install_self_forcing_hooks(pipeline, str(config)),
        model_name="Self-Forcing",
    )


if __name__ == "__main__":
    main()
