# Third-Party Backends

This directory is the default location for upstream model repositories. It is intentionally ignored by git; only this README is tracked.

VAR-Q does not vendor or patch third-party source code. Public entrypoints locate backends through `third_party/<repo>` and install runtime hooks after the model is constructed.

## Required For Public Hook Entry Points

```bash
git clone https://github.com/FoundationVision/VAR third_party/VAR
git clone https://github.com/FoundationVision/Infinity third_party/Infinity
git clone https://github.com/FoundationVision/InfinityStar third_party/InfinityStar
```

Install each backend's environment according to its official upstream README. VAR-Q does not define or pin the CUDA/PyTorch/attention stack for these repositories.

## Optional For Retained Configs

```bash
git clone https://github.com/guandeh17/Self-Forcing third_party/Self-Forcing
git clone https://github.com/NVlabs/LongLive third_party/LongLive
```

## Weights

Model checkpoints are not tracked by this repository and are not stored in public JSON configs. Put them wherever your backend loader expects, or pass paths through the public scripts.

Wan weights used by some video setups can be placed under `third_party/wan_models/`:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir third_party/wan_models/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False
```
