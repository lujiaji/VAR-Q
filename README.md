# ⚡ VAR-Q: KV-Cache Quantization for Visual Autoregressive Generation

**VAR-Q** is a lightweight KV-cache quantization method for efficient visual autoregressive generation.
It reduces inference-time KV-cache memory while preserving generation quality, and is designed to attach to existing visual autoregressive model implementations without modifying their source code.

<p align="center">
  <img src="assets/VAR-Q-performance.png" alt="VAR-Q performance summary" width="820">
</p>

## ✨ Highlights

- **Runtime hook integration**: VAR-Q is installed in memory after a backend model is built; users do not need to patch third-party repositories.
- **Main VAR-Q method**: supports `VARQ`, all `G_*` grouping variants, ratio-controlled grouping, pre-RoPE control, and low-bit KV cache packing/unpacking.
- **Clean ablation boundary**: KIVI, FLexGen, and KVQuant live under `ablation/`; they do not share runtime implementation with `VAR_Q`.
- **Backend-friendly release**: third-party model repositories, checkpoints, generated media, local tests, and experiment scratch files are ignored by default.
- **Minimal core dependency**: the VAR-Q core only depends on PyTorch/Triton-level tensor operations; backend-specific environments should follow the upstream model repositories.

## 🧩 Supported Backends

| Backend | Upstream repository | Default checkout | Integration |
| --- | --- | --- | --- |
| VAR | https://github.com/FoundationVision/VAR | `third_party/VAR` | Runtime hook |
| Infinity | https://github.com/FoundationVision/Infinity | `third_party/Infinity` | Runtime hook |
| InfinityStar | https://github.com/FoundationVision/InfinityStar | `third_party/InfinityStar` | Runtime hook |
| Self-Forcing | https://github.com/guandeh17/Self-Forcing | `third_party/Self-Forcing` | Runtime hook adapter / config launcher |
| LongLive | https://github.com/NVlabs/LongLive | `third_party/LongLive` | Runtime hook adapter / config launcher |

VAR-Q and its ablation methods are both routed through the hook layer for supported backends. The hook layer only decides where to attach and which quantizer to build; the VAR-Q implementation remains in `VAR_Q/`, and ablations remain in `ablation/`.

## 📦 Repository Layout

```text
VAR-Q/
├── VAR_Q/                  # VAR-Q core and runtime hooks
│   ├── quant.py            # VARQ, G_* grouping, ratio logic, pre-RoPE control
│   ├── pack_unpack.py      # Low-bit pack/unpack utilities
│   ├── config_loader.py    # Public JSON normalization
│   ├── paths.py            # Relative third-party path helpers
│   └── hooks/              # SmoothQuant-style hook installer
├── ablation/               # KIVI / FLexGen / KVQuant implementations
├── Benchmark/              # Public evaluation entrypoints
├── configs/                # Curated JSON configs
├── scripts/                # Public inference and evaluation launchers
├── third_party/README.md   # Upstream checkout instructions
├── requirements-varq.txt   # Lightweight VAR-Q convenience environment
└── LICENSE
```

Ignored local-only paths include `third_party/`, `tests/`, `temp/`, benchmark outputs, model weights, and generated images/videos.

## 🚀 Installation

Clone VAR-Q:

```bash
git clone <this-repo-url> VAR-Q
cd VAR-Q
```

Install the third-party backend environment from its official repository first. Then install the minimal VAR-Q core requirements:

```bash
conda activate <your-backend-env>
pip install -r requirements-varq.txt
```

VAR-Q itself only needs lightweight PyTorch/Triton-compatible tensor support. VAR, Infinity, InfinityStar, Self-Forcing, and LongLive may pin different CUDA, PyTorch, `flash-attn`, `xformers`, tokenizer, or evaluation package versions; those requirements belong to the official upstream repositories, not to VAR-Q.

## 🧱 Third-Party Models

Clone upstream repositories into `third_party/`:

```bash
git clone https://github.com/FoundationVision/VAR third_party/VAR
git clone https://github.com/FoundationVision/Infinity third_party/Infinity
git clone https://github.com/FoundationVision/InfinityStar third_party/InfinityStar
git clone https://github.com/guandeh17/Self-Forcing third_party/Self-Forcing
git clone https://github.com/NVlabs/LongLive third_party/LongLive
```

No `git apply`, patch marker, or source edit is required. Runtime entrypoints load the third-party model, call `install_varq_hooks(...)`, and then run the backend's normal inference path.

Checkpoints are intentionally not stored in JSON configs. Provide them through command-line arguments, environment variables, or the upstream backend's native loader.

## 🪝 Runtime Hook API

VAR-Q follows a SmoothQuant-style runtime replacement design. The installer scans supported attention modules, stores the original methods in a handle, and replaces the instance-level KV-cache path in memory.

```python
from VAR_Q.hooks import install_varq_hooks, remove_varq_hooks

handle = install_varq_hooks(
    model,
    model_type="var",  # "var", "infinity", or "infinitystar"
    quant_config={
        "enable": True,
        "q_bits": 4,
        "quant_method": "VARQ",
        "qkv_format": "BLHc",
        "pack_to_int32": True,
    },
)

# Run backend inference.

remove_varq_hooks(handle)
```

Use `quant_method="VARQ"` or any `G_*` method for the main method. Use `KIVI`, `FLexGen`, or `KVQuant` in public configs for ablations; the loader normalizes them to isolated ablation implementations.

## 🎯 Inference Scripts

All public scripts resolve paths relative to the VAR-Q repository root.

VAR:

```bash
export VARQ_VAE_CKPT=/path/to/vae_ch160v4096z32.pth
export VARQ_VAR_CKPT_TEMPLATE='/path/to/var_d{}.pth'
bash scripts/inference_VAR.sh configs/var/varq/base/VAR-VARQ-8.json scripts/output/var
```

Infinity:

```bash
export INFINITY_MODEL_PATH=/path/to/infinity_model
export INFINITY_TEXT_ENCODER_CKPT=/path/to/text_encoder
export INFINITY_PN=1M
bash scripts/inference_Infinity.sh \
  configs/infinity/varq/base/Infinity-VARQ-8.json \
  "a cinematic photograph of a red fox in snow" \
  scripts/output/infinity.png
```

InfinityStar:

```bash
export INFINITYSTAR_CHECKPOINTS_DIR=/path/to/infinitystar/checkpoints
bash scripts/inference_InfinityStar.sh \
  configs/infinitystar/varq/base/InfinityStar-VARQ-8.json \
  --output scripts/output/infinitystar_varq_demo.mp4
```

Self-Forcing runs in the official upstream Self-Forcing environment. Pass the upstream inference command after `--`:

```bash
bash scripts/inference_SelfForcing.sh \
  configs/self_forcing/varq/base/SF-VARQ-4.json \
  -- python <upstream_self_forcing_inference.py> <upstream args>
```

LongLive follows the same pattern:

```bash
bash scripts/inference_LongLive.sh \
  configs/longlive/varq/base/LL-VARQ-4.json \
  -- python <upstream_longlive_inference.py> <upstream args>
```

If a required third-party checkout is missing, launchers fail early and print the expected `third_party/<repo>` path.

## 🧪 Evaluation

VAR evaluation:

```bash
bash scripts/eval_VAR.sh \
  configs/var/varq/base/VAR-VARQ-8.json \
  /path/to/VIRTUAL_imagenet256_labeled.npz
```

Infinity evaluation:

```bash
bash scripts/eval_Infinity.sh geneval \
  configs/infinity/varq/base/Infinity-VARQ-8.json
```

Supported Infinity evaluation tasks are `geneval`, `dpg`, and `imagereward`.

## ⚙️ Configs

Curated JSON configs live under:

```text
configs/<backend>/<family>/<topic>/*.json
```

The `quantization` block is intentionally backend-agnostic and can be reused when calling `install_varq_hooks(...)` directly. Full JSON files are still kept per backend because model loaders use different `qkv_format`, sequence layout, image/video schedule, and grouping defaults.

Retained public configs include:

| Backend | VAR-Q configs | Ablation configs | Extra configs |
| --- | --- | --- | --- |
| VAR | 8/4/3-bit | KIVI 4-bit, FLexGen 4-bit | `G_HEAD_DIM` |
| Infinity | 8/4/3/2-bit | KIVI 4-bit, FLexGen 4-bit | `G_HEAD_DIM` |
| InfinityStar | 8/4-bit | KIVI 4-bit, FLexGen 4-bit | `G_HEAD_DIM`, ratio 1/2 and 1/3 |
| Self-Forcing | 8/4/3-bit | KIVI 4-bit, FLexGen 4-bit | `G_HEAD_DIM`, 4/3-bit ratio 1/2, 1/4, 1/8 |
| LongLive | 8/4/3-bit | KIVI 4-bit, FLexGen 4-bit | `G_HEAD_DIM`, 4/3-bit ratio 1/2, 1/4, 1/8 |

For next-frame video backends such as Self-Forcing and LongLive, `max_scale_seq_len=1560` is the default grouping unit. If `compression_ratio` is omitted, it defaults to `1`, so the group length is 1560. `compression_ratio=3` represents grouping the full 4680-token chunk.

## ✅ Development Checks

Local tests are kept in ignored `tests/` and are not part of the public package surface:

```bash
python -m unittest discover -s tests -p 'test_runtime_hooks.py'
python -m unittest discover -s tests -p 'test_quant_boundaries.py'
python -m py_compile VAR_Q/*.py VAR_Q/hooks/*.py ablation/*.py scripts/*.py
```

The smoke tests instantiate clean upstream-style attention modules and verify that VAR-Q can attach without changing third-party source files.

## 🗺️ TODO

- Support more visual autoregressive backends.
- Improve GPU memory fragmentation behavior during long generation.
- Add more backend version signatures for robust hook detection.
- Add fused kernels for common low-bit KV packing/unpacking paths.
- Expand end-to-end smoke tests for video generation backends.
- Add config inheritance/snippet support so repeated quantization blocks can be shared more compactly.

## 📚 Citation

If this repository is useful for your research, please cite the VAR-Q paper. The BibTeX entry will be added after the final publication metadata is available.

## 📄 License

This repository is released under the MIT License. Third-party model repositories, checkpoints, datasets, and generated assets are governed by their own licenses.
