# VAR-Q: Unified Quantization for Visual Autoregressive Models

**VAR-Q (Visual Autoregressive Quantization)** is an efficient and flexible quantization framework for **Vision‑Autoregressive (VAR)** and related transformer models. It targets **KV/activation/weight** memory reduction and **inference acceleration**, while keeping generation quality stable. The framework supports **per‑head**, **per‑dim**, **group‑wise** and whole‑tensor strategies, with optional statistics collection and CSV reporting.

---

## ✨ Features

* **Multiple quant rules**: `per-tensor`, `per-channel`, `per-head`, `per-dim`, and hybrid/grouped variants.
* **Low‑bit packing**: built‑in (PyTorch/Triton) INT8/4/2 packing & unpacking utilities.
* **Range & error analysis**: on‑demand min/max range stats and **L2‑normalized error** comparison, saved to **CSV**.
* **Plug‑and‑play**: drop into existing VAR/Infinity code paths for PTQ-style quant.

---

## 📁 Repository Structure

```
VAR-Q/
├── VAR_Q/                # Core library (quant ops, pack/unpack, helpers)
├── VAR/                  # Upstream VAR (adapted for VAR-Q integration)
├── Infinity/             # Upstream Infinity (adapted for VAR-Q integration)
├── Benchmark/            # Reports, figures, and evaluation artifacts
├── scripts/              # Shell entrypoints (quick-start & benchmark .sh)
└── README.md
```

> Notes
>
> * Use `VAR_Q/` as your import root when integrating in Python.
> * Check `scripts/` for runnable **.sh** examples (quantization, inference, benchmarking).

---

## ⚙️ Installation

### 1) Create a clean environment

We recommend a single, pinned environment to avoid version drift (CUDA/Torch/Triton consistent across all modules):

```bash
conda create -n varq python=3.10 -y
conda activate varq

# Core deps
pip install torch torchvision
pip install triton omegaconf pandas tqdm
```

### 2) (Optional) Enable inference with **VAR** / **Infinity**

If you plan to run the original **VAR** or **Infinity** generators with VAR‑Q, you must install **their own requirements** under their **respective directories**:

```bash
# Install VAR dependencies
cd VAR
pip install -r requirements.txt
cd ..

# Install Infinity dependencies
cd Infinity
pip install -r requirements.txt
cd ..
```

> Keep CUDA/Torch versions aligned with the environment you created in step 1.

---

## 🚀 Quick Start (run the shell scripts in `scripts/`)

List available scripts and pick one for your use case:

```bash
ls -1 scripts/*.sh
```

Run directly (examples below—use the exact `.sh` names in your repo):

```bash
# End‑to‑end inference with a VAR/Infinity pipeline (after installing their reqs)
bash scripts/infer_var_example.sh
bash scripts/infer_infinity_example.sh
```

> Most scripts are self‑documented: run with `-h` or open the file to see flags.

---

## 📊 Benchmarking (also via `scripts/`)

All benchmarking is provided as shell entrypoints under `scripts/`.

```bash
# VAR-side benchmarking
bash scripts/eval_VAR.sh

# Infinity-side benchmarking
bash scripts/eval_Infinity.sh
```

Generated artifacts (CSV/figures) are saved to `Benchmark/outputs/` by default unless overridden by script flags.

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 📬 Contact

**Author:** Jiaji Lu Boxun Xu

**Affiliation:** Peng Li's lab, ECE, UCSB

<p align="center">⭐ If you find VAR‑Q useful, please give it a star! ⭐</p>
