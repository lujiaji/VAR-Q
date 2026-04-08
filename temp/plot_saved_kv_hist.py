#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已保存的 npz K/V 文件中读取指定 index（如 39）的数据并绘制“科研风”直方图。

默认读取：
  /data/jiaji_lu/Infinity/8b_QKV/ori/K/K_ori_{idx}.npz
  /data/jiaji_lu/Infinity/8b_QKV/ori/V/V_ori_{idx}.npz

示例：
python scripts/plot_saved_kv_hist.py \
  --k_dir /data/jiaji_lu/Infinity/8b_QKV/ori/K \
  --v_dir /data/jiaji_lu/Infinity/8b_QKV/ori/V \
  --idx 39 \
  --out_dir /data/jiaji_lu/Infinity/8b_QKV/ori/vis_39 \
  --bins 300 \
  --clip_q 0.999 \
  --stride 10 \
  --logy
"""

import os
import json
import argparse
from typing import Tuple, Optional, Dict, Any

import numpy as np


def _load_npz(path: str, key: str) -> np.ndarray:
    z = np.load(path)
    if key not in z.files:
        raise KeyError(f"{path} 内不包含 key={key}，实际 keys={z.files}")
    return z[key]


def _as_1d(a: np.ndarray, stride: int) -> np.ndarray:
    x = a.reshape(-1)
    if stride and stride > 1:
        x = x[:: int(stride)]
    return x


def _clip_by_quantile(x: np.ndarray, clip_q: Optional[float]) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    if clip_q is None:
        return x, None
    q = float(clip_q)
    if not (0.5 < q <= 1.0):
        raise ValueError("--clip_q 需要在 (0.5, 1.0] 范围内")
    if q == 1.0:
        return x, None
    lo = np.quantile(x, 1.0 - q)
    hi = np.quantile(x, q)
    return np.clip(x, lo, hi), (float(lo), float(hi))


def _set_research_rcparams():
    # 不依赖 seaborn；尽量用 matplotlib 自身参数打造“科研风”
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
        }
    )


def _plot_hist(
    x: np.ndarray,
    *,
    bins: int,
    title: str,
    xlabel: str,
    out_png: str,
    out_pdf: str,
    logy: bool,
    clip_range: Optional[Tuple[float, float]],
):
    import matplotlib.pyplot as plt

    mu = float(x.mean())
    sig = float(x.std())
    x_min = float(x.min())
    x_max = float(x.max())

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    # 频次直方图：纵轴为计数（frequency），不是概率密度
    ax.hist(x, bins=int(bins), density=False, color="#2A6F97", alpha=0.85, edgecolor="none")
    ax.axvline(mu, color="#8B0000", linewidth=1.4, label=f"$\\mu$={mu:.4g}")
    ax.axvline(mu - sig, color="#8B0000", linewidth=1.0, linestyle="--", alpha=0.9, label=f"$\\sigma$={sig:.4g}")
    ax.axvline(mu + sig, color="#8B0000", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.axvline(mu - 3.0 * sig, color="#5A189A", linewidth=1.1, linestyle="-.", alpha=0.95, label="$\\pm 3\\sigma$")
    ax.axvline(mu + 3.0 * sig, color="#5A189A", linewidth=1.1, linestyle="-.", alpha=0.95)

    # range 竖线（min/max），用更浅的样式避免喧宾夺主
    ax.axvline(x_min, color="#6c757d", linewidth=1.0, linestyle=":", alpha=0.9, label="min/max")
    ax.axvline(x_max, color="#6c757d", linewidth=1.0, linestyle=":", alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    # 用户要求：不使用 logy（保持线性频次坐标）

    # range 标注（min/max）
    # ax.text(
    #     0.98,
    #     0.92,
    #     f"range=[{x_min:.4g}, {x_max:.4g}]",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=9,
    # )

    if clip_range is not None:
        lo, hi = clip_range
        ax.text(
            0.98,
            0.98,
            f"clipped to [{lo:.4g}, {hi:.4g}]",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_dir", type=str, default="/data/jiaji_lu/Infinity/8b_QKV/ori/K")
    parser.add_argument("--v_dir", type=str, default="/data/jiaji_lu/Infinity/8b_QKV/ori/V")
    parser.add_argument("--idx", type=int, default=39)
    parser.add_argument("--key", type=str, default="data")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1, help="展平后下采样步长（1=全量；10=取 1/10 点）")
    parser.add_argument("--clip_q", type=float, default=None, help="已弃用：该脚本默认不做裁剪（传了也会忽略）")
    parser.add_argument("--logy", action="store_true", help="已弃用：该脚本固定使用线性 y 轴（传了也会忽略）")
    args = parser.parse_args()

    k_path = os.path.join(args.k_dir, f"K_ori_{args.idx}.npz")
    v_path = os.path.join(args.v_dir, f"V_ori_{args.idx}.npz")
    if not os.path.exists(k_path):
        raise FileNotFoundError(k_path)
    if not os.path.exists(v_path):
        raise FileNotFoundError(v_path)

    # matplotlib 可能不在所有环境默认装好：给出清晰报错
    try:
        import matplotlib  # noqa: F401
        _set_research_rcparams()
    except Exception as e:
        raise RuntimeError(
            "缺少 matplotlib（或无法导入）。请在你的科研环境里安装/启用 matplotlib 后再运行该脚本。"
        ) from e

    K = _load_npz(k_path, args.key)
    V = _load_npz(v_path, args.key)

    if args.logy:
        print("[Warn] --logy 已弃用且会被忽略：该脚本固定使用线性 y 轴（frequency）")

    k1 = _as_1d(K, args.stride).astype(np.float32, copy=False)
    v1 = _as_1d(V, args.stride).astype(np.float32, copy=False)

    # 用户需求：不做 clip，显示完整分布（即使传了 --clip_q 也忽略）
    if args.clip_q is not None and float(args.clip_q) != 1.0:
        print(f"[Warn] --clip_q 已弃用且会被忽略：clip_q={args.clip_q}（将展示完整分布）")
    k_plot, k_clip = k1, None
    v_plot, v_clip = v1, None

    os.makedirs(args.out_dir, exist_ok=True)

    _plot_hist(
        k_plot,
        bins=args.bins,
        title=f"K distribution (idx={args.idx})",
        xlabel="K value",
        out_png=os.path.join(args.out_dir, f"K_hist_{args.idx}.png"),
        out_pdf=os.path.join(args.out_dir, f"K_hist_{args.idx}.pdf"),
        logy=False,
        clip_range=k_clip,
    )
    _plot_hist(
        v_plot,
        bins=args.bins,
        title=f"V distribution (idx={args.idx})",
        xlabel="V value",
        out_png=os.path.join(args.out_dir, f"V_hist_{args.idx}.png"),
        out_pdf=os.path.join(args.out_dir, f"V_hist_{args.idx}.pdf"),
        logy=False,
        clip_range=v_clip,
    )

    meta: Dict[str, Any] = {
        "k_path": k_path,
        "v_path": v_path,
        "key": args.key,
        "idx": int(args.idx),
        "bins": int(args.bins),
        "stride": int(args.stride),
        "clip_q": None,
        "logy": False,
        "K_shape": list(K.shape),
        "V_shape": list(V.shape),
        "K_dtype": str(K.dtype),
        "V_dtype": str(V.dtype),
        "K_minmax_mean_std": [float(k1.min()), float(k1.max()), float(k1.mean()), float(k1.std())],
        "V_minmax_mean_std": [float(v1.min()), float(v1.max()), float(v1.mean()), float(v1.std())],
        "K_clip_range": None if k_clip is None else [k_clip[0], k_clip[1]],
        "V_clip_range": None if v_clip is None else [v_clip[0], v_clip[1]],
    }
    with open(os.path.join(args.out_dir, f"meta_{args.idx}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] saved:")
    print(" -", os.path.join(args.out_dir, f"K_hist_{args.idx}.png"))
    print(" -", os.path.join(args.out_dir, f"V_hist_{args.idx}.png"))
    print(" -", os.path.join(args.out_dir, f"meta_{args.idx}.json"))


if __name__ == "__main__":
    main()


