#!/usr/bin/env python3

"""
python /home/jiaji_lu/AR/VAR-Q/temp/visualize_qkv_hist.py --dump-dir /data/jiaji_lu/WM/infinitystar/qkv/sageattn_qk_int8_pv_fp16_triton/ --layer 35
"""



"""
从单层（block）的 Q/K/V 展平后画直方图（柱状，300 bins），各输出一张 PNG。

默认与 InfinityStar 的 dump 一致（环境变量 INFINITYSTAR_QKV_DUMP_DIR 或 --dump-dir）：
  每个 block 一个 ``block_{idx:02d}.pt``，内容为 dict，含 ``q`` / ``k`` / ``v``（bf16 张量），
  ``layout`` 为 ``B_H_L_D``。见 InfinityStar ``infinity/models/basic.py`` 中 INFINITYSTAR_QKV_DUMP。

仍支持旧版 analyze_kv 风格的三个独立 ``.npz``（键 ``data``）：用 ``--q/--k/--v`` 指定。

示例：
  python visualize_qkv_hist.py --list-blocks
  python visualize_qkv_hist.py --layer 19
  python visualize_qkv_hist.py --dump-dir /path/to/dump --layer 19
  python visualize_qkv_hist.py --pt /path/to/block_19.pt
  python visualize_qkv_hist.py --rule 'per-head+per-dim' --layer 19
  python visualize_qkv_hist.py --q /path/q.npz --k /path/k.npz --v /path/v.npz --out-dir .
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch


def _default_dump_dir() -> str:
    d = os.environ.get("INFINITYSTAR_QKV_DUMP_DIR", "").strip()
    if d:
        return d
    return "/data/jiaji_lu/WM/Infinitystar/qkv/sageattn_qk_int8_pv_fp16_triton"


_BLOCK_PT_RE = re.compile(r"^block_(\d+)\.pt$", re.IGNORECASE)


def scan_block_pt_files(dump_dir: str) -> list[tuple[int, str]]:
    """列出 ``dump_dir`` 下 ``block_<整数>.pt``，返回按索引排序的 (idx, path)。"""
    if not os.path.isdir(dump_dir):
        return []
    out: list[tuple[int, str]] = []
    for name in os.listdir(dump_dir):
        m = _BLOCK_PT_RE.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(dump_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def resolve_block_pt(dump_dir: str, layer: int) -> str:
    """
    解析某层的 ``.pt`` 路径：先试 ``block_{layer:02d}.pt``、``block_{layer}.pt``，
    再在目录中扫描 ``block_*.pt`` 按索引匹配。
    """
    layer = int(layer)
    if not dump_dir:
        raise FileNotFoundError("dump 目录为空；请设置 --dump-dir 或 INFINITYSTAR_QKV_DUMP_DIR。")
    if not os.path.isdir(dump_dir):
        raise FileNotFoundError(
            f"dump 目录不存在: {dump_dir}\n"
            "请把 --dump-dir 或环境变量 INFINITYSTAR_QKV_DUMP_DIR 设成你实际保存 block_*.pt 的目录 "
            "（例如推理时终端里 du -sh 的那个目录）。"
        )

    candidates = [
        os.path.join(dump_dir, f"block_{layer:02d}.pt"),
        os.path.join(dump_dir, f"block_{layer}.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    scanned = scan_block_pt_files(dump_dir)
    by_idx = {idx: p for idx, p in scanned}
    if layer in by_idx:
        return by_idx[layer]

    avail = [idx for idx, _ in scanned]
    lines = [
        f"找不到 layer={layer} 对应的文件（已尝试: {candidates[0]} 与 {candidates[1]}）。",
    ]
    if avail:
        lines.append(f"该目录下已有的 block 索引: {avail}（共 {len(avail)} 个）。")
        lines.append(f"可改用例如: --layer {avail[-1]}（最后一层）。")
    else:
        lines.append(f"该目录下未发现任何 block_<n>.pt；请确认 dump 根目录是否选对、文件是否尚未生成。")
    lines.append("若文件在子目录中，请把 --dump-dir 指到包含 block_*.pt 的那一层，或用 --pt 直接指定 .pt 路径。")
    raise FileNotFoundError("\n".join(lines))


def default_paths_for_layer(layer: int) -> tuple[str, str, str]:
    """与 AR/VAR_Q/analyze_kv.py 中 ori 路径一致（独立 npz，旧流程）。"""
    return (
        f"/data/jiaji_lu/kv/q/q_ori_{layer}.npz",
        f"/data/jiaji_lu/kv/ori/k_ori_{layer}.npz",
        f"/data/jiaji_lu/kv/ori/v_ori_{layer}.npz",
    )


def rule_paths_for_layer(rule: str, layer: int) -> tuple[str, str, str]:
    """与 analyze_kv.py 中带 rule 的量化路径一致。"""
    return (
        f"/data/jiaji_lu/kv/q/q_q_{rule}_{layer}.npz",
        f"/data/jiaji_lu/kv/{rule}/q_k_{rule}_{layer}.npz",
        f"/data/jiaji_lu/kv/{rule}/q_v_{rule}_{layer}.npz",
    )


def _tensor_to_flat_f64(t: torch.Tensor | object) -> np.ndarray:
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.detach().to(torch.float64).numpy().ravel()


def load_pt_dict(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_flat(path: str, *, tensor_key: str | None = None) -> np.ndarray:
    """从 ``.npz``（键 ``data``）或 ``.pt``（dict 中的 ``q``/``k``/``v``）读入并展平为 float64。"""
    if path.endswith(".pt"):
        if not tensor_key:
            raise ValueError(".pt 需要指定 tensor_key（q/k/v）")
        obj = load_pt_dict(path)
        return _tensor_to_flat_f64(obj[tensor_key])
    arr = np.load(path)["data"]
    return np.asarray(arr, dtype=np.float64).ravel()


def _format_stat(x: float) -> str:
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-4):
        return f"{x:.6e}"
    return f"{x:.6g}"


def plot_hist_bar(
    values: np.ndarray,
    bins: int,
    title: str,
    out_path: str,
    figsize: tuple[float, float] = (11, 6),
    show_stats: bool = True,
) -> None:
    v = np.asarray(values, dtype=np.float64).ravel()
    mean = float(np.mean(v))
    std = float(np.std(v))
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    lo3 = mean - 3.0 * std
    hi3 = mean + 3.0 * std

    counts, edges = np.histogram(v, bins=bins)
    width = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=figsize)
    if show_stats:
        ax.axvspan(lo3, hi3, color="gold", alpha=0.18, zorder=0)
    ax.bar(centers, counts, width=width * 0.98, align="center", edgecolor="none", zorder=2, color="steelblue", alpha=0.85)
    if show_stats:
        ax.axvline(mean, color="crimson", linewidth=1.4, zorder=3)
        ax.axvline(vmin, color="#444", linestyle="--", linewidth=1.1, zorder=3, alpha=0.85)
        ax.axvline(vmax, color="#444", linestyle="--", linewidth=1.1, zorder=3, alpha=0.85)
        ax.axvline(lo3, color="darkgreen", linestyle=":", linewidth=1.2, zorder=3)
        ax.axvline(hi3, color="darkgreen", linestyle=":", linewidth=1.2, zorder=3)
        stats_txt = (
            f"mean = {_format_stat(mean)}\n"
            f"std  = {_format_stat(std)}\n"
            f"min  = {_format_stat(vmin)}\n"
            f"max  = {_format_stat(vmax)}\n"
            f"3σ   = [{_format_stat(lo3)}, {_format_stat(hi3)}]"
        )
        ax.text(
            0.98,
            0.97,
            stats_txt,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="floralwhite", edgecolor="0.65", alpha=0.95),
        )
        leg_handles = [
            Line2D([0], [0], color="crimson", linewidth=1.4, label="mean"),
            Line2D([0], [0], color="#444", linestyle="--", linewidth=1.1, label="min / max"),
            Line2D([0], [0], color="darkgreen", linestyle=":", linewidth=1.2, label="μ ± 3σ"),
            Patch(facecolor="gold", alpha=0.35, edgecolor="none", label="μ±3σ range"),
        ]
        ax.legend(handles=leg_handles, loc="upper left", fontsize=8, framealpha=0.92)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q/K/V distribution histogram（300 bins）")
    parser.add_argument("--layer", type=int, default=None, help="层索引；与默认模板联用。默认取最后一层（见 --last-layer）")
    parser.add_argument(
        "--last-layer",
        type=int,
        default=19,
        help="当未指定 --layer 且未显式给 npz 路径时，使用的「最后一层」索引（默认 19，对应 20 层时最后一层）",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default=None,
        help="若设置则使用 analyze_kv 中带 rule 的 q/k/v 路径模板（独立 .npz），而非 InfinityStar 的 block_*.pt",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="InfinityStar QKV dump 根目录；默认取环境变量 INFINITYSTAR_QKV_DUMP_DIR，否则为 WM/Infinitystar/qkv/sageattn_qk_int8_pv_fp16_triton",
    )
    parser.add_argument(
        "--legacy-npz",
        action="store_true",
        help="使用旧路径模板 /data/jiaji_lu/kv/... 下的 q_ori/k_ori/v_ori 独立 npz，而非 block_XX.pt",
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="直接指定单个 block_*.pt（与 --dump-dir/--layer 二选一；优先级最高）",
    )
    parser.add_argument(
        "--list-blocks",
        action="store_true",
        help="列出 --dump-dir（或默认目录）下的 block_*.pt 后退出",
    )
    parser.add_argument("--q", type=str, default=None, help="Q 文件路径（.npz 键 data，或 .pt 键 q）")
    parser.add_argument("--k", type=str, default=None, help="K 文件路径（.npz 键 data，或 .pt 键 k）")
    parser.add_argument("--v", type=str, default=None, help="V 文件路径（.npz 键 data，或 .pt 键 v）")
    parser.add_argument("--bins", type=int, default=300, help="直方图 bin 数量")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__))),
        help="输出 PNG 目录（默认为本脚本所在目录，即 VAR-Q/temp）",
    )
    parser.add_argument("--prefix", type=str, default="qkv_dist", help="输出文件名前缀")
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="不绘制 mean/std/3σ/min/max 标注与参考线（仅柱状图）",
    )
    args = parser.parse_args()

    dump_dir_for_list = args.dump_dir if args.dump_dir is not None else _default_dump_dir()
    if args.list_blocks:
        scanned = scan_block_pt_files(dump_dir_for_list)
        if not os.path.isdir(dump_dir_for_list):
            print(f"目录不存在: {dump_dir_for_list}", file=sys.stderr)
            sys.exit(1)
        print(f"dump_dir: {os.path.abspath(dump_dir_for_list)}")
        if not scanned:
            print("未发现 block_<n>.pt 文件。")
        else:
            for idx, p in scanned:
                print(f"  block_{idx:02d} -> {p}")
        sys.exit(0)

    if args.q and args.k and args.v:
        q_path, k_path, v_path = args.q, args.k, args.v
        layer_tag = "custom"

        def _flat_for(role: str, p: str) -> np.ndarray:
            if p.endswith(".pt"):
                return load_flat(p, tensor_key=role)
            return load_flat(p)

        specs = [
            ("q", q_path, _flat_for("q", q_path)),
            ("k", k_path, _flat_for("k", k_path)),
            ("v", v_path, _flat_for("v", v_path)),
        ]
    else:
        if args.q or args.k or args.v:
            parser.error("请同时提供 --q、--k、--v，或三者都不提供以使用默认路径模板")
        layer = args.last_layer if args.layer is None else args.layer
        if args.rule:
            q_path, k_path, v_path = rule_paths_for_layer(args.rule, layer)
            layer_tag = f"{layer}_{args.rule}"
            specs = [
                ("q", q_path, load_flat(q_path)),
                ("k", k_path, load_flat(k_path)),
                ("v", v_path, load_flat(v_path)),
            ]
        elif args.legacy_npz:
            q_path, k_path, v_path = default_paths_for_layer(layer)
            layer_tag = str(layer)
            specs = [
                ("q", q_path, load_flat(q_path)),
                ("k", k_path, load_flat(k_path)),
                ("v", v_path, load_flat(v_path)),
            ]
        else:
            dump_dir = args.dump_dir if args.dump_dir is not None else _default_dump_dir()
            if args.pt:
                pt_path = os.path.abspath(args.pt)
                if not os.path.isfile(pt_path):
                    raise FileNotFoundError(f"找不到 --pt 文件: {pt_path}")
                layer_tag = str(layer) if args.layer is not None else os.path.splitext(os.path.basename(pt_path))[0]
            else:
                pt_path = resolve_block_pt(dump_dir, layer)
                layer_tag = str(layer)
            pt_obj = load_pt_dict(pt_path)
            specs = [
                ("q", pt_path, _tensor_to_flat_f64(pt_obj["q"])),
                ("k", pt_path, _tensor_to_flat_f64(pt_obj["k"])),
                ("v", pt_path, _tensor_to_flat_f64(pt_obj["v"])),
            ]

    os.makedirs(args.out_dir, exist_ok=True)

    for name, path, flat in specs:
        out_png = os.path.join(args.out_dir, f"{args.prefix}_{name}_layer{layer_tag}.png")
        plot_hist_bar(
            flat,
            bins=args.bins,
            title=f"{name.upper()} distribution (layer={layer_tag}, n={flat.size}, bins={args.bins})",
            out_path=out_png,
            show_stats=not args.no_stats,
        )
        print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
