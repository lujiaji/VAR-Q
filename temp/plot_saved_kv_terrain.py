#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""


python /home/jiaji_lu/AR/VAR-Q/temp/plot_saved_kv_terrain.py   --dump-dir /data/jiaji_lu/WM/infinitystar/qkv/sageattn_qk_int8_pv_fp16_triton   --layer 35   --mode qkv   --b 0   --out-dir /home/jiaji_lu/AR/VAR-Q/temp/tmp   --seq_stride 5   --feat_stride 2   --transform none



"""



"""
绘制 Q/K/V 的 3D“地形图”(terrain/surface)，形态类似：
  x: Feature Dim (H×D)，即 H 与 head 维 c 展平
  y: Sequence (L)
  z: Value

数据源（二选一）：
  A) InfinityStar dump：目录下 ``block_{layer:02d}.pt``，dict 含 ``q``/``k``/``v``，形状 ``B_H_L_D``（即 BHLc）
  B) 旧版 npz：``K_ori_{idx}.npz`` / ``V_ori_{idx}.npz``，key 默认 ``data``

原始网格很大时需用 ``--seq_stride`` / ``--feat_stride`` 下采样。
可选剪裁：``--feat_range a:b`` 只取特征子区间；``--max_seq N`` 只取前 N 个 token。
二者默认不写表示 **全特征维 (0:H×D)**、**全序列长 L**（仍受 stride 稀疏采样，非逐元素）。

示例（pt dump，输出到本脚本旁 ``tmp/``，画 Q+K+V）：
  python plot_saved_kv_terrain.py \\
    --dump-dir /data/jiaji_lu/WM/infinitystar/qkv/sageattn_qk_int8_pv_fp16_triton \\
    --layer 35 --mode qkv --b 0 \\
    --seq_stride 20 --feat_stride 8 \\
    --transform none

示例（旧 npz，只画 K/V）：
  python plot_saved_kv_terrain.py --idx 39 --mode both \\
    --k_dir .../K --v_dir .../V --out_dir ./tmp --seq_stride 20 --feat_stride 8
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TMP = os.path.join(_SCRIPT_DIR, "tmp")

_BLOCK_PT_RE = re.compile(r"^block_(\d+)\.pt$", re.IGNORECASE)


def scan_block_pt_files(dump_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(dump_dir):
        return []
    out: List[Tuple[int, str]] = []
    for name in os.listdir(dump_dir):
        m = _BLOCK_PT_RE.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(dump_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def resolve_block_pt(dump_dir: str, layer: int) -> str:
    layer = int(layer)
    if not dump_dir:
        raise FileNotFoundError("需要 --dump-dir 或 --pt。")
    if not os.path.isdir(dump_dir):
        raise FileNotFoundError(f"dump 目录不存在: {dump_dir}")
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
    msg = f"找不到 block_{layer:02d}.pt；目录已有索引: {avail}"
    raise FileNotFoundError(msg)


def load_pt_dict(path: str) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("加载 .pt 需要安装 torch。")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pt_tensor_to_numpy_bhld(t: Any) -> np.ndarray:
    if torch is not None and isinstance(t, torch.Tensor):
        x = t.detach().cpu().float().numpy()
    else:
        x = np.asarray(t, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"期望 Q/K/V 形状 (B,H,L,D)，实际 {x.shape}")
    return x


def _set_research_rcparams():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def _load_npz(path: str, key: str) -> np.ndarray:
    z = np.load(path)
    if key not in z.files:
        raise KeyError(f"{path} 内不包含 key={key}，实际 keys={z.files}")
    return z[key]


def _to_LF(a_bhld: np.ndarray, b: int) -> np.ndarray:
    """
    (B,H,L,D) -> (L, F=H*D) for batch index b
    """
    if a_bhld.ndim != 4:
        raise ValueError(f"期望 4 维 (B,H,L,D)，实际 shape={a_bhld.shape}")
    B, H, L, D = a_bhld.shape
    if not (0 <= b < B):
        raise ValueError(f"b={b} 越界，B={B}")
    x_hld = a_bhld[b]  # (H,L,D)
    x_lhd = np.transpose(x_hld, (1, 0, 2))  # (L,H,D)
    return x_lhd.reshape(L, H * D)  # (L,F)


def _transform_values(x: np.ndarray, mode: str, scale: float) -> np.ndarray:
    """
    对数值做可选变换（用于增强可视化对比）。

    - none: 不变换
    - signed_log1p: sign(x) * log(1 + |x|/scale)   （支持负值）
    - log1p_pos: log(1 + max(x,0)/scale)           （仅正半轴；负值截为 0）
    """
    mode = str(mode or "none").lower()
    if mode in ("none", "raw", "identity"):
        return x
    s = float(scale)
    if s <= 0:
        raise ValueError(f"--transform_scale 必须 > 0，当前 {scale}")
    x32 = x.astype(np.float32, copy=False)
    if mode in ("signed_log1p", "signed_log"):
        return np.sign(x32) * np.log1p(np.abs(x32) / s)
    if mode in ("log1p_pos", "log_pos"):
        return np.log1p(np.maximum(x32, 0.0) / s)
    raise ValueError(f"未知 --transform 模式: {mode}")


def _ratio1_patch_nums_by_L(L: int) -> List[int]:
    """
    ratio=1 时常用的 patch_num 序列（h=w=patch_num）。
    通过 L=sum(p^2) 反推使用哪一档；如果不匹配则默认按 1M。
    """
    cand = {
        "0.06M": [1, 2, 4, 6, 8, 12, 16],                               # sum=425
        "0.25M": [1, 2, 4, 6, 8, 12, 16, 20, 24, 32],                   # sum=2521
        "1M":    [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64],       # sum=10521
    }
    for _, ps in cand.items():
        if int(L) == int(sum(p * p for p in ps)):
            return ps
    return cand["1M"]


def _scale_split_starts_from_patch_nums(patch_nums: List[int]) -> List[int]:
    """
    给定每个 scale 的 patch_num，返回每个 scale 的起始 token index（0-based）。
    例如 [1,2,4] -> [0,1,5]
    """
    starts = [0]
    acc = 0
    for p in patch_nums[:-1]:
        acc += int(p) * int(p)
        starts.append(acc)
    return starts


def _downsample(mat_lf: np.ndarray, seq_stride: int, feat_stride: int, max_seq: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (seq_idx, feat_idx, z) 其中 z 为 (len(seq_idx), len(feat_idx))
    """
    L, F = mat_lf.shape
    if max_seq is not None:
        L = min(L, int(max_seq))
        mat_lf = mat_lf[:L]

    seq_stride = max(1, int(seq_stride))
    feat_stride = max(1, int(feat_stride))
    seq_idx = np.arange(0, L, seq_stride, dtype=np.int32)
    feat_idx = np.arange(0, F, feat_stride, dtype=np.int32)
    z = mat_lf[np.ix_(seq_idx, feat_idx)].astype(np.float32, copy=False)
    return seq_idx, feat_idx, z


def _parse_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    解析 "a:b"（半开区间 [a,b)），返回 (a,b)。
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    if ":" not in t:
        raise ValueError(f"range 格式应为 a:b（半开区间），实际: {s}")
    a, b = t.split(":", 1)
    return int(a), int(b)


def _parse_ranges(s: Optional[str]) -> List[Tuple[int, int]]:
    """
    解析 "a:b,c:d" -> [(a,b),(c,d)]，均为半开区间 [a,b)。
    """
    if s is None:
        return []
    t = str(s).strip()
    if not t:
        return []
    parts = [p.strip() for p in t.split(",") if p.strip()]
    out: List[Tuple[int, int]] = []
    for p in parts:
        out.append(_parse_range(p))  # type: ignore[arg-type]
    return out


def _parse_int_list(s: Optional[str]) -> List[int]:
    """
    解析 "a,b,c" -> [a,b,c]
    """
    if s is None:
        return []
    t = str(s).strip()
    if not t:
        return []
    out: List[int] = []
    for p in t.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _build_feat_idx(
    *,
    F: int,
    feat_stride: int,
    feat_range: Optional[Tuple[int, int]],
    extra_feats: List[int],
) -> np.ndarray:
    """
    构建 feature index 网格（用于 x 轴），并确保高亮 feature 一定被包含（即使 stride 采样跳过了它）。
    """
    feat_stride = max(1, int(feat_stride))
    if feat_range is None:
        lo, hi = 0, F
    else:
        lo, hi = feat_range
    lo = max(0, int(lo))
    hi = min(F, int(hi))
    if not (0 <= lo < hi <= F):
        raise ValueError(f"feat_range 越界或无效：{feat_range}，F={F}")

    base = np.arange(lo, hi, feat_stride, dtype=np.int32)
    if extra_feats:
        extras = np.asarray([f for f in extra_feats if lo <= f < hi], dtype=np.int32)
        if extras.size:
            feat_idx = np.unique(np.concatenate([base, extras], axis=0))
            feat_idx.sort()
            return feat_idx
    return base


def _build_seq_idx(
    *,
    L: int,
    seq_stride: int,
    max_seq: Optional[int],
    extra_seq: List[int],
) -> np.ndarray:
    """
    构建 sequence index 网格（用于 y 轴），并确保高亮 sequence 边界一定被包含。
    """
    if max_seq is not None:
        L_eff = min(int(max_seq), int(L))
    else:
        L_eff = int(L)
    if L_eff <= 0:
        raise ValueError(f"无效的 L_eff={L_eff}")

    seq_stride = max(1, int(seq_stride))
    base = np.arange(0, L_eff, seq_stride, dtype=np.int32)
    if extra_seq:
        extras = np.asarray([s for s in extra_seq if 0 <= s < L_eff], dtype=np.int32)
        if extras.size:
            seq_idx = np.unique(np.concatenate([base, extras], axis=0))
            seq_idx.sort()
            return seq_idx
    return base


def _plot_surface(
    *,
    seq_idx: np.ndarray,
    feat_idx: np.ndarray,
    z: np.ndarray,
    # title: str,
    out_png: str,
    out_pdf: str,
    elev: float,
    azim: float,
    cmap: str,
    highlight_feats: Optional[List[int]] = None,
    highlight_ranges: Optional[List[Tuple[int, int]]] = None,
    highlight_seq_ranges: Optional[List[Tuple[int, int]]] = None,
    highlight_lift_frac: float = 0.12,
    aux_seq_ranges: Optional[List[Tuple[int, int]]] = None,
    scale_split_ys: Optional[List[int]] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.colors import Normalize

    # meshgrid: X=feature, Y=sequence
    X, Y = np.meshgrid(feat_idx, seq_idx)
    # 为避免高亮层与底图共面导致的 z-fighting，给高亮层一个很小的 z-offset
    z_range = float(np.nanmax(z) - np.nanmin(z))
    z_eps = (0.005 * z_range) if z_range > 0 else 1e-4
    # “浮空”抬升：让标注区域看起来像从底图抽出来悬浮在上方
    lift_frac = float(highlight_lift_frac)
    z_lift = (lift_frac * z_range) if z_range > 0 else (20.0 * z_eps)
    # 同时把底图整体下移一点、标注层整体上移一点，确保“标注在最顶层”不会被遮挡
    z_base = z - z_eps
    # 统一颜色归一化：按“未上移前的原始 z 数值”做映射
    # 需求：上移只改变几何位置（z+lift），颜色仍保持未上移时的 colormap 对应关系
    norm = Normalize(vmin=float(np.nanmin(z)), vmax=float(np.nanmax(z)))
    # norm = Normalize(vmin=-2, vmax=2)

    cmap_obj = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(11.0, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # 颜色与几何解耦：用未上移的 z 计算 facecolors；用 z_base / z+lift 作为几何高度
    face_base = cmap_obj(norm(z))
    face_base[..., 3] = 0.50  # 未标注区域透明度 70%
    surf = ax.plot_surface(
        X,
        Y,
        z_base,
        facecolors=face_base,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    # ax.set_title(title, pad=12)
    ax.set_xlabel("Feature Dim (H×D)")
    ax.set_ylabel("Sequence (S)")
    ax.set_zlabel("Value")
    ax.view_init(elev=elev, azim=azim)

    # ---------- 辅助边界面：用于标记标注/分割边界 ----------
    z_min = float(np.nanmin(z_base))
    # 标注区域边界辅助面：黑色
    aux_color = "#D00000"
    aux_alpha = 0.3
    # scale 分割/辅助线：红色 + 更高不透明度 + 更粗轮廓线
    split_color = "#D00000"
    split_plane_alpha = 0.1
    split_line_alpha = 0.1
    split_linewidth = 0.5

    def _plane_x(x0: int):
        # 垂直平面：x = x0, 从底部 z_min 拉到该 x 处“浮空平面”的局部高度
        cols = np.where(feat_idx == int(x0))[0]
        if not cols.size:
            return
        c = int(cols[0])
        z_top_col = (z[:, c].astype(np.float32, copy=False) + z_eps + z_lift)
        Yp = np.repeat(seq_idx[:, None], 2, axis=1).astype(np.float32)
        Xp = np.full_like(Yp, float(x0), dtype=np.float32)
        Zp = np.stack(
            [np.full_like(seq_idx, z_min, dtype=np.float32), z_top_col],
            axis=1,
        )
        ax.plot_surface(Xp, Yp, Zp, color=aux_color, alpha=aux_alpha, linewidth=0, shade=False, antialiased=False)

    def _plane_y(y0: int):
        # 垂直平面：y = y0, 从底部 z_min 拉到该 y 处“浮空平面”的局部高度
        rows = np.where(seq_idx == int(y0))[0]
        if not rows.size:
            return
        r = int(rows[0])
        z_top_row = (z[r, :].astype(np.float32, copy=False) + z_eps + z_lift)
        Xp = np.repeat(feat_idx[None, :], 2, axis=0).astype(np.float32)
        Yp = np.full_like(Xp, float(y0), dtype=np.float32)
        Zp = np.stack(
            [np.full_like(feat_idx, z_min, dtype=np.float32), z_top_row],
            axis=0,
        )
        ax.plot_surface(Xp, Yp, Zp, color=aux_color, alpha=aux_alpha, linewidth=0, shade=False, antialiased=False)

    # sequence(你说的 scale) 区间边界（每个区间两条）
    if aux_seq_ranges:
        for (a, b) in aux_seq_ranges:
            s_lo, s_hi = int(a), int(b) - 1
            if s_lo in set(map(int, seq_idx.tolist())):
                _plane_y(s_lo)
            if s_hi in set(map(int, seq_idx.tolist())):
                _plane_y(s_hi)

    # ratio=1 的 scale 切分：用“幕布面”从底部拉到地形面，显示每个 scale 分段
    if scale_split_ys:
        for i, y0 in enumerate(scale_split_ys):
            # 不显示前 5 个 scale 的辅助线/幕布面
            if i < 5:
                continue
            y0 = int(y0)
            rows = np.where(seq_idx == y0)[0]
            if not rows.size:
                continue
            r = int(rows[0])
            Xc = np.repeat(feat_idx[None, :].astype(np.float32), 2, axis=0)
            Yc = np.full_like(Xc, float(y0), dtype=np.float32)
            # scale 分割线要求“顶到原始平面”，不用浮空高度
            z_top_row = z_base[r, :].astype(np.float32, copy=False)
            Zc = np.stack(
                [np.full_like(feat_idx, z_min, dtype=np.float32), z_top_row],
                axis=0,
            )
            # 更透明的幕布面 + 更明显的橙色轮廓线
            ax.plot_surface(Xc, Yc, Zc, color=split_color, alpha=split_plane_alpha, linewidth=0, shade=False, antialiased=False)
            ax.plot(
                feat_idx.astype(np.float32),
                np.full_like(feat_idx, float(y0), dtype=np.float32),
                z_top_row,
                color=split_color,
                alpha=split_line_alpha,
                linewidth=split_linewidth,
            )

    # ---------- 连接墙面：让浮空区域更像“从底图中抽出来” ----------
    wall_alpha = 0.2
    wall_color = "#D00000"

    def _wall_feat(x0: int):
        # 在 feature=x0 位置，把底图与浮空高度连接起来（沿 sequence）
        cols = np.where(feat_idx == int(x0))[0]
        if not cols.size:
            return
        c = int(cols[0])
        Z0 = z_base[:, c].astype(np.float32, copy=False)
        Z1 = (z[:, c] + z_eps + z_lift).astype(np.float32, copy=False)
        Xw = np.repeat(np.asarray([[float(x0), float(x0)]], dtype=np.float32), seq_idx.shape[0], axis=0)
        Yw = np.repeat(seq_idx[:, None].astype(np.float32), 2, axis=1)
        Zw = np.stack([Z0, Z1], axis=1)
        ax.plot_surface(Xw, Yw, Zw, color=wall_color, alpha=wall_alpha, linewidth=0, shade=False, antialiased=False)

    def _wall_seq(y0: int):
        # 在 sequence=y0 位置，把底图与浮空高度连接起来（沿 feature）
        rows = np.where(seq_idx == int(y0))[0]
        if not rows.size:
            return
        r = int(rows[0])
        Z0 = z_base[r, :].astype(np.float32, copy=False)
        Z1 = (z[r, :] + z_eps + z_lift).astype(np.float32, copy=False)
        Xw = np.repeat(feat_idx[None, :].astype(np.float32), 2, axis=0)
        Yw = np.full_like(Xw, float(y0), dtype=np.float32)
        Zw = np.stack([Z0, Z1], axis=0)
        ax.plot_surface(Xw, Yw, Zw, color=wall_color, alpha=wall_alpha, linewidth=0, shade=False, antialiased=False)

    if z_lift > 0:
        # 对 feature 高亮区间边界做连接墙（两条）
        if highlight_ranges:
            for (a, b) in highlight_ranges:
                _wall_feat(int(a))
                _wall_feat(int(b) - 1)
        # 对 sequence 高亮区间边界做连接墙（两条）
        if highlight_seq_ranges:
            for (a, b) in highlight_seq_ranges:
                _wall_seq(int(a))
                _wall_seq(int(b) - 1)

    # --- 高亮：多个 feature（多条曲线）---
    if highlight_feats:
        # 用同一个 colormap 给线段着色（随 z 变化），且线条完全不透明
        for i, hf in enumerate(highlight_feats):
            hf = int(hf)
            cols = np.where(feat_idx == hf)[0]
            if not cols.size:
                continue
            c = int(cols[0])
            xs = np.full_like(seq_idx, hf, dtype=np.float32)
            ys = seq_idx.astype(np.float32)
            zs = (z[:, c] + z_eps + z_lift).astype(np.float32, copy=False)
            # segments: (N-1, 2, 3)
            pts0 = np.stack([xs[:-1], ys[:-1], zs[:-1]], axis=1)
            pts1 = np.stack([xs[1:], ys[1:], zs[1:]], axis=1)
            segs = np.stack([pts0, pts1], axis=1)
            # 颜色按“未上移前的原始 z 值”计算
            zc = z[:, c].astype(np.float32, copy=False)
            seg_z = (zc[:-1] + zc[1:]) * 0.5
            colors = cmap_obj(norm(seg_z))
            lc = Line3DCollection(segs, colors=colors, linewidths=2.0, alpha=1.0)
            ax.add_collection3d(lc)

    # --- 高亮：feature 区间（叠加半透明 surface）---
    if highlight_ranges:
        for (a, b) in highlight_ranges:
            a_i, b_i = int(a), int(b)
            if b_i <= a_i:
                continue
            m = (feat_idx >= a_i) & (feat_idx < b_i)
            if not np.any(m):
                continue
            X2, Y2 = np.meshgrid(feat_idx[m], seq_idx)
            Z2 = z[:, m] + z_eps + z_lift
            face2 = cmap_obj(norm(z[:, m]))
            face2[..., 3] = 1.0  # 标注区域完全不透明
            ax.plot_surface(
                X2,
                Y2,
                Z2,
                facecolors=face2,
                linewidth=0,
                antialiased=True,
                shade=True,
            )

    # --- 高亮：sequence 区间整片（跨所有 feature，叠加半透明 surface）---
    if highlight_seq_ranges:
        for (a, b) in highlight_seq_ranges:
            a_i, b_i = int(a), int(b)
            if b_i <= a_i:
                continue
            m = (seq_idx >= a_i) & (seq_idx < b_i)
            if not np.any(m):
                continue
            X3, Y3 = np.meshgrid(feat_idx, seq_idx[m])
            Z3 = z[m, :] + z_eps + z_lift
            face3 = cmap_obj(norm(z[m, :]))
            face3[..., 3] = 1.0  # 标注区域完全不透明
            ax.plot_surface(
                X3,
                Y3,
                Z3,
                facecolors=face3,
                linewidth=0,
                antialiased=True,
                shade=True,
            )

    # colorbar
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.06)
    cbar.set_label("Value")

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def _parse_mode_tags(mode: str) -> List[str]:
    """mode -> list of 'q'|'k'|'v'"""
    m = str(mode).strip().lower()
    presets = {
        "q": ["q"],
        "k": ["k"],
        "v": ["v"],
        "qk": ["q", "k"],
        "qv": ["q", "v"],
        "kv": ["k", "v"],
        "both": ["k", "v"],
        "qkv": ["q", "k", "v"],
        "all": ["q", "k", "v"],
    }
    if m in presets:
        return presets[m]
    raise ValueError(f"未知 --mode: {mode}；可选: q,k,v,qk,qv,kv,qkv,all,both")


def run_terrain(
    tag: str,
    a_bhld: np.ndarray,
    *,
    out_tag: str,
    source_ref: str,
    args: Any,
    feat_range: Optional[Tuple[int, int]],
    hl_feats: List[int],
    hl_ranges: List[Tuple[int, int]],
    hl_seq_ranges: List[Tuple[int, int]],
    meta: Dict[str, Any],
) -> None:
    mat_raw = _to_LF(a_bhld, args.b)
    mat = _transform_values(mat_raw, mode=args.transform, scale=args.transform_scale)

    extra_feats: List[int] = []
    if hl_feats:
        extra_feats.extend([int(x) for x in hl_feats])
    if hl_ranges:
        for (aa, bb) in hl_ranges:
            extra_feats.extend([int(aa), int(bb - 1)])

    L, F = mat.shape
    feat_idx = _build_feat_idx(F=F, feat_stride=args.feat_stride, feat_range=feat_range, extra_feats=extra_feats)

    extra_seq: List[int] = []
    if hl_seq_ranges:
        for (aa, bb) in hl_seq_ranges:
            extra_seq.extend([int(aa), int(bb - 1)])

    scale_split_ys: Optional[List[int]] = None
    if tag == "v" and args.v_show_ratio1_scale_splits:
        patch_nums = _ratio1_patch_nums_by_L(L)
        starts = _scale_split_starts_from_patch_nums(patch_nums)
        scale_split_ys = [s for s in starts[1:] if 0 <= s < L]
        extra_seq.extend(scale_split_ys)

    seq_idx = _build_seq_idx(L=L, seq_stride=args.seq_stride, max_seq=args.max_seq, extra_seq=extra_seq)
    z = mat[np.ix_(seq_idx, feat_idx)].astype(np.float32, copy=False)

    out_png = os.path.join(args.out_dir, f"{tag.upper()}_terrain_{out_tag}.png")
    out_pdf = os.path.join(args.out_dir, f"{tag.upper()}_terrain_{out_tag}.pdf")
    _plot_surface(
        seq_idx=seq_idx,
        feat_idx=feat_idx,
        z=z,
        out_png=out_png,
        out_pdf=out_pdf,
        elev=args.elev,
        azim=args.azim,
        cmap=args.cmap,
        highlight_feats=hl_feats if hl_feats else None,
        highlight_ranges=hl_ranges if hl_ranges else None,
        highlight_seq_ranges=hl_seq_ranges if hl_seq_ranges else None,
        highlight_lift_frac=float(args.highlight_lift_frac),
        aux_seq_ranges=hl_seq_ranges if hl_seq_ranges else None,
        scale_split_ys=scale_split_ys,
    )
    meta[f"{tag}_path"] = source_ref
    meta[f"{tag}_shape"] = list(a_bhld.shape)
    meta[f"{tag}_dtype"] = str(a_bhld.dtype)
    meta[f"{tag}_z_shape"] = list(z.shape)
    meta[f"{tag}_raw_minmax_mean_std"] = [
        float(mat_raw.min()),
        float(mat_raw.max()),
        float(mat_raw.mean()),
        float(mat_raw.std()),
    ]
    meta[f"{tag}_transformed_minmax_mean_std"] = [
        float(mat.min()),
        float(mat.max()),
        float(mat.mean()),
        float(mat.std()),
    ]
    print("[OK] saved:", out_png)


def main():
    parser = argparse.ArgumentParser(
        description="Q/K/V 3D 地形图：pt dump（block_*.pt）或 npz（K/V_ori）。下采样 --seq_stride/--feat_stride，剪裁 --feat_range/--max_seq，模式 --mode。"
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="InfinityStar QKV dump 目录（其下 block_<layer>.pt）；可与环境变量 INFINITYSTAR_QKV_DUMP_DIR 二选一",
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="直接指定单个 .pt（优先级高于 --dump-dir + --layer）",
    )
    parser.add_argument("--layer", type=int, default=None, help="pt 模式下的 block 层号（如 35 -> block_35.pt）")
    parser.add_argument("--k_dir", type=str, default="/data/jiaji_lu/Infinity/8b_QKV/ori/K")
    parser.add_argument("--v_dir", type=str, default="/data/jiaji_lu/Infinity/8b_QKV/ori/V")
    parser.add_argument("--idx", type=int, default=39, help="npz 模式：文件名索引 K_ori_{idx}.npz；pt 模式未给 --layer 时可作 block 号")
    parser.add_argument("--key", type=str, default="data", help="npz 数组键名")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=_DEFAULT_TMP,
        help=f"输出目录（默认：本脚本旁 tmp/ = {_DEFAULT_TMP}）",
    )
    parser.add_argument("--b", type=int, default=0, help="batch 下标（BHLc 中取第几个 batch）")

    parser.add_argument("--seq_stride", type=int, default=20, help="序列维 L 下采样步长（越大越快、越粗）")
    parser.add_argument("--feat_stride", type=int, default=8, help="特征维 H×D 下采样步长（越大越快、越粗）")
    parser.add_argument(
        "--max_seq",
        type=int,
        default=None,
        help="序列维剪裁：只画前 max_seq 个 token；默认不写表示使用全长 L",
    )
    parser.add_argument(
        "--feat_range",
        type=str,
        default=None,
        help="特征维截取 [a,b)（半开）；默认不写表示使用全特征维 0..H×D-1",
    )

    parser.add_argument("--q_highlight_feats", type=str, default=None, help="Q：高亮 feature 索引，逗号分隔")
    parser.add_argument("--q_highlight_ranges", type=str, default=None, help="Q：高亮 feature 区间 a:b,c:d（半开）")
    parser.add_argument("--q_highlight_seq_ranges", type=str, default=None, help="Q：高亮 sequence 区间 a:b")
    parser.add_argument("--k_highlight_feats", type=str, default=None, help="K：高亮 feature 索引")
    parser.add_argument("--k_highlight_ranges", type=str, default=None, help="K：高亮 feature 区间")
    parser.add_argument("--k_highlight_seq_ranges", type=str, default=None, help="K：高亮 sequence 区间")
    parser.add_argument("--v_highlight_feats", type=str, default=None, help="V：高亮 feature 索引")
    parser.add_argument("--v_highlight_ranges", type=str, default=None, help="V：高亮 feature 区间")
    parser.add_argument("--v_highlight_seq_ranges", type=str, default=None, help="V：高亮 sequence 区间")

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["q", "k", "v", "qk", "qv", "kv", "qkv", "all", "both"],
        help="绘制哪些张量：q/k/v 组合或 both(=K+V，兼容旧 npz)",
    )
    parser.add_argument("--elev", type=float, default=28.0, help="3D 视角 elevation")
    parser.add_argument("--azim", type=float, default=-62.0, help="3D 视角 azimuth")
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--highlight_lift_frac", type=float, default=0.8, help="高亮区域相对高度范围的抬升比例")
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["none", "signed_log1p", "log1p_pos"],
        help="数值模式：none 原始；signed_log1p 符号 log；log1p_pos 仅正部 log",
    )
    parser.add_argument("--transform_scale", type=float, default=1.0, help="log 类变换的分母 scale（|x|/scale）")
    parser.add_argument(
        "--v_show_ratio1_scale_splits",
        action="store_true",
        help="仅在 V 图上绘制 ratio=1 的 scale 分界幕布",
    )
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401

        _set_research_rcparams()
    except Exception as e:
        raise RuntimeError("缺少 matplotlib（或无法导入）。请在你的环境里安装/启用 matplotlib。") from e

    env_dump = os.environ.get("INFINITYSTAR_QKV_DUMP_DIR", "").strip()
    dump_dir = args.dump_dir or env_dump or None
    use_pt = bool(args.pt) or bool(dump_dir)

    tags = _parse_mode_tags(args.mode)
    if not use_pt and ("q" in tags):
        raise ValueError("当前为 npz 模式，数据中没有 Q；请改用 --dump-dir/--pt 加载 block_*.pt，或不要选含 q 的 --mode。")

    os.makedirs(args.out_dir, exist_ok=True)

    if use_pt:
        layer_or_idx = int(args.layer) if args.layer is not None else int(args.idx)
        out_tag = str(layer_or_idx)
    else:
        out_tag = str(int(args.idx))

    meta: Dict[str, Any] = {
        "input": "pt" if use_pt else "npz",
        "out_tag": out_tag,
        "idx": int(args.idx),
        "layer": args.layer,
        "key": args.key,
        "b": int(args.b),
        "seq_stride": int(args.seq_stride),
        "feat_stride": int(args.feat_stride),
        "max_seq": args.max_seq,
        "feat_range": args.feat_range,
        "elev": float(args.elev),
        "azim": float(args.azim),
        "cmap": args.cmap,
        "mode": args.mode,
        "highlight_lift_frac": float(args.highlight_lift_frac),
        "transform": args.transform,
        "transform_scale": float(args.transform_scale),
        "v_show_ratio1_scale_splits": bool(args.v_show_ratio1_scale_splits),
    }

    feat_range = _parse_range(args.feat_range)
    q_ranges = _parse_ranges(args.q_highlight_ranges)
    k_ranges = _parse_ranges(args.k_highlight_ranges)
    v_ranges = _parse_ranges(args.v_highlight_ranges)
    q_seq = _parse_ranges(args.q_highlight_seq_ranges)
    k_seq = _parse_ranges(args.k_highlight_seq_ranges)
    v_seq = _parse_ranges(args.v_highlight_seq_ranges)
    q_feats = _parse_int_list(args.q_highlight_feats)
    k_feats = _parse_int_list(args.k_highlight_feats)
    v_feats = _parse_int_list(args.v_highlight_feats)

    if use_pt:
        pt_path = os.path.abspath(args.pt) if args.pt else resolve_block_pt(dump_dir, layer_or_idx)
        meta["pt_path"] = pt_path
        obj = load_pt_dict(pt_path)
        q_np = pt_tensor_to_numpy_bhld(obj["q"])
        k_np = pt_tensor_to_numpy_bhld(obj["k"])
        v_np = pt_tensor_to_numpy_bhld(obj["v"])
        arr_map = {"q": q_np, "k": k_np, "v": v_np}
        hl_map = {
            "q": (q_feats, q_ranges, q_seq),
            "k": (k_feats, k_ranges, k_seq),
            "v": (v_feats, v_ranges, v_seq),
        }
        for tag in tags:
            run_terrain(
                tag,
                arr_map[tag],
                out_tag=out_tag,
                source_ref=pt_path,
                args=args,
                feat_range=feat_range,
                hl_feats=hl_map[tag][0],
                hl_ranges=hl_map[tag][1],
                hl_seq_ranges=hl_map[tag][2],
                meta=meta,
            )
    else:
        k_path = os.path.join(args.k_dir, f"K_ori_{args.idx}.npz")
        v_path = os.path.join(args.v_dir, f"V_ori_{args.idx}.npz")
        if "k" in tags and not os.path.exists(k_path):
            raise FileNotFoundError(k_path)
        if "v" in tags and not os.path.exists(v_path):
            raise FileNotFoundError(v_path)
        if "k" in tags:
            a = _load_npz(k_path, args.key)
            run_terrain(
                "k",
                a,
                out_tag=out_tag,
                source_ref=k_path,
                args=args,
                feat_range=feat_range,
                hl_feats=k_feats,
                hl_ranges=k_ranges,
                hl_seq_ranges=k_seq,
                meta=meta,
            )
        if "v" in tags:
            a = _load_npz(v_path, args.key)
            run_terrain(
                "v",
                a,
                out_tag=out_tag,
                source_ref=v_path,
                args=args,
                feat_range=feat_range,
                hl_feats=v_feats,
                hl_ranges=v_ranges,
                hl_seq_ranges=v_seq,
                meta=meta,
            )

    meta_path = os.path.join(args.out_dir, f"meta_terrain_{out_tag}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[OK] saved:", meta_path)


if __name__ == "__main__":
    main()


