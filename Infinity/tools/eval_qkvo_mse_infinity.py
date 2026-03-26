#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import torch

_INFINITY_ROOT = osp.dirname(osp.dirname(__file__))
_VARQ_ROOT = osp.dirname(_INFINITY_ROOT)
sys.path.append(_INFINITY_ROOT)
sys.path.append(_VARQ_ROOT)

from tools.run_infinity import _import_basic, _import_dynamic_resolution, gen_one_img
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_args_from_config(args_cmd):
    cfg = _load_json(args_cmd.config_file)
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})
    ckpt_cfg = cfg.get("checkpoints", {})

    args = SimpleNamespace()
    args.model_type = model_cfg.get("model_type", "infinity_2b")
    args.model_path = args_cmd.model_path or ckpt_cfg.get("model_path", "")
    args.vae_path = args_cmd.vae_path or ckpt_cfg.get("vae_ckpt", "")
    args.text_encoder_ckpt = args_cmd.text_encoder_ckpt

    args.checkpoint_type = args_cmd.checkpoint_type
    args.enable_model_cache = int(args_cmd.enable_model_cache)
    args.cache_dir = args_cmd.cache_dir
    args.bf16 = int(args_cmd.bf16)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.pn = args_cmd.pn
    args.h_div_w_template = args_cmd.h_div_w_template
    args.text_channels = int(args_cmd.text_channels)
    if args_cmd.vae_type is None:
        if args.model_type == "infinity_8b":
            args.vae_type = 14
        elif args.model_type == "infinity_2b":
            args.vae_type = 32
        else:
            args.vae_type = 32
    else:
        args.vae_type = int(args_cmd.vae_type)
    args.apply_spatial_patchify = int(args_cmd.apply_spatial_patchify)
    args.use_flex_attn = int(args_cmd.use_flex_attn)
    args.rope2d_each_sa_layer = int(args_cmd.rope2d_each_sa_layer)
    args.rope2d_normalized_by_hw = int(args_cmd.rope2d_normalized_by_hw)
    args.use_scale_schedule_embedding = int(args_cmd.use_scale_schedule_embedding)
    args.use_bit_label = int(args_cmd.use_bit_label)
    args.add_lvl_embeding_only_first_block = int(args_cmd.add_lvl_embeding_only_first_block)

    args.seed = int(infer_cfg.get("seed", args_cmd.seed))
    args.cfg = float(infer_cfg.get("cfg", args_cmd.cfg))
    args.tau = float(infer_cfg.get("tau", args_cmd.tau))
    args.cfg_insertion_layer = int(args_cmd.cfg_insertion_layer)
    args.sampling_per_bits = int(args_cmd.sampling_per_bits)
    args.enable_positive_prompt = int(args_cmd.enable_positive_prompt)

    args.enable_quantization = 0
    args.q_bits = 8
    args.quant_method = args_cmd.quant_method
    args.qkv_format = args_cmd.qkv_format
    args.rescale_qk = 0
    return args


def build_scale_schedule(args):
    dynamic_resolution_h_w, _ = _import_dynamic_resolution()
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    return [(1, h, w) for (_, h, w) in scale_schedule]


def _list_dump_files(run_dir: str):
    files = [osp.join(run_dir, fn) for fn in os.listdir(run_dir) if fn.endswith(".pt")]
    files.sort()
    return files


def compute_mse_from_dump_pair(base_dir: str, var_dir: str):
    base_files = _list_dump_files(base_dir)
    var_files = _list_dump_files(var_dir)
    if len(base_files) != len(var_files):
        raise RuntimeError(f"Dump file count mismatch: baseline={len(base_files)} variant={len(var_files)}")

    sse = {"q": 0.0, "k": 0.0, "v": 0.0, "o": 0.0}
    cnt = {"q": 0, "k": 0, "v": 0, "o": 0}

    for fb, fv in zip(base_files, var_files):
        tb = torch.load(fb, map_location="cpu")
        tv = torch.load(fv, map_location="cpu")
        for key in ("q", "k", "v", "o"):
            d = tv[key].float() - tb[key].float()
            sse[key] += float((d * d).sum().item())
            cnt[key] += int(d.numel())

    out = {f"mse_{k}": (sse[k] / cnt[k] if cnt[k] > 0 else 0.0) for k in ("q", "k", "v", "o")}
    out["mse_mean"] = (out["mse_q"] + out["mse_k"] + out["mse_v"] + out["mse_o"]) / 4.0
    out["num_calls"] = len(base_files)
    return out


def _to_head_first(x: torch.Tensor, qkv_format: str, tensor_name: str, num_heads: int):
    """
    Return a tensor shaped as [H, N] so per-head MSE can be aggregated.
    """
    if tensor_name in ("q", "k", "v"):
        if qkv_format == "BLHc":
            # [B, L, H, c] -> [H, B*L*c]
            xh = x.permute(2, 0, 1, 3).contiguous()
        else:
            # [B, H, L, c] -> [H, B*L*c]
            xh = x.permute(1, 0, 2, 3).contiguous()
        return xh.reshape(xh.shape[0], -1)

    # o: [B, L, C] -> [H, B*L*c], with C=H*c
    B, L, C = x.shape
    if C % num_heads != 0:
        raise RuntimeError(f"Cannot split O by heads: C={C}, H={num_heads}")
    c = C // num_heads
    xh = x.view(B, L, num_heads, c).permute(2, 0, 1, 3).contiguous()
    return xh.reshape(num_heads, -1)


def enumerate_baseline_head_scale_keys(base_dir: str):
    keys = set()
    for fb in _list_dump_files(base_dir):
        tb = torch.load(fb, map_location="cpu")
        qkv_format = tb.get("qkv_format", "BHLc")
        scale_ind = tb.get("scale_ind", "unknown")
        if qkv_format == "BLHc":
            num_heads = int(tb["q"].shape[2])
        else:
            num_heads = int(tb["q"].shape[1])
        for tensor_name in ("q", "k", "v", "o"):
            for h in range(num_heads):
                keys.add((str(scale_ind), int(h), tensor_name))
    return sorted(list(keys), key=lambda t: (t[0], t[1], t[2]))


def compute_head_scale_mse_from_dump_pair(base_dir: str, var_dir: str, mode_name: str):
    base_files = _list_dump_files(base_dir)
    var_files = _list_dump_files(var_dir)
    if len(base_files) != len(var_files):
        raise RuntimeError(f"Dump file count mismatch: baseline={len(base_files)} variant={len(var_files)}")

    stat = {}
    for fb, fv in zip(base_files, var_files):
        tb = torch.load(fb, map_location="cpu")
        tv = torch.load(fv, map_location="cpu")
        scale_ind = str(tv.get("scale_ind", tb.get("scale_ind", "unknown")))
        qkv_format = tv.get("qkv_format", tb.get("qkv_format", "BHLc"))
        if qkv_format == "BLHc":
            num_heads = int(tv["q"].shape[2])
        else:
            num_heads = int(tv["q"].shape[1])

        for tensor_name in ("q", "k", "v", "o"):
            hb = _to_head_first(tb[tensor_name].float(), qkv_format, tensor_name, num_heads)
            hv = _to_head_first(tv[tensor_name].float(), qkv_format, tensor_name, num_heads)
            d = hv - hb
            for h in range(num_heads):
                key = (scale_ind, int(h), tensor_name)
                if key not in stat:
                    stat[key] = {"sse": 0.0, "count": 0}
                stat[key]["sse"] += float((d[h] * d[h]).sum().item())
                stat[key]["count"] += int(d[h].numel())

    rows = []
    for (scale_ind, h, tensor_name), s in sorted(stat.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        cnt = s["count"]
        mse = s["sse"] / cnt if cnt > 0 else 0.0
        rows.append(
            {
                "mode": mode_name,
                "scale_ind": scale_ind,
                "head_idx": h,
                "tensor": tensor_name,
                "mse": mse,
                "count": cnt,
            }
        )
    return rows


def run_one_dump(args_base, prompt: str, mode_name: str, enable_quant: int, q_bits: int, rescale_qk: int, dump_root: str):
    basic = _import_basic()
    args = SimpleNamespace(**vars(args_base))
    args.enable_quantization = int(enable_quant)
    args.q_bits = int(q_bits)
    args.rescale_qk = int(rescale_qk)

    basic.enable_qkvo_dump_capture(dump_root, mode_name)

    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    scale_schedule = build_scale_schedule(args)

    with torch.no_grad():
        _ = gen_one_img(
            infinity_test=infinity,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            g_seed=args.seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=args.cfg,
            tau_list=args.tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=args.enable_positive_prompt,
        )

    basic.disable_qkvo_dump_capture()
    del infinity, vae, text_encoder, text_tokenizer
    torch.cuda.empty_cache()


def save_outputs(rows, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = osp.join(output_dir, f"qkvo_mse_table_{ts}.csv")
    md_path = osp.join(output_dir, f"qkvo_mse_table_{ts}.md")
    json_path = osp.join(output_dir, f"qkvo_mse_table_{ts}.json")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
        f.write("\n")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    return csv_path, md_path, json_path


def save_detailed_outputs(rows, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = osp.join(output_dir, f"qkvo_mse_head_scale_{ts}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _load_rows_from_any_table(path: str):
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    if path.endswith(".csv"):
        return pd.read_csv(path).to_dict(orient="records")
    if path.endswith(".md"):
        # markdown table parser is intentionally omitted; prefer csv/json inputs.
        raise ValueError("Please provide --reuse_table as .csv or .json for merging.")
    raise ValueError("Unsupported --reuse_table format; use .csv or .json")


def _merge_rows_keep_order(base_rows, patch_rows):
    patch_map = {r["mode"]: r for r in patch_rows}
    merged = []
    for r in base_rows:
        mode = r.get("mode")
        if mode in patch_map:
            merged.append(patch_map.pop(mode))
        else:
            merged.append(r)
    for _, r in patch_map.items():
        merged.append(r)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q/K/V/O MSE for Infinity VAR-Q modes (offline dump + compare).")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--text_encoder_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Benchmark/outputs/Infinity/qkvo_mse")
    parser.add_argument("--dump_dir", type=str, default="")
    parser.add_argument("--mode_set", type=str, default="all", choices=["all", "rescale_only"])
    parser.add_argument("--reuse_table", type=str, default="",
                        help="Existing csv/json table to merge into (keep previous rows, update rerun rows).")
    parser.add_argument("--write_head_scale_csv", type=int, default=1, choices=[0, 1])

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--vae_path", type=str, default="")
    parser.add_argument("--checkpoint_type", type=str, default="torch_shard", choices=["torch", "torch_shard"])
    parser.add_argument("--enable_model_cache", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cache_dir", type=str, default="/dev/shm")
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--cfg_insertion_layer", type=int, default=0)
    parser.add_argument("--sampling_per_bits", type=int, default=1)
    parser.add_argument("--enable_positive_prompt", type=int, default=0, choices=[0, 1])

    parser.add_argument("--pn", type=str, default="1M", choices=["0.06M", "0.25M", "1M"])
    parser.add_argument("--h_div_w_template", type=float, default=1.0)
    parser.add_argument("--text_channels", type=int, default=2048)
    parser.add_argument("--vae_type", type=int, default=None)
    parser.add_argument("--apply_spatial_patchify", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_flex_attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rope2d_each_sa_layer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rope2d_normalized_by_hw", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--use_scale_schedule_embedding", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_bit_label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--add_lvl_embeding_only_first_block", type=int, default=0, choices=[0, 1])

    parser.add_argument("--quant_method", type=str, default="G_SCALE_HEAD_DIM")
    parser.add_argument("--qkv_format", type=str, default="BHLc", choices=["BLHc", "BHLc"])
    args_cmd = parser.parse_args()

    args_base = build_args_from_config(args_cmd)
    if not args_base.model_path or not args_base.vae_path:
        raise ValueError("model_path/vae_path is empty. Please set in config file or pass --model_path/--vae_path.")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dump_root = args_cmd.dump_dir or osp.join(args_cmd.output_dir, f"dumps_{ts}")
    os.makedirs(dump_root, exist_ok=True)

    baseline_tag = "baseline_raw_infinity"
    all_modes = [
        ("varq-8", 1, 8, 0),
        ("varq-4", 1, 4, 0),
        ("varq-2", 1, 2, 0),
        ("rescale+varq-8", 1, 8, 1),
        ("rescale+varq-4", 1, 4, 1),
        ("rescale+varq-2", 1, 2, 1),
    ]
    if args_cmd.mode_set == "rescale_only":
        eval_modes = [m for m in all_modes if m[0].startswith("rescale+")]
    else:
        eval_modes = all_modes

    print(f"[Run] {baseline_tag}")
    run_one_dump(args_base, args_cmd.prompt, baseline_tag, 0, 0, 0, dump_root)

    new_rows = [{
        "mode": baseline_tag,
        "enable_quantization": 0,
        "q_bits": 0,
        "rescale_qk": 0,
        "mse_q": 0.0,
        "mse_k": 0.0,
        "mse_v": 0.0,
        "mse_o": 0.0,
        "mse_mean": 0.0,
    }]
    detailed_rows = []

    base_dir = osp.join(dump_root, baseline_tag)
    if args_cmd.write_head_scale_csv:
        base_keys = enumerate_baseline_head_scale_keys(base_dir)
        for scale_ind, h, tensor_name in base_keys:
            detailed_rows.append(
                {
                    "mode": baseline_tag,
                    "scale_ind": scale_ind,
                    "head_idx": h,
                    "tensor": tensor_name,
                    "mse": 0.0,
                    "count": 0,
                }
            )

    for mode_name, enable_quant, q_bits, rescale_qk in eval_modes:
        print(f"[Run] {mode_name}")
        run_one_dump(args_base, args_cmd.prompt, mode_name, enable_quant, q_bits, rescale_qk, dump_root)
        var_dir = osp.join(dump_root, mode_name)
        mse = compute_mse_from_dump_pair(base_dir, var_dir)
        new_rows.append({
            "mode": mode_name,
            "enable_quantization": int(enable_quant),
            "q_bits": int(q_bits),
            "rescale_qk": int(rescale_qk),
            "mse_q": mse["mse_q"],
            "mse_k": mse["mse_k"],
            "mse_v": mse["mse_v"],
            "mse_o": mse["mse_o"],
            "mse_mean": mse["mse_mean"],
            "num_calls": mse["num_calls"],
        })
        if args_cmd.write_head_scale_csv:
            detailed_rows.extend(compute_head_scale_mse_from_dump_pair(base_dir, var_dir, mode_name))

    if args_cmd.reuse_table:
        base_rows = _load_rows_from_any_table(args_cmd.reuse_table)
        rows = _merge_rows_keep_order(base_rows, new_rows)
    else:
        rows = new_rows

    csv_path, md_path, json_path = save_outputs(rows, args_cmd.output_dir)
    detail_csv_path = None
    if args_cmd.write_head_scale_csv and len(detailed_rows) > 0:
        detail_csv_path = save_detailed_outputs(detailed_rows, args_cmd.output_dir)
    print(f"[Done] CSV:  {csv_path}")
    print(f"[Done] MD:   {md_path}")
    print(f"[Done] JSON: {json_path}")
    if detail_csv_path is not None:
        print(f"[Done] HeadScale CSV: {detail_csv_path}")
    print(f"[Done] Dumps: {dump_root}")


if __name__ == "__main__":
    main()
