from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

from VAR_Q.config_loader import VARQConfig


def add_infinity_config_file_argument(parser: argparse.ArgumentParser) -> None:
    if "--config_file" not in parser._option_string_actions:
        parser.add_argument("--config_file", type=str, default=None, help="Path to VAR-Q JSON config.")


def apply_infinity_config(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config_file = getattr(args, "config_file", None) or getattr(args, "varq_config", None)
    if not config_file:
        return vars(args), getattr(args, "ablation_config", {})

    config = VARQConfig(config_file)
    model_config = config.get_model_config()
    quant_config = config.get_quantization_config()
    ablation_config = config.get_ablation_config()
    inference_config = config.get_inference_config()

    if "model_type" in model_config:
        args.model_type = model_config["model_type"]

    args.enable_quantization = int(bool(quant_config.get("enable", False)))
    args.q_bits = int(quant_config.get("q_bits", getattr(args, "q_bits", 8)))
    args.q_bits_k = quant_config.get("q_bits_k", None)
    args.q_bits_v = quant_config.get("q_bits_v", None)
    args.pack_to_int32 = bool(quant_config.get("pack_to_int32", True))
    args.pack_to_int32_k = quant_config.get("pack_to_int32_k", None)
    args.pack_to_int32_v = quant_config.get("pack_to_int32_v", None)
    args.quant_method = quant_config.get("quant_method", getattr(args, "quant_method", "VARQ"))
    args.qkv_format = quant_config.get("qkv_format", getattr(args, "qkv_format", "BHLc"))
    args.kivi_group_size = int(quant_config.get("kivi_group_size", 128))
    args.kivi_cali_k_group_size = int(quant_config.get("kivi_cali_k_group_size", 128))
    args.kivi_cali_v_group_size = int(quant_config.get("kivi_cali_v_group_size", 128))
    args.compression_ratio = float(quant_config.get("compression_ratio", quant_config.get("ratio", 1.0)))
    args.max_scale_seq_len = int(quant_config.get("max_scale_seq_len", 0) or 0)
    args.rescale_qk = int(bool(quant_config.get("rescale_qk", False)))
    args.enable_fused_kv_flashattn = int(bool(quant_config.get("enable_fused_kv_flashattn", False)))
    args.outlier_ratio = float(quant_config.get("outlier_ratio", 0.0))
    args.outlier_mode = quant_config.get("outlier_mode", "ratio")
    args.outlier_n_sigma = float(quant_config.get("outlier_n_sigma", 3.0))
    args.ablation_config = ablation_config

    if "cfg" in inference_config:
        args.cfg = str(inference_config["cfg"])
    if "tau" in inference_config:
        args.tau = inference_config["tau"]
    if "seed" in inference_config:
        args.seed = inference_config["seed"]
    if "h_div_w" in inference_config:
        args.h_div_w_template = float(inference_config["h_div_w"])
    if "enable_positive_prompt" in inference_config:
        args.enable_positive_prompt = int(inference_config["enable_positive_prompt"])
    if getattr(args, "seed_override", None) is not None:
        args.seed = args.seed_override

    return quant_config, ablation_config
