from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_flexgen import DEFAULT_FLexGen_CACHE_DIR, FLexGenLinearQuantizer, WeightQuantizationConfig, build_weight_quantization_config


_BLOCK_PREFIX = "block"
_EXCLUDED_LINEAR_NAME_TOKENS = (
    "ada",
    "gate_msa",
    "gate_mlp",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _moviegen_prompt_path() -> Path:
    return _repo_root() / "Benchmark" / "MovieGen" / "MovieGenVideoBench_200.txt"


def _geneval_prompt_path() -> Path:
    return _repo_root() / "Benchmark" / "GenEval" / "prompts" / "evaluation_metadata.jsonl"


def _dpg_prompt_path() -> Path:
    return _repo_root() / "Benchmark" / "DPG" / "DPG_prompts.jsonl"


def _resolve_weight_cfg(raw_cfg: Any) -> WeightQuantizationConfig:
    if isinstance(raw_cfg, WeightQuantizationConfig):
        raw_cfg.validate()
        return raw_cfg
    return build_weight_quantization_config(raw_cfg)


def _stable_model_tag(tag: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag).strip("_")
    clean = clean[:96] if clean else "model"
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
    return f"{clean}-{digest}"


def _cache_paths(cfg: WeightQuantizationConfig, model_tag: str) -> Tuple[Path, Path]:
    root = Path(cfg.cache_dir or DEFAULT_FLexGen_CACHE_DIR).expanduser().resolve()
    cache_dir = root / _stable_model_tag(model_tag)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "flexgen_q4_g128_fake.pt", cache_dir / ".flexgen.lock"


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_plain_prompts(path: Path, limit: int) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")][:limit]


def _read_jsonl_prompts(path: Path, limit: int) -> List[str]:
    if not path.exists():
        return []
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
            if len(prompts) >= limit:
                break
    return prompts


def _build_moviegen_prompts(limit: int) -> List[str]:
    return _read_plain_prompts(_moviegen_prompt_path(), limit)


def _build_infinity_calibration_prompts() -> List[str]:
    prompts = _read_jsonl_prompts(_geneval_prompt_path(), 16)
    prompts.extend(_read_jsonl_prompts(_dpg_prompt_path(), 16))
    return prompts[:32]


def _deterministic_hidden(batch: int, seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    seq = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    feat = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(0)
    hidden = torch.sin(seq / 17.0 + feat / 29.0) + 0.5 * torch.cos(seq / 37.0 - feat / 13.0)
    return hidden.unsqueeze(0).expand(batch, -1, -1).contiguous().to(dtype=dtype)


def _iter_batch_slices(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _collect_block_linears(block: nn.Module) -> List[Tuple[str, nn.Module]]:
    linears: List[Tuple[str, nn.Module]] = []
    for name, module in block.named_modules():
        if name == "":
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        lowered = name.lower()
        if any(token in lowered for token in _EXCLUDED_LINEAR_NAME_TOKENS):
            continue
        linears.append((name, module))
    return linears


def _resolve_target_blocks(model: nn.Module, family: str) -> List[nn.Module]:
    if family == "var":
        return list(model.blocks)
    if family in {"infinity", "infinitystar"}:
        return list(model.unregistered_blocks if hasattr(model, "unregistered_blocks") else model.blocks)
    if family in {"self_forcing", "longlive", "wan"}:
        return list(model.blocks)
    raise ValueError(f"Unsupported FLexGen family: {family}")


def _build_cache_key(block_index: int, linear_name: str) -> str:
    return f"{_BLOCK_PREFIX}_{block_index}.{linear_name}.weight"


def _load_cached_weights(blocks: Sequence[nn.Module], payload: Dict[str, torch.Tensor]) -> None:
    for block_index, block in enumerate(blocks):
        for linear_name, module in _collect_block_linears(block):
            key = _build_cache_key(block_index, linear_name)
            tensor = payload.get(key)
            if tensor is None:
                continue
            module.weight.data.copy_(tensor.to(device=module.weight.device, dtype=module.weight.dtype))


def _snapshot_quantized_weights(blocks: Sequence[nn.Module]) -> Dict[str, torch.Tensor]:
    payload: Dict[str, torch.Tensor] = {}
    for block_index, block in enumerate(blocks):
        for linear_name, module in _collect_block_linears(block):
            payload[_build_cache_key(block_index, linear_name)] = module.weight.detach().cpu()
    return payload


@dataclass
class BlockwisePlan:
    family: str
    blocks: List[nn.Module]
    states: List[Dict[str, Any]]
    run_block: Callable[[int, nn.Module, Dict[str, Any]], torch.Tensor]
    cleanup: Optional[Callable[[], None]] = None

    def close(self) -> None:
        if self.cleanup is not None:
            self.cleanup()
            self.cleanup = None


def _quantize_plan(plan: BlockwisePlan, cfg: WeightQuantizationConfig) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"family": plan.family, "blocks": []}
    try:
        for block_index, block in enumerate(plan.blocks):
            block_info = {"block_index": block_index, "linears": []}
            linears = _collect_block_linears(block)
            for linear_name, linear in linears:
                quantizer = FLexGenLinearQuantizer(linear, cfg)

                def _hook(_module, inputs, _output, flexgen=quantizer):
                    if not inputs:
                        return
                    flexgen.add_batch(inputs[0].detach())

                handle = linear.register_forward_hook(_hook)
                try:
                    with torch.no_grad():
                        for state in plan.states:
                            plan.run_block(block_index, block, state)
                finally:
                    handle.remove()

                info = quantizer.quantize()
                info["linear"] = linear_name
                block_info["linears"].append(info)

            with torch.no_grad():
                for state in plan.states:
                    state["x"] = plan.run_block(block_index, block, state).detach()

            summary["blocks"].append(block_info)
            torch.cuda.empty_cache()
        return summary
    finally:
        plan.close()


def _build_var_plan(model: nn.Module) -> BlockwisePlan:
    blocks = list(model.blocks)
    labels = list(range(32))
    states: List[Dict[str, Any]] = []
    device = model.pos_1LC.device
    ed = model.L
    token_count = ed - model.first_l
    embedding_weight = getattr(model.vae_quant_proxy[0], "embedding", None)
    embedding_table = getattr(embedding_weight, "weight", None)

    for batch_labels in _iter_batch_slices(labels, 8):
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
        batch = len(batch_labels)
        if isinstance(embedding_table, torch.Tensor):
            token_ids = (
                torch.arange(token_count, device=device).unsqueeze(0)
                + batch_labels_tensor.unsqueeze(1)
            ) % embedding_table.shape[0]
            x_tokens = embedding_table[token_ids]
        else:
            x_tokens = _deterministic_hidden(batch, token_count, model.Cvae, device, torch.float32)

        with torch.cuda.amp.autocast(enabled=False):
            sos = cond_bd = model.class_emb(batch_labels_tensor)
            sos = sos.unsqueeze(1).expand(batch, model.first_l, -1) + model.pos_start.expand(batch, model.first_l, -1)
            x_blc = torch.cat((sos, model.word_embed(x_tokens.float())), dim=1)
            x_blc += model.lvl_embed(model.lvl_1L[:, :ed].expand(batch, -1)) + model.pos_1LC[:, :ed]
            attn_bias = model.attn_bias_for_masking[:, :, :ed, :ed].to(dtype=x_blc.dtype, device=device)
            cond = model.shared_ada_lin(cond_bd).to(dtype=x_blc.dtype, device=device)
        states.append({"x": x_blc.detach(), "cond": cond.detach(), "attn_bias": attn_bias.detach()})

    def run_block(_block_index: int, block: nn.Module, state: Dict[str, Any]) -> torch.Tensor:
        return block(x=state["x"], cond_BD=state["cond"], attn_bias=state["attn_bias"])

    return BlockwisePlan(family="var", blocks=blocks, states=states, run_block=run_block)


def _build_infinity_plan(model: nn.Module, args: Any) -> BlockwisePlan:
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
    from transformers import AutoTokenizer, T5EncoderModel

    blocks = list(model.unregistered_blocks if hasattr(model, "unregistered_blocks") else model.blocks)
    prompts = _build_infinity_calibration_prompts()
    batch_size = 2 if "8b" in str(getattr(args, "model_type", "")).lower() else 4
    device = model.pos_start.device
    scale_template = float(getattr(args, "h_div_w_template", 1.0))
    matched_ratio = h_div_w_templates[torch.argmin(torch.abs(torch.tensor(h_div_w_templates) - scale_template)).item()]
    scale_schedule = [(1, h, w) for _, h, w in dynamic_resolution_h_w[matched_ratio][args.pn]["scales"]]
    total_tokens = sum(int(pt * ph * pw) for pt, ph, pw in scale_schedule)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_ckpt, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_ckpt, torch_dtype=torch.float16).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    states: List[Dict[str, Any]] = []
    cleanup_items = [text_encoder]

    for prompt_batch in _iter_batch_slices(prompts, batch_size):
        tokens = text_tokenizer(
            text=list(prompt_batch),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device=device, non_blocking=True)
        mask = tokens.attention_mask.to(device=device, non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
        lens = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        max_seqlen_k = max(lens)
        kv_compact = torch.cat([feat[:length] for feat, length in zip(text_features.unbind(0), lens)], dim=0)

        x_prefix = _deterministic_hidden(len(prompt_batch), total_tokens, model.d_vae, device, torch.float32)
        with torch.amp.autocast("cuda", enabled=False):
            kv_compact = model.text_norm(kv_compact).contiguous()
            sos = cond_bd = model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
            kv_proj = model.text_proj_for_ca(kv_compact).contiguous()
            ca_kv = (kv_proj, cu_seqlens_k, max_seqlen_k)
            cond = model.shared_ada_lin(cond_bd).contiguous()
            sos = sos.unsqueeze(1).expand(len(prompt_batch), 1, -1) + model.pos_start.expand(len(prompt_batch), 1, -1)
            x_blc = torch.cat((sos, model.word_embed(model.norm0_ve(x_prefix))), dim=1)
            l_end = x_blc.shape[1]
            need_to_pad = ((l_end + model.pad_to_multiplier - 1) // model.pad_to_multiplier) * model.pad_to_multiplier - l_end
            if model.customized_flash_attn:
                attn_bias = (model.Infinity_visible_kvlen[:l_end], model.Infinity_invisible_qlen[:l_end])
            elif model.use_flex_attn:
                if need_to_pad:
                    x_blc = F.pad(x_blc, (0, 0, 0, need_to_pad))
                attn_bias = None
            else:
                d = torch.cat([torch.full((pt * ph * pw,), i, device=device) for i, (pt, ph, pw) in enumerate(scale_schedule)]).view(1, l_end, 1)
                d_t = d.transpose(1, 2)
                attn_bias = torch.where(d >= d_t, 0.0, -torch.inf).reshape(1, 1, l_end, l_end)
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_blc = F.pad(x_blc, (0, 0, 0, need_to_pad))
                attn_bias = attn_bias.to(dtype=x_blc.dtype)
            attn_fn = model.attn_fn_compile_dict[tuple(scale_schedule)] if model.use_flex_attn else None
        states.append(
            {
                "x": x_blc.detach(),
                "cond": cond.detach(),
                "ca_kv": ca_kv,
                "attn_bias": attn_bias,
                "attn_fn": attn_fn,
                "scale_schedule": scale_schedule,
                "need_to_pad": need_to_pad,
            }
        )

    def run_block(block_index: int, block: nn.Module, state: Dict[str, Any]) -> torch.Tensor:
        x_in = state["x"]
        if model.add_lvl_embeding_only_first_block and block_index == 0:
            x_in = model.add_lvl_embeding_for_x_BLC(x_in, state["scale_schedule"], state["need_to_pad"])
        if not model.add_lvl_embeding_only_first_block:
            x_in = model.add_lvl_embeding_for_x_BLC(x_in, state["scale_schedule"], state["need_to_pad"])
        return block(
            x=x_in,
            cond_BD=state["cond"],
            ca_kv=state["ca_kv"],
            attn_bias_or_two_vector=state["attn_bias"],
            attn_fn=state["attn_fn"],
            scale_schedule=state["scale_schedule"],
            rope2d_freqs_grid=model.rope2d_freqs_grid,
        )

    def cleanup() -> None:
        for item in cleanup_items:
            try:
                item.cpu()
            except Exception:
                pass
        torch.cuda.empty_cache()

    return BlockwisePlan(family="infinity", blocks=blocks, states=states, run_block=run_block, cleanup=cleanup)


def _build_infinitystar_plan(model: nn.Module, args: Any) -> BlockwisePlan:
    from transformers import AutoTokenizer, T5EncoderModel

    prompts = _build_moviegen_prompts(8)
    if not prompts:
        prompts = [f"InfinityStar calibration prompt {idx}" for idx in range(8)]
    blocks = list(model.unregistered_blocks if hasattr(model, "unregistered_blocks") else model.blocks)
    batch_size = 2
    device = model.word_embed.weight.device
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_ckpt, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_ckpt, torch_dtype=torch.float16).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    states: List[Dict[str, Any]] = []

    for prompt_batch in _iter_batch_slices(prompts, batch_size):
        tokens = text_tokenizer(
            text=list(prompt_batch),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device=device, non_blocking=True)
        mask = tokens.attention_mask.to(device=device, non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
        lens = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        max_seqlen_k = max(lens) if lens else 1
        kv_compact = torch.cat([feat[:length] for feat, length in zip(text_features.unbind(0), lens)], dim=0)

        with torch.amp.autocast("cuda", enabled=False):
            prefix_tokens, _ = model.prepare_text_conditions(
                (kv_compact, lens, cu_seqlens_k, max_seqlen_k),
                cfg_list=[1.0],
                B=len(prompt_batch),
                negative_label_B_or_BLT=None,
                text_maxlen_this_iter=max_seqlen_k,
            )
            rope_cache = model.rope2d_freqs_grid["freqs_text"][:, :, :, :, :max_seqlen_k].to(prefix_tokens.device)
        states.append(
            {
                "x": prefix_tokens.detach(),
                "rope_cache": rope_cache.detach(),
            }
        )

    def run_block(_block_index: int, block: nn.Module, state: Dict[str, Any]) -> torch.Tensor:
        return block(
            x=state["x"],
            cond_BD=None,
            ca_kv=None,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            scale_schedule=[],
            rope2d_freqs_grid=state["rope_cache"],
            scale_ind="t0",
            context_info=None,
            last_repetition_step=True,
        )

    def cleanup() -> None:
        try:
            text_encoder.cpu()
        except Exception:
            pass
        torch.cuda.empty_cache()

    return BlockwisePlan(family="infinitystar", blocks=blocks, states=states, run_block=run_block, cleanup=cleanup)


def _build_wan_plan(model: nn.Module, text_encoder: nn.Module, prompt_limit: int, family: str) -> BlockwisePlan:
    from wan.modules.model import sinusoidal_embedding_1d

    prompts = _build_moviegen_prompts(prompt_limit)
    blocks = list(model.blocks)
    device = model.patch_embedding.weight.device
    frame_count = 3
    latent_channels = int(getattr(model.patch_embedding, "in_channels", 16))
    latent_h = 60
    latent_w = 104
    states: List[Dict[str, Any]] = []

    if getattr(model, "block_mask", None) is None:
        if getattr(model, "independent_first_frame", False):
            model.block_mask = model._prepare_blockwise_causal_attn_mask_i2v(
                device,
                num_frames=frame_count,
                frame_seqlen=latent_h * latent_w // (model.patch_size[1] * model.patch_size[2]),
                num_frame_per_block=model.num_frame_per_block,
                local_attn_size=model.local_attn_size,
            )
        else:
            model.block_mask = model._prepare_blockwise_causal_attn_mask(
                device,
                num_frames=frame_count,
                frame_seqlen=latent_h * latent_w // (model.patch_size[1] * model.patch_size[2]),
                num_frame_per_block=model.num_frame_per_block,
                local_attn_size=model.local_attn_size,
            )

    for prompt_batch in _iter_batch_slices(prompts, 1):
        cond = text_encoder(text_prompts=list(prompt_batch))["prompt_embeds"]
        bsz = cond.shape[0]
        patch_dtype = model.patch_embedding.weight.dtype
        noise = _deterministic_hidden(bsz, frame_count * latent_h * latent_w, latent_channels, device, patch_dtype)
        noise = noise.view(bsz, frame_count, latent_h, latent_w, latent_channels).permute(0, 4, 1, 2, 3).contiguous()
        x_list = [model.patch_embedding(u.unsqueeze(0)) for u in noise]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x_list])
        x_list = [u.flatten(2).transpose(1, 2) for u in x_list]
        seq_lens = torch.tensor([u.size(1) for u in x_list], dtype=torch.long, device=device)
        max_seq_len = int(seq_lens.max().item())
        x = torch.cat([torch.cat([u, u.new_zeros(1, max_seq_len - u.size(1), u.size(2))], dim=1) for u in x_list])
        timestep = torch.full((bsz, frame_count), 500, dtype=torch.long, device=device)
        e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, timestep.flatten()).type_as(x))
        e0 = model.time_projection(e).unflatten(1, (6, model.dim)).unflatten(dim=0, sizes=timestep.shape)
        text_dtype = next(model.text_embedding.parameters()).dtype
        context = model.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))]) for u in cond]).to(dtype=text_dtype)
        )
        states.append(
            {
                "x": x.detach(),
                "kwargs": {
                    "e": e0.detach(),
                    "seq_lens": seq_lens.detach(),
                    "grid_sizes": grid_sizes.detach(),
                    "freqs": model.freqs.to(device),
                    "context": context.detach(),
                    "context_lens": None,
                    "block_mask": model.block_mask,
                },
            }
        )

    def run_block(_block_index: int, block: nn.Module, state: Dict[str, Any]) -> torch.Tensor:
        return block(state["x"], **state["kwargs"])

    return BlockwisePlan(family=family, blocks=blocks, states=states, run_block=run_block)


def _build_synthetic_plan(model: nn.Module, family: str) -> BlockwisePlan:
    blocks = _resolve_target_blocks(model, family)
    device = next(model.parameters()).device
    hidden_dim = int(getattr(model, "C", getattr(model, "dim", 1536)))
    states = [{"x": _deterministic_hidden(1, 256, hidden_dim, device, torch.float32)}]

    def run_block(_block_index: int, block: nn.Module, state: Dict[str, Any]) -> torch.Tensor:
        try:
            return block(state["x"])
        except Exception:
            return state["x"]

    return BlockwisePlan(family=f"{family}_synthetic", blocks=blocks, states=states, run_block=run_block)


def _build_plan(model: nn.Module, family: str, **kwargs) -> BlockwisePlan:
    if family == "var":
        return _build_var_plan(model)
    if family == "infinity":
        return _build_infinity_plan(model, kwargs["args"])
    if family == "infinitystar":
        return _build_infinitystar_plan(model, kwargs["args"])
    if family in {"self_forcing", "longlive"}:
        return _build_wan_plan(model, kwargs["text_encoder"], 16 if family == "self_forcing" else 8, family)
    return _build_synthetic_plan(model, family)


def maybe_apply_flexgen(
    model: nn.Module,
    raw_cfg: Any,
    model_tag: str,
    family: str,
    **kwargs,
) -> Dict[str, Any]:
    cfg = _resolve_weight_cfg(raw_cfg)
    if not cfg.enable:
        return {"applied": False, "reason": "disabled"}

    cache_path, lock_path = _cache_paths(cfg, model_tag)
    blocks = _resolve_target_blocks(model, family)
    with _file_lock(lock_path):
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            _load_cached_weights(blocks, payload.get("weights", {}))
            return {
                "applied": True,
                "source": "cache",
                "cache_path": str(cache_path),
                "model_tag": model_tag,
                "family": family,
            }

        plan = _build_plan(model, family, **kwargs)
        summary = _quantize_plan(plan, cfg)
        payload = {
            "family": family,
            "model_tag": model_tag,
            "config": {
                "q_bits": cfg.q_bits,
                "group_size": cfg.group_size,
                "sym": cfg.sym,
                "block_size": cfg.block_size,
                "percdamp": cfg.percdamp,
                "act_order": cfg.act_order,
                "static_groups": cfg.static_groups,
                "runtime_form": cfg.runtime_form,
            },
            "weights": _snapshot_quantized_weights(blocks),
            "summary": summary,
        }
        torch.save(payload, cache_path)
        return {
            "applied": True,
            "source": "fresh",
            "cache_path": str(cache_path),
            "model_tag": model_tag,
            "family": family,
            "summary": summary,
        }
