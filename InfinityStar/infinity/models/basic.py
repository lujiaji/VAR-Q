# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Definitions of blocks of VAR transformer model.
"""

import math
import os
import sys
import time
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from infinity.models.rope import apply_rotary_emb
from infinity.utils.sequence_parallel import sp_all_to_all, SequenceParallelManager as sp_manager

try:
    from VAR_Q.quant import VAR_Q
    from VAR_Q.quant_infinitystar import InfinityStarVARQ
except ImportError:
    # Fallback: when launched from InfinityStar root, VAR-Q repo root may be missing in sys.path.
    varq_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if varq_repo_root not in sys.path:
        sys.path.append(varq_repo_root)
    try:
        from VAR_Q.quant import VAR_Q
        from VAR_Q.quant_infinitystar import InfinityStarVARQ
    except ImportError:
        VAR_Q = None
        InfinityStarVARQ = None

# Import SageAttention
from sageattention import sageattn,sageattn_qk_int8_pv_fp8_cuda,sageattn_qk_int8_pv_fp16_triton,sageattn_qk_int8_pv_fp16_cuda,sageattn_qk_int8_pv_fp8_cuda_sm90,sageattn_varlen
# Import flash_attn's fused ops
try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    fused_mlp_func = None
    flash_fused_op_installed = False
    
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


_NUMERICS_DEBUG = os.environ.get("INFINITYSTAR_NUMERICS_DEBUG", "0") == "1"


def _format_debug_tensor_stats(tensor: torch.Tensor) -> str:
    t = tensor.detach().float()
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    finite_mask = torch.isfinite(t)
    if bool(finite_mask.any()):
        finite_vals = t[finite_mask]
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
        absmax_val = float(finite_vals.abs().max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
        absmax_val = float("nan")
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"nan={nan_count} inf={inf_count} min={min_val:.6g} max={max_val:.6g} absmax={absmax_val:.6g}"
    )


def check_debug_tensor_finite(name: str, tensor: Optional[torch.Tensor], context: str) -> None:
    if not _NUMERICS_DEBUG or tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return
    if bool(torch.isfinite(tensor).all()):
        return
    msg = f"[INFINITYSTAR_NUMERICS] non-finite tensor `{name}` at {context}: {_format_debug_tensor_stats(tensor)}"
    print(msg, flush=True)
    raise RuntimeError(msg)


def check_debug_prob_tensor(name: str, tensor: Optional[torch.Tensor], context: str) -> None:
    if not _NUMERICS_DEBUG or tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return
    check_debug_tensor_finite(name, tensor, context)
    min_val = float(tensor.detach().float().min().item())
    if min_val < 0:
        msg = (
            f"[INFINITYSTAR_NUMERICS] invalid probability tensor `{name}` at {context}: "
            f"min={min_val:.6g}, {_format_debug_tensor_stats(tensor)}"
        )
        print(msg, flush=True)
        raise RuntimeError(msg)


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))
    
    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)
    
    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = -1
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=self.heuristic,
                process_group=None,
            ))
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)
    
    def forward(self, x):
        return self.drop(self.fc2( F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12, num_key_value_heads=-1,
        use_flex_attn=False, 
        pad_to_multiplier=1, rope2d_normalized_by_hw=0,
        mask_type='var', context_frames=1000000, steps_per_frame=4,
        arch='var',
        qwen_qkvo_bias=False,
        q_bits=8, quant_method='G_SCALE_HEAD_DIM', qkv_format='BHLc', enable_quantization=False, rescale_qk=False,
        enable_fused_kv_flashattn=False,
        enable_sageattn=False, sageattn_type='sageattn',
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_key_value_heads == -1 or num_heads % num_key_value_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else num_heads
        self.arch = arch
        if self.arch == 'qwen':
            self.q_proj = nn.Linear(embed_dim, self.num_heads*self.head_dim, bias=qwen_qkvo_bias)
            self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
            self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
            self.o_proj = nn.Linear(self.num_heads*self.head_dim, embed_dim, bias=qwen_qkvo_bias)
            self.q_norm = FastRMSNorm(self.head_dim)
            self.k_norm = FastRMSNorm(self.head_dim)
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        else:
            raise ValueError(f'arch {self.arch} not supported')
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = {}    # kv caching: only used during inference
        self.cached_v = {}    # kv caching: only used during inference
        self.k_varq = None
        self.v_varq = None

        self.q_bits = q_bits
        self.quant_method = quant_method
        self.qkv_format = qkv_format
        self.enable_quantization = bool(enable_quantization)
        self.rescale_qk = bool(rescale_qk)
        self.enable_fused_kv_flashattn = bool(enable_fused_kv_flashattn)
        self.using_sageattn = bool(enable_sageattn)
        self.sageattn_type = str(sageattn_type)
        if self.enable_fused_kv_flashattn and int(self.q_bits) != 4:
            raise ValueError("enable_fused_kv_flashattn requires int4 quantization (q_bits == 4).")
        self._varq_available = self.enable_quantization and (VAR_Q is not None) and (InfinityStarVARQ is not None)
        self._sageattn_impl = None
        if self.using_sageattn:
            if self.sageattn_type == 'sageattn_qk_int8_pv_fp8_cuda':
                self._sageattn_impl = sageattn_qk_int8_pv_fp8_cuda
                # print("using sageattn_qk_int8_pv_fp8_cuda")
            elif self.sageattn_type == 'sageattn':
                self._sageattn_impl = sageattn
                # print("using sageattn")
            elif self.sageattn_type == 'sageattn_qk_int8_pv_fp16_triton':
                self._sageattn_impl = sageattn_qk_int8_pv_fp16_triton
                # print("using sageattn_qk_int8_pv_fp16_triton")
            elif self.sageattn_type == 'sageattn_qk_int8_pv_fp16_cuda':
                self._sageattn_impl = sageattn_qk_int8_pv_fp16_cuda
                # print("using sageattn_qk_int8_pv_fp16_cuda")
            elif self.sageattn_type == 'sageattn_qk_int8_pv_fp8_cuda_sm90':
                self._sageattn_impl = sageattn_qk_int8_pv_fp8_cuda_sm90
                # print("using sageattn_qk_int8_pv_fp8_cuda_sm90")
            elif self.sageattn_type == 'sageattn_varlen':
                self._sageattn_impl = sageattn_varlen
                # print("using sageattn_varlen")
            else:
                raise ValueError(f"Unsupported sageattn_type: {self.sageattn_type}")
            if self._sageattn_impl is None:
                print(
                    f"Warning: SageAttention ({self.sageattn_type}) not found, fallback to flash_attn."
                )
                self.using_sageattn = False
        self.debug_block_id = -1
        self._varq_debug = os.environ.get("VARQ_DEBUG_LOG", "0") == "1"
        self._varq_debug_block = int(os.environ.get("VARQ_DEBUG_BLOCK_IDX", "0"))

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.mask_type = mask_type
        self.context_frames = context_frames
        self.steps_per_frame = steps_per_frame
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = {}
        self.cached_v = {}
        if enable and self._varq_available:
            self.k_varq = InfinityStarVARQ(
                quant_bits=self.q_bits,
                qkv_format='BHLc',
                quant_method=self.quant_method,
                rescale_qk=self.rescale_qk,
                dequant_dtype='bf16',
            )
            self.v_varq = InfinityStarVARQ(
                quant_bits=self.q_bits,
                qkv_format='BHLc',
                quant_method=self.quant_method,
                rescale_qk=self.rescale_qk,
                dequant_dtype='bf16',
            )
        else:
            self.k_varq = None
            self.v_varq = None

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, rope2d_freqs_grid=[], scale_schedule=[], scale_ind=0, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be sharded (L = raw_seq_len//sp_size)
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be sharded
        """
        # x: fp32
        B, L, C = x.shape

        if self.arch == 'qwen':
            hidden_states = x
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # batch, num_key_value_heads, slen, head_dim
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # batch, num_key_value_heads, slen, head_dim

            if sp_manager.sp_on():
                # Headnum need to be sharded and L needs to be gathered
                # [B, H, raw_L/sp, C] --> [B, H/sp, raw_L, C]
                sdim = 1
                gdim = 2
                L = L * sp_manager.get_sp_size()
                C = C // sp_manager.get_sp_size()
                query_states = sp_all_to_all(query_states, sdim, gdim)
                key_states = sp_all_to_all(key_states, sdim, gdim)
                value_states = sp_all_to_all(value_states, sdim, gdim)

            query_states, key_states = apply_rotary_emb(query_states, key_states, rope2d_freqs_grid)
            if self.caching:    # kv caching: only used during inference
                if self._varq_available and self.k_varq is not None and self.v_varq is not None:
                    if last_repetition_step:
                        if isinstance(scale_ind, int):
                            self.k_varq.cache_scale(scale_ind, key_states, overwrite=True, return_dequant=False)
                            self.v_varq.cache_scale(scale_ind, value_states, overwrite=True, return_dequant=False)
                        else:
                            self.cached_k[scale_ind] = key_states
                            self.cached_v[scale_ind] = value_states

                    if isinstance(scale_ind, int):
                        ref_scale_inds = context_info[scale_ind]['ref_sids'] + ref_text_scale_inds
                        if len(ref_scale_inds) > 0:
                            k_parts, v_parts = [], []
                            for sid in ref_scale_inds:
                                if isinstance(sid, int):
                                    k_parts.append(self.k_varq.get_scale(sid))
                                    v_parts.append(self.v_varq.get_scale(sid))
                                else:
                                    k_parts.append(self.cached_k[sid])
                                    v_parts.append(self.cached_v[sid])
                            k_parts.append(key_states)
                            v_parts.append(value_states)
                            key_states = torch.cat(k_parts, dim=2)
                            value_states = torch.cat(v_parts, dim=2)

                        # Release scales that are no longer referenced.
                        ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
                        for si in range(len(context_info)):
                            for ref_si in context_info[si]['ref_sids']:
                                ref_scale_2_last_use_scale[ref_si] = si
                        for ref_si in range(scale_ind):
                            if ref_scale_2_last_use_scale[ref_si] < scale_ind:
                                self.k_varq.clear_scales([ref_si])
                                self.v_varq.clear_scales([ref_si])
                        if self._varq_debug and self.debug_block_id == self._varq_debug_block:
                            k_bytes = self.k_varq.cache_bytes()
                            v_bytes = self.v_varq.cache_bytes()
                            live_k = self.k_varq.live_scale_ids()
                            total_mb = (k_bytes["total_bytes"] + v_bytes["total_bytes"]) / 1024 / 1024
                            print(
                                f"[VARQ][block={self.debug_block_id:02d}][scale={scale_ind:02d}] "
                                f"refs={ref_scale_inds} live_scales={live_k} "
                                f"k_MB={k_bytes['total_bytes']/1024/1024:.2f} v_MB={v_bytes['total_bytes']/1024/1024:.2f} "
                                f"kv_total_MB={total_mb:.2f}"
                            )
                else:
                    if last_repetition_step:
                        self.cached_k[scale_ind] = key_states
                        self.cached_v[scale_ind] = value_states
                    if isinstance(scale_ind, int):
                        ref_scale_inds = context_info[scale_ind]['ref_sids'] + ref_text_scale_inds
                        key_states = torch.cat([self.cached_k[ind] for ind in ref_scale_inds] + [key_states], dim=2)
                        value_states = torch.cat([self.cached_v[ind] for ind in ref_scale_inds] + [value_states], dim=2)

                        ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
                        for si in range(len(context_info)):
                            for ref_si in context_info[si]['ref_sids']:
                                ref_scale_2_last_use_scale[ref_si] = si
                        for ref_si in range(scale_ind):
                            if (ref_scale_2_last_use_scale[ref_si] < scale_ind) and (self.cached_k[ref_si] is not None):
                                tmpk, tmpv = self.cached_k[ref_si], self.cached_v[ref_si]
                                self.cached_k[ref_si], self.cached_v[ref_si] = None, None
                                del tmpk, tmpv

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            scale = self.head_dim**-0.5
            # Dump Q/K/V (post-RoPE, post-KV-cache concat) for analysis; see INFINITYSTAR_QKV_DUMP_DIR.
            _qkv_dump_dir = os.environ.get("INFINITYSTAR_QKV_DUMP_DIR", "").strip()
            if (
                _qkv_dump_dir
                and self.caching
                and isinstance(scale_schedule, (list, tuple))
                and len(scale_schedule) > 0
                and isinstance(scale_ind, int)
                and scale_ind == len(scale_schedule) - 1
                and last_repetition_step
            ):
                os.makedirs(_qkv_dump_dir, exist_ok=True)
                _path = os.path.join(_qkv_dump_dir, f"block_{int(self.debug_block_id):02d}.pt")
                torch.save(
                    {
                        "q": query_states.detach().cpu().to(torch.bfloat16).contiguous(),
                        "k": key_states.detach().cpu().to(torch.bfloat16).contiguous(),
                        "v": value_states.detach().cpu().to(torch.bfloat16).contiguous(),
                        "layout": "B_H_L_D",
                        "scale_ind": scale_ind,
                        "block_idx": int(self.debug_block_id),
                        "head_dim": self.head_dim,
                        "num_heads": self.num_heads,
                    },
                    _path,
                )
                if int(self.debug_block_id) == 0:
                    print(
                        f"[INFINITYSTAR_QKV_DUMP] last scale si={scale_ind} -> {os.path.abspath(_qkv_dump_dir)}"
                    )
            # print(f"q.shape:{query_states.shape}, k.shape:{key_states.shape}, v.shape:{value_states.shape}")
            if self.use_flex_attn and attn_fn is not None:
                attn_output = attn_fn(query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states, scale=scale).transpose(1, 2).reshape(B, L, C)
            else:
                _attn_ctx = (
                    f"SelfAttention block={self.debug_block_id} scale={scale_ind} "
                    f"B={B} L={L} sage={int(self.using_sageattn)} q_bits={self.q_bits} "
                    f"quant={int(self.enable_quantization)}"
                )
                check_debug_tensor_finite("query_states_pre_attn", query_states, _attn_ctx)
                check_debug_tensor_finite("key_states_pre_attn", key_states, _attn_ctx)
                check_debug_tensor_finite("value_states_pre_attn", value_states, _attn_ctx)
                # Prefer fused int4 KV path only for strict Phase-1 constraints; otherwise fallback to baseline.
                use_fused_int4 = (
                    self.enable_fused_kv_flashattn
                    and self._varq_available
                    and self.k_varq is not None
                    and self.v_varq is not None
                    and isinstance(scale_ind, int)
                    and (not sp_manager.sp_on())
                    and self.q_bits == 4
                )
                if use_fused_int4:
                    ref_scale_inds = context_info[scale_ind]['ref_sids'] + ref_text_scale_inds
                    can_fuse = (
                        len(ref_scale_inds) == 1
                        and isinstance(ref_scale_inds[0], int)
                        and ref_scale_inds[0] in self.k_varq.live_scale_ids()
                    )
                    if can_fuse:
                        sid = ref_scale_inds[0]
                        k_q, k_s, k_meta = self.k_varq.get_scale_quantized(sid)
                        v_q, v_s, v_meta = self.v_varq.get_scale_quantized(sid)
                        can_fuse = (
                            k_meta is not None and v_meta is not None
                            and int(k_meta.get("bits", -1)) == 4
                            and int(v_meta.get("bits", -1)) == 4
                            and k_s.dim() == 4 and v_s.dim() == 4
                            and k_s.size(2) > 0 and v_s.size(2) > 0
                        )
                    if can_fuse:
                        from flash_attn import flash_attn_func_quant_kv_int4
                        # query_states/key_states/value_states are [B, H, L, D], FA expects [B, L, H, D]
                        q_fa = query_states.permute([0, 2, 1, 3]).to(torch.bfloat16)
                        attn_output = flash_attn_func_quant_kv_int4(
                            q_fa,
                            k_q.permute([0, 2, 1, 3]).contiguous(),
                            v_q.permute([0, 2, 1, 3]).contiguous(),
                            k_s[:, :, 0, :].contiguous(),
                            v_s[:, :, 0, :].contiguous(),
                            quant_group_id=0,
                            softmax_scale=scale,
                        )
                        attn_output = attn_output.reshape(B, L, C)
                    else:
                        from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
                        attn_output = flash_attn_func(query_states.permute([0,2,1,3]).to(torch.bfloat16), key_states.permute([0,2,1,3]).to(torch.bfloat16), value_states.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=scale)
                        attn_output = attn_output.reshape(B, L, C)
                else:
                    # fa2, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
                    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
                    if self.using_sageattn and self._sageattn_impl is not None:
                        # SageAttention expects (B, H, N, D) when tensor_layout="HND".
                        attn_output = self._sageattn_impl(
                            query_states.to(torch.bfloat16),
                            key_states.to(torch.bfloat16),
                            value_states.to(torch.bfloat16),
                            tensor_layout="HND",
                            is_causal=False,
                            sm_scale=scale,
                        ).transpose(1, 2).reshape(B, L, C)
                        check_debug_tensor_finite("sageattn_output", attn_output, _attn_ctx)
                    else:
                        attn_output = flash_attn_func(
                            query_states.permute([0, 2, 1, 3]).to(torch.bfloat16),
                            key_states.permute([0, 2, 1, 3]).to(torch.bfloat16),
                            value_states.permute([0, 2, 1, 3]).to(torch.bfloat16),
                            softmax_scale=scale,
                        )
                        attn_output = attn_output.reshape(B, L, C)
                        check_debug_tensor_finite("flashattn_output", attn_output, _attn_ctx)

                # fa3, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
                # from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func
                # attn_output = flash_attn_func(query_states.permute([0,2,1,3]).to(torch.bfloat16), key_states.permute([0,2,1,3]).to(torch.bfloat16), value_states.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=scale)
                # attn_output = attn_output[0].reshape(B, L, C)
                
                # slow attn
                # attn_output = slow_attn(query=query_states, key=key_states, value=value_states, scale=scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
            if sp_manager.sp_on():
                # [B, raw_L, C/sp] --> [B, raw_L/sp, C]
                sdim = 1
                gdim = 2
                attn_output = sp_all_to_all(attn_output, sdim, gdim)

            attn_output = self.o_proj(attn_output)
            check_debug_tensor_finite("attn_output_after_o_proj", attn_output, _attn_ctx)

            return attn_output
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim), this way
        
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
        v = v.contiguous()                                                  # bf16

        if sp_manager.sp_on():
            # Headnum need to be sharded and L needs to be gathered
            # [B, H, raw_L/sp, C] --> [B, H/sp, raw_L, C]
            sdim = 1
            gdim = 2

            L = L * sp_manager.get_sp_size()
            C = C // sp_manager.get_sp_size()

            q = sp_all_to_all(q, sdim, gdim)
            k = sp_all_to_all(k, sdim, gdim)
            v = sp_all_to_all(v, sdim, gdim)

        def rescale_qk(q: torch.Tensor, k: torch.Tensor):
            range_q = q.abs().max()
            range_k = k.abs().max()
            theta = torch.sqrt((range_k + 1e-6) / (range_q + 1e-6))
            q = q * theta
            k = k / theta
            return q, k
        q, k = apply_rotary_emb(q, k, rope2d_freqs_grid) #, freqs_cis=freqs_cis)
        if self.caching:    # kv caching: only used during inference
            if last_repetition_step:
                self.cached_k.append(k)
                self.cached_v.append(v)
            if scale_ind >= 0:
                ref_scale_inds = context_info[scale_ind]['ref_sids']
                k = torch.cat([self.cached_k[0]] + [self.cached_k[ind+1] for ind in ref_scale_inds] + [k], dim=L_dim)
                v = torch.cat([self.cached_v[0]] + [self.cached_v[ind+1] for ind in ref_scale_inds] + [v], dim=L_dim)

            ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
            for si in range(len(context_info)):
                for ref_si in context_info[si]['ref_sids']:
                    ref_scale_2_last_use_scale[ref_si] = si
            for ref_si in range(scale_ind):
                if (ref_scale_2_last_use_scale[ref_si] < scale_ind) and (self.cached_k[ref_si+1] is not None):
                    tmpk, tmpv = self.cached_k[ref_si+1], self.cached_v[ref_si+1]
                    self.cached_k[ref_si+1], self.cached_v[ref_si+1] = None, None
                    del tmpk, tmpv
        
        # if self.cos_attn: q, k are in fp32; v is in bf16
        # else: q, k, v are in bf16
        
        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q.to(v.dtype), k.to(v.dtype), v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            # oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
            # fa2, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
            from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
            oup = flash_attn_func(q.permute([0,2,1,3]).to(torch.bfloat16), k.permute([0,2,1,3]).to(torch.bfloat16), v.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=self.scale)
            oup = oup.reshape(B, L, C)
        # oup: bf16

        if sp_manager.sp_on():
            # [B, raw_L, C/sp] --> [B, raw_L/sp, C]
            sdim = 1
            gdim = 2
            oup = sp_all_to_all(oup, sdim, gdim)

        return self.proj_drop(self.proj(oup))
    
class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        cond_dim,
        num_heads,
        num_key_value_heads,
        mlp_ratio=4.0,
        use_flex_attn=False,
        pad_to_multiplier=1,
        rope2d_normalized_by_hw=False,
        mask_type="",
        context_frames=-1,
        steps_per_frame=-1,
        arch="var",
        qwen_qkvo_bias=False,
        inject_sync=False,
        q_bits=8,
        quant_method='G_SCALE_HEAD_DIM',
        qkv_format='BHLc',
        enable_quantization=False,
        rescale_qk=False,
        enable_fused_kv_flashattn=False,
        enable_sageattn=False,
        sageattn_type='sageattn',
    ):
        super(SelfAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.arch=arch
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_key_value_heads=num_key_value_heads,
            use_flex_attn=use_flex_attn, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            mask_type=mask_type, context_frames=context_frames, steps_per_frame=steps_per_frame, arch=arch, qwen_qkvo_bias=qwen_qkvo_bias,
                q_bits=q_bits, quant_method=quant_method, qkv_format=qkv_format, enable_quantization=enable_quantization, rescale_qk=rescale_qk,
                enable_fused_kv_flashattn=enable_fused_kv_flashattn,
                enable_sageattn=enable_sageattn, sageattn_type=sageattn_type,
        )
        if self.arch == 'qwen':
            self.mlp = Qwen3MLP(hidden_size=embed_dim, intermediate_size=round(embed_dim * mlp_ratio / 256) * 256)
            self.input_layernorm = FastRMSNorm(embed_dim)
            self.post_attention_layernorm = FastRMSNorm(embed_dim)
            self.inject_sync = inject_sync
        else:
            raise ValueError(f'arch {self.arch} not supported')
        
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, rope2d_freqs_grid=[], scale_schedule=[], scale_ind=0, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        _rec = getattr(self, '_latency_recorder', None)
        if _rec is not None:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        residual = x
        hidden_states = x
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds)
        hidden_states = residual + hidden_states

        if _rec is not None:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if _rec is not None:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()
            _rec['attn'] += _t1 - _t0
            _rec['ffn'] += _t2 - _t1

        return hidden_states
    

if __name__ == '__main__':
    pass
