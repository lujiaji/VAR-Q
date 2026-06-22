import torch
import torch.nn as nn
import pytest

from VAR_Q.hooks import VideoKVCacheAdapter, install_varq_hooks, is_hooked, remove_varq_hooks


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_idx = 0
        self.num_heads = 2
        self.head_dim = 4
        self.mat_qkv = nn.Linear(8, 24, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(8))
        self.zero_k_bias = nn.Parameter(torch.zeros(8), requires_grad=False)
        self.v_bias = nn.Parameter(torch.zeros(8))
        self.proj = nn.Linear(8, 8)
        self.proj_drop = nn.Identity()
        self.scale = self.head_dim**-0.5
        self.caching = False
        self.using_flash = False
        self.using_xform = False
        self.attn_l2_norm = False

    def kv_caching(self, enable: bool):
        self.caching = enable
        return self

    def forward(self, x, attn_bias=None):
        return self.proj(x)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SelfAttention()


class RenamedAttention(SelfAttention):
    pass


class NoHitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = RenamedAttention()


def test_mock_attention_install_quantize_and_remove_hook():
    model = MockModel()
    original_forward = model.attn.forward

    handle = install_varq_hooks(
        model,
        "var",
        {
            "enable": True,
            "q_bits": 4,
            "quant_method": "VARQ",
            "qkv_format": "BLHc",
            "pack_to_int32": True,
        },
        require_hits=True,
    )

    assert is_hooked(model)
    assert handle.hit_count >= 1
    assert len(handle.modules) == 1
    assert model.attn.forward.__func__ is not original_forward.__func__

    model.attn.kv_caching(True)
    x = torch.randn(1, 3, 8)
    y = model.attn(x, attn_bias=None)

    assert y.shape == x.shape
    assert model.attn.k_quant is not None
    assert model.attn.v_quant is not None

    remove_varq_hooks(handle)

    assert not is_hooked(model)
    assert model.attn.forward.__func__ is original_forward.__func__


def test_install_varq_hooks_reports_zero_hits():
    model = NoHitModel()

    with pytest.raises(RuntimeError, match="patched 0 attention modules.*scanned .*modules.*SelfAttention"):
        install_varq_hooks(
            model,
            "var",
            {
                "enable": True,
                "q_bits": 4,
                "quant_method": "VARQ",
                "qkv_format": "BLHc",
                "pack_to_int32": True,
            },
            require_hits=True,
        )

    handle = install_varq_hooks(
        model,
        "var",
        {
            "enable": True,
            "q_bits": 4,
            "quant_method": "VARQ",
            "qkv_format": "BLHc",
            "pack_to_int32": True,
        },
        require_hits=False,
    )

    assert handle.hit_count == 0
    assert handle.modules == []
    remove_varq_hooks(handle)
    assert not is_hooked(model)


def test_runtime_hook_rejects_qkv_layout_head_mismatch():
    model = MockModel()
    handle = install_varq_hooks(
        model,
        "var",
        {
            "enable": True,
            "q_bits": 4,
            "quant_method": "VARQ",
            "qkv_format": "BLHc",
            "pack_to_int32": True,
        },
        require_hits=True,
    )
    model.attn.kv_caching(True)

    model.attn.num_heads = 1
    model.attn.head_dim = 8
    with pytest.raises(RuntimeError, match="expected layout=BHLc.*q\\.shape"):
        model.attn(torch.randn(1, 3, 8), attn_bias=None)

    remove_varq_hooks(handle)


def test_video_kv_cache_adapter_skip_last_scale():
    adapter = VideoKVCacheAdapter(
        {
            "enable": True,
            "q_bits": 4,
            "quant_method": "VARQ",
            "qkv_format": "BLHc",
            "pack_to_int32": True,
            "max_scale_seq_len": 4,
            "skip_cache_last_scale": True,
        }
    )
    k0 = torch.randn(1, 4, 2, 8)
    v0 = torch.randn(1, 4, 2, 8)
    k1 = torch.randn(1, 4, 2, 8)
    v1 = torch.randn(1, 4, 2, 8)

    out_k0, out_v0 = adapter.update(k0, v0, scale_idx=0, num_scales=2)
    assert out_k0.shape == k0.shape
    assert out_v0.shape == v0.shape

    out_k1, out_v1 = adapter.update(k1, v1, scale_idx=1, num_scales=2)
    assert out_k1.shape[1] == k0.shape[1] + k1.shape[1]
    assert out_v1.shape[1] == v0.shape[1] + v1.shape[1]

    stats = adapter.memory_breakdown()
    assert stats["packed_kv_bytes"] > 0
    assert stats["dequant_workspace_bytes"] == 0
