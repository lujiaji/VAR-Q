import torch
import torch.nn as nn

from VAR_Q.hooks import install_varq_hooks, is_hooked, remove_varq_hooks


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
    )

    assert is_hooked(model)
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
