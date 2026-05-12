#!/usr/bin/env python3
import torch
import torch.nn as nn

from VAR_Q.hooks import install_varq_hooks, remove_varq_hooks


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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockModel().to(device)
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
    model.attn.kv_caching(True)
    x = torch.randn(1, 4, 8, device=device)
    y = model.attn(x, attn_bias=None)
    remove_varq_hooks(handle)

    if y.shape != x.shape:
        raise RuntimeError(f"unexpected hook output shape: {tuple(y.shape)} != {tuple(x.shape)}")

    print(f"runtime hook smoke passed on {device}: output_shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
