"""The flag must select the fused path; default must be unchanged."""
from VAR_Q.hooks import runtime


def test_flag_default_off():
    cfg = {}
    assert bool(cfg.get("enable_fused_kv_flashattn", False)) is False


def test_dispatch_helper_exists():
    assert hasattr(runtime, "_should_use_fused_kv_attn")


def test_dispatch_requires_flag_and_last_scale():
    f = runtime._should_use_fused_kv_attn
    assert f(enabled=True, is_last_scale=True, qkv_format="BHLc", bits=8) is True
    assert f(enabled=False, is_last_scale=True, qkv_format="BHLc", bits=8) is False
    assert f(enabled=True, is_last_scale=False, qkv_format="BHLc", bits=8) is False
    for bits in (2, 3, 4, 6, 8):
        assert f(enabled=True, is_last_scale=True, qkv_format="BHLc", bits=bits) is True
    assert f(enabled=True, is_last_scale=True, qkv_format="BHLc", bits=5) is False
