"""The flag must select the fused path; default must be unchanged."""
from VAR_Q.hooks import runtime


def test_flag_default_off():
    cfg = {}
    assert bool(cfg.get("enable_fused_kv_flashattn", False)) is False


def test_dispatch_helper_exists():
    # A small pure-python dispatcher we can unit-test without a GPU/model.
    assert hasattr(runtime, "_should_use_fused_kv_attn")


def test_dispatch_requires_flag_and_last_scale():
    f = runtime._should_use_fused_kv_attn
    assert f(enabled=True, is_last_scale=True, qkv_format="BHLc", bits=8) is True
    assert f(enabled=False, is_last_scale=True, qkv_format="BHLc", bits=8) is False
    # fusion is defined for the last (two-segment) step only in v1
    assert f(enabled=True, is_last_scale=False, qkv_format="BHLc", bits=8) is False
    # v1 supports q8 only
    assert f(enabled=True, is_last_scale=True, qkv_format="BHLc", bits=4) is False
