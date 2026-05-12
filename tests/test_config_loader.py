import json

from VAR_Q.config_loader import VARQConfig


def test_config_loader_reads_public_quantization_block(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"model_type": "mock"},
                "quantization": {
                    "enable": True,
                    "q_bits": 4,
                    "quant_method": "FLexGen",
                    "qkv_format": "BLHc",
                    "pack_to_int32": True,
                },
                "inference": {"device": "cpu", "seed": 0},
            }
        ),
        encoding="utf-8",
    )

    cfg = VARQConfig(str(config_path))

    assert cfg.get_model_config()["model_type"] == "mock"
    assert cfg.get_quantization_config()["quant_method"] == "ABL_KV_FLexGen"
    assert cfg.get_checkpoint_config() == {}
    assert cfg.get_device() == "cpu"


def test_config_loader_defaults_optional_sections(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"model_type": "var"},
                "quantization": {"enable": False, "q_bits": 8, "quant_method": "VARQ"},
                "inference": {"device": "cpu"},
            }
        ),
        encoding="utf-8",
    )

    cfg = VARQConfig(str(config_path))

    assert cfg.get_ablation_config() == {}
    assert cfg.get_weight_quantization_config() == {}
    assert cfg.get_batch_processing_config() == {}
