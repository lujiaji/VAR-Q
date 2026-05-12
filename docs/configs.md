# VAR-Q Configs

VAR-Q JSON configs are intentionally small. Model weights, dataset paths, generated outputs, and backend-specific dependency settings should not be stored in these files.

## `quantization` Block

```json
{
  "quantization": {
    "enable": true,
    "q_bits": 4,
    "quant_method": "VARQ",
    "qkv_format": "BLHc",
    "pack_to_int32": true,
    "compression_ratio": 1.0,
    "max_scale_seq_len": 1560,
    "rescale_qk": false
  }
}
```

| Field | Meaning |
| --- | --- |
| `enable` | Enables runtime KV-cache quantization. If false, hooks can be installed but quantization is bypassed. |
| `q_bits` | Number of bits used for K/V cache values. Pack-supported values are `2`, `3`, `4`, `6`, and `8`. |
| `q_bits_k`, `q_bits_v` | Optional K/V-specific bit widths. If omitted, both use `q_bits`. |
| `quant_method` | Quantization strategy. Use `VARQ` for the main method, `G_*` for grouping ablations, and `KIVI` / `FLexGen` / `KVQuant` for ablation baselines. |
| `qkv_format` | Input tensor layout for Q/K/V states: `BLHc` means batch, sequence, head, head-dim; `BHLc` means batch, head, sequence, head-dim. |
| `pack_to_int32` | Packs low-bit int values into int32 storage. This reduces cache memory and is the default path for supported bit widths. |
| `pack_to_int32_k`, `pack_to_int32_v` | Optional K/V-specific packing flags. |
| `compression_ratio` / `ratio` | Controls sequence grouping. `1.0` means the default grouping unit. Smaller values split more finely; larger values group longer chunks when `max_scale_seq_len` is provided. |
| `max_scale_seq_len` | Optional reference sequence length for chunked grouping. For next-frame video configs, `1560` is the default group unit. |
| `rescale_qk` | Enables optional Q/K range rescaling before attention. Default is false. |
| `kivi_group_size` | Group size used by KIVI-style ablation quantizers. |
| `kivi_cali_k_group_size`, `kivi_cali_v_group_size` | Calibration group sizes for KIVI calibration variants. |

## Method Boundaries

`VARQ` is the main method implemented in `VAR_Q`. It quantizes runtime KV-cache tensors and is not a weight-only quantization method.

`G_*` methods are grouping ablations of VAR-Q. They are useful for measuring how quantization granularity affects quality and memory:

| Method | Purpose |
| --- | --- |
| `G_TENSOR` | One scale for the whole tensor. |
| `G_SCALE_HEAD_DIM` | Scale/group-aware quantization across scale, head, and head dimension. |
| `G_HEAD_DIM` | Quantizes after concatenating cached and current KV along head/dim groups. |
| `G_SCALE` | Groups by generation scale. |
| `G_TOKEN` | Groups along token positions. |
| `G_TOKEN_HEAD` | Groups by token and head. |

`KIVI`, `FLexGen`, and `KVQuant` are ablation baselines implemented under `ablation/`. Public configs may use these names directly; the loader normalizes them to internal ablation method names. Ablation code must remain separated from the main `VAR_Q` implementation.

## Backend-Specific Fields

Full JSON files remain backend-specific because model families differ in tensor layout, schedule, prompt handling, and video chunking semantics. The `quantization` block is backend-agnostic enough to reuse manually through:

```python
from VAR_Q.hooks import install_varq_hooks

install_varq_hooks(model, model_type="var", quant_config=config["quantization"])
```
