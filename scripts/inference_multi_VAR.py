import os
import os.path as osp
import sys
import torch, torchvision
import random
import argparse
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

THIS_DIR = osp.dirname(osp.abspath(__file__))
VARQ_ROOT = osp.dirname(THIS_DIR)
if VARQ_ROOT not in sys.path:
    sys.path.append(VARQ_ROOT)

from VAR_Q.hooks import install_varq_hooks
from VAR_Q.paths import prepend_sys_path, require_third_party_repo
from VAR_Q.profiling import (
    collect_varq_memory_breakdown,
    format_memory_breakdown,
    reset_cuda_memory_stats,
)

VAR_REPO_ROOT = require_third_party_repo("VAR", "https://github.com/FoundationVision/VAR")
prepend_sys_path([VARQ_ROOT, VAR_REPO_ROOT.parent])
os.chdir(VARQ_ROOT)

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from VAR_Q.config_loader import load_varq_config
from VAR.models import build_vae_var_from_config
from VAR.utils.misc import create_npz_from_sample_folder

parser = argparse.ArgumentParser(description='VAR-Q Multi-Image Inference')
parser.add_argument(
    "--config",
    type=str,
    default="configs/var/varq/base/VAR-VARQ-8.json",
    help="Path to config file",
)
parser.add_argument("--total_iters", type=int, default=None, help="Total number of iterations (overrides config)")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size per iteration (overrides config)")
parser.add_argument("--save_path", type=str, default='Benchmark/output/VAR/images', help="Save path for generated images")
parser.add_argument("--profile_memory", action="store_true", help="Print VAR-Q cache and CUDA allocator memory stats")
parser.add_argument("--skip_npz", action="store_true", help="Skip ImageNet-style NPZ export for smoke runs")
parser.add_argument("--vae_ckpt", type=str, default=os.environ.get("VARQ_VAE_CKPT"), help="Path to VAR VAE checkpoint")
parser.add_argument(
    "--var_ckpt_template",
    type=str,
    default=os.environ.get("VARQ_VAR_CKPT_TEMPLATE"),
    help="Path template for VAR checkpoint, e.g. /path/to/var_d{}.pth",
)
args = parser.parse_args()

config = load_varq_config(args.config)

if 'batch_processing' not in config.config:
    config.config['batch_processing'] = {
        'total_iters': 1000,
        'batch_size': 50
    }

if args.total_iters is not None:
    config.config['batch_processing']['total_iters'] = args.total_iters
if args.batch_size is not None:
    config.config['batch_processing']['batch_size'] = args.batch_size

model_depth = config.get_model_config()['depth']
assert model_depth in {16, 20, 24, 30, 36}


if not args.vae_ckpt or not args.var_ckpt_template:
    raise ValueError(
        "VAR checkpoint paths are not stored in public JSON configs. "
        "Pass --vae_ckpt and --var_ckpt_template, or set VARQ_VAE_CKPT and VARQ_VAR_CKPT_TEMPLATE."
    )
vae_ckpt = args.vae_ckpt
var_ckpt = args.var_ckpt_template.format(model_depth)

def require_checkpoint(local_path: str):
    if osp.exists(local_path):
        return
    raise FileNotFoundError(
        f"Required checkpoint not found: {local_path}. "
        "This script no longer auto-downloads checkpoints; pass the local weights path via CLI or environment variables."
    )

require_checkpoint(vae_ckpt)
require_checkpoint(var_ckpt)

device = config.get_device()
print(f"Using device: {device}")

print("Building models from configuration...")
vae, var = build_vae_var_from_config(config.config, device=device)

print("Loading checkpoints...")
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
if config.get_quantization_config().get("enable", False):
    install_varq_hooks(
        var,
        "var",
        config.get_quantization_config(),
        ablation_config=config.get_ablation_config(),
    )
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)

print(f'Model preparation finished.')
if args.profile_memory:
    reset_cuda_memory_stats()

inference_config = config.get_inference_config()
seed = inference_config['seed']
cfg = inference_config['cfg']
top_k = inference_config['top_k']
top_p = inference_config['top_p']
more_smooth = inference_config['more_smooth']

batch_config = config.config.get('batch_processing', {})
images_per_iter = batch_config.get('batch_size', 50)
total_iters = batch_config.get('total_iters', 1000)
save_path = args.save_path

os.makedirs(save_path, exist_ok=True)

print(f"Inference parameters:")
print(f"  Model depth: {model_depth}")
print(f"  Seed: {seed}")
print(f"  CFG: {cfg}")
print(f"  Top-k: {top_k}")
print(f"  Top-p: {top_p}")
print(f"  More smooth: {more_smooth}")
print(f"  Batch size: {images_per_iter}")
print(f"  Total iterations: {total_iters}")
print(f"  Save path: {save_path}")

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
        for step in range(total_iters):
            print(f"Generating {images_per_iter} images for class {step}...")
            label_B = torch.full((images_per_iter,), step, dtype=torch.long, device=device)
            result = var.autoregressive_infer_cfg(
                B = images_per_iter, 
                label_B = label_B, 
                cfg = cfg, 
                top_k = top_k, 
                top_p = top_p, 
                g_seed = seed, 
                more_smooth = more_smooth
            )
            if args.profile_memory:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                stats = collect_varq_memory_breakdown(var)
                print("[VAR-Q memory] " + format_memory_breakdown(stats))
            for i in range(images_per_iter):
                img = result[i].clone()
                img = img.permute(1, 2, 0).mul_(255).cpu().numpy()
                img = PImage.fromarray(img.astype(np.uint8))
                img.save(os.path.join(save_path, f"iters{step}_img{i}.png"))

if args.skip_npz:
    print("Image generation complete >> Skip npz export")
else:
    print(f"Image generation complete >> Generating npz")
    npz_path = create_npz_from_sample_folder(save_path)
    print(f"Image generation complete >> Generate npz -->  {npz_path}")
