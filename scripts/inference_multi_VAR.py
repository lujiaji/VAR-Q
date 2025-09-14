import os
import os.path as osp
import sys
import torch, torchvision
import random
import argparse
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

# Add VAR-Q directory to Python path
project_root = 'VAR-Q'
sys.path.append(project_root)
os.chdir(project_root)

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

# Import configuration system and models
from VAR_Q.config_loader import load_varq_config
from VAR.models import build_vae_var_from_config
from VAR.utils.misc import create_npz_from_sample_folder

# Create simple argparser - most parameters now in config
parser = argparse.ArgumentParser(description='VAR-Q Multi-Image Inference')
parser.add_argument("--config", type=str, default=None, help="Path to config file")
parser.add_argument("--total_iters", type=int, default=None, help="Total number of iterations (overrides config)")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size per iteration (overrides config)")
parser.add_argument("--save_path", type=str, default='Benchmark/outputs/VAR/images', help="Save path for generated images")
args = parser.parse_args()

# Load configuration
config = load_varq_config(args.config)

# Add batch processing parameters to config if not present
if 'batch_processing' not in config.config:
    config.config['batch_processing'] = {
        'total_iters': 1000,
        'batch_size': 50
    }

# Override config with command line arguments
if args.total_iters is not None:
    config.config['batch_processing']['total_iters'] = args.total_iters
if args.batch_size is not None:
    config.config['batch_processing']['batch_size'] = args.batch_size

# Get final configuration values
model_depth = config.get_model_config()['depth']
assert model_depth in {16, 20, 24, 30, 36}


# Get checkpoint paths from config
vae_ckpt, var_ckpt = config.get_checkpoint_paths(model_depth)
hf_home = config.get_checkpoint_config()['hf_home']

# Download checkpoints if they don't exist
if not osp.exists(vae_ckpt):
    print(f"Downloading VAE checkpoint from {hf_home}/{osp.basename(vae_ckpt)}")
    os.system(f'wget {hf_home}/{osp.basename(vae_ckpt)}')
if not osp.exists(var_ckpt):
    print(f"Downloading VAR checkpoint from {hf_home}/{osp.basename(var_ckpt)}")
    os.system(f'wget {hf_home}/{osp.basename(var_ckpt)}')

# Get device from config
device = config.get_device()
print(f"Using device: {device}")

# Build models using configuration
print("Building models from configuration...")
vae, var = build_vae_var_from_config(config.config, device=device)

# Load checkpoints
print("Loading checkpoints...")
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'Model preparation finished.')

# Get inference parameters from config
inference_config = config.get_inference_config()
seed = inference_config['seed']
cfg = inference_config['cfg']
top_k = inference_config['top_k']
top_p = inference_config['top_p']
more_smooth = inference_config['more_smooth']

# Get batch processing parameters
batch_config = config.config.get('batch_processing', {})
images_per_iter = batch_config.get('batch_size', 50)
total_iters = batch_config.get('total_iters', 1000)
save_path = args.save_path

# Create save directory if it doesn't exist
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

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
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
            for i in range(images_per_iter):
                img = result[i].clone()
                img = img.permute(1, 2, 0).mul_(255).cpu().numpy()
                img = PImage.fromarray(img.astype(np.uint8))
                img.save(os.path.join(save_path, f"iters{step}_img{i}.png"))

print(f"Image generation complete >> Generating npz")
npz_path = create_npz_from_sample_folder(save_path)
print(f"Image generation complete >> Generate npz -->  {npz_path}")