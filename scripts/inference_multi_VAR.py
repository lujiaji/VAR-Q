import os
import os.path as osp
import torch, torchvision
import random
import argparse
import numpy as np
from ..VAR.utils.misc import create_npz_from_sample_folder
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from ..VAR.models import VQVAE, build_vae_var

#create argparser
parser = argparse.ArgumentParser()
parser.add_argument("--model_depth", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cfg", type=float, default=2.0)
parser.add_argument("--top_k", type=int, default=600)
parser.add_argument("--more_smooth", action="store_true", help="True for more smooth output")
parser.add_argument("--total_iters", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--save_path", type=str, default='../Benchmark/output/VAR/images')
args = parser.parse_args()

model_depth = args.model_depth
assert model_depth in {16, 20, 24, 30, 36}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{model_depth}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
if model_depth == 36:
    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32) # model with depth 36 generates 512*512px images
    FOR_512px = True
else:
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16) # model with other depth generates 256*256px images
    FOR_512px = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=model_depth, shared_aln=FOR_512px,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

# set args
seed = args.seed #@param {type:"number"}
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = args.cfg #@param {type:"slider", min:1, max:10, step:0.1}
top_k = args.top_k #@param {type:"slider", min:600, max:900, step:1}
class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
more_smooth = args.more_smooth # True for more smooth output
images_per_iter = args.batch_size
total_iters = args.total_iters
save_path = args.save_path

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
                top_p = 0.95, 
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