import os
import os.path as osp
import time
import argparse
import json
import cv2
import numpy as np
import csv
import sys
from pytorch_lightning import seed_everything

# Add project paths
project_root = 'VAR-Q'
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Infinity'))

from infinity.utils.csv_util import load_csv_as_dicts, write_dicts2csv_file
from tools.run_infinity import *
from tools.run_infinity import _import_dynamic_resolution
from conf import HF_TOKEN, HF_HOME

# Add VAR_Q to path for config loading
var_q_path = 'VAR-Q/VAR_Q'
if os.path.exists(var_q_path):
    sys.path.append(var_q_path)
    from config_loader import VARQConfig

# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='DPG_prompts.jsonl')
    # Note: q_bits, quant_method, qkv_format, config_file are already defined in add_common_arguments
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config_file and os.path.exists(args.config_file):
        print(f"[Config] Loading configuration from {args.config_file}")
        try:
            config = VARQConfig(args.config_file)
            model_config = config.get_model_config()
            quant_config = config.get_quantization_config()
            inference_config = config.get_inference_config()
            checkpoint_config = config.get_checkpoint_config()
            
            # Override args with config values
            if 'model_type' in model_config:
                args.model_type = model_config['model_type']
            if 'model_path' in checkpoint_config:
                args.model_path = checkpoint_config['model_path']
            if 'vae_ckpt' in checkpoint_config:
                args.vae_path = checkpoint_config['vae_ckpt']
            if 'enable' in quant_config:
                args.enable_quantization = int(quant_config['enable'])
            if 'q_bits' in quant_config:
                args.q_bits = quant_config['q_bits']
            if 'quant_method' in quant_config:
                args.quant_method = quant_config['quant_method']
            if 'qkv_format' in quant_config:
                args.qkv_format = quant_config['qkv_format']
            if 'cfg' in inference_config:
                args.cfg = inference_config['cfg']
            if 'tau' in inference_config:
                args.tau = inference_config['tau']
            if 'seed' in inference_config:
                args.seed = inference_config['seed']
            if 'h_div_w' in inference_config:
                args.h_div_w_template = inference_config['h_div_w']
            
            # Set default text encoder path if not provided
            if not hasattr(args, 'text_encoder_ckpt') or not args.text_encoder_ckpt:
                args.text_encoder_ckpt = 'YOUR_PATH/flan-t5-xl'
            
            # Set model-specific parameters based on model type (only if not already set)
            if not hasattr(args, 'vae_type') or args.vae_type == 1:  # 1 is the default from add_common_arguments
                if args.model_type == "infinity_2b":
                    args.vae_type = 32
                    args.apply_spatial_patchify = 0
                    args.checkpoint_type = "torch"
                elif args.model_type == "infinity_8b":
                    args.vae_type = 14
                    args.apply_spatial_patchify = 1
                    args.checkpoint_type = "torch_shard"
                else:
                    # Default to 2b configuration
                    args.vae_type = 32
                    args.apply_spatial_patchify = 0
                    args.checkpoint_type = "torch"
                
            print(f"[Config] Configuration loaded successfully!")
            print(f"[Config] Model: {args.model_type}")
            print(f"[Config] VAR-Q Quantization: {'enabled' if args.enable_quantization else 'disabled'}")
            if args.enable_quantization:
                print(f"[Config]   - q_bits: {args.q_bits}")
                print(f"[Config]   - quant_method: {args.quant_method}")
                print(f"[Config]   - qkv_format: {args.qkv_format}")
        except Exception as e:
            print(f"[Error] Failed to load configuration: {e}")
            print("[Warning] Continuing with command-line arguments...")
    else:
        print("[Info] No configuration file provided, using command-line arguments")

    os.makedirs(args.outdir,exist_ok=True)
    grid_dir=osp.join(args.outdir,'dpg_images')
    os.makedirs(grid_dir,exist_ok=True)

    # parse cfg
    if isinstance(args.cfg, str):
        args.cfg = list(map(float, args.cfg.split(',')))
        if len(args.cfg) == 1:
            args.cfg = args.cfg[0]
    # If cfg is already a number from config, keep it as is
    
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    total=len(metadatas);print(f'total {total} prompts')

    if 'infinity' in args.model_type:
        # Load dynamic resolution for Infinity
        print("[Infinity] Loading dynamic resolution...")
        dynamic_resolution_h_w, h_div_w_templates = _import_dynamic_resolution()
        print("[Infinity] Dynamic resolution loaded successfully!")
        
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)

    rows=[]

    for index, metadata in enumerate(metadatas):
        seed_everything(args.seed)
        prompt = metadata['prompt']
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        tau = args.tau
        cfg = args.cfg
        images = []
        for sample_j in range(args.n_samples):
            print(f"Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
            t1 = time.time()
            # Use h_div_w from config or args
            h_div_w = getattr(args, 'h_div_w_template', 1.0)
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
            image = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type)
            t2 = time.time()
            print(f'{args.model_type} infer one image takes {t2-t1:.2f}s')
            images.append(image)

        #now prepare the DPG images
        arrs=[im.cpu().numpy() if hasattr(im, 'cpu') else np.array(im) for im in images]
        h, w = arrs[0].shape[:2]
        grid = np.zeros((2*h, 2*w, 3), dtype=arrs[0].dtype)
        grid[0:h,   0:w  ] = arrs[0]
        grid[0:h,   w:2*w] = arrs[1]
        grid[h:2*h, 0:w  ] = arrs[2]
        grid[h:2*h, w:2*w] = arrs[3]
        #save images
        item_id = f"{index:05}"
        out_img = osp.join(grid_dir, f"{item_id}.png")
        cv2.imwrite(out_img, grid)

        rows.append({
            'item_id': item_id,
            'text': prompt,
            'keywords': '',
            'proposition_id': '0',
            'dependency': '0',
            'category_broad': '',
            'category_detailed': '',
            'tuple': '',
            'question_natural_language':''
        })
    csv_path ='Benchmark/DPG/dpg_metadata.csv'
    fieldnames = [
        'item_id','text','keywords','proposition_id',
        'dependency','category_broad','category_detailed',
        'tuple','question_natural_language'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"save csv as: {csv_path}")

