import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys

# Add project paths
project_root = 'VAR-Q'
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Infinity'))

import cv2
import tqdm
import torch
import numpy as np
from pytorch_lightning import seed_everything

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
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--metadata_file', type=str, default='benchmark-prompts.json')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
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

    # parse cfg
    if isinstance(args.cfg, str):
        args.cfg = list(map(float, args.cfg.split(',')))
        if len(args.cfg) == 1:
            args.cfg = args.cfg[0]
    # If cfg is already a number from config, keep it as is
    
    with open(args.metadata_file) as fp:
        metadatas = json.load(fp)

    if args.model_type == 'sdxl':
        from diffusers import DiffusionPipeline
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
    elif args.model_type == 'sd3':
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    elif args.model_type == 'pixart_sigma':
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        ).to("cuda")
    elif args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
        # Load dynamic resolution for Infinity
        print("[Infinity] Loading dynamic resolution...")
        dynamic_resolution_h_w, h_div_w_templates = _import_dynamic_resolution()
        print("[Infinity] Dynamic resolution loaded successfully!")
        
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)

        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])
    
    save_metadatas = []
    for index, metadata in enumerate(metadatas):
        seed_everything(args.seed)
        prompt_id = metadata['id']
        prompt = metadata['prompt']
        sample_path = os.path.join(args.outdir, prompt_id)
        os.makedirs(sample_path, exist_ok=True)
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        tau = args.tau
        cfg = args.cfg
        if args.rewrite_prompt:
            refined_prompt = prompt_rewriter.rewrite(prompt)
            input_key_val = extract_key_val(refined_prompt)
            prompt = input_key_val['prompt']
            print(f'prompt: {prompt}, refined_prompt: {refined_prompt}')
        
        images = []
        for _ in range(args.n_samples):
            t1 = time.time()
            if args.model_type == 'sdxl':
                image = base(
                    prompt=prompt,
                    num_inference_steps=40,
                    denoising_end=0.8,
                    output_type="latent",
                ).images
                image = refiner(
                    prompt=prompt,
                    num_inference_steps=40,
                    denoising_start=0.8,
                    image=image,
                ).images[0]
            elif args.model_type == 'sd3':
                image = pipe(
                    prompt,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev_schnell':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif args.model_type == 'pixart_sigma':
                image = pipe(prompt).images[0]
            elif 'infinity' in args.model_type:
                # Use h_div_w from config or args
                h_div_w = getattr(args, 'h_div_w_template', 1.0)
                h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
                scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
                image = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type)
            else:
                raise ValueError
            t2 = time.time()
            print(f'{args.model_type} infer one image takes {t2-t1:.2f}s')
            images.append(image)
        
        os.makedirs(sample_path, exist_ok=True)
        metadata['gen_image_paths'] = []
        for i, image in enumerate(images):
            save_file_path = os.path.join(sample_path, f"{prompt_id}_{i}.jpg")
            if 'infinity' in args.model_type:
                cv2.imwrite(save_file_path, image.cpu().numpy())
            else:
                image.save(save_file_path)
            metadata['gen_image_paths'].append(save_file_path)
        print(save_file_path)
        save_metadatas.append(metadata)

        save_metadata_file_path = os.path.join(args.outdir, "metadata.jsonl")
        with open(save_metadata_file_path, "w") as fp:
            json.dump(save_metadatas, fp)
