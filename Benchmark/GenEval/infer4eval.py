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
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from VAR_Q.paths import require_third_party_repo
from VAR_Q.hooks import install_varq_hooks
from VAR_Q.infinity_config import add_infinity_config_file_argument, apply_infinity_config

_INFINITY_ROOT = str(require_third_party_repo('Infinity', 'https://github.com/FoundationVision/Infinity'))
if _INFINITY_ROOT not in sys.path:
    sys.path.append(_INFINITY_ROOT)

import cv2
import tqdm
import torch
import numpy as np
from pytorch_lightning import seed_everything

from infinity.utils.csv_util import load_csv_as_dicts, write_dicts2csv_file
from tools.run_infinity import *
from tools.run_infinity import _import_dynamic_resolution
from conf import HF_TOKEN, HF_HOME


# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='prompts/evaluation_metadata.jsonl')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--load_rewrite_prompt_cache', type=int, default=1, choices=[0,1])
    add_infinity_config_file_argument(parser)
    args = parser.parse_args()
    
    apply_infinity_config(args)

    # parse cfg
    if isinstance(args.cfg, str):
        args.cfg = list(map(float, args.cfg.split(',')))
        if len(args.cfg) == 1:
            args.cfg = args.cfg[0]
    # If cfg is already a number from config, keep it as is
    
    metadata_file = args.metadata_file
    if not os.path.isabs(metadata_file):
        metadata_file = os.path.join(_THIS_DIR, metadata_file)
    with open(metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    prompt_rewrite_cache_file = osp.join('GenEval', 'prompt_rewrite_cache.json')
    if osp.exists(prompt_rewrite_cache_file):
        with open(prompt_rewrite_cache_file, 'r') as f:
            prompt_rewrite_cache = json.load(f)
    else:
        prompt_rewrite_cache = {}

    if args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
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
        install_varq_hooks(
            infinity,
            "infinity",
            vars(args),
            ablation_config=getattr(args, "ablation_config", None),
        )

        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])

    for index, metadata in enumerate(metadatas):
        seed_everything(args.seed)
        outpath = os.path.join(args.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        # Idempotent skip: if all n_samples already on disk, skip this prompt.
        existing = [f for f in os.listdir(sample_path) if f.endswith('.jpg') or f.endswith('.png')]
        if len(existing) >= args.n_samples:
            print(f"Prompt ({index: >3}/{len(metadatas)}) skip (cached): '{prompt}'")
            continue
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        tau = args.tau
        cfg = args.cfg
        if args.rewrite_prompt:
            old_prompt = prompt
            if args.load_rewrite_prompt_cache and prompt in prompt_rewrite_cache:
                prompt = prompt_rewrite_cache[prompt]
            else:
                refined_prompt = prompt_rewriter.rewrite(prompt)
                input_key_val = extract_key_val(refined_prompt)
                prompt = input_key_val['prompt']
                prompt_rewrite_cache[prompt] = prompt
            print(f'old_prompt: {old_prompt}, refined_prompt: {prompt}')
            
        images = []
        for sample_j in range(args.n_samples):
            print(f"Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
            t1 = time.time()
            if args.model_type == 'flux_1_dev':
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
        for i, image in enumerate(images):
            save_file = os.path.join(sample_path, f"{i:05}.jpg")
            if 'infinity' in args.model_type:
                cv2.imwrite(save_file, image.cpu().numpy())
            else:
                image.save(save_file)
    
        with open(prompt_rewrite_cache_file, 'w') as f:
            json.dump(prompt_rewrite_cache, f, indent=2)
