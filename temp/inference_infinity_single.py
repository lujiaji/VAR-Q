#!/usr/bin/env python3
"""
Infinity Single Image Inference Script with Timing
Based on the notebook configuration and model paths
"""

import random
import torch
import cv2
import numpy as np
import os
import os.path as osp
import sys
import argparse
import time
import json
from datetime import datetime

# Set project root and add to path
project_root = '/home/jiaji_lu/AR/VAR-Q'
os.chdir(project_root)
sys.path.append('/home/jiaji_lu/AR/VAR-Q')
sys.path.append('/home/jiaji_lu/AR/VAR-Q/Infinity')

# Import Infinity modules
from Infinity.tools.run_infinity import *
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

def load_config(config_file):
    """Load configuration from JSON file"""
    print(f"[Config] Loading configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        model_config = config_data.get('model', {})
        inference_config = config_data.get('inference', {})
        checkpoint_config = config_data.get('checkpoints', {})
        
        print(f"[Config] Configuration loaded successfully!")
        print(f"[Config] Model: {model_config.get('model_type')}")
        print(f"[Config] Original Infinity inference (no quantization)")
        
        return model_config, inference_config, checkpoint_config
        
    except Exception as e:
        print(f"[Error] Failed to load configuration: {e}")
        raise

def create_args_from_config(model_config, inference_config, checkpoint_config):
    """Create args object from configuration"""
    
    # Determine model-specific parameters based on model type
    model_type = model_config.get('model_type', 'infinity_2b')
    
    if model_type == "infinity_2b":
        vae_type = 32
        apply_spatial_patchify = 0
        checkpoint_type = "torch"
    elif model_type == "infinity_8b":
        vae_type = 14
        apply_spatial_patchify = 1
        checkpoint_type = "torch_shard"
    else:
        # Default to 2b configuration
        vae_type = 32
        apply_spatial_patchify = 0
        checkpoint_type = "torch"
    
    # Create args object from configuration
    args = argparse.Namespace(
        # Model configuration
        model_type=model_type,
        pn='1M',  # Default to 1M, can be overridden in config if needed
        
        # Checkpoint paths from config
        model_path=checkpoint_config.get('model_path'),
        vae_path=checkpoint_config.get('vae_ckpt'),
        text_encoder_ckpt='/data/boxunxu/Infinity/flan-t5-xl',  # Update this path as needed
        
        # Model architecture
        vae_type=vae_type,
        apply_spatial_patchify=apply_spatial_patchify,
        checkpoint_type=checkpoint_type,
        
        # Model behavior
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_channels=2048,
        h_div_w_template=inference_config.get('h_div_w', 1.0),
        use_flex_attn=0,
        
        # System settings
        cache_dir='/dev/shm',
        seed=inference_config.get('seed', 0),
        bf16=1,
        save_file='tmp.jpg',
        enable_model_cache=0,  # Disable model caching by default
        
        # Additional required parameters
        cfg_insertion_layer=0,
        enable_positive_prompt=0,
        cfg=inference_config.get('cfg', 3.0),
        tau=inference_config.get('tau', 0.5),
    )
    
    print(f"[Args] Arguments created from configuration:")
    print(f"  - Model: {args.model_type}")
    print(f"  - Model path: {args.model_path}")
    print(f"  - VAE path: {args.vae_path}")
    print(f"  - VAE type: {args.vae_type}")
    print(f"  - Original Infinity (no quantization)")
    print(f"  - CFG: {args.cfg}")
    print(f"  - Tau: {args.tau}")
    print(f"  - Seed: {args.seed}")
    
    return args

def load_models(args):
    """Load all models with timing"""
    print("\n" + "="*50)
    print("[Model Loading] Starting model loading...")
    
    model_load_start = time.time()
    
    # Load text encoder
    print("[Model Loading] Loading text encoder...")
    text_tokenizer_start = time.time()
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    text_tokenizer_time = time.time() - text_tokenizer_start
    print(f"[Model Loading] Text encoder loaded in {text_tokenizer_time:.2f}s")
    
    # Load VAE
    print("[Model Loading] Loading VAE...")
    vae_start = time.time()
    vae = load_visual_tokenizer(args)
    vae_time = time.time() - vae_start
    print(f"[Model Loading] VAE loaded in {vae_time:.2f}s")
    
    # Load Infinity transformer
    print("[Model Loading] Loading Infinity transformer...")
    transformer_start = time.time()
    infinity = load_transformer(vae, args)
    transformer_time = time.time() - transformer_start
    print(f"[Model Loading] Infinity transformer loaded in {transformer_time:.2f}s")
    
    total_model_load_time = time.time() - model_load_start
    print(f"[Model Loading] Total model loading time: {total_model_load_time:.2f}s")
    print("="*50)
    
    return text_tokenizer, text_encoder, vae, infinity, total_model_load_time

def generate_image(infinity, vae, text_tokenizer, text_encoder, prompt, args, inference_config):
    """Generate a single image with timing"""
    print(f"\n[Inference] Starting inference...")
    print(f"[Inference] Prompt: {prompt}")
    
    # Use inference parameters from configuration
    cfg = inference_config.get('cfg', 3.0)
    tau = inference_config.get('tau', 0.5)
    h_div_w = inference_config.get('h_div_w', 1.0)  # aspect ratio, height:width
    seed = inference_config.get('seed', 0)
    enable_positive_prompt = inference_config.get('enable_positivee_prompt', 0)  # Note: config has typo
    
    print(f"[Inference] Using parameters:")
    print(f"  - CFG: {cfg}")
    print(f"  - Tau: {tau}")
    print(f"  - H/W ratio: {h_div_w}")
    print(f"  - Seed: {seed}")
    print(f"  - Enable positive prompt: {enable_positive_prompt}")
    
    # Load dynamic resolution
    print("[Inference] Loading dynamic resolution...")
    # dynamic_resolution_h_w and h_div_w_templates are already imported
    
    # Prepare scale schedule
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    
    # Reset attention timing statistics before inference
    infinity.reset_attention_timing_stats()
    
    # Start inference timing
    inference_start = time.time()
    
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
    )
    
    inference_time = time.time() - inference_start
    
    # Get attention timing statistics
    attention_stats = infinity.get_attention_timing_stats()
    
    print(f"[Inference] Image generation completed in {inference_time:.2f}s")
    
    # Print attention timing breakdown
    total_attention_time = attention_stats['total_attention_time']
    attention_percentage = (total_attention_time / inference_time) * 100 if inference_time > 0 else 0
    
    print(f"\n[Attention Timing Analysis]")
    print(f"  Total Attention Time: {total_attention_time:.4f}s ({attention_percentage:.1f}% of inference)")
    
    self_attn = attention_stats['self_attention']
    cross_attn = attention_stats['cross_attention']
    
    if self_attn['total_calls'] > 0:
        self_percentage = (self_attn['total_time'] / inference_time) * 100 if inference_time > 0 else 0
        print(f"  Self-Attention: {self_attn['total_time']:.4f}s ({self_percentage:.1f}% of inference)")
        print(f"    - Total calls: {self_attn['total_calls']}")
        print(f"    - Average per call: {self_attn['avg_time']:.4f}s")
    
    if cross_attn['total_calls'] > 0:
        cross_percentage = (cross_attn['total_time'] / inference_time) * 100 if inference_time > 0 else 0
        print(f"  Cross-Attention: {cross_attn['total_time']:.4f}s ({cross_percentage:.1f}% of inference)")
        print(f"    - Total calls: {cross_attn['total_calls']}")
        print(f"    - Average per call: {cross_attn['avg_time']:.4f}s")
    
    return generated_image, inference_time, attention_stats

def save_image(generated_image, output_path):
    """Save generated image"""
    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, generated_image.cpu().numpy())
    print(f'[Save] Image saved to {osp.abspath(output_path)}')

def main():
    parser = argparse.ArgumentParser(description='Infinity Single Image Inference with Timing')
    parser.add_argument('--config', type=str, 
                       default='/home/jiaji_lu/AR/VAR-Q/temp/Infinity-VAR_Q-8.json',
                       help='Path to configuration file')
    parser.add_argument('--prompt', type=str, 
                       default='alien spaceship enterprise',
                       help='Text prompt for image generation')
    parser.add_argument('--output', type=str,
                       default='/home/jiaji_lu/AR/VAR-Q/temp/generated_image.jpg',
                       help='Output image path')
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device ID')
    
    args_cmd = parser.parse_args()
    
    # Set CUDA device
    torch.cuda.set_device(args_cmd.device)
    print(f"[System] Using CUDA device: {args_cmd.device}")
    
    # Record start time
    total_start_time = time.time()
    
    try:
        # Load configuration
        model_config, inference_config, checkpoint_config = load_config(args_cmd.config)
        
        # Create args from config
        args = create_args_from_config(model_config, inference_config, checkpoint_config)
        
        # Load models
        text_tokenizer, text_encoder, vae, infinity, model_load_time = load_models(args)
        
        # Generate image
        generated_image, inference_time, attention_stats = generate_image(
            infinity, vae, text_tokenizer, text_encoder, 
            args_cmd.prompt, args, inference_config
        )
        
        # Save image
        save_image(generated_image, args_cmd.output)
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Print timing summary
        print("\n" + "="*50)
        print("[Timing Summary]")
        print(f"  Model Loading Time: {model_load_time:.2f}s")
        print(f"  Inference Time (Pure): {inference_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Overhead Time: {total_time - model_load_time - inference_time:.2f}s")
        print("="*50)
        
        # Save timing results to file
        timing_results = {
            'timestamp': datetime.now().isoformat(),
            'prompt': args_cmd.prompt,
            'config_file': args_cmd.config,
            'model_type': args.model_type,
            'quantization_enabled': False,
            'inference_mode': 'Original Infinity (no quantization)',
            'timings': {
                'model_loading_time': model_load_time,
                'inference_time': inference_time,
                'total_time': total_time,
                'overhead_time': total_time - model_load_time - inference_time,
                'attention_breakdown': {
                    'total_attention_time': attention_stats['total_attention_time'],
                    'attention_percentage': (attention_stats['total_attention_time'] / inference_time) * 100 if inference_time > 0 else 0,
                    'self_attention': attention_stats['self_attention'],
                    'cross_attention': attention_stats['cross_attention']
                }
            }
        }
        
        timing_file = args_cmd.output.replace('.jpg', '_timing.json')
        with open(timing_file, 'w') as f:
            json.dump(timing_results, f, indent=2)
        print(f"[Timing] Results saved to {timing_file}")
        
    except Exception as e:
        print(f"[Error] Script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
