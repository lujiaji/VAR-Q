from typing import Tuple, Dict, Any
import torch.nn as nn

from .quant import VectorQuantizer2
from .var import VAR
from .vqvae import VQVAE


def build_vae_var(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp


def build_vae_var_from_config(config: Dict[str, Any], device: str = None) -> Tuple[VQVAE, VAR]:
    """
    Build VAE and VAR models from configuration dictionary
    
    Args:
        config: Configuration dictionary containing model parameters
        device: Device to place models on. If None, uses config['inference']['device']
        
    Returns:
        Tuple of (VQVAE, VAR) models
    """
    # Extract configurations
    model_config = config.get('model', {})
    inference_config = config.get('inference', {})
    quantization_config = config.get('quantization', {})
    
    # Get device
    if device is None:
        device = inference_config.get('device', 'cuda')
    
    # Get model depth and calculate derived parameters
    depth = model_config.get('depth', 16)
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # Set shared_aln based on model depth (depth 36 generates 512px images)
    shared_aln = (depth == 36)
    
    # Get patch numbers based on depth
    if depth == 36:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)  # for 512px images
    else:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)   # for 256px images
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # Build VAE with hard-coded parameters (as in original VAR)
    vae_local = VQVAE(
        vocab_size=4096,
        z_channels=32, 
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=patch_nums
    ).to(device)
    
    # Build VAR with hard-coded parameters (as in original VAR)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        num_classes=1000,
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=shared_aln,
        cond_drop_rate=0.1,
        attn_l2_norm=True,
        patch_nums=patch_nums,
        flash_if_available=True,
        fused_if_available=True,
        q_bits=quantization_config.get('q_bits', 8),
        quant_method=quantization_config.get('quant_method', 'G_SCALE_HEAD_DIM'),
        qkv_format=quantization_config.get('qkv_format', 'BLHc'),
        enable_quantization=quantization_config.get('enable', False),
    ).to(device)
    
    # Initialize weights with hard-coded parameters
    var_wo_ddp.init_weights(
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1
    )
    
    return vae_local, var_wo_ddp
