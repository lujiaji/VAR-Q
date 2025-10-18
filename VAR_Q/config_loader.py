"""
Configuration loader for VAR-Q model
"""
import json
import os
from typing import Dict, Any, Tuple
import torch


class VARQConfig:
    """Configuration manager for VAR-Q model (supports both VAR and Infinity)"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from JSON file
        
        Args:
            config_path: Path to config.json file. If None, uses default path.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration parameters"""
        return self.config['model'].copy()
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration parameters"""
        return self.config['quantization'].copy()
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration parameters"""
        return self.config['inference'].copy()
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint configuration parameters"""
        return self.config['checkpoints'].copy()
    
    
    def get_checkpoint_paths(self, model_depth: int = None) -> Tuple[str, str]:
        """
        Get checkpoint paths for VAE and model (VAR/Infinity)
        
        Args:
            model_depth: Model depth (for VAR models, optional for Infinity)
            
        Returns:
            Tuple of (vae_ckpt_path, model_ckpt_path)
        """
        checkpoint_config = self.get_checkpoint_config()
        vae_ckpt = checkpoint_config['vae_ckpt']
        
        # For Infinity models, use model_path directly
        if 'model_path' in checkpoint_config:
            model_ckpt = checkpoint_config['model_path']
        # For VAR models, use template with model_depth
        elif 'var_ckpt_template' in checkpoint_config and model_depth is not None:
            model_ckpt = checkpoint_config['var_ckpt_template'].format(model_depth)
        else:
            raise ValueError("Either 'model_path' or 'var_ckpt_template' with model_depth must be provided")
        
        return vae_ckpt, model_ckpt
    
    def get_device(self) -> str:
        """Get device configuration"""
        device = self.config['inference']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device
    
    def get_batch_processing_config(self) -> Dict[str, Any]:
        """Get batch processing configuration parameters"""
        return self.config.get('batch_processing', {}).copy()
    
    def is_infinity_model(self) -> bool:
        """Check if this is an Infinity model configuration"""
        model_config = self.get_model_config()
        return 'infinity' in model_config.get('model_type', '').lower()
    
    def is_var_model(self) -> bool:
        """Check if this is a VAR model configuration"""
        return not self.is_infinity_model()
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary containing configuration updates
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
    
    def save_config(self, save_path: str = None):
        """
        Save current configuration to file
        
        Args:
            save_path: Path to save config. If None, saves to original path.
        """
        if save_path is None:
            save_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)


def load_varq_config(config_path: str = None) -> VARQConfig:
    """
    Convenience function to load VAR-Q configuration
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        VARQConfig instance
    """
    return VARQConfig(config_path)
