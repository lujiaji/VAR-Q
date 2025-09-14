"""
Configuration loader for VAR-Q model
"""
import json
import os
from typing import Dict, Any, Tuple
import torch


class VARQConfig:
    """Configuration manager for VAR-Q model"""
    
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
    
    
    def get_checkpoint_paths(self, model_depth: int) -> Tuple[str, str]:
        """
        Get checkpoint paths for VAE and VAR models
        
        Args:
            model_depth: Model depth
            
        Returns:
            Tuple of (vae_ckpt_path, var_ckpt_path)
        """
        checkpoint_config = self.get_checkpoint_config()
        vae_ckpt = checkpoint_config['vae_ckpt']
        var_ckpt = checkpoint_config['var_ckpt_template'].format(model_depth)
        return vae_ckpt, var_ckpt
    
    def get_device(self) -> str:
        """Get device configuration"""
        device = self.config['inference']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device
    
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
