import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Configuration loader for YAML config files"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration"""
        return self.config.get('lora', {})
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration"""
        return self.config.get('optimizer', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.config.get('hardware', {})
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration"""
        return self.config.get('cache', {})
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get WandB configuration"""
        return self.config.get('wandb', {})
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config
    
    def merge_configs(self, base_config: str, override_config: str) -> Dict[str, Any]:
        """Merge two configuration files, with override_config taking precedence"""
        base_loader = ConfigLoader(base_config)
        override_loader = ConfigLoader(override_config)
        
        merged_config = base_loader.get_all_config()
        override_config_dict = override_loader.get_all_config()
        
        # Recursively merge configurations
        self._merge_dicts(merged_config, override_config_dict)
        
        return merged_config
    
    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value


def load_config_from_env() -> ConfigLoader:
    """Load configuration from environment variable XFL_CONFIG"""
    config_path = os.getenv('XFL_CONFIG')
    if not config_path:
        raise ValueError("XFL_CONFIG environment variable not set")
    
    return ConfigLoader(config_path)


def setup_environment_from_config(config: ConfigLoader):
    """Set up environment variables from configuration"""
    hardware_config = config.get_hardware_config()
    cache_config = config.get_cache_config()
    wandb_config = config.get_wandb_config()
    
    # Set hardware environment variables
    if 'cuda_visible_devices' in hardware_config:
        os.environ['CUDA_VISIBLE_DEVICES'] = hardware_config['cuda_visible_devices']
    
    if 'tokenizers_parallelism' in hardware_config:
        os.environ['TOKENIZERS_PARALLELISM'] = str(hardware_config['tokenizers_parallelism']).lower()
    
    # Set cache environment variables
    if 'hf_hub_cache' in cache_config:
        os.environ['HF_HUB_CACHE'] = cache_config['hf_hub_cache']
    
    # Set WandB environment variables
    if wandb_config.get('enabled', False) and wandb_config.get('api_key'):
        os.environ['WANDB_API_KEY'] = wandb_config['api_key'] 