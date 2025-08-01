#!/usr/bin/env python3
"""
Example script demonstrating how to use the configuration system.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import ConfigLoader, setup_environment_from_config


def main():
    """Example usage of the configuration system"""
    
    # Example 1: Load configuration from file
    print("=== Example 1: Loading configuration from file ===")
    config_path = "./train/config/dupli_human.yaml"
    
    if os.path.exists(config_path):
        config = ConfigLoader(config_path)
        print(f"Loaded config from: {config_path}")
        
        # Access different configuration sections
        model_config = config.get_model_config()
        lora_config = config.get_lora_config()
        training_config = config.get_training_config()
        
        print(f"Model config: {model_config}")
        print(f"LoRA config: {lora_config}")
        print(f"Training config: {training_config}")
    else:
        print(f"Config file not found: {config_path}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Load configuration from environment variable
    print("=== Example 2: Loading configuration from environment ===")
    os.environ['XFL_CONFIG'] = "./train/config/default.yaml"
    
    try:
        from config import load_config_from_env
        config = load_config_from_env()
        print(f"Loaded config from environment: {config.config_path}")
        
        # Setup environment variables from config
        setup_environment_from_config(config)
        print("Environment variables set from config")
        
    except Exception as e:
        print(f"Error loading config from environment: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Merge configurations
    print("=== Example 3: Merging configurations ===")
    base_config = "./train/config/default.yaml"
    override_config = "./train/config/dupli_human.yaml"
    
    if os.path.exists(base_config) and os.path.exists(override_config):
        config = ConfigLoader(base_config)
        merged_config = config.merge_configs(base_config, override_config)
        print("Merged configuration:")
        print(f"  Base: {base_config}")
        print(f"  Override: {override_config}")
        print(f"  Result: {merged_config}")
    else:
        print("One or both config files not found for merging example")


if __name__ == "__main__":
    main() 