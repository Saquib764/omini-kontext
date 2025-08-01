import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from src.model import create_model_config, train_model
from src.data import FluxOminiKontextDataset



def main():
    """Example training script for Flux Omini Kontext LoRA"""
    
    # Create datasets
    train_dataset = FluxOminiKontextDataset()
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4
    )
    
    # Create model configuration
    config = create_model_config()
    
    # Customize configuration if needed
    config.update({
        "flux_pipe_id": "black-forest-labs/FLUX.1-Kontext-dev",
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "optimizer_config": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999)
            }
        },
        "gradient_checkpointing": True
    })
    
    # Train the model
    model, trainer = train_model(
        train_dataloader=train_dataloader,
        config=config,
        max_epochs=10,
        save_path="./lora_weights"
    )
    
    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main() 