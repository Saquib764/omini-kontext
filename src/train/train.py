#!/usr/bin/env python3
"""
Training script for Flux Omini Kontext model with YAML configuration support.
"""

import os
import sys
import argparse
import lightning as L
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from model import FluxOminiKontextModel
from config import load_config_from_env, setup_environment_from_config
from data import create_dataloaders


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Flux Omini Kontext model")
    parser.add_argument("--config", type=str, help="Path to config file (overrides XFL_CONFIG)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        os.environ['XFL_CONFIG'] = args.config
    
    try:
        config = load_config_from_env()
        setup_environment_from_config(config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    print(f"Using configuration from: {config.config_path}")
    
    # Extract configurations
    model_config = config.get_model_config()
    lora_config = config.get_lora_config()
    optimizer_config = config.get_optimizer_config()
    training_config = config.get_training_config()
    data_config = config.get_data_config()
    logging_config = config.get_logging_config()
    hardware_config = config.get_hardware_config()
    
    # Create model
    print("Initializing model...")
    model = FluxOminiKontextModel(
        flux_pipe_id=model_config.get('flux_pipe_id', 'black-forest-labs/FLUX.1-Kontext-dev'),
        lora_config=lora_config,
        device=model_config.get('device', 'cuda'),
        dtype=getattr(torch, model_config.get('dtype', 'bfloat16')),
        model_config=model_config,
        optimizer_config=optimizer_config,
        gradient_checkpointing=model_config.get('gradient_checkpointing', False),
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_data_path=data_config.get('train_data_path', './data/train'),
        val_data_path=data_config.get('val_data_path', './data/val'),
        batch_size=training_config.get('batch_size', 2),
        num_workers=data_config.get('num_workers', 2),
        pin_memory=data_config.get('pin_memory', True),
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    save_path = logging_config.get('save_path', './lora_weights')
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="lora-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=logging_config.get('save_top_k', 3),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = L.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=logging_config.get('early_stopping_patience', 3),
        mode="min",
    )
    callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = L.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=training_config.get('max_epochs', 10),
        accelerator=training_config.get('accelerator', 'gpu'),
        devices=training_config.get('devices', 1),
        precision=training_config.get('precision', 'bf16-mixed'),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        callbacks=callbacks,
        log_every_n_steps=logging_config.get('log_every_n_steps', 10),
        val_check_interval=logging_config.get('val_every_n_epochs', 1.0),
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume
    )
    
    # Save final LoRA weights
    final_save_path = os.path.join(save_path, "final_lora")
    print(f"Saving final LoRA weights to: {final_save_path}")
    model.save_lora(final_save_path)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    import torch
    main() 