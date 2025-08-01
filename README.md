# Flux Omini Kontext LoRA Training

This repository contains a Lightning model for training LoRA weights on the Flux Omini Kontext pipeline. The model takes two images (input image and reference image) plus text as input and outputs one image.

## Features

- **Lightning-based training**: Uses PyTorch Lightning for easy training and validation
- **LoRA fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **Multi-image input**: Supports both input image and reference image with position deltas
- **Text conditioning**: Supports text prompts for guided generation
- **Gradient checkpointing**: Memory-efficient training
- **Multiple optimizers**: Support for AdamW, Prodigy, and SGD

## Model Architecture

The `FluxOminiKontextModel` is based on the Flux Omini Kontext pipeline and includes:

- **Input processing**: Encodes both input and reference images
- **Text encoding**: Processes text prompts using CLIP and T5 encoders
- **LoRA layers**: Trainable low-rank adaptation layers
- **Diffusion training**: Noise prediction training with MSE loss

## Installation

```bash
pip install torch lightning diffusers transformers peft prodigyopt
```

## Usage

### Basic Training

```python
from src.model import FluxOminiKontextModel, create_model_config

# Create model configuration
config = create_model_config()

# Initialize model
model = FluxOminiKontextModel(**config)

# Train with Lightning Trainer
trainer = L.Trainer(max_epochs=10, accelerator="gpu")
trainer.fit(model, train_dataloader, val_dataloader)
```

### Custom Configuration

```python
config = {
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
}
```

### Data Format

The model expects batches with the following format:

```python
batch = {
    "input_image": torch.Tensor,      # Main input image [B, C, H, W]
    "reference_image": torch.Tensor,   # Reference image [B, C, H, W]
    "prompt": List[str],              # Text prompts
    "reference_delta": List[List[int]], # Position deltas [[x, y], ...]
    "look_ahead": List[int]           # Look ahead parameters
}
```

### Saving and Loading LoRA Weights

```python
# Save LoRA weights
model.save_lora("./lora_weights")

# Load existing LoRA weights
model = FluxOminiKontextModel(
    flux_pipe_id="black-forest-labs/FLUX.1-Kontext-dev",
    lora_path="./lora_weights"
)
```

## Training Script

Use the provided training script:

```bash
python example_training.py
```

## Model Components

### FluxOminiKontextModel

The main Lightning module that handles:

- **Pipeline loading**: Loads the Flux Omini Kontext pipeline
- **LoRA initialization**: Sets up LoRA layers for fine-tuning
- **Training loop**: Implements the training and validation steps
- **Loss computation**: MSE loss for noise prediction
- **Optimizer configuration**: Supports multiple optimizer types

### Key Methods

- `init_lora()`: Initialize LoRA layers
- `save_lora()`: Save LoRA weights
- `step()`: Forward pass and loss computation
- `training_step()`: Training step
- `validation_step()`: Validation step

## Configuration Options

### LoRA Configuration

- `r`: Rank of LoRA layers (default: 16)
- `lora_alpha`: Alpha parameter for LoRA (default: 32)
- `target_modules`: Modules to apply LoRA to
- `lora_dropout`: Dropout rate for LoRA layers
- `bias`: Bias handling in LoRA layers

### Optimizer Configuration

- `type`: Optimizer type ("AdamW", "Prodigy", "SGD")
- `params`: Optimizer parameters (learning rate, weight decay, etc.)

### Model Configuration

- `flux_pipe_id`: HuggingFace model ID
- `gradient_checkpointing`: Enable gradient checkpointing
- `device`: Device to use ("cuda", "cpu")
- `dtype`: Data type (torch.bfloat16, torch.float16, torch.float32)

## Example Training Pipeline

1. **Prepare data**: Create datasets with input images, reference images, and prompts
2. **Configure model**: Set up LoRA and optimizer configurations
3. **Initialize model**: Create the FluxOminiKontextModel instance
4. **Train**: Use Lightning Trainer for training
5. **Save weights**: Save the trained LoRA weights
6. **Inference**: Use the trained weights for inference

## Notes

- The model freezes the base pipeline components (text encoders, VAE) and only trains LoRA layers
- Gradient checkpointing is enabled by default for memory efficiency
- The model supports both training from scratch and loading existing LoRA weights
- Position deltas allow precise control over reference image placement

## Dependencies

- PyTorch
- PyTorch Lightning
- Diffusers
- Transformers
- PEFT
- ProdigyOpt (optional)
- PIL
- NumPy 