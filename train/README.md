# Omini-Kontext Training 🛠️

This guide provides comprehensive instructions for training LoRA (Low-Rank Adaptation) models on the Omini-Kontext pipeline. The training system is built on PyTorch Lightning and supports efficient fine-tuning of the Flux Omini-Kontext model.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## 🎯 Overview

Omini-Kontext is a multi-image diffusion model that takes:
- **Input image**: The main image to be modified
- **Reference image**: A reference image for context
- **Text prompt**: Descriptive text for guidance
- **Reference delta**: Position coordinates for reference image placement

The training system uses LoRA to efficiently fine-tune the model while keeping the base model frozen.

## 🔧 Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended: 24GB+ VRAM)
- **HuggingFace account** for model access
- **Weights & Biases account** (optional, for experiment tracking)

## 📦 Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd omini-kontext
```

2. **Install dependencies**:
```bash
# Install diffusers from GitHub (required for FluxKontext pipeline)
pip install git+https://github.com/huggingface/diffusers

# Install training-specific requirements
pip install -r train/requirements.txt

# Or install core requirements
pip install -r requirements.txt
```

## 📊 Data Preparation

**⚠️ Data preparation guide coming soon!**

### Data Structure

Your training data should be organized as follows:

```
data/
├── start/          # Input images (960x512)
├── reference/      # Reference images (512x512)
└── end/           # Target images (896x512)
```

### Data Format Requirements

- **Input images**: 960×512 pixels, RGB format
- **Reference images**: 512×512 pixels, RGB format  
- **Target images**: 896×512 pixels, RGB format
- **Supported formats**: JPG, JPEG, PNG, GIF

### Creating Your Dataset

1. **Prepare image triplets**:
   - Each training example needs 3 corresponding images
   - Ensure filenames match across directories
   - Use consistent naming convention (e.g., `001.jpg`, `002.jpg`)

2. **Image preprocessing**:
```python
from PIL import Image

# Resize input image
input_img = Image.open("input.jpg").resize((960, 512)).convert("RGB")

# Resize reference image  
reference_img = Image.open("reference.jpg").resize((512, 512)).convert("RGB")

# Resize target image
target_img = Image.open("target.jpg").resize((896, 512)).convert("RGB")
```

3. **Custom dataset** (optional):
   If you want to use HuggingFace datasets, modify `src/train/data.py`:

```python
# Example: Using HuggingFace datasets
dataset = load_and_concatenate_datasets(
    dataset_names=["your-dataset-name"],
    source_field_values=["your-source"],
    split="train"
)
```

## ⚙️ Configuration

### Default Configuration

The training uses YAML configuration files. The default config is in `train/config/default.yaml`:

```yaml
flux_path: "black-forest-labs/FLUX.1-Kontext-dev"
dtype: "bfloat16"

train:
  batch_size: 1
  accumulate_grad_batches: 1
  dataloader_workers: 5
  save_interval: 1000
  sample_interval: 100
  max_steps: -1
  gradient_checkpointing: false
  save_path: "runs"
  
  condition_type: "subject"
  
  resume_training_from_last_checkpoint: false
  resume_training_from_checkpoint_path: "runs/20250127-114531/ckpt/1000"
  
  dataset:
    init_size: 512
    reference_size: 512
    target_size: 512
    image_size: 512
    padding: 8
    drop_text_prob: 0.1
    drop_image_prob: 0.1
  
  wandb:
    project: "OminiKontextControl"
  
  lora_config:
    r: 24
    lora_alpha: 24
    init_lora_weights: "gaussian"
    target_modules: "(.*x_embedder|.*transformer_blocks\\.[0-9]+\\.(norm|norm1)\\.linear|.*transformer_blocks\\.[0-9]+\\.attn\\.(to_k|to_q|to_v|to_add_out)|.*transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*single_transformer_blocks\\.[0-9]+\\.attn\\.to_out|.*single_transformer_blocks\\.[0-9]+\\.(proj_mlp|proj_out)|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.0\\.proj|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1_context\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff_context\\.net\\.0\\.proj|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff_context\\.net\\.2|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.(to_add_out|add_k_proj|add_q_proj|add_v_proj))"
  
  optimizer:
    type: "Prodigy"
    params:
      lr: 1
      use_bias_correction: true
      safeguard_warmup: true
      weight_decay: 0.01
```

### Key Configuration Parameters

#### Training Parameters
- `batch_size`: Batch size for training (default: 1)
- `accumulate_grad_batches`: Gradient accumulation steps (default: 1)
- `max_steps`: Maximum training steps (-1 for unlimited)
- `save_interval`: Save checkpoint every N steps
- `sample_interval`: Generate samples every N steps

#### LoRA Configuration
- `r`: LoRA rank (default: 24)
- `lora_alpha`: LoRA alpha parameter (default: 24)
- `target_modules`: Regex pattern for modules to apply LoRA to
- `init_lora_weights`: Weight initialization method

#### Optimizer Configuration
- `type`: Optimizer type ("Prodigy", "AdamW", "SGD")
- `lr`: Learning rate
- `weight_decay`: Weight decay coefficient

### Creating Custom Configuration

1. **Copy default config**:
```bash
cp train/config/default.yaml train/config/my_config.yaml
```

2. **Modify parameters**:
```yaml
# Example custom configuration
train:
  batch_size: 2
  max_steps: 5000
  lora_config:
    r: 16
    lora_alpha: 32
  optimizer:
    type: "AdamW"
    params:
      lr: 1e-4
      weight_decay: 0.01
```

## 🚀 Training

### Quick Start

1. **Set environment variables**:
```bash
export XFL_CONFIG=./train/config/default.yaml
export HF_HUB_CACHE=./cache
export TOKENIZERS_PARALLELISM=true
```

2. **Run training**:
```bash
# Using the provided script
bash train/script/default.sh

# Or directly with accelerate
accelerate launch --main_process_port 41353 -m src.train.train
```

### Multi-GPU Training

1. **Configure GPU devices**:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

2. **Run multi-GPU training**:
```bash
accelerate launch --multi_gpu --num_processes=4 -m src.train.train
```

### Resume Training

1. **From last checkpoint**:
```yaml
train:
  resume_training_from_last_checkpoint: true
```

2. **From specific checkpoint**:
```yaml
train:
  resume_training_from_checkpoint_path: "runs/20250127-114531/ckpt/1000"
```

### Training Scripts

#### Default Training Script
```bash
# train/script/default.sh
export XFL_CONFIG=./train/config/default.yaml
export HF_HUB_CACHE=./cache
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41353 -m src.train.train
```

#### Custom Training Script
```bash
#!/bin/bash
# train/script/custom.sh

# Set configuration
export XFL_CONFIG=./train/config/my_config.yaml
export HF_HUB_CACHE=./cache
export TOKENIZERS_PARALLELISM=true

# Set GPU devices
export CUDA_VISIBLE_DEVICES=0,1

# Set Weights & Biases API key (optional)
export WANDB_API_KEY='your-wandb-api-key'

# Run training
accelerate launch --multi_gpu --num_processes=2 -m src.train.train
```

## 📈 Monitoring

### Weights & Biases Integration

1. **Set up WANDB**:
```bash
export WANDB_API_KEY='your-wandb-api-key'
```

2. **Configure in YAML**:
```yaml
train:
  wandb:
    project: "OminiKontextControl"
    name: "my-experiment"
```

3. **Monitor metrics**:
- Loss curves
- Learning rate schedules
- Generated samples
- Model checkpoints

### Local Monitoring

Training outputs are saved to:
- `runs/{timestamp}/`: Training session directory
- `runs/{timestamp}/ckpt/`: Model checkpoints
- `runs/{timestamp}/config.yaml`: Training configuration
- `runs/{timestamp}/samples/`: Generated samples

### Checkpoint Management

- **Automatic saving**: Every `save_interval` steps
- **Best model**: Based on validation loss
- **Latest checkpoint**: For resuming training

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```yaml
# Reduce batch size
train:
  batch_size: 1
  gradient_checkpointing: true

# Use gradient accumulation
train:
  accumulate_grad_batches: 4
```

#### 2. Slow Training
```yaml
# Increase batch size if memory allows
train:
  batch_size: 2

# Reduce dataloader workers
train:
  dataloader_workers: 2
```

#### 3. Poor Convergence
```yaml
# Adjust learning rate
train:
  optimizer:
    type: "AdamW"
    params:
      lr: 1e-4

# Increase LoRA rank
train:
  lora_config:
    r: 32
    lora_alpha: 64
```

#### 4. Data Loading Issues
- Verify image paths and formats
- Check file permissions
- Ensure consistent naming across directories

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=.
python -m src.train.train --debug
```

## 🎯 Advanced Usage

### Custom Datasets

1. **Modify dataset class**:
```python
# src/train/data.py
class CustomFluxOminiKontextDataset(Dataset):
    def __init__(self, data_path, **kwargs):
        # Custom initialization
        pass
    
    def __getitem__(self, idx):
        # Custom data loading
        return {
            "input_image": input_tensor,
            "target_image": target_tensor,
            "reference_image": reference_tensor,
            "prompt": prompt,
            "reference_delta": delta,
        }
```

2. **Update training script**:
```python
# src/train/train.py
dataset = CustomFluxOminiKontextDataset(
    data_path="path/to/data",
    # Additional parameters
)
```

### Hyperparameter Tuning

1. **Learning rate scheduling**:
```yaml
train:
  optimizer:
    type: "AdamW"
    params:
      lr: 1e-4
      scheduler:
        type: "cosine"
        warmup_steps: 1000
```

2. **LoRA parameter tuning**:
```yaml
train:
  lora_config:
    r: 16  # Try: 8, 16, 24, 32
    lora_alpha: 32  # Usually 2*r
    lora_dropout: 0.1
```

### Model Evaluation

1. **Generate samples during training**:
```yaml
train:
  sample_interval: 500
  save_samples: true
```

2. **Custom evaluation metrics**:
```python
# Add to src/train/callbacks.py
class CustomEvaluationCallback(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Custom evaluation logic
        pass
```

### Production Deployment

1. **Export trained LoRA**:
```python
# Save LoRA weights
model.save_lora("./trained_lora")

# Load for inference
model = FluxOminiKontextModel(
    flux_pipe_id="black-forest-labs/FLUX.1-Kontext-dev",
    lora_path="./trained_lora"
)
```

2. **Optimize for inference**:
```python
# Enable inference optimizations
model.eval()
torch.set_grad_enabled(False)
```

## 📚 Additional Resources

- [Flux Model Documentation](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [OminiControl GitHub Repository](https://github.com/Yuanshi9815/OminiControl)

## 🤝 Contributing

For issues, questions, or contributions:
1. Check existing issues
2. Create detailed bug reports
3. Submit pull requests with clear descriptions
4. Follow the project's coding standards

---

**Happy Training! 🎉**
