# Flux Omini Kontext LoRA Training

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-yellow.svg)](https://lightning.ai)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A comprehensive Lightning-based training framework for fine-tuning the Flux Omini Kontext pipeline using LoRA (Low-Rank Adaptation). This project enables efficient training of multi-image diffusion models that can generate images from input images, reference images, and text prompts.


## 🎨 Generated Samples

### Character Insertion

The following examples demonstrate how the trained model can insert cartoon characters into existing scenes:

| Scene | Reference Character | Generated Result |
|-------|-------------------|------------------|
| ![Scene 1](assets/scene_01.png) | ![Boy Reference](assets/boy_reference_512.png) | ![Output 1](assets/output_01.png) |
| ![Scene 2](assets/scene_02.png) | ![Boy Reference](assets/boy_reference_512.png) | ![Output 2](assets/output_02.png) |


More comming soon!

## 🌟 Features

- **🚀 Lightning-based Training**: Built on PyTorch Lightning for scalable and efficient training
- **🎯 LoRA Fine-tuning**: Memory-efficient training with only 0.1% additional parameters
- **🖼️ Multi-image Input**: Supports both input image and reference image with position deltas
- **📝 Text Conditioning**: Advanced text prompt processing with CLIP and T5 encoders
- **⚡ Gradient Checkpointing**: Memory-efficient training for large models
- **🔧 Multiple Optimizers**: Support for AdamW, Prodigy, and SGD optimizers
- **📊 Comprehensive Monitoring**: Built-in logging and experiment tracking
- **🎨 Flexible Resolution**: Support for various image resolutions and aspect ratios

## 🚀 Quick Start

### Setup Environment

```bash
# Create conda environment
conda create -n omini-kontext python=3.10
conda activate omini-kontext

# Install dependencies
pip install -r requirements.txt
```

### Basic Training

```python
from src.pipeline_flux_omini_kontext import FluxOminiKontextPipeline
import torch

# Initialize pipeline
pipe = FluxOminiKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Generate image
image = pipe(
    image=input_image,
    reference=reference_image,
    reference_delta=[0, 0],
    prompt="Your text prompt here",
    guidance_scale=3.5
).images[0]
```

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended: 24GB+ VRAM)
- **PyTorch 2.0+**
- **HuggingFace account** for model access

### Install Dependencies

```bash
# Core requirements
pip install torch>=2.0.0 lightning>=2.0.0

# Install diffusers from GitHub (required for FluxKontext pipeline)
pip install git+https://github.com/huggingface/diffusers

# Training-specific requirements
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
from src.pipeline_flux_omini_kontext import FluxOminiKontextPipeline

# Test pipeline loading
pipe = FluxOminiKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev"
)
print("✅ Installation successful!")
```

## 🎯 Usage

### Basic Inference

```python
from diffusers.utils import load_image
from src.pipeline_flux_omini_kontext import FluxOminiKontextPipeline
import torch

# Load pipeline
pipe = FluxOminiKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load images
input_image = load_image("path/to/input.jpg")
reference_image = load_image("path/to/reference.jpg")

# Generate
result = pipe(
    image=input_image,
    reference=reference_image,
    reference_delta=[0, 0],  # Position delta for reference
    prompt="A beautiful landscape with mountains",
    guidance_scale=3.5,
    num_inference_steps=28
)

# Save result
result.images[0].save("output.png")
```

### Advanced Usage

```python
# Custom configuration
pipe = FluxOminiKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)

# Enable memory optimizations
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# Generate with custom parameters
result = pipe(
    image=input_image,
    reference=reference_image,
    reference_delta=[64, 32],  # Offset reference by 64px right, 32px down
    prompt="Your detailed prompt here",
    prompt_2="Additional prompt for T5 encoder",
    negative_prompt="blurry, low quality",
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(42)
)
```

**Note**: The `reference_delta` value is specific to the trained LoRA and depends on the settings used during training. The recommended value is calculated as `(1024 + 512) // 16 = 96`. This formula accounts for the image dimensions and the model's internal scaling factor.

## 🛠️ Training

### Data Preparation

Your training data should be organized as follows:

```
data/
├── start/          # Input images (960x512)
├── reference/      # Reference images (512x512)
└── end/           # Target images (896x512)
```

### Training Configuration

```python
# Training config
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

### Start Training

```bash
# Basic training
python train/script/train.py --config train/config/basic.yaml

# Multi-GPU training
python train/script/train.py --config train/config/multi_gpu.yaml

# Resume training
python train/script/train.py --config train/config/resume.yaml --resume_from_checkpoint path/to/checkpoint.ckpt
```

### Training Monitoring

```python
# Monitor with TensorBoard
tensorboard --logdir runs/

# Monitor with Weights & Biases
wandb login
python train/script/train.py --config train/config/wandb.yaml
```

## 📚 Examples

### Character Insertion

See `examples/character_insert.ipynb` for a complete example of inserting characters into scenes.

**Trained Model**: Check out the [omini-kontext-character](https://huggingface.co/saquiboye/omini-kontext-character) model on Hugging Face, which is specifically trained to insert cartoon characters into existing scenes.


## 🏗️ Model Architecture

The Flux Omini Kontext pipeline consists of several key components:

### Base model
Flux Kontext dev model

### LoRA Integration

```python
# LoRA layers are applied to attention modules
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

# LoRA configuration
lora_config = {
    "r": 16,                    # Rank
    "lora_alpha": 32,           # Alpha scaling
    "lora_dropout": 0.1,        # Dropout rate
    "bias": "none",             # Bias handling
    "task_type": "CAUSAL_LM"    # Task type
}
```

### Training Process

1. **Input Processing**: Encode input and reference images
2. **Text Encoding**: Process prompts with CLIP and T5
3. **LoRA Forward**: Apply LoRA layers during forward pass
4. **Noise Prediction**: Train to predict noise
5. **Loss Computation**: MSE loss on noise prediction

## ⚙️ Configuration

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | PIL.Image | None | Input image |
| `reference` | PIL.Image | None | Reference image |
| `reference_delta` | List[int] | [0, 0] | Position offset for reference (specific to trained LoRA, recommended: (1024+512)//16 = 96) |
| `prompt` | str | None | Text prompt |
| `prompt_2` | str | None | Secondary text prompt |
| `guidance_scale` | float | 3.5 | Classifier-free guidance scale |
| `num_inference_steps` | int | 28 | Number of denoising steps |
| `height` | int | 1024 | Output height |
| `width` | int | 1024 | Output width |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 1e-4 | Learning rate |
| `batch_size` | int | 1 | Training batch size |
| `max_epochs` | int | 10 | Maximum training epochs |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `warmup_steps` | int | 100 | Learning rate warmup steps |


## 📋 To-do

- [ ] **Add ways to control location and scale of the reference character**
- [ ] **Speed up by removing irrelevant pixels**
- [ ] **Deploy a public demo**
- [ ] **Deploy a replicate version**
- [x] **Basic training script**
- [x] **Basic inference script**

### Model Plans
- [ ] **Person Models**: Develop models for realistic human subjects
- [ ] **Clothes Models**: Create models for clothing and fashion items
- [ ] **Subject Models**: Train models for specific objects and items
- [x] **Character Models**: Train specialized models for anime/cartoon characters

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Black Forest Labs** for the FLUX.1-Kontext-dev model
- **HuggingFace** for the diffusers library
- **PyTorch Lightning** for the training framework
- **PEFT** for LoRA implementation
- **[OminiControl](https://github.com/Yuanshi9815/OminiControl)** for the universal control framework for Diffusion Transformers

## 📚 References

```bibtex
@article{omini-kontext,
  title={OminiKontext: Multi-image references for image to image instruction models},
  author={Saquib Alam},
  year={2025}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Full Documentation](https://your-docs-url.com)

---

**Made with ❤️ for the AI community** 