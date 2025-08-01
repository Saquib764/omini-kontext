# Flux Omini Kontext Training Configuration

This directory contains the configuration system for training Flux Omini Kontext models with LoRA fine-tuning.

## Configuration Files

### Available Configurations

- `default.yaml` - Basic configuration for single GPU training
- `dupli_human.yaml` - Configuration optimized for dupli_human dataset
- `high_memory.yaml` - High memory configuration for powerful GPUs

### Configuration Structure

Each YAML file contains the following sections:

#### Model Configuration
```yaml
model:
  flux_pipe_id: "black-forest-labs/FLUX.1-Kontext-dev"
  device: "cuda"
  dtype: "bfloat16"
  gradient_checkpointing: true
```

#### LoRA Configuration
```yaml
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
```

#### Training Configuration
```yaml
training:
  max_epochs: 10
  batch_size: 4
  gradient_clip_val: 1.0
  precision: "bf16-mixed"
  accelerator: "gpu"
  devices: 1
```

#### Data Configuration
```yaml
data:
  train_data_path: "./data/train"
  val_data_path: "./data/val"
  num_workers: 4
  pin_memory: true
```

#### Hardware Configuration
```yaml
hardware:
  cuda_visible_devices: "0,1"
  main_process_port: 41353
  tokenizers_parallelism: true
```

## Usage

### Using the Shell Script

1. **Basic training with default config:**
   ```bash
   ./train.sh
   ```

2. **Training with specific config:**
   ```bash
   ./train.sh ./train/config/dupli_human.yaml
   ```

3. **Resume from checkpoint:**
   ```bash
   ./train.sh ./train/config/dupli_human.yaml /path/to/checkpoint.ckpt
   ```

### Using Python Directly

```bash
# Set environment variables
export XFL_CONFIG=./train/config/dupli_human.yaml
export HF_HUB_CACHE=./cache
export TOKENIZERS_PARALLELISM=true

# Run training
accelerate launch --main_process_port 41353 -m src.train.train
```

### Command Line Arguments

The training script supports the following arguments:

- `--config`: Path to config file (overrides XFL_CONFIG environment variable)
- `--resume`: Path to checkpoint file to resume from

## Creating Custom Configurations

1. Copy an existing config file:
   ```bash
   cp train/config/default.yaml train/config/my_config.yaml
   ```

2. Modify the parameters in your new config file

3. Run training with your custom config:
   ```bash
   ./train.sh train/config/my_config.yaml
   ```

## Environment Variables

The following environment variables are automatically set by the configuration system:

- `XFL_CONFIG`: Path to the configuration file
- `HF_HUB_CACHE`: HuggingFace cache directory
- `CUDA_VISIBLE_DEVICES`: GPU devices to use
- `TOKENIZERS_PARALLELISM`: Enable tokenizer parallelism
- `WANDB_API_KEY`: WandB API key (if enabled)

## Output

Training outputs are saved to:
- `./lora_weights/` - LoRA weights and checkpoints
- `./cache/` - HuggingFace model cache
- `./logs/` - Training logs

## Troubleshooting

### Common Issues

1. **Config file not found:**
   - Ensure the config file path is correct
   - Check that the file exists in the `train/config/` directory

2. **CUDA out of memory:**
   - Reduce batch size in the config
   - Enable gradient checkpointing
   - Use a lower precision (bf16-mixed)

3. **Import errors:**
   - Ensure all dependencies are installed
   - Check that the src directory is in the Python path

### Debug Mode

To run with debug information:
```bash
export XFL_CONFIG=./train/config/default.yaml
python -m src.train.train --config ./train/config/default.yaml
``` 