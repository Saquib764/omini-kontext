#!/bin/bash

# Flux Omini Kontext Training Script
# This script sets up the environment and runs the training process

# Exit on any error
set -e

# Default configuration
DEFAULT_CONFIG="./train/config/default.yaml"

# Parse command line arguments
CONFIG_FILE=${1:-$DEFAULT_CONFIG}
RESUME_CHECKPOINT=${2:-""}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -la ./train/config/
    exit 1
fi

echo "=== Flux Omini Kontext Training ==="
echo "Config file: $CONFIG_FILE"
echo "Resume checkpoint: $RESUME_CHECKPOINT"
echo ""

# Set environment variables from config
export XFL_CONFIG="$CONFIG_FILE"
export HF_HUB_CACHE="./cache"

# Set WandB API key if provided (uncomment and set your key)
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

# Set tokenizers parallelism
export TOKENIZERS_PARALLELISM=true

# Create necessary directories
mkdir -p ./cache
mkdir -p ./lora_weights
mkdir -p ./logs

echo "Environment variables set:"
echo "  XFL_CONFIG=$XFL_CONFIG"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo ""

# Run the training
echo "Starting training with accelerate..."
accelerate launch --main_process_port 41353 -m src.train.train ${RESUME_CHECKPOINT:+--resume $RESUME_CHECKPOINT}

echo ""
echo "Training completed!"
echo "Check the logs and saved models in ./lora_weights/" 