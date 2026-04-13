#!/bin/bash
# ==============================================================================
# MODEL-W Training Script for Lambda Cloud
# ==============================================================================
#
# Usage:
#   ./scripts/train_lambda.sh [MODEL_SIZE] [BATCH_SIZE] [MAX_STEPS]
#
# Example:
#   ./scripts/train_lambda.sh base 32 100000
#
# Recommended Lambda instances:
#   - 1x A100 (40GB): model_size=base, batch_size=32
#   - 1x A100 (80GB): model_size=large, batch_size=48
#   - 8x A100 (80GB): model_size=xl, batch_size=24 (per GPU)
#
# ==============================================================================

set -e

# Configuration
MODEL_SIZE=${1:-"base"}
BATCH_SIZE=${2:-32}
MAX_STEPS=${3:-100000}

# Paths
DATA_DIR="${DATA_DIR:-/home/ubuntu/data/lakh_midi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/ubuntu/checkpoints}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/cache}"

# Wandb (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-model-w}"

echo "=============================================="
echo "MODEL-W Training on Lambda Cloud"
echo "=============================================="
echo "Model size:    $MODEL_SIZE"
echo "Batch size:    $BATCH_SIZE"
echo "Max steps:     $MAX_STEPS"
echo "Data dir:      $DATA_DIR"
echo "Checkpoint:    $CHECKPOINT_DIR"
echo "=============================================="

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Launch training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching distributed training..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m modelw.trainer \
        --data_dir="$DATA_DIR" \
        --model_size="$MODEL_SIZE" \
        --batch_size=$BATCH_SIZE \
        --max_steps=$MAX_STEPS \
        --checkpoint_dir="$CHECKPOINT_DIR"
else
    echo "Launching single GPU training..."
    python -m modelw.trainer \
        --data_dir="$DATA_DIR" \
        --model_size="$MODEL_SIZE" \
        --batch_size=$BATCH_SIZE \
        --max_steps=$MAX_STEPS \
        --checkpoint_dir="$CHECKPOINT_DIR"
fi

echo "Training complete!"

