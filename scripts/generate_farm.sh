#!/bin/bash
# ==============================================================================
# MIDI Generation Farm
# ==============================================================================
#
# Generate millions of synthetic MIDI files for training larger models.
#
# Usage:
#   ./scripts/generate_farm.sh [NUM_SAMPLES] [OUTPUT_DIR] [MODEL_PATH]
#
# Example:
#   ./scripts/generate_farm.sh 1000000 ./synthetic_data ./checkpoints/best_model.pt
#
# ==============================================================================

set -e

# Configuration
NUM_SAMPLES=${1:-100000}
OUTPUT_DIR=${2:-"./synthetic_data"}
MODEL_PATH=${3:-"./checkpoints/best_model.pt"}
TOKENIZER_PATH=${4:-"./checkpoints/tokenizer"}

# Generation parameters
BATCH_SIZE=${BATCH_SIZE:-16}
TEMPERATURE=${TEMPERATURE:-0.9}
TOP_P=${TOP_P:-0.92}

echo "=============================================="
echo "MIDI Generation Farm"
echo "=============================================="
echo "Target samples: $NUM_SAMPLES"
echo "Output dir:     $OUTPUT_DIR"
echo "Model:          $MODEL_PATH"
echo "Batch size:     $BATCH_SIZE"
echo "Temperature:    $TEMPERATURE"
echo "=============================================="

# Detect GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Estimate time
SAMPLES_PER_SEC=50  # Conservative estimate
EST_HOURS=$(echo "scale=1; $NUM_SAMPLES / $SAMPLES_PER_SEC / 3600 / $NUM_GPUS" | bc)
echo "Estimated time: ~${EST_HOURS} hours"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run generation
python -c "
from modelw.generate import GenerationFarm, GenerationConfig

config = GenerationConfig(
    batch_size=$BATCH_SIZE,
    temperature=$TEMPERATURE,
    top_p=$TOP_P,
)

farm = GenerationFarm(
    model_path='$MODEL_PATH',
    tokenizer_path='$TOKENIZER_PATH',
    output_dir='$OUTPUT_DIR',
    num_gpus=$NUM_GPUS,
    config=config,
)

farm.run(total_samples=$NUM_SAMPLES)
"

echo ""
echo "Generation complete!"
echo "Output: $OUTPUT_DIR"

# Count generated files
MIDI_COUNT=$(find "$OUTPUT_DIR" -name "*.mid" | wc -l)
echo "Generated $MIDI_COUNT MIDI files"

