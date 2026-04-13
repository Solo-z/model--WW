#!/bin/bash
# ==============================================================================
# Lambda Cloud Instance Setup Script
# ==============================================================================
#
# Run this script after launching a new Lambda Cloud instance to set up
# the environment for training MODEL-W.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/YOUR_REPO/MODEL-W/main/scripts/setup_lambda.sh | bash
#
# Or clone and run:
#   git clone https://github.com/YOUR_REPO/MODEL-W.git
#   cd MODEL-W
#   ./scripts/setup_lambda.sh
#
# ==============================================================================

set -e

echo "=============================================="
echo "MODEL-W Lambda Cloud Setup"
echo "=============================================="

# Update system
echo "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq wget curl git bc

# Create directories
echo "Creating directories..."
mkdir -p ~/data
mkdir -p ~/checkpoints
mkdir -p ~/cache
mkdir -p ~/synthetic_data

# Install Python dependencies
echo "Installing Python dependencies..."
cd ~/MODEL-W || cd ~
pip install --upgrade pip
pip install -e . || pip install -r requirements.txt

# Install flash-attention (optional, for faster training)
echo "Installing flash-attention..."
pip install flash-attn --no-build-isolation || echo "Flash attention install failed (optional)"

# Check CUDA
echo ""
echo "CUDA check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Check PyTorch
echo ""
echo "PyTorch check:"
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download the Lakh MIDI dataset:"
echo "   python scripts/download_lakh.py --output ~/data/lakh_midi"
echo ""
echo "2. Start training:"
echo "   ./scripts/train_lambda.sh base 32 100000"
echo ""
echo "3. Or with custom settings:"
echo "   DATA_DIR=~/data/lakh_midi CHECKPOINT_DIR=~/checkpoints ./scripts/train_lambda.sh large 24 200000"
echo ""
echo "=============================================="

