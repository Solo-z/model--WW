#!/usr/bin/env bash
# ROOM — isolated venv with matching torch + torchaudio (CUDA 12.x wheels).
#
# Use on Lambda / Ubuntu when apt installs python3-torch (CUDA 12) but pip
# pulled torchaudio built for CUDA 13 → libcudart.so.13 errors.
#
# Usage (from repo root):
#   bash scripts/bootstrap_venv_room.sh
#   source .venv/bin/activate
#   python scripts/setup_room.py
#   python app.py --share --port 7870

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "[venv] Creating .venv ..."
  "$PY" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate
echo "[venv] Upgrading pip ..."
python -m pip install -U pip wheel

echo "[venv] Installing PyTorch stack (cu124, CUDA 12.x) — same major/minor for torch + torchaudio ..."
python -m pip install "numpy>=1.26,<2"
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "[venv] Installing MODEL-W package + Gradio ..."
python -m pip install -e .
python -m pip install "gradio>=4.0"

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then run:             python scripts/setup_room.py"
echo "Then:                 python app.py --share --port 7870"