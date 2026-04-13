#!/bin/bash
# MODEL-W — Deploy to Lambda Cloud GPU
#
# SSH into your Lambda instance, then:
#   git clone <your-repo> && cd MODEL-W && bash scripts/deploy_lambda.sh
#
# This will:
#   1. Install deps
#   2. Clone + install ACE-Step
#   3. Launch the web UI with a public share link
#
# The share link works for ~72 hours — send it to your investor.

set -e

echo ""
echo "  MODEL-W — Lambda Deploy"
echo "  ========================"
echo ""

# 1. Install MODEL-W deps
echo "[1/4] Installing MODEL-W..."
pip install -e . 2>/dev/null || pip install -r requirements.txt

# 2. Clone and install ACE-Step
if [ ! -d "models/ace-step" ]; then
    echo "[2/4] Cloning ACE-Step 1.5..."
    git clone https://github.com/ACE-Step/ACE-Step-1.5.git models/ace-step
else
    echo "[2/4] ACE-Step already cloned"
fi

echo "[3/4] Installing ACE-Step..."
pip install -e models/ace-step

# Write config
cat > .env.acestep << 'EOF'
ACESTEP_ROOT=./models/ace-step
ACESTEP_DIT_CONFIG=acestep-v15-turbo
ACESTEP_LM_MODEL=acestep-5Hz-lm-1.7B
ACESTEP_LM_BACKEND=vllm
EOF

# 3. Install vllm for the LM (Lambda has CUDA)
pip install vllm 2>/dev/null || echo "vllm install failed, will use pt backend"

# 4. Launch with public link
echo ""
echo "[4/4] Launching MODEL-W..."
echo ""
echo "  Models will auto-download on first generation (~4GB)"
echo "  A public share link will appear below — send it to anyone"
echo ""

python app.py --share
