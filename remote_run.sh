#!/bin/bash
set -e

# 1. Activate Environment
export PATH=$HOME/.local/bin:$PATH
source .venv/bin/activate

# 2. Generate Data (if needed)
# We check if data exists to avoid re-generating every time, 
# but for now let's force generation to ensure PyClaw is used.
echo ">>> Generating Data with PyClaw..."
python src/setup/generate_parametric_data.py

# 3. Run Training
echo ">>> Starting Training..."
# Ensure GPU is visible and allow growth
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
python src/train_autoregressive.py

# 4. Run Verification
echo ">>> Running Rollout Verification..."
python src/verify_rollout.py

echo ">>> All tasks completed."
