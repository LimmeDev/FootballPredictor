#!/usr/bin/env bash
# Colab one-click setup script for FootballPredictor
# Usage inside a Colab cell:
#   !bash setup_colab.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Optional: pass SKIP_STATSBOMB=1 to skip heavy event data
# -----------------------------------------------------------------------------
SKIP_STATSBOMB="${SKIP_STATSBOMB:-0}"

# -----------------------------------------------------------------------------
# 1. Install Python dependencies (GPU-enabled LightGBM)
# -----------------------------------------------------------------------------
python -m pip uninstall -y lightgbm -q || true
python -m pip install -q lightgbm --no-binary lightgbm \
  --config-settings=cmake.define.USE_GPU=ON
python -m pip install -q -r requirements.txt

# -----------------------------------------------------------------------------
# 2. Download data
# -----------------------------------------------------------------------------
if [[ "$SKIP_STATSBOMB" == "1" ]]; then
  export SKIP_STATSBOMB=1
fi
python src/download_data.py

# -----------------------------------------------------------------------------
# 3. Build features + train model (GPU)
# -----------------------------------------------------------------------------
python src/make_features.py
python src/train_model.py

echo "\n✓ Colab setup finished — model saved under models/football_predictor.txt" 