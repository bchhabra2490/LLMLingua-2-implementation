#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/../checkpoints/token-compressor}"
WEIGHTS_FILE="$CHECKPOINT_DIR/model.safetensors"

echo "Installing Python dependencies..."
pip install -r "$ROOT_DIR/backend/requirements.txt"

echo "Building React frontend..."
cd "$ROOT_DIR/frontend"
npm ci
npm run build

if [ ! -f "$WEIGHTS_FILE" ]; then
  if [ -n "${MODEL_URL:-}" ]; then
    echo "Downloading model weights from MODEL_URL..."
    mkdir -p "$CHECKPOINT_DIR"
    RESOLVED_URL="${MODEL_URL%/}"
    if [[ "$RESOLVED_URL" =~ ^https?://huggingface.co/[^/]+/[^/]+$ ]]; then
      RESOLVED_URL="${RESOLVED_URL}/resolve/main/model.safetensors"
    fi
    curl -fsSL "$RESOLVED_URL" -o "$WEIGHTS_FILE"
  elif [ -n "${HF_MODEL_ID:-}" ]; then
    echo "Downloading checkpoint from Hugging Face: $HF_MODEL_ID"
    PYTHONPATH="$ROOT_DIR/backend" python3 - <<PY
from pathlib import Path
from model_utils import download_from_huggingface

download_from_huggingface("$HF_MODEL_ID", Path("$CHECKPOINT_DIR"))
PY
  else
    echo "Warning: $WEIGHTS_FILE not found. Set HF_MODEL_ID or MODEL_URL in Render."
  fi
fi

echo "Build complete."
