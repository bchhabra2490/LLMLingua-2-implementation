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

if [ ! -f "$WEIGHTS_FILE" ] && [ -n "${MODEL_URL:-}" ]; then
  echo "Downloading model weights..."
  mkdir -p "$CHECKPOINT_DIR"
  curl -fsSL "$MODEL_URL" -o "$WEIGHTS_FILE"
elif [ ! -f "$WEIGHTS_FILE" ]; then
  echo "Warning: $WEIGHTS_FILE not found. Set MODEL_URL in Render to download weights at build time."
fi

echo "Build complete."
