#!/usr/bin/env bash
# Upload the fine-tuned compressor to Hugging Face Hub so Render can download it.
#
# Usage:
#   pip install huggingface_hub
#   huggingface-cli login
#   ./scripts/publish-model.sh your-username/llmlingua2-compressor
#
# Then set this on Render:
#   MODEL_URL=https://huggingface.co/your-username/llmlingua2-compressor/resolve/main/model.safetensors

set -euo pipefail

REPO_ID="${1:-}"
if [ -z "$REPO_ID" ]; then
  echo "Usage: $0 <huggingface-repo-id>"
  echo "Example: $0 bchhabra2490/llmlingua2-compressor"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="$ROOT_DIR/../checkpoints/token-compressor/checkpoint-960"

if [ ! -f "$CHECKPOINT_DIR/model.safetensors" ]; then
  CHECKPOINT_DIR="$ROOT_DIR/../checkpoints/token-compressor"
fi

echo "Uploading from $CHECKPOINT_DIR to hf://$REPO_ID"
python3 - <<PY
from huggingface_hub import HfApi

api = HfApi()
repo_id = "$REPO_ID"
checkpoint = "$CHECKPOINT_DIR"

api.create_repo(repo_id, exist_ok=True)
api.upload_folder(
    folder_path=checkpoint,
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["checkpoint-*"],
)
print(f"Uploaded. Set MODEL_URL to:")
print(f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors")
PY
