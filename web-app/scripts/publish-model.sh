#!/usr/bin/env bash
# Upload the fine-tuned compressor to Hugging Face Hub so Render can download it.
#
# Usage:
#   pip install huggingface_hub
#   huggingface-cli login
#   ./scripts/publish-model.sh your-username/llmlingua2-compressor
#
# Then set this on Render:
#   HF_MODEL_ID=your-username/llmlingua2-compressor
#   MODEL_URL=https://huggingface.co/your-username/llmlingua2-compressor

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

export REPO_ID CHECKPOINT_DIR
python3 <<'PY'
import os

from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
checkpoint = os.environ["CHECKPOINT_DIR"]

api = HfApi()
api.create_repo(repo_id, exist_ok=True)
api.upload_folder(
    folder_path=checkpoint,
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["checkpoint-*"],
)

print("Uploaded. Use either env var on Render:")
print(f"  HF_MODEL_ID={repo_id}")
print(f"  MODEL_URL=https://huggingface.co/{repo_id}")
PY
