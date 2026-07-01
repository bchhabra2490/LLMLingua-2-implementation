"""Resolve checkpoint path and download model weights when needed."""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "token-compressor"
WEIGHTS_FILENAME = "model.safetensors"


def checkpoint_root() -> Path:
    override = os.environ.get("CHECKPOINT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CHECKPOINT_ROOT


def resolve_latest_checkpoint(root: Path | None = None) -> Path:
    root = root or checkpoint_root()
    checkpoints = [
        p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)
    ]
    if checkpoints:
        return max(checkpoints, key=lambda p: int(p.name.split("-", 1)[1]))
    return root


def weights_path(root: Path | None = None) -> Path:
    return resolve_latest_checkpoint(root) / WEIGHTS_FILENAME


def ensure_model_weights(root: Path | None = None) -> Path:
    root = root or checkpoint_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = weights_path(root)

    if destination.exists():
        return destination

    model_url = os.environ.get("MODEL_URL", "").strip()
    if not model_url:
        raise RuntimeError(
            f"{WEIGHTS_FILENAME} not found at {destination}. "
            "Set the MODEL_URL environment variable to a direct download URL "
            "(for example a Hugging Face file URL or cloud storage link)."
        )

    print(f"Downloading model weights to {destination}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".safetensors.part")
    urllib.request.urlretrieve(model_url, tmp_path)
    tmp_path.replace(destination)
    print("Model weights downloaded.")
    return destination
