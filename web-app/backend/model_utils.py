"""Resolve checkpoint path and download model weights when needed."""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "token-compressor"
WEIGHTS_FILENAME = "model.safetensors"
DEFAULT_HF_MODEL_ID = "bchhabra2490/llmlingua2-compressor"


def checkpoint_root() -> Path:
    override = os.environ.get("CHECKPOINT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CHECKPOINT_ROOT


def hf_model_id() -> str:
    return os.environ.get("HF_MODEL_ID", DEFAULT_HF_MODEL_ID).strip()


def normalize_model_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return url

    hf_repo_match = re.fullmatch(r"https?://huggingface\.co/([^/]+/[^/]+)", url)
    if hf_repo_match:
        repo_id = hf_repo_match.group(1)
        return f"https://huggingface.co/{repo_id}/resolve/main/{WEIGHTS_FILENAME}"

    return url


def resolve_latest_checkpoint(root: Path | None = None) -> Path:
    root = root or checkpoint_root()
    if not root.exists():
        return root

    checkpoints = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    if checkpoints:
        return max(checkpoints, key=lambda p: int(p.name.split("-", 1)[1]))
    return root


def weights_path(root: Path | None = None) -> Path:
    return resolve_latest_checkpoint(root) / WEIGHTS_FILENAME


def download_from_huggingface(model_id: str, root: Path) -> Path:
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint from Hugging Face: {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(root),
        local_dir_use_symlinks=False,
    )
    destination = weights_path(root)
    if not destination.exists():
        raise RuntimeError(f"Downloaded {model_id} but {WEIGHTS_FILENAME} was not found in {root}.")
    print(f"Checkpoint ready at {root}")
    return destination


def download_from_url(url: str, destination: Path) -> Path:
    print(f"Downloading model weights to {destination}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".safetensors.part")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(destination)
    print("Model weights downloaded.")
    return destination


def ensure_model_weights(root: Path | None = None) -> Path:
    root = root or checkpoint_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = weights_path(root)

    if destination.exists():
        return destination

    model_url = os.environ.get("MODEL_URL", "").strip()
    if model_url:
        return download_from_url(normalize_model_url(model_url), destination)

    model_id = hf_model_id()
    if model_id:
        return download_from_huggingface(model_id, root)

    raise RuntimeError(
        f"{WEIGHTS_FILENAME} not found at {destination}. "
        "Upload the checkpoint to Hugging Face or set MODEL_URL / HF_MODEL_ID."
    )
