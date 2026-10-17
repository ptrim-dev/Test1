"""Locating and downloading the ONNX models the studio needs."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import onnxruntime

ASSETS = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"

# name -> (url, approximate size in MB, what it does)
MODELS = {
    "inswapper_128_fp16.onnx": (f"{ASSETS}/inswapper_128_fp16.onnx", 265, "face swapper"),
    "gfpgan_1.4.onnx": (f"{ASSETS}/gfpgan_1.4.onnx", 325, "face restoration"),
}


def models_dir() -> Path:
    path = Path(os.environ.get("FACESWAP_MODELS", "~/.faceswap_studio/models")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(name: str) -> Path:
    """Return the local path of a model, downloading it on first use."""
    path = models_dir() / name
    if path.exists():
        return path
    url, size_mb, purpose = MODELS[name]
    print(f"Downloading {name} ({purpose}, ~{size_mb} MB) to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def providers() -> list[str]:
    """Best available ONNX Runtime execution providers, GPU first."""
    available = onnxruntime.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available
