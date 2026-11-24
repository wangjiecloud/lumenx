"""Local model path resolution.

All local weights live in a FLAT layout under ``LUMENX_MODELS_DIR``::

    <models_dir>/sdxl-base/
    <models_dir>/flux1-schnell/
    <models_dir>/ltx-video/
    <models_dir>/Fun-CosyVoice3-0.5B-2512/
    ...

The directory can be overridden with ``LUMENX_MODELS_DIR``.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_MODELS_DIR = "/Volumes/ExternalSSD/codespace/SuperJAI/projects/yue/data/models"

REQUIRED_PATHS: Dict[str, List[str]] = {
    "sdxl-base": ["model_index.json", "unet", "vae", "text_encoder", "text_encoder_2"],
    "flux1-schnell": [
        "model_index.json",
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "vae",
    ],
    "sdxl-inpaint": ["model_index.json", "unet", "vae", "text_encoder"],
    "flux1-kontext-dev": ["model_index.json", "transformer", "text_encoder_2", "vae"],
    "ip-adapter": [
        "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "sdxl_models/image_encoder",
    ],
    "ip-adapter-faceid": ["ip-adapter-faceid-plusv2_sdxl.bin"],
    "antelopev2": ["scrfd_10g_bnkps.onnx", "glintr100.onnx"],
    "ltx-video": ["model_index.json", "transformer", "vae", "text_encoder"],
    "wan21-vace-1.3b": [
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ],
    "whisper-large-v3-turbo": ["model.safetensors", "config.json"],
    "realesrgan": ["RealESRGAN_x4.pth"],
    "birefnet": ["model.safetensors", "config.json"],
    "Fun-CosyVoice3-0.5B-2512": ["llm.pt", "flow.pt", "hift.pt"],
    "Qwen2.5-VL-3B-Instruct-4bit": ["model.safetensors", "config.json"],
}


def models_dir() -> Path:
    raw = os.getenv("LUMENX_MODELS_DIR") or DEFAULT_MODELS_DIR
    return Path(raw).expanduser()


def model_path(key: str, required: bool = True) -> Path:
    path = models_dir() / key
    if required and not path.exists():
        raise FileNotFoundError(
            f"本地模型目录缺失: {path}\n"
            f"  请把模型拷贝到 {models_dir()}/ 下，或设置 LUMENX_MODELS_DIR。"
        )
    return path


def check_models(keys: Iterable[str] = None) -> Dict[str, Tuple[bool, List[str]]]:
    """Return {key: (ok, missing_relative_paths)} without importing torch."""
    result: Dict[str, Tuple[bool, List[str]]] = {}
    for key in keys or REQUIRED_PATHS.keys():
        base = model_path(key, required=False)
        if not base.exists():
            result[key] = (False, ["<model dir missing>"])
            continue
        missing = [
            rel for rel in REQUIRED_PATHS.get(key, []) if not (base / rel).exists()
        ]
        result[key] = (not missing, missing)
    return result
