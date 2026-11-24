"""Runtime helpers for local model inference (device / dtype / memory / flags).

Everything here is lazy: importing this module must not import torch.
"""

import gc
import logging
import os
import subprocess
from typing import Any, Callable, Dict, Optional

from .paths import models_dir

logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def has_runtime_deps() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def backend_mode(kind: str) -> str:
    """kind: image | video | tts. Returns auto | local | dashscope."""
    return (os.getenv(f"{kind.upper()}_BACKEND") or "auto").strip().lower()


def local_enabled(kind: str) -> bool:
    mode = backend_mode(kind)
    if mode == "dashscope":
        return False
    if mode == "local":
        return True
    # auto: use local only when deps + model dir are present
    if kind == "tts":
        return _tts_worker_available()
    return has_runtime_deps() and models_dir().exists()


def _tts_worker_available() -> bool:
    """CosyVoice runs in its own venv, so torch in the API process is irrelevant."""
    python = os.getenv("COSYVOICE_PYTHON") or (
        "/Volumes/ExternalSSD/codespace/SuperJAI/projects/yue/.venv-cosyvoice/bin/python"
    )
    return (
        os.path.isfile(python) and (models_dir() / "Fun-CosyVoice3-0.5B-2512").exists()
    )


def device() -> str:
    requested = (os.getenv("LOCAL_DEVICE") or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def dtype(name: str = "auto"):
    import torch

    requested = (os.getenv("LOCAL_DTYPE") or name or "auto").strip().lower()
    if requested == "auto":
        marker = device()
        if marker == "cuda":
            return torch.bfloat16
        if marker == "mps":
            return torch.float16
        return torch.float32
    return {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(requested, torch.float16)


def total_ram_gb() -> Optional[float]:
    try:
        if os.uname().sysname == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            return int(out) / (1024**3)
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return None


def low_vram() -> bool:
    if env_bool("LOCAL_LOW_VRAM", False):
        return True
    if env_bool("LOCAL_HIGH_VRAM", False):
        return False
    ram = total_ram_gb()
    return bool(ram and ram <= 24)


def describe() -> Dict[str, Any]:
    ram = total_ram_gb()
    info: Dict[str, Any] = {
        "models_dir": str(models_dir()),
        "models_dir_exists": models_dir().exists(),
        "device": device(),
        "low_vram": low_vram(),
        "total_ram_gb": round(ram, 1) if ram else None,
        "deps": has_runtime_deps(),
        "image_backend": backend_mode("image"),
        "video_backend": backend_mode("video"),
        "tts_backend": backend_mode("tts"),
    }
    if has_runtime_deps():
        import torch

        info["torch"] = torch.__version__
        info["mps"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        info["cuda"] = torch.cuda.is_available()
        try:
            import diffusers
            import transformers

            info["diffusers"] = diffusers.__version__
            info["transformers"] = transformers.__version__
        except Exception:
            pass
    return info


class ManagedCache:
    """Very small LRU-ish cache that can unload heavy pipelines.

    On a 16 GB machine only one large pipeline should be resident at a time,
    so ``clear()`` is called before loading a different family.
    """

    def __init__(self, max_loaded: Optional[int] = None):
        self._items: Dict[str, Any] = {}
        self._max = max_loaded or int(
            os.getenv("LOCAL_MAX_LOADED") or (1 if low_vram() else 2)
        )

    def get(self, key: str, factory: Callable[[], Any]):
        if key in self._items:
            return self._items[key]
        while len(self._items) >= self._max:
            oldest = next(iter(self._items))
            logger.info("Unloading local pipeline: %s", oldest)
            self._items.pop(oldest, None)
            _release_memory()
        item = factory()
        self._items[key] = item
        return item

    def peek(self, key: str):
        return self._items.get(key)

    def clear(self) -> None:
        self._items.clear()
        _release_memory()

    def keys(self):
        return list(self._items.keys())


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


_release_memory = release_memory


CACHE = ManagedCache()


def to_device(pipe):
    """Move a diffusers pipeline to the configured device with safety tweaks."""
    marker = device()
    try:
        pipe.to(marker)
    except Exception as exc:  # e.g. unsupported op on MPS
        logger.warning("pipe.to(%s) failed (%s); falling back to cpu", marker, exc)
        pipe.to("cpu")
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        if low_vram() and hasattr(pipe, fn):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass
    return pipe
