"""Local model loading layer.

Two backends coexist:
    * LLM / VLM      -> LiteLLM proxy (see ``src/apps/comic_gen/llm_adapter.py``)
    * Everything else -> local weights under ``LUMENX_MODELS_DIR`` (this package)

Public helpers are lazy and safe to import without torch installed.
"""

from .paths import check_models, model_path, models_dir
from .runtime import describe, device, dtype, local_enabled
from .image import LocalImageModel, get_local_image_model
from .video import LocalVideoModel, get_local_video_model
from .audio import LocalCosyVoiceTTS, get_local_tts
from .asr import transcribe
from .post import remove_background, upscale_image

__all__ = [
    "check_models",
    "model_path",
    "models_dir",
    "describe",
    "device",
    "dtype",
    "local_enabled",
    "LocalImageModel",
    "get_local_image_model",
    "LocalVideoModel",
    "get_local_video_model",
    "LocalCosyVoiceTTS",
    "get_local_tts",
    "transcribe",
    "upscale_image",
    "remove_background",
]
