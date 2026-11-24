"""Local ASR (Whisper) for reference-audio transcription."""

import logging
from typing import Optional

from .paths import model_path

logger = logging.getLogger(__name__)

_PIPE = None


def _device():
    from .runtime import device

    marker = device()
    if marker == "cuda":
        return 0
    if marker == "mps":
        return "mps"
    return -1


def transcribe(audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe an audio file with the local Whisper model."""
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline

        _PIPE = pipeline(
            "automatic-speech-recognition",
            model=str(model_path("whisper-large-v3-turbo")),
            device=_device(),
        )
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = _PIPE(audio_path, **kwargs)
    return result.get("text", "").strip()
