"""Translate Chinese storyboard/video prompts into English for local models.

SDXL/LTX understand English prompts far better than Chinese. This helper is
lazy and cached at the call site (frames store `image_prompt_en`).
"""
import logging

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)
_adapter = None


def _get_adapter() -> LLMAdapter:
    global _adapter
    if _adapter is None:
        _adapter = LLMAdapter()
    return _adapter


def to_english_prompt(prompt_zh: str, kind: str = "image") -> str:
    """Convert a Chinese prompt into a concise English generation prompt."""
    text = (prompt_zh or "").strip()
    if not text:
        return ""
    instruction = (
        "Convert the following Chinese storyboard prompt into a concise English "
        "image-generation prompt (30-60 words). Keep proper nouns by pinyin, use common "
        "English costume/architecture/camera terms, and do not add commentary. "
        "Output only the English prompt.\n\n" + text
    )
    try:
        out = _get_adapter().chat([{"role": "user", "content": instruction}])
        return (out or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prompt translation failed: %s", exc)
        return ""
