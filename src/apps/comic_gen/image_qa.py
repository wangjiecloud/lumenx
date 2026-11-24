"""Auto-QA for local image generation.

Two layers:
  * lint_prompt(): deterministic, cheap. Moves forbidden terms out of the
    positive prompt and guarantees required anchors (e.g. Chinese architecture,
    no people for scenes).
  * verify_image(): VLM check (LiteLLM proxy, image_input) returning a JSON
    verdict + fix terms. Used to optionally regenerate a failed image.
"""
import base64
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)
_adapter = None

SCENE_FORBIDDEN = [
    "modern road", "asphalt", "car", "vehicle", "power line", "electric pole",
    "modern building", "concrete", "skyscraper", "Japanese", "tatami", "shoji",
    "kimono", "engawa",
]
SCENE_REQUIRED = ["Chinese architecture", "no people", "no modern elements"]


def _get_adapter() -> LLMAdapter:
    global _adapter
    if _adapter is None:
        _adapter = LLMAdapter()
    return _adapter


def lint_prompt(prompt: str, kind: str = "image") -> Tuple[str, str]:
    """Return (positive_prompt, extra_negative_terms) with deterministic fixes."""
    text = prompt or ""
    extra_negative = []
    if kind in ("scene", "frame"):
        for bad in SCENE_FORBIDDEN:
            if re.search(re.escape(bad), text, re.IGNORECASE):
                text = re.sub(re.escape(bad), "", text, flags=re.IGNORECASE)
                extra_negative.append(bad)
    if kind == "scene":
        for anchor in reversed(SCENE_REQUIRED):
            if anchor.lower() not in text.lower():
                text = anchor + ", " + text
    elif kind == "frame":
        if "no modern elements" not in text.lower():
            text = "no modern elements, " + text
    text = re.sub(r",\s*,+", ",", text).strip(", ")
    return text, ", ".join(extra_negative)


def verify_image(path: str, expectation: str, kind: str = "image") -> Optional[Dict[str, Any]]:
    """Ask the VLM whether an image meets the expectation. Returns dict or None."""
    try:
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        prompt = (
            "你是图像质检员。判断这张图是否满足要求：" + (expectation or "") + "。\n"
            "检查：性别是否正确、是否单人（角色图）、是否古代中国、是否出现现代元素或日式元素。\n"
            '只输出 JSON：{"ok": true/false, "issues": ["..."], "fix_terms": "英文负向词，逗号分隔"}'
        )
        out = _get_adapter().chat(
            [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
            response_format={"type": "json_object"},
        )
        text = (out or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image QA verify failed: %s", exc)
        return None


def auto_qa_enabled() -> bool:
    return (os.getenv("AUTO_QA") or "check").lower() not in ("0", "off", "false")


def auto_qa_regen_enabled() -> bool:
    return (os.getenv("AUTO_QA") or "check").lower() in ("regen", "fix", "repair")
