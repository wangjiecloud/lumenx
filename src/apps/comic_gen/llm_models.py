"""LiteLLM proxy model catalogue (page-selectable LLM/VLM models).

The corporate LiteLLM gateway does not allow the client to list deployments
(`GET /v1/models` requires an `sk-` virtual key), so we ship the catalogue that
mirrors the proxy's model list and allow overriding it via env:

    LITELLM_MODELS=deepseek-v4.1-flash,qwen3.8-flash,claude-sonnet-4.6

Default model resolution: env ``LITELLM_MODEL`` > catalogue default.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

CATALOGUE_PATH = Path(__file__).resolve().parents[3] / "config" / "litellm_models.json"

_FALLBACK = {
    "default": "deepseek-v4.1-flash",
    "models": [
        {
            "id": "deepseek-v4.1-flash",
            "name": "DeepSeek V4.1 Flash",
            "capabilities": ["tool_use", "image_input"],
            "roles": ["chat"],
        },
    ],
}


def _load_catalogue() -> Dict[str, Any]:
    try:
        return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(_FALLBACK)


def list_models() -> List[Dict[str, Any]]:
    catalogue = _load_catalogue()
    models = catalogue.get("models") or list(_FALLBACK["models"])

    override = (os.getenv("LITELLM_MODELS") or "").strip()
    if override:
        wanted = [m.strip() for m in override.split(",") if m.strip()]
        by_id = {m.get("id"): m for m in models}
        models = [
            by_id.get(
                model_id,
                {
                    "id": model_id,
                    "name": model_id,
                    "capabilities": [],
                    "roles": ["chat"],
                },
            )
            for model_id in wanted
        ]
    return models


def default_model() -> str:
    explicit = (os.getenv("LITELLM_MODEL") or "").strip()
    if explicit:
        return explicit
    return _load_catalogue().get("default") or _FALLBACK["default"]


def provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "dashscope").strip().lower()


def is_proxy_provider() -> bool:
    return provider() == "litellm"
