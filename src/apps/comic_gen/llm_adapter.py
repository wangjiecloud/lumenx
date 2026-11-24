"""
LLM Adapter - Unified interface for LiteLLM proxy, DashScope and OpenAI-compatible APIs.

Supports three providers (via LLM_PROVIDER env var):
  - litellm: Corporate LiteLLM gateway behind the wsproxy continue-proxy
    (OpenAI-compatible chat completions, auth token injected server-side).
  - dashscope: Alibaba Cloud DashScope via OpenAI-compatible endpoint
  - openai: Any OpenAI-compatible API (OpenAI, DeepSeek, Ollama, etc.)

Configuration via environment variables:
  LLM_PROVIDER=litellm|dashscope|openai
  LITELLM_BASE_URL=http://10.99.99.2:9090/litellm/v1
  LITELLM_API_KEY=placeholder
  LITELLM_MODEL=deepseek-v4.1-flash
  DASHSCOPE_API_KEY=...
  OPENAI_API_KEY=...
  OPENAI_BASE_URL=https://api.openai.com/v1
  OPENAI_MODEL=gpt-4o
"""

import os
import logging
from typing import Dict, List, Optional, Any

from ...utils.endpoints import get_provider_base_url

logger = logging.getLogger(__name__)

LITELLM_DEFAULT_BASE_URL = "http://10.99.99.2:9090/litellm/v1"
LITELLM_DEFAULT_API_KEY = "placeholder"
LITELLM_DEFAULT_MODEL = "deepseek-v4.1-flash"
# 网关在远端（高延迟 ~600ms），且默认模型是 reasoning 模型，超时要放宽
LITELLM_TIMEOUT_SECONDS = 300.0
LITELLM_MAX_RETRIES = 2


class LLMAdapter:
    """Unified LLM call interface supporting LiteLLM proxy, DashScope and OpenAI-compatible APIs."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "dashscope").lower()
        self._client = None
        logger.info(f"LLM Adapter initialized with provider: {self.provider}")

    @property
    def is_configured(self) -> bool:
        if self.provider == "litellm":
            return bool(self._litellm_base_url())
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        return bool(os.getenv("DASHSCOPE_API_KEY"))

    def _litellm_base_url(self) -> str:
        return (os.getenv("LITELLM_BASE_URL") or LITELLM_DEFAULT_BASE_URL).rstrip("/")

    def _get_client(self):
        """Get or create the OpenAI-compatible client (lazy, cached)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai>=1.0.0"
                )

            if self.provider == "litellm":
                self._client = OpenAI(
                    api_key=os.getenv("LITELLM_API_KEY") or LITELLM_DEFAULT_API_KEY,
                    base_url=self._litellm_base_url(),
                    timeout=LITELLM_TIMEOUT_SECONDS,
                    max_retries=LITELLM_MAX_RETRIES,
                    default_headers={"User-Agent": "curl/8.0"},
                )
            elif self.provider == "openai":
                self._client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    default_headers={"User-Agent": "curl/8.0"},
                )
            else:
                # DashScope uses OpenAI-compatible endpoint
                self._client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url=f"{get_provider_base_url('DASHSCOPE')}/compatible-mode/v1",
                    default_headers={"User-Agent": "curl/8.0"},
                )
        return self._client

    # DashScope qwen 系列：首选 qwen3.7-plus（最新），不可用时回退到 qwen3.6-plus，
    # 最终回退到 qwen-plus alias（始终指向最新稳定通用版）。
    # 维护 fallback chain 而不是硬写一个名字，避免新版本上下线时整条 LLM 链断掉。
    _DASHSCOPE_MODEL_FALLBACK_CHAIN = ["qwen3.7-plus", "qwen3.6-plus", "qwen-plus"]

    def _get_default_model(self) -> str:
        if self.provider == "litellm":
            return os.getenv("LITELLM_MODEL") or LITELLM_DEFAULT_MODEL
        if self.provider == "openai":
            return os.getenv("OPENAI_MODEL", "gpt-4o")
        return self._DASHSCOPE_MODEL_FALLBACK_CHAIN[0]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Send a chat completion request and return the response content.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            model: Model name override (uses provider default if None)
            response_format: Optional {"type": "json_object"} constraint

        Returns:
            The assistant's response content as a string.

        Raises:
            RuntimeError: If the API call fails.
        """
        client = self._get_client()

        # 显式 model override 路径：单次尝试，失败就抛。
        if model:
            return self._chat_once(client, model, messages, response_format)

        # Provider 默认路径：DashScope 走 fallback chain，其余单次尝试。
        if self.provider in ("openai", "litellm"):
            return self._chat_once(
                client, self._get_default_model(), messages, response_format
            )

        last_err: Optional[Exception] = None
        for idx, candidate in enumerate(self._DASHSCOPE_MODEL_FALLBACK_CHAIN):
            try:
                return self._chat_once(client, candidate, messages, response_format)
            except RuntimeError as e:
                # 仅在 "模型不存在 / 不可用" 类错误时回退；其他错误（鉴权、限流、网络）
                # 直接抛，不浪费第二次重试。判定关键字宽松匹配 DashScope 文案。
                msg = str(e).lower()
                is_model_unavailable = any(
                    k in msg
                    for k in (
                        "model not found",
                        "invalidmodel",
                        "model_not_found",
                        "no such model",
                        "not supported",
                        "modelnotfound",
                        "404",
                    )
                )
                last_err = e
                if (
                    is_model_unavailable
                    and idx < len(self._DASHSCOPE_MODEL_FALLBACK_CHAIN) - 1
                ):
                    next_candidate = self._DASHSCOPE_MODEL_FALLBACK_CHAIN[idx + 1]
                    logger.warning(
                        "DashScope model %s unavailable (%s); falling back to %s",
                        candidate,
                        e,
                        next_candidate,
                    )
                    continue
                raise
        # 理论上不可达（最后一次失败已 raise），保留兜底
        raise last_err if last_err else RuntimeError("DashScope: no models available")

    def _chat_once(
        self,
        client,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]],
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            provider_label = {
                "litellm": "LiteLLM",
                "openai": "OpenAI",
            }.get(self.provider, "DashScope")
            raise RuntimeError(f"{provider_label} API error: {e}") from e
