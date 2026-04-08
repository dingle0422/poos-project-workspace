"""统一 LLM 调用客户端，支持 OpenAI / Anthropic / MiniMax 三种后端。

根据 LLMConfig.provider 自动选择对应的 SDK 进行调用，
对外暴露统一的 chat / chat_json 接口，屏蔽底层差异。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)

# MiniMax Anthropic-compatible API 端点
MINIMAX_BASE_URL = "https://api.minimaxi.com/anthropic"


class LLMClient:
    """统一的 LLM 调用封装，支持 OpenAI / Anthropic / MiniMax 三后端。"""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None
        self._init_client()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """根据 provider 初始化对应 SDK 客户端。"""
        if self._config.provider == LLMProvider.OPENAI:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "请安装 openai 包: pip install openai",
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self._config.api_key}
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url
            self._client = OpenAI(**kwargs)

        elif self._config.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "请安装 anthropic 包: pip install anthropic",
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self._config.api_key}
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url
            self._client = anthropic.Anthropic(**kwargs)

        elif self._config.provider == LLMProvider.MINIMAX:
            # MiniMax 使用 Anthropic 兼容格式
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "请安装 anthropic 包: pip install anthropic",
                ) from exc
            kwargs: dict[str, Any] = {
                "api_key": self._config.api_key,
                "base_url": self._config.base_url or MINIMAX_BASE_URL,
            }
            self._client = anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送聊天消息，返回纯文本回复。"""
        if self._config.provider == LLMProvider.OPENAI:
            return self._chat_openai(messages)
        return self._chat_anthropic(messages)

    def chat_json(self, messages: list[dict[str, str]]) -> Any:
        """发送聊天消息，自动解析返回的 JSON。"""
        text = self.chat(messages)
        return self._extract_json(text)

    # ------------------------------------------------------------------
    # OpenAI 后端
    # ------------------------------------------------------------------

    def _chat_openai(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Anthropic / MiniMax 后端（共用）
    # ------------------------------------------------------------------

    def _chat_anthropic(self, messages: list[dict[str, str]]) -> str:
        """Anthropic SDK 要求 system 消息单独传参，此处做转换。"""
        system_msg = ""
        user_messages: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "messages": user_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if self._config.temperature:
            kwargs["temperature"] = self._config.temperature

        response = self._client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""

    # ------------------------------------------------------------------
    # JSON 提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Any:
        """从模型输出中提取并解析 JSON。"""
        match = _JSON_BLOCK_RE.search(text)
        if match:
            return json.loads(match.group(1).strip())
        stripped = text.strip()
        return json.loads(stripped)
