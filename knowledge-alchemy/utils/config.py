"""全局配置管理。

通过环境变量或直接赋值配置 LLM 后端和各阶段参数。
支持 OpenAI / Anthropic / MiniMax 三种 Provider，运行时按需切换。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MINIMAX = "minimax"


@dataclass
class LLMConfig:
    """LLM 调用参数。"""
    provider: LLMProvider = LLMProvider.OPENAI
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096
    base_url: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY", "")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            else:
                self.api_key = os.getenv("MINIMAX_API_KEY", "")

    def call_llm(self, system: str, user: str) -> str:
        """直接调用 LLM，返回文本。"""
        from .llm_client import LLMClient
        client = LLMClient(self)
        return client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])


@dataclass
class StageThresholds:
    """各阶段质量阈值。"""
    # 第一阶段: 最少原子化单元数(低于则警告)
    min_units_per_qa: int = 1
    # 第二阶段: 逻辑矛盾严重度过滤(仅 high 会阻断)
    conflict_block_severity: str = "high"
    # 第三阶段: 大纲覆盖率最低阈值
    min_coverage: float = 0.6
    # 第四阶段: 每层最小字数
    l1_min_words: int = 200
    l2_min_words: int = 400
    l3_min_words: int = 600


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置。"""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    cache_dir: str | None = None


@dataclass
class AlchemyConfig:
    """知识炼金系统顶层配置。"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    thresholds: StageThresholds = field(default_factory=StageThresholds)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    domain: str = "通用行业"
    verbose: bool = True
    # 人机协作 hook: 在这些阶段暂停等待专家审核
    expert_review_stages: list[str] = field(
        default_factory=lambda: ["abstraction", "synthesis"],
    )
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> AlchemyConfig:
        """从环境变量快速构建配置。"""
        provider_str = os.getenv("KA_LLM_PROVIDER", "minimax").lower()
        if provider_str == "anthropic":
            provider = LLMProvider.ANTHROPIC
        elif provider_str == "openai":
            provider = LLMProvider.OPENAI
        else:
            provider = LLMProvider.MINIMAX

        model_default = (
            "claude-sonnet-4-20250514" if provider == LLMProvider.ANTHROPIC
            else "MiniMax-M2.7" if provider == LLMProvider.MINIMAX
            else "gpt-4o"
        )

        return cls(
            llm=LLMConfig(
                provider=provider,
                model=os.getenv("KA_LLM_MODEL", model_default),
                temperature=float(os.getenv("KA_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("KA_MAX_TOKENS", "8192")),
                base_url=os.getenv("KA_BASE_URL"),
            ),
            domain=os.getenv("KA_DOMAIN", "通用行业"),
            verbose=os.getenv("KA_VERBOSE", "1") == "1",
        )
