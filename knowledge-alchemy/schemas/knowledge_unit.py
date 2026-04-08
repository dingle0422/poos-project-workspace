"""知识原子数据结构 —— 整个系统的数据契约层。

所有 Agent 之间的数据流转均通过此模块中定义的类型完成,
确保四阶段流水线的输入/输出格式严格对齐。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields as dataclass_fields
import dataclasses
from enum import Enum
from typing import Any


class LayerLevel(str, Enum):
    """教科书三层级体系。"""
    L1_INTRO = "L1_导论"
    L2_SOP = "L2_标准SOP"
    L3_CASE = "L3_案例深度"


@dataclass
class QAPair:
    """原始问答对 —— 流水线的最小输入单元。"""
    question: str
    answer: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class KnowledgeUnit:
    """最小知识单元 —— 第一阶段输出的原子结构。

    五要素模型: scenario + pain_point + trigger + actors + authority
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    original_q: str = ""      # 原始问题
    original_a: str = ""       # 原始回答
    key_concept: str = ""      # 核心概念（3-5字）
    scope: str = ""            # 知识单元的范围边界
    scenario: str = ""         # 业务场景
    pain_point: str = ""       # 用户痛点
    trigger: str = ""          # 触发条件
    actors: list[str] = field(default_factory=list)    # 涉及主体
    authority: str = ""        # 合规依据
    tags: list[str] = field(default_factory=list)      # 语义标签
    source: str = ""           # 来源
    cluster_id: str = ""       # 语义聚类后的簇编号
    embedding: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        # 提供 dict() 支持：把所有字段变成字典
        pass

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return [f.name for f in dataclasses.fields(self)]

    def __iter__(self):
        for f in dataclasses.fields(self):
            yield f.name, getattr(self, f.name)


@dataclass
class IfThenRule:
    """IF-THEN 逻辑规则 —— 第二阶段的判断树节点。"""
    condition: str         # IF 部分
    action: str            # THEN 部分
    exceptions: list[str] = field(default_factory=list)  # ELSE / 例外
    confidence: float = 1.0
    source_unit_ids: list[str] = field(default_factory=list)


@dataclass
class LogicSkeleton:
    """逻辑骨架 —— 同一标签下归纳出的通用逻辑结构。"""
    tag: str
    summary: str
    rules: list[IfThenRule] = field(default_factory=list)
    source_unit_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class ConflictReport:
    """逻辑矛盾报告。"""
    unit_id_a: str
    unit_id_b: str
    description: str
    severity: str = "medium"  # low / medium / high
    suggested_resolution: str = ""


@dataclass
class OutlineNode:
    """教科书大纲节点 —— 树形结构。"""
    title: str
    level: int             # 0=根, 1=章, 2=节, 3=小节
    children: list[OutlineNode] = field(default_factory=list)
    mapped_skeleton_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class KnowledgeGap:
    """知识空白描述。"""
    outline_node_id: str
    outline_title: str
    missing_description: str
    suggested_query: str = ""  # 用于 RAG 检索的建议查询


@dataclass
class TextBlock:
    """最终文本块 —— 第四阶段输出。"""
    outline_node_id: str
    layer: LayerLevel
    title: str
    body: str
    cross_references: list[str] = field(default_factory=list)
    glossary_terms: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class QualityMetrics:
    """质量评价指标。"""
    coverage: float = 0.0       # 大纲节点被填充的比例
    redundancy: float = 0.0     # 重复知识点的比例
    consistency: float = 0.0    # 无矛盾知识点的比例
    detail: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Coverage={self.coverage:.2%}  "
            f"Redundancy={self.redundancy:.2%}  "
            f"Consistency={self.consistency:.2%}"
        )


@dataclass
class AlchemyResult:
    """流水线最终输出 —— 聚合所有阶段产物。"""
    knowledge_units: list[KnowledgeUnit] = field(default_factory=list)
    logic_skeletons: list[LogicSkeleton] = field(default_factory=list)
    conflicts: list[ConflictReport] = field(default_factory=list)
    outline: OutlineNode | None = None
    gaps: list[KnowledgeGap] = field(default_factory=list)
    text_blocks: list[TextBlock] = field(default_factory=list)
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
