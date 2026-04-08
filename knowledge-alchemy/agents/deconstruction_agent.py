"""第一阶段Agent - 去噪与结构化标签

负责将非结构化的问答对转化为可被机器处理的"知识元数据"。
"""

from typing import Optional
from schemas.knowledge_unit import KnowledgeUnit, QAPair
from utils.config import LLMConfig
from utils.embedder import Embedder
import json
import re


class DeconstructionAgent:
    """第一阶段：去噪与结构化标签

    本阶段负责将非结构化的对话变成可被机器处理的"知识元数据"。

    核心任务：
    1. 要素提取：从Q&A中提取五要素（场景/痛点/触发/主体/依据）
    2. 原子化拆解：将长问答拆分为多个"最小知识单元"
    3. 多维打标：自动生成行业标签（语义聚类）

    设计理念：
    - 每个知识单元应该是"自包含"的，即包含回答一个问题所需的全部上下文
    - 不要预设死板的分类，让AI先进行语义聚类，观察数据自然分布
    """

    def __init__(self):
        self._embedder: Optional[Embedder] = None

    @property
    def embedder(self) -> Optional[Embedder]:
        """延迟初始化embedder（需要时才加载模型）"""
        return self._embedder

    def extract_elements(self, qa: QAPair, llm_config: LLMConfig) -> dict:
        """提取五要素

        从单条Q&A中提取：
        - scenario: 业务场景（这件事发生在什么背景下）
        - pain_point: 核心痛点（用户遇到了什么麻烦）
        - trigger: 触发条件（什么情况会引发这个问题）
        - actors: 涉及主体（谁参与了这个场景）
        - authority: 合规依据（有没有法规/制度支撑）

        Args:
            qa: 原始问答对

        Returns:
            五要素字典
        """
        prompt = f"""请从以下问答对中提取结构化要素：

问题：{qa.question}
回答：{qa.answer}

请提取以下五个要素（如果某个要素在问答中没有提及，标注为"未明确"）：

1. 业务场景（Scenario）：这件事发生在什么业务背景下？
2. 核心痛点（Pain Point）：提问者遇到了什么具体问题或困扰？
3. 触发条件（Trigger）：什么情况或条件引发了这个痛点？
4. 涉及主体（Actors）：哪些人或组织参与了这个场景？
5. 合规依据（Authority）：回答中引用的法规、制度、行业标准是什么？

输出格式（严格JSON）：
{{
  "scenario": "...",
  "pain_point": "...",
  "trigger": "...",
  "actors": [...],
  "authority": [...]
}}"""

        response = llm_config.call_llm(
            system="你是一位专业的数据分析师，擅长从文本中提取结构化要素。严格输出JSON格式。",
            user=prompt
        )

        elements = self._extract_json(response)
        if elements:
            return elements

        # Fallback: 返回原始要素
        return {
            "scenario": "未明确",
            "pain_point": "未明确",
            "trigger": "未明确",
            "actors": [],
            "authority": []
        }

    def atomize(self, qa: QAPair, llm_config: LLMConfig) -> list[KnowledgeUnit]:
        """将长问答拆分为多个"最小知识单元"

        一个关于"合同违约"的回答可能包含：
        - 违约认定标准
        - 赔偿计算方式
        - 证据保全方法
        - 诉讼时效规定

        这些应该被拆分为独立的知识单元，而不是作为一个整体。

        Args:
            qa: 原始问答对

        Returns:
            知识单元列表
        """
        prompt = f"""请将以下问答对拆分为多个"最小知识单元"。

每个知识单元应该：
- 回答一个具体、单一的问题
- 包含完整的上下文（谁、什么、何时、如何）
- 可以独立被引用和检索

问题：{qa.question}
回答：{qa.answer}

请拆分为N个知识单元（N根据内容复杂度决定，通常2-5个）：

输出格式（严格JSON数组）：
[
  {{
    "atom_id": 1,
    "question": "这个单元回答的具体问题",
    "answer": "完整的回答内容",
    "key_concept": "核心概念（3-5个字）",
    "scope": "这个知识单元的范围边界"
  }},
  ...
]"""

        response = llm_config.call_llm(
            system="你是一位专业的知识工程师，擅长将复杂内容拆解为原子化知识单元。严格输出JSON数组格式。",
            user=prompt
        )

        atoms = self._extract_json(response)
        if isinstance(atoms, list):
            units = []
            for atom in atoms:
                unit = KnowledgeUnit(
                    id=f"{qa.source or 'unknown'}_{atom.get('atom_id', 0)}",
                    original_q=f"{qa.question}（原始：{atom.get('question', '')}）",
                    original_a=atom.get("answer", ""),
                    key_concept=atom.get("key_concept", ""),
                    scope=atom.get("scope", ""),
                    scenario="", pain_point="", trigger="",
                    actors=[], authority="", tags=[]
                )
                units.append(unit)
            return units

        # Fallback: 将整个回答作为一个单元
        return [
            KnowledgeUnit(
                id=f"{qa.source or 'unknown'}_1",
                original_q=qa.question,
                original_a=qa.answer,
                key_concept="综合",
                scope="通用",
                scenario="", pain_point="", trigger="",
                actors=[], authority="", tags=[]
            )
        ]

    def extract_and_atomize(self, qa: QAPair,
                           llm_config: LLMConfig) -> list[KnowledgeUnit]:
        """一步完成要素提取+原子化拆解

        这是最常用的方法，先提取五要素，再做原子化拆分，
        最后将五要素信息注入到每个知识单元中。

        Args:
            qa: 原始问答对
            llm_config: LLM配置

        Returns:
            知识单元列表（已注入五要素）
        """
        # Step 1: 提取五要素
        elements = self.extract_elements(qa, llm_config)

        # Step 2: 原子化拆解
        atoms = self.atomize(qa, llm_config)

        # Step 3: 将五要素注入每个知识单元
        for atom in atoms:
            atom.scenario = elements.get("scenario", "")
            atom.pain_point = elements.get("pain_point", "")
            atom.trigger = elements.get("trigger", "")
            atom.actors = elements.get("actors", [])
            atom.authority = elements.get("authority", [])
            # 继承原始来源和标签
            atom.source = qa.source or ""

        return atoms

    def semantic_tagging(self, units: list[KnowledgeUnit],
                         llm_config: LLMConfig,
                         use_embedding: bool = False) -> list[KnowledgeUnit]:
        """多维语义打标

        不预设死板的分类，而是让AI观察数据自然分布后生成标签。

        Args:
            units: 知识单元列表
            use_embedding: 是否使用embedding聚类辅助打标（需要加载模型）

        Returns:
            更新了tags的知识单元列表
        """
        if not units:
            return units

        # 如果超过10个单元，先做embedding聚类
        if use_embedding and len(units) >= 5:
            try:
                self._embedder = Embedder()
                texts = [f"{u.original_q} {u.original_a}" for u in units]
                vectors = self._embedder.encode(texts)
                cluster_labels = self._embedder.cluster(vectors, n_clusters=min(10, len(units) // 2 + 1))

                for unit, label in zip(units, cluster_labels):
                    unit.tags.append(f"cluster_{label}")
            except Exception:
                pass

        # LLM语义打标
        units_str = json.dumps([{"id": u.id, "q": u.original_q, "a": u.original_a[:100]} for u in units],
                              ensure_ascii=False, indent=2)

        prompt = f"""请为以下知识单元生成语义标签。

标签应该反映：
- 主题领域（如：劳动法、合同法、税务）
- 业务类型（如：员工管理、业务招待）
- 处理阶段（如：事前预防、事中处理、事后救济）
- 难易程度（如：基础、进阶、高阶）

知识单元：
{units_str}

请生成3-5个标签覆盖这些单元的主题分布，输出JSON：
{{
  "tagging": {{
    "单元ID1": ["标签1", "标签2"],
    "单元ID2": ["标签1", "标签3"]
  }},
  "tag_description": {{
    "标签1": "这个标签的含义和适用范围"
  }}
}}"""

        response = llm_config.call_llm(
            system="你是一位专业的知识分类专家，负责生成有意义的语义标签。严格输出JSON格式。",
            user=prompt
        )

        tagging = self._extract_json(response)
        if tagging and "tagging" in tagging:
            for unit in units:
                if unit.id in tagging["tagging"]:
                    unit.tags.extend(tagging["tagging"][unit.id])
                # 去重
                unit.tags = list(dict.fromkeys(unit.tags))

        return units

    def _extract_json(self, text: str) -> dict | list | None:
        """从LLM响应中提取JSON"""
        json_match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
