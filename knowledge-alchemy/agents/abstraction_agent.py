"""第二阶段Agent - 逻辑抽象与模式识别"""

import json
import re
from typing import Optional
from schemas.knowledge_unit import KnowledgeUnit
from utils.config import LLMConfig


class AbstractionAgent:
    """第二阶段：逻辑抽象与模式识别

    本阶段负责从"个案"上升到"方法论"，把"肉"剔除，留下"骨架"。

    核心任务：
    1. 逻辑归纳（Induction）：同标签下的多个案例 → 通用逻辑骨架
    2. IF-THEN 转化：将具体回复 → 逻辑判断树
    3. 矛盾检测：扫描潜在逻辑矛盾，提请人工裁决

    设计理念：
    - 从碎片化案例中提取可复用的逻辑框架
    - 区分"变量"（因场景而异）和"定量"（不变的规则）
    - 让AI做"外科手术式"的归纳，而非笼统概括
    """

    def __init__(self):
        pass

    def logical_induction(self, units: list[KnowledgeUnit],
                         llm_config: LLMConfig) -> dict:
        """逻辑归纳 - 从多个案例中归纳通用逻辑骨架

        输入：同一标签或主题下的多个知识单元
        输出：该业务处理的通用逻辑架构，包括：
        - 变量：该环节中哪些因素会影响处理方式
        - 定量：无论什么情况下都成立的规则

        示例：
        输入：5个关于"合同违约处理"的问答
        输出：{
            "logic_framework": "合同违约处理5步法",
            "variables": ["违约情形", "损失大小", "合同约定条款"],
            "constants": ["必须先书面通知", "诉讼时效3年", "证据是核心"]
        }
        """
        if not units:
            return {"logic_framework": "", "variables": [], "constants": []}

        units_json = json.dumps([
            {
                "id": u.id,
                "question": u.original_q,
                "answer": u.original_a,
                "key_concept": u.key_concept,
            }
            for u in units
        ], ensure_ascii=False, indent=2)

        prompt = f"""你是一位专业的业务逻辑分析师。请分析以下{len(units)}个相关问答案例，
从中归纳出该业务处理的通用逻辑骨架。

【分析要求】
1. 识别所有案例中的共同处理步骤（定量/流程）
2. 识别因案例而异的因素（变量）
3. 指出哪些是"不变的核心规则"，哪些是"可调整的参数"

【案例列表】
{units_json}

【输出格式 - 严格JSON】
{{
  "logic_framework": "逻辑骨架的简要描述（一句话）",
  "common_steps": ["步骤1", "步骤2", ...],
  "variables": ["变量1", "变量2", ...],
  "constants": ["定量规则1", "定量规则2", ...],
  "summary": "对该业务逻辑的归纳总结（2-3句话）"
}}"""

        response = llm_config.call_llm(
            system="你是一位专业的业务逻辑分析师，擅长从案例中归纳方法论。严格输出JSON格式。",
            user=prompt
        )

        result = self._extract_json(response)
        if result:
            return result

        return {"logic_framework": "", "variables": [], "constants": [], "common_steps": []}

    def if_then_conversion(self, unit: KnowledgeUnit,
                          llm_config: LLMConfig) -> dict:
        """IF-THEN 逻辑判断树转化

        将具体的回复内容转化为逻辑判断树形式。

        示例：
        输入："装修漏水怎么办？首先要拍照留证据，然后联系装修公司，
              如果确认是装修问题可以要求返工或赔偿，协商不成可以起诉。"
        输出：{
            "if": "家庭装修中水路施工出现质量问题",
            "when": "水路施工违约",
            "then": [
                "1. 证据保全 -> 拍照/视频记录",
                "2. 责任判定 -> 确认违约方",
                "3. 协商 -> 要求返工或赔偿",
                "4. 诉讼 -> 向法院起诉（可选）"
            ],
            "depends_on": ["是否有合同", "损失大小"]
        }
        """
        prompt = f"""请将以下问答对转化为 IF-THEN 逻辑判断树。

【问答内容】
问题：{unit.original_q}
回答：{unit.original_a}

【转化规则】
- IF：触发条件（什么情况下触发这个处理逻辑）
- WHEN：时间/阶段条件（什么时候适用）
- THEN：处理步骤（按顺序列出）
- DEPENDS_ON：判断依赖（需要先确认的条件）

【输出格式 - 严格JSON】
{{
  "if": "触发条件描述",
  "when": "适用阶段或场景",
  "then": ["步骤1", "步骤2", "步骤3"],
  "depends_on": ["判断条件1", "判断条件2"],
  "example": "该逻辑适用的具体案例场景"
}}"""

        response = llm_config.call_llm(
            system="你是一位专业的逻辑结构分析师，擅长将自然语言转化为IF-THEN逻辑树。严格输出JSON格式。",
            user=prompt
        )

        result = self._extract_json(response)
        if result:
            return result

        return {"if": "", "when": "", "then": [], "depends_on": [], "example": ""}

    def conflict_detection(self, units: list[KnowledgeUnit],
                          llm_config: LLMConfig,
                          threshold: float = 0.7) -> list[dict]:
        """逻辑矛盾检测

        扫描同一问题或相关问题的不同回答，标记潜在的逻辑矛盾。

        矛盾类型：
        1. 事实矛盾：两个回答对同一事实的描述相反
        2. 时效冲突：规则已变更但旧回答未更新
        3. 范围冲突：适用条件描述不一致

        Args:
            units: 知识单元列表
            llm_config: LLM配置
            threshold: 矛盾置信度阈值，超过才报告

        Returns:
            矛盾对列表，每项包含矛盾描述和置信度
        """
        if len(units) < 2:
            return []

        # 构建问答对列表用于矛盾检测
        pairs = []
        for i, u in enumerate(units):
            pairs.append({
                "index": i,
                "id": u.id,
                "key_concept": u.key_concept,
                "question": u.original_q,
                "answer": u.original_a[:300],  # 截断避免token过多
            })

        pairs_json = json.dumps(pairs, ensure_ascii=False, indent=2)

        prompt = f"""请检测以下问答对之间的逻辑矛盾。

【问答对列表】
{pairs_json}

【矛盾检测要求】
1. 找出回答中存在直接冲突的对（如：一个是"可以"，另一个是"不可以"）
2. 找出时效性冲突（如：引用了已废止的法规）
3. 找出适用范围不一致的情况（如：一个说"必须"，另一个说"一般不需要"）

【输出格式 - 严格JSON数组】
[
  {{
    "index_a": 0,
    "index_b": 1,
    "conflict_type": "fact|time|scope",
    "description": "矛盾的具体描述",
    "confidence": 0.85,
    "severity": "high|medium|low",
    "resolution_suggestion": "建议如何解决这个矛盾"
  }},
  ...
]

如果没有检测到矛盾，返回空数组 []。"""

        response = llm_config.call_llm(
            system="你是一位严格的知识审计员，擅长检测文本中的逻辑矛盾。严格输出JSON数组格式。",
            user=prompt
        )

        result = self._extract_json(response)
        if isinstance(result, list):
            # 过滤低置信度矛盾
            return [r for r in result if r.get("confidence", 0) >= threshold]

        return []

    def identify_variables_vs_constants(self, units: list[KnowledgeUnit],
                                       llm_config: LLMConfig) -> dict:
        """区分"变量"与"定量"

        这是一个更精细的归纳任务：
        - 定量：无论什么情况下都成立的规则（如：诉讼时效）
        - 变量：因具体情况而异的判断（如：赔偿金额）

        用于后续构建可配置的逻辑模板。
        """
        if not units:
            return {"variables": [], "constants": []}

        units_str = "\n".join([
            f"[{i}] {u.original_q}\n    答: {u.original_a[:200]}"
            for i, u in enumerate(units)
        ])

        prompt = f"""请分析以下问答对，区分其中的"定量"（不变规则）和"变量"（因情况而异的因素）。

【问答对】
{units_str}

【输出格式 - 严格JSON】
{{
  "constants": [
    {{"rule": "规则描述", "source": "来自哪个问答"}}
  ],
  "variables": [
    {{"factor": "变量描述", "affected_cases": ["受影响的场景列表"]}}
  ]
}}"""

        response = llm_config.call_llm(
            system="你是一位专业的法律逻辑分析师，擅长区分不变规则与可变因素。严格输出JSON格式。",
            user=prompt
        )

        result = self._extract_json(response)
        return result if result else {"constants": [], "variables": []}

    def _extract_json(self, text: str) -> dict | list | None:
        """从LLM响应中提取JSON"""
        json_match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
