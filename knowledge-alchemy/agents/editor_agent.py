"""第四阶段Agent - 层级化编排与润色"""

from typing import TypedDict
from utils.config import LLMConfig
import json


class LayeredContent(TypedDict):
    """三层知识结构"""
    l1_intro: str       # L1 导论：核心原理阐述
    l2_sop: str         # L2 标准流程：业务SOP
    l3_cases: str       # L3 疑难穿插：案例深度


class CrossReferences(TypedDict):
    """交叉引用结构"""
    glossary: dict[str, str]      # 术语表 {术语: 定义}
    index: list[str]             # 索引词列表
    跳转链接: dict[str, str]     # 章节跳转 {源: 目标}


class EditorAgent:
    """第四阶段：层级化编排与润色

    本阶段负责：
    1. 语体转换 - 口语 -> 规范教科书陈述
    2. 分层撰写 - L1/L2/L3 三层内容
    3. 交叉引用 - 术语表、索引、章节跳转
    """

    SYSTEM_PROMPT = """你是一位专业的知识编辑，擅长将口语化内容转化为规范的教科书陈述。

风格要求：
- 口语化的"你/咱们" -> 规范的"从业人员/应当/必须"
- 主观建议口吻 -> 客观陈述口吻
- 保留实务指导性的核心信息，去除口语杂质

输出格式：直接输出转化后的文本，不要额外解释。"""

    LAYERING_PROMPT = """请将以下知识内容按教科书标准进行三层撰写：

L1 导论层：阐述该章节的核心原理和基础概念，用权威、沉稳的学术语气介绍。
L2 标准流程层：基于内容梳理标准业务SOP，用"第1步/第2步/..."的规范格式输出。
L3 疑难穿插层：将原始的典型问答对作为"案例链接"或"深度思考"模块嵌入，用【案例】标签标注。

内容：
{content}

输出格式（严格遵循）：
【L1 导论】
<L1内容>

【L2 标准流程】
<L2内容>

【L3 疑难穿插】
<L3内容>"""

    CROSS_REF_PROMPT = """请为以下教科书内容生成交叉引用：

1. 提取所有专业术语，给出简要定义（术语表）
2. 列出需要建立索引的关键词
3. 识别应建立章节跳转链接的地方（格式：源描述 -> 目标章节）

内容：
{content}

输出JSON格式：
{{
  "glossary": {{"术语1": "定义1", "术语2": "定义2"}},
  "index": ["索引词1", "索引词2"],
  "跳转链接": {{"源描述1": "目标章节1", "源描述2": "目标章节2"}}
}}"""

    def __init__(self):
        pass

    def style_transfer(self, content: str, llm_config: LLMConfig) -> str:
        """将口语化咨询建议转化为规范教科书陈述

        示例：
        输入："你签合同的时候可得注意了，别傻乎乎的什么都不看就签了"
        输出："从业人员在与相对方签订合同时，应当认真审查合同各项条款..."
        """
        response = llm_config.call_llm(
            system=self.SYSTEM_PROMPT,
            user=f"请将以下内容转化为规范的教科书陈述：\n\n{content}"
        )
        return response

    def layering(self, content: dict, llm_config: LLMConfig) -> LayeredContent:
        """执行三层撰写：L1导论 / L2标准SOP / L3案例深度"""
        content_str = str(content)

        response = llm_config.call_llm(
            system="你是一位专业的教科书编辑。严格按照指定的格式输出三层内容。",
            user=self.LAYERING_PROMPT.format(content=content_str)
        )

        # 解析三层结构
        result: LayeredContent = {"l1_intro": "", "l2_sop": "", "l3_cases": ""}
        current_key = None

        for line in response.split("\n"):
            if "【L1 导论】" in line or "<L1内容>" in line:
                current_key = "l1_intro"
            elif "【L2 标准流程】" in line or "<L2内容>" in line:
                current_key = "l2_sop"
            elif "【L3 疑难穿插】" in line or "<L3内容>" in line:
                current_key = "l3_cases"
            elif current_key and line.strip() and not line.startswith("【"):
                result[current_key] += line + "\n"

        # 清理并返回
        for key in result:
            result[key] = result[key].strip()
        return result

    def cross_reference(self, textbook: dict, llm_config: LLMConfig) -> CrossReferences:
        """生成术语表、索引、章节跳转链接"""
        content = str(textbook)

        response = llm_config.call_llm(
            system="你是一位专业的知识编辑，擅长建立知识间的交叉引用。严格按照JSON格式输出。",
            user=self.CROSS_REF_PROMPT.format(content=content)
        )

        # 尝试从响应中提取JSON
        import re
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return CrossReferences(
                    glossary=data.get("glossary", {}),
                    index=data.get("index", []),
                    跳转链接=data.get("跳转链接", {})
                )
            except json.JSONDecodeError:
                pass

        # Fallback
        return CrossReferences(glossary={}, index=[], 跳转链接={})

    def edit_final(self, layered: LayeredContent, cross_ref: CrossReferences,
                   llm_config: LLMConfig) -> dict:
        """最终润色：将三层内容与交叉引用整合成最终教科书"""
        integration_prompt = f"""请将以下三层内容整合成最终教科书文本，并嵌入交叉引用。

L1 导论：
{layered['l1_intro']}

L2 标准流程：
{layered['l2_sop']}

L3 疑难穿插：
{layered['l3_cases']}

术语表：
{json.dumps(cross_ref['glossary'], ensure_ascii=False, indent=2)}

请输出整合后的完整教科书章节文本。"""

        response = llm_config.call_llm(
            system="你是一位专业的教科书编辑，负责最终文字润色和格式整合。",
            user=integration_prompt
        )

        return {
            "content": response,
            "glossary": cross_ref["glossary"],
            "index": cross_ref["index"],
            "cross_refs": cross_ref["跳转链接"]
        }
