"""第三阶段Agent - 知识补全与系统合成"""

from typing import TypedDict
import json
import re
from utils.config import LLMConfig


class TextbookOutline(TypedDict):
    """教科书大纲结构"""
    title: str
    domain: str
    chapters: list[dict]


class GapAnalysisResult(TypedDict):
    """空白分析结果"""
    gaps: list[str]
    filled_by_rag: list[dict]
    unfilled: list[str]


class SynthesisAgent:
    """第三阶段：知识补全与系统合成

    本阶段是整个流水线的"架构师"角色，负责：
    1. 根据行业标准生成理想的教科书大纲（Top-down）
    2. 将前两阶段提炼的逻辑点填充到大纲中（Mapping）
    3. 对比理想大纲与已有素材，识别知识空白（Gap Analysis）
    4. 通过RAG检索外部知识补全空白（RAG Completion）

    核心设计理念：
    - 不再依赖原始问答对的"幸存者偏差"
    - 而是以"完整知识体系"为参照，逆向补充缺失
    """

    SYSTEM_PROMPT = """你是一位专业的知识架构师，擅长根据行业标准构建完整的知识体系大纲。
你生成的教科书大纲应当：
- 符合该领域公认的知识结构
- 层次分明（章/节/目）
- 覆盖该领域从基础到进阶的核心内容
- 既包含原理性内容，也包含实务性内容"""

    OUTLINE_GENERATION_PROMPT = """请为「{domain}」领域生成一本教科书的大纲（Table of Contents）。

要求：
- 生成5-8章，每章2-4节
- 覆盖该领域的基础概念、标准流程、常见场景、疑难处理
- 用JSON格式输出，结构如下：

{{
  "title": "教科书标题",
  "domain": "{domain}",
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "第一章标题",
      "sections": [
        {{"section_number": 1.1, "title": "第一节标题", "subsections": []}},
        {{"section_number": 1.2, "title": "第二节标题", "subsections": []}}
      ]
    }}
  ]
}}"""

    MAPPING_PROMPT = """你是一位知识编辑，负责将具体的知识内容填充到教科书大纲中。

理想大纲：
{outline}

已有知识素材：
{knowledge_points}

任务：
1. 逐章检查已有素材能覆盖大纲中的哪些章节
2. 将相关素材分配到对应章节
3. 用JSON格式输出映射结果：

{{
  "coverage": {{
    "章节编号": ["相关素材索引列表"],
    ...
  }},
  "unmatched_chapters": ["未覆盖的章节列表"],
  "matched_knowledge_points": ["已匹配的素材索引列表"],
  "unmatched_knowledge_points": ["未匹配的素材索引列表（孤岛知识）"]
}}"""

    GAP_ANALYSIS_PROMPT = """你是一位严格的知识审计员，负责识别知识体系中的空白。

理想大纲覆盖的章节：
{outline_chapters}

已有素材已覆盖的章节：
{covered_chapters}

已有素材提炼的知识点：
{knowledge_points}

任务：
1. 对比理想大纲与已有覆盖，找出缺失的内容领域
2. 用中文描述每个知识空白，格式："关于'XXX'（在'YYY章节'下）的系统性描述缺失"
3. 输出JSON：

{{
  "gaps": [
    "关于'XXX基础概念'（在第N章第M节下）的系统性描述缺失，建议补充：具体缺失内容",
    ...
  ],
  "gap_locations": {{
    "gap_1": ["章节位置1", "章节位置2"]
  }}
}}"""

    RAG_COMPLETION_PROMPT = """以下是需要补充的知识空白：

{gaps}

请针对每个空白，用检索外部知识的方式生成补充内容。
每个空白输出一段规范的专业描述（200-400字）。

输出JSON格式：
{{
  "completions": [
    {{
      "gap": "空白描述",
      "content": "补充的专业内容"
    }}
  ]
}}"""

    def __init__(self):
        pass

    def generate_outline(self, domain: str, llm_config: LLMConfig) -> TextbookOutline:
        """根据行业标准生成教科书大纲（Top-down Structure）

        这是第三阶段的起点。不看任何问答对，直接由AI根据
        该领域的公认知识体系生成一份"理想大纲"。

        这个大纲代表了"一个合格的从业者应该掌握的所有知识"，
        后续所有步骤都以此为参照标准。
        """
        response = llm_config.call_llm(
            system=self.SYSTEM_PROMPT,
            user=self.OUTLINE_GENERATION_PROMPT.format(domain=domain)
        )

        # 解析JSON响应
        outline = self._extract_json(response)
        if outline:
            return TextbookOutline(
                title=outline.get("title", f"{domain}知识手册"),
                domain=domain,
                chapters=outline.get("chapters", [])
            )

        # Fallback: 生成简单大纲
        return TextbookOutline(
            title=f"{domain}知识手册",
            domain=domain,
            chapters=[
                {"chapter_number": 1, "title": "基础概念", "sections": [
                    {"section_number": 1.1, "title": "定义与范围", "subsections": []}
                ]},
                {"chapter_number": 2, "title": "标准流程", "sections": [
                    {"section_number": 2.1, "title": "流程概述", "subsections": []}
                ]},
            ]
        )

    def knowledge_mapping(self, outline: TextbookOutline,
                          knowledge_points: list[dict],
                          llm_config: LLMConfig) -> dict:
        """将第二阶段提炼的逻辑点填充到大纲（Mapping）

        这是一个匹配过程：将"碎片化的知识矿石"嵌入"理想的大理石雕像"中。
        可能出现的情况：
        1. 知识矿石恰好嵌入一个凹槽 → 完美匹配
        2. 知识矿石太大需要打磨 → 部分匹配
        3. 知识矿石找不到位置 → 孤岛知识（留待后续处理）
        4. 大纲有凹槽但没有矿石 → 知识空白
        """
        outline_str = json.dumps(outline, ensure_ascii=False, indent=2)
        points_str = json.dumps(knowledge_points, ensure_ascii=False, indent=2,
                               default=str)

        response = llm_config.call_llm(
            system="你是一位专业的知识架构师，负责将知识内容映射到大纲中。严格输出JSON格式。",
            user=self.MAPPING_PROMPT.format(
                outline=outline_str,
                knowledge_points=points_str
            )
        )

        mapping = self._extract_json(response)
        if mapping:
            return mapping

        return {
            "coverage": {},
            "unmatched_chapters": [],
            "matched_knowledge_points": [],
            "unmatched_knowledge_points": [str(i) for i in range(len(knowledge_points))]
        }

    def gap_analysis(self, outline: TextbookOutline,
                     covered_chapters: list[str],
                     knowledge_points: list[dict],
                     llm_config: LLMConfig) -> list[str]:
        """识别知识空白（Gap Analysis）

        这是最关键的步骤之一。
        AI对比"理想大纲"与"已有素材"，明确指出：
        - "为了形成体系，目前缺失关于'XXX基础概念'的实务描述"
        - "关于'YYY前期准备'的操作指引尚未覆盖"

        这些空白是后续RAG补全或人工补充的目标。
        """
        all_chapters = self._flatten_outline(outline)
        covered_set = set(covered_chapters)

        uncovered = [ch for ch in all_chapters if ch not in covered_set]
        points_str = json.dumps(knowledge_points, ensure_ascii=False, indent=2,
                              default=str)

        response = llm_config.call_llm(
            system="你是一位严格的知识审计员，负责识别知识体系中的空白。用JSON格式输出。",
            user=self.GAP_ANALYSIS_PROMPT.format(
                outline_chapters=json.dumps(all_chapters, ensure_ascii=False, indent=2),
                covered_chapters=json.dumps(covered_set, ensure_ascii=False, indent=2),
                knowledge_points=points_str
            )
        )

        result = self._extract_json(response)
        if result and "gaps" in result:
            return result["gaps"]

        # Fallback: 基于章节对比生成简单gap描述
        gaps = [f"关于'{ch}'章节的系统性描述缺失" for ch in uncovered]
        return gaps

    def rag_completion(self, gaps: list[str],
                       llm_config: LLMConfig) -> list[dict]:
        """针对知识空白，通过RAG检索外部知识进行补全

        这是一个预留接口。当前版本返回空列表，
        表示这些gap需要通过其他方式（如人工补充）来填补。

        完整RAG实现需要：
        1. 连接外部知识库（行业法规库、ISO标准库等）
        2. 对每个gap进行向量化检索
        3. 将检索到的外部内容整合后返回

        TODO: 实现RAG补全逻辑
        """
        if not gaps:
            return []

        # 当前版本：提示需要外部RAG
        # 后续可接入：法规数据库、行业标准文档、Wikipedia等
        return [
            {
                "gap": gap,
                "content": "",
                "status": "pending_rag",
                "suggestion": f"需要通过RAG检索外部知识库补充: {gap}"
            }
            for gap in gaps
        ]

    def build_textbook_structure(self, outline: TextbookOutline,
                                 mapping: dict,
                                 knowledge_points: list[dict],
                                 conflicts: list[dict]) -> dict:
        """将所有元素整合成教科书结构

        这是第三阶段的最终输出：
        一份有骨架（大纲）、有内容（填充的知识）、有标注（空白和矛盾）的半成品教科书。
        """
        chapters = []
        coverage = mapping.get("coverage", {})

        for chapter in outline.get("chapters", []):
            chapter_num = chapter.get("chapter_number", 0)
            chapter_title = chapter.get("title", "")

            matched_indices = coverage.get(str(chapter_num), [])
            matched_points = []

            for idx in matched_indices:
                try:
                    if 0 <= idx < len(knowledge_points):
                        matched_points.append(knowledge_points[idx])
                except (ValueError, TypeError):
                    pass

            chapters.append({
                "chapter_number": chapter_num,
                "title": chapter_title,
                "knowledge_units": matched_points,
                "conflict_notes": [
                    c for c in conflicts
                    if c.get("chapter") == chapter_num
                ]
            })

        return {
            "outline": dict(outline),
            "chapters": chapters,
            "unmatched_knowledge": mapping.get("unmatched_knowledge_points", []),
            "coverage_rate": len(chapters) / max(len(outline.get("chapters", [])), 1)
        }

    def _flatten_outline(self, outline: TextbookOutline) -> list[str]:
        """将大纲展平为章节标题列表"""
        chapters = []
        for ch in outline.get("chapters", []):
            ch_title = ch.get("title", "")
            chapters.append(f"第{ch.get('chapter_number', '?')}章 {ch_title}")
            for sec in ch.get("sections", []):
                chapters.append(f"  {sec.get('section_number', '')} {sec.get('title', '')}")
        return chapters

    def _extract_json(self, text: str) -> dict | None:
        """从LLM响应中提取JSON"""
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
