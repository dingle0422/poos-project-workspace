"""第三阶段 Prompt —— 知识补全与系统合成 (Synthesis)

设计意图:
  以 Top-down 方式生成教科书大纲,再将已有逻辑骨架映射到大纲中。
  通过 gap analysis 发现缺失,为后续 RAG 补全提供检索线索。
  大纲结构采用嵌套 JSON 表达,便于程序化处理。
"""


class SynthesisPrompts:
    """管理第三阶段所有 Prompt 模板。"""

    # ── 大纲生成 ──────────────────────────────────────────────
    # 设计意图: 根据行业领域和已有标签,生成结构化的教科书大纲。
    # 大纲应遵循"总→分→合"的经典教材编排逻辑。
    GENERATE_OUTLINE_SYSTEM = (
        "你是一位教材编审专家。根据给定的行业领域和知识标签列表,"
        "生成一份结构化的教科书大纲。\n\n"
        "大纲要求:\n"
        "- 3-5个一级章节(level=1)\n"
        "- 每个一级章节下2-4个二级节(level=2)\n"
        "- 必要时添加三级小节(level=3)\n"
        "- 遵循从基础到高级、从理论到实践的编排逻辑\n"
        "- 覆盖提供的所有标签主题\n\n"
        "以嵌套 JSON 格式输出:\n"
        "{\n"
        '  "title": "根节点标题",\n'
        '  "level": 0,\n'
        '  "children": [\n'
        "    {\n"
        '      "title": "第一章标题",\n'
        '      "level": 1,\n'
        '      "children": [\n'
        '        {"title": "第一节标题", "level": 2, "children": []}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    GENERATE_OUTLINE_USER = (
        "行业领域: {domain}\n\n"
        "已有知识标签:\n{tags_text}\n\n"
        "请生成教科书大纲。"
    )

    # ── 知识映射 ──────────────────────────────────────────────
    # 设计意图: 将逻辑骨架自动分配到大纲的对应节点。
    # 这是一个"分类+匹配"任务,需要模型理解骨架语义和大纲层级。
    KNOWLEDGE_MAPPING_SYSTEM = (
        "你是一位知识编排专家。给定一份教科书大纲和一组逻辑骨架,"
        "将每个逻辑骨架映射到最合适的大纲节点。\n\n"
        "映射原则:\n"
        "- 一个骨架可以映射到多个节点(如果内容跨章节)\n"
        "- 优先映射到最具体的层级(叶节点优先)\n"
        "- 如果找不到合适的节点,标记为 'unmapped'\n\n"
        "输出 JSON 数组:\n"
        "[\n"
        '  {"skeleton_id": "xxx", "outline_node_ids": ["yyy", "zzz"]},\n'
        '  {"skeleton_id": "aaa", "outline_node_ids": ["unmapped"]}\n'
        "]"
    )

    KNOWLEDGE_MAPPING_USER = (
        "## 教科书大纲\n{outline_text}\n\n"
        "## 逻辑骨架列表\n{skeletons_text}\n\n"
        "请完成知识映射。"
    )

    # ── 空白分析 ──────────────────────────────────────────────
    # 设计意图: 比较"理想大纲"和"已填充内容",
    # 找出缺少素材的节点,生成 RAG 检索查询建议。
    GAP_ANALYSIS_SYSTEM = (
        "你是一位知识审计师。对比教科书大纲与已映射的知识内容,"
        "识别哪些大纲节点缺少足够的知识素材。\n\n"
        "对每个存在知识空白的节点,输出:\n"
        "- outline_node_id: 节点ID\n"
        "- outline_title: 节点标题\n"
        "- missing_description: 缺失了什么内容\n"
        "- suggested_query: 建议用于检索补全的搜索查询\n\n"
        "以 JSON 数组格式输出。如果所有节点都已充分覆盖,返回空数组 []。"
    )

    GAP_ANALYSIS_USER = (
        "## 教科书大纲(含映射信息)\n{outline_with_mapping}\n\n"
        "请进行知识空白分析。"
    )

    # ── RAG 补全(预留接口的 Prompt) ──────────────────────────
    # 设计意图: 给定一个知识空白和检索到的外部文本段落,
    # 要求模型提炼出与现有知识体系一致的补充内容。
    RAG_COMPLETION_SYSTEM = (
        "你是一位知识补全专家。根据检索到的参考资料,"
        "为指定的知识空白生成补充内容。\n\n"
        "要求:\n"
        "- 内容应与现有知识体系风格一致\n"
        "- 标注信息来源\n"
        "- 以五要素格式输出(scenario/pain_point/trigger/actors/authority)\n"
        "- 如果参考资料不足以回答,明确标注为 'insufficient_source'\n\n"
        "输出 JSON 格式,同 KnowledgeUnit 结构。"
    )

    RAG_COMPLETION_USER = (
        "## 知识空白\n{gap_description}\n\n"
        "## 检索到的参考资料\n{retrieved_text}\n\n"
        "请生成补充知识单元。"
    )

    @classmethod
    def format_outline(
        cls, domain: str, tags_text: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.GENERATE_OUTLINE_SYSTEM},
            {"role": "user", "content": cls.GENERATE_OUTLINE_USER.format(
                domain=domain, tags_text=tags_text,
            )},
        ]

    @classmethod
    def format_mapping(
        cls, outline_text: str, skeletons_text: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.KNOWLEDGE_MAPPING_SYSTEM},
            {"role": "user", "content": cls.KNOWLEDGE_MAPPING_USER.format(
                outline_text=outline_text, skeletons_text=skeletons_text,
            )},
        ]

    @classmethod
    def format_gap_analysis(
        cls, outline_with_mapping: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.GAP_ANALYSIS_SYSTEM},
            {"role": "user", "content": cls.GAP_ANALYSIS_USER.format(
                outline_with_mapping=outline_with_mapping,
            )},
        ]

    @classmethod
    def format_rag_completion(
        cls, gap_description: str, retrieved_text: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.RAG_COMPLETION_SYSTEM},
            {"role": "user", "content": cls.RAG_COMPLETION_USER.format(
                gap_description=gap_description,
                retrieved_text=retrieved_text,
            )},
        ]
