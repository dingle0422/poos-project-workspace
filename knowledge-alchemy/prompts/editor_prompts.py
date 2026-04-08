"""第四阶段 Prompt —— 层级化编排与润色 (Layering)

设计意图:
  将结构化的逻辑骨架和知识点转化为可出版的教科书文本。
  三层体系(L1导论/L2标准SOP/L3案例深度)分别面向不同读者,
  要求模型在同一主题上产出三种不同深度和语体的文本。
"""


class EditorPrompts:
    """管理第四阶段所有 Prompt 模板。"""

    # ── 语体转换 ──────────────────────────────────────────────
    # 设计意图: 原始QA内容多为口语化表达,
    # 需要转化为规范的教科书语体,同时保持信息完整性。
    STYLE_TRANSFER_SYSTEM = (
        "你是一位专业的学术写作编辑。将口语化的问答内容"
        "改写为规范的教科书语体。\n\n"
        "改写原则:\n"
        "- 消除口语标记词(比如说、就是、对吧、然后呢)\n"
        "- 使用规范的书面用语和专业术语\n"
        "- 保持信息完整,不遗漏关键细节\n"
        "- 添加必要的逻辑连接词\n"
        "- 段落结构清晰,每段聚焦一个要点\n\n"
        "直接输出改写后的文本,不要添加额外说明。"
    )

    STYLE_TRANSFER_USER = "请将以下内容改写为教科书语体:\n\n{content}"

    # ── 三层撰写 ──────────────────────────────────────────────
    # 设计意图: L1面向初学者(概览),L2面向从业者(操作指南),
    # L3面向专家(深度案例分析)。三层采用不同 system prompt
    # 引导模型调整语言深度和信息密度。
    LAYERING_L1_SYSTEM = (
        "你是一位教材撰写专家,正在撰写【L1 导论层】内容。\n\n"
        "L1 导论层要求:\n"
        "- 面向初学者,语言通俗易懂\n"
        "- 300-500字\n"
        "- 解释核心概念,不涉及具体操作步骤\n"
        "- 用类比或生活化举例帮助理解\n"
        "- 以 '本节要点' 开头,列出2-3个学习目标\n"
        "- 以 '延伸阅读' 结尾,提示读者可深入的方向\n\n"
        "直接输出教材文本。"
    )

    LAYERING_L2_SYSTEM = (
        "你是一位教材撰写专家,正在撰写【L2 标准SOP层】内容。\n\n"
        "L2 标准SOP层要求:\n"
        "- 面向从业者,提供可执行的操作指南\n"
        "- 500-800字\n"
        "- 包含明确的步骤编号(Step 1, Step 2...)\n"
        "- 标注每个步骤的注意事项和常见错误\n"
        "- 使用表格或列表整理关键参数\n"
        "- 引用相关法规或行业标准条款号\n\n"
        "直接输出教材文本。"
    )

    LAYERING_L3_SYSTEM = (
        "你是一位教材撰写专家,正在撰写【L3 案例深度层】内容。\n\n"
        "L3 案例深度层要求:\n"
        "- 面向专家,深度剖析真实或典型案例\n"
        "- 800-1200字\n"
        "- 包含: 案例背景 → 问题诊断 → 解决方案 → 效果评估 → 经验总结\n"
        "- 提供数据支撑(可合理虚构统计数据)\n"
        "- 讨论边界条件和例外情况\n"
        "- 引导读者思考: 结尾附1-2个讨论题\n\n"
        "直接输出教材文本。"
    )

    LAYERING_USER = (
        "章节标题: {title}\n\n"
        "相关逻辑骨架:\n{skeleton_text}\n\n"
        "原始知识点:\n{knowledge_text}\n\n"
        "请撰写该层级的教材内容。"
    )

    # ── 交叉引用生成 ──────────────────────────────────────────
    # 设计意图: 教科书的交叉引用和术语索引是专业性的重要体现,
    # 帮助读者在不同章节间建立知识网络。
    CROSS_REFERENCE_SYSTEM = (
        "你是一位教材索引编辑。分析给定的教材章节列表,"
        "生成交叉引用关系和术语索引。\n\n"
        "输出 JSON 格式:\n"
        "{\n"
        '  "cross_references": [\n'
        '    {"from_section": "章节A标题", "to_section": "章节B标题", '
        '"reason": "引用原因"}\n'
        "  ],\n"
        '  "glossary": [\n'
        '    {"term": "术语", "definition": "定义", '
        '"appears_in": ["章节A", "章节B"]}\n'
        "  ]\n"
        "}"
    )

    CROSS_REFERENCE_USER = (
        "以下是教材的所有章节标题和摘要:\n\n{sections_text}\n\n"
        "请生成交叉引用和术语索引。"
    )

    @classmethod
    def format_style_transfer(cls, content: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.STYLE_TRANSFER_SYSTEM},
            {"role": "user", "content": cls.STYLE_TRANSFER_USER.format(
                content=content,
            )},
        ]

    @classmethod
    def format_layering(
        cls,
        layer: str,
        title: str,
        skeleton_text: str,
        knowledge_text: str,
    ) -> list[dict[str, str]]:
        system_map = {
            "L1": cls.LAYERING_L1_SYSTEM,
            "L2": cls.LAYERING_L2_SYSTEM,
            "L3": cls.LAYERING_L3_SYSTEM,
        }
        return [
            {"role": "system", "content": system_map[layer]},
            {"role": "user", "content": cls.LAYERING_USER.format(
                title=title,
                skeleton_text=skeleton_text,
                knowledge_text=knowledge_text,
            )},
        ]

    @classmethod
    def format_cross_reference(
        cls, sections_text: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.CROSS_REFERENCE_SYSTEM},
            {"role": "user", "content": cls.CROSS_REFERENCE_USER.format(
                sections_text=sections_text,
            )},
        ]
