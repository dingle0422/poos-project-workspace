"""第一阶段 Prompt —— 去噪与结构化标签 (Deconstruction)

设计意图:
  将非结构化的问答文本拆解为标准化的五要素知识原子。
  Prompt 采用"角色设定 + 任务描述 + 输出格式约束"三段式结构,
  确保 LLM 输出可直接被 JSON 解析,减少后处理成本。
"""


class DeconstructionPrompts:
    """管理第一阶段所有 Prompt 模板。"""

    # ── 五要素提取 ──────────────────────────────────────────────
    # 设计意图: 用显式 JSON Schema 约束输出格式,
    # 并通过角色设定引导模型以"知识工程师"视角审视问答文本。
    EXTRACT_ELEMENTS_SYSTEM = (
        "你是一位资深知识工程师。你的任务是从一段问答对话中精确提取以下五个要素:\n"
        "1. scenario — 业务场景: 这段对话发生在什么业务背景下?\n"
        "2. pain_point — 用户痛点: 用户核心想解决的问题是什么?\n"
        "3. trigger — 触发条件: 在什么情况下这个问题会被触发?\n"
        "4. actors — 涉及主体: 哪些角色/实体参与其中? (列表)\n"
        "5. authority — 合规依据: 回答中引用了哪些法规、政策或行业标准? 如无则填'无'\n\n"
        "你还需要为该问答对生成3-5个语义标签(tags)。\n\n"
        "严格以 JSON 格式输出,不要包含任何其他文字。JSON Schema:\n"
        "{\n"
        '  "scenario": "string",\n'
        '  "pain_point": "string",\n'
        '  "trigger": "string",\n'
        '  "actors": ["string"],\n'
        '  "authority": "string",\n'
        '  "tags": ["string"]\n'
        "}"
    )

    EXTRACT_ELEMENTS_USER = (
        "请从以下问答对中提取五要素和标签:\n\n"
        "【问题】\n{question}\n\n"
        "【回答】\n{answer}"
    )

    # ── 原子化拆分 ──────────────────────────────────────────────
    # 设计意图: 长问答往往包含多个独立知识点,
    # 需要拆分为互不依赖的原子单元以便后续独立管理。
    # 通过要求输出 JSON 数组来天然支持多条结果。
    ATOMIZE_SYSTEM = (
        "你是一位知识拆解专家。给定一段较长的问答对话,"
        "你需要将其拆分为多个独立的最小知识单元。\n\n"
        "拆分原则:\n"
        "- 每个知识单元应当是自包含的,不依赖其他单元即可理解\n"
        "- 一个知识单元只包含一个核心知识点\n"
        "- 保留原始语境中的关键信息(如条件、例外)\n"
        "- 如果原始内容只有一个知识点,返回包含一个元素的数组即可\n\n"
        "为每个知识单元提取五要素和标签,以 JSON 数组格式输出:\n"
        "[\n"
        "  {\n"
        '    "scenario": "string",\n'
        '    "pain_point": "string",\n'
        '    "trigger": "string",\n'
        '    "actors": ["string"],\n'
        '    "authority": "string",\n'
        '    "tags": ["string"],\n'
        '    "content": "该知识点的精简描述"\n'
        "  }\n"
        "]"
    )

    ATOMIZE_USER = (
        "请将以下问答对拆分为多个最小知识单元:\n\n"
        "【问题】\n{question}\n\n"
        "【回答】\n{answer}"
    )

    # ── 语义标签生成(轻量版,聚类前的初步标签) ────────────────
    # 设计意图: 为已提取的知识单元补充更细粒度的标签,
    # 作为后续 embedding 聚类的"软约束"。
    TAG_GENERATION_SYSTEM = (
        "你是一位知识分类专家。根据给定的知识单元内容,"
        "生成3-7个层次化的语义标签。\n\n"
        "标签应覆盖: 行业领域 > 业务模块 > 具体主题\n"
        '以 JSON 数组输出,如: ["电商", "售后", "退货运费"]'
    )

    TAG_GENERATION_USER = (
        "请为以下知识单元生成语义标签:\n\n"
        "场景: {scenario}\n"
        "痛点: {pain_point}\n"
        "内容: {content}"
    )

    @classmethod
    def format_extract(cls, question: str, answer: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.EXTRACT_ELEMENTS_SYSTEM},
            {"role": "user", "content": cls.EXTRACT_ELEMENTS_USER.format(
                question=question, answer=answer,
            )},
        ]

    @classmethod
    def format_atomize(cls, question: str, answer: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.ATOMIZE_SYSTEM},
            {"role": "user", "content": cls.ATOMIZE_USER.format(
                question=question, answer=answer,
            )},
        ]

    @classmethod
    def format_tag(
        cls, scenario: str, pain_point: str, content: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.TAG_GENERATION_SYSTEM},
            {"role": "user", "content": cls.TAG_GENERATION_USER.format(
                scenario=scenario, pain_point=pain_point, content=content,
            )},
        ]
