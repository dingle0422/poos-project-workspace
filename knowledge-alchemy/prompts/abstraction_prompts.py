"""第二阶段 Prompt —— 逻辑抽象与模式识别 (Abstraction)

设计意图:
  将第一阶段产出的知识原子从"具体案例"提升为"通用逻辑"。
  采用 Few-shot 策略: 在 system prompt 中内嵌 1-2 个示例,
  引导模型输出结构化的逻辑骨架和 IF-THEN 规则。
"""


class AbstractionPrompts:
    """管理第二阶段所有 Prompt 模板。"""

    # ── 逻辑归纳 ──────────────────────────────────────────────
    # 设计意图: 将同标签下的多条案例归纳为一条通用逻辑。
    # Few-shot 示例帮助模型理解"从具体到抽象"的思维方式。
    LOGICAL_INDUCTION_SYSTEM = (
        "你是一位逻辑归纳专家。给定同一标签下的多个知识单元,"
        "你需要归纳出一条通用的逻辑骨架(Logic Skeleton)。\n\n"
        "逻辑骨架应当:\n"
        "- 覆盖所有输入案例的共性\n"
        "- 抽象掉具体的数字、名称,保留逻辑结构\n"
        "- 标注适用范围和例外情况\n\n"
        "## Few-shot 示例\n\n"
        "输入知识单元 (标签: 退货运费):\n"
        '  1. "7天内退货,卖家承担运费"\n'
        '  2. "质量问题退货,卖家承担运费"\n'
        '  3. "非质量问题退货,买家承担运费"\n\n'
        "输出:\n"
        "```json\n"
        "{\n"
        '  "tag": "退货运费",\n'
        '  "summary": "退货运费的承担方取决于退货原因和时间窗口",\n'
        '  "rules": [\n'
        "    {\n"
        '      "condition": "商品存在质量问题 OR 在法定退货期内",\n'
        '      "action": "由卖家承担退货运费",\n'
        '      "exceptions": ["买家人为损坏", "定制商品"]\n'
        "    },\n"
        "    {\n"
        '      "condition": "非质量问题 AND 超出法定退货期",\n'
        '      "action": "由买家承担退货运费",\n'
        '      "exceptions": ["卖家自愿承担"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "请严格以上述 JSON 格式输出。"
    )

    LOGICAL_INDUCTION_USER = (
        "请对以下标签「{tag}」下的知识单元进行逻辑归纳:\n\n"
        "{units_text}"
    )

    # ── IF-THEN 转化 ──────────────────────────────────────────
    # 设计意图: 将自然语言回复转化为可执行的判断树,
    # 使知识可以被规则引擎或决策系统直接消费。
    IF_THEN_SYSTEM = (
        "你是一位决策逻辑设计师。将给定的知识内容转化为 IF-THEN 规则集。\n\n"
        "## Few-shot 示例\n\n"
        '输入: "如果订单超过30天未确认收货,平台自动确认。'
        '但如果买家已申请售后,则暂停自动确认。"\n\n'
        "输出:\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "condition": "订单超过30天未确认收货 AND 买家未申请售后",\n'
        '    "action": "平台自动确认收货",\n'
        '    "exceptions": [],\n'
        '    "confidence": 0.95\n'
        "  },\n"
        "  {\n"
        '    "condition": "订单超过30天未确认收货 AND 买家已申请售后",\n'
        '    "action": "暂停自动确认,等待售后处理结果",\n'
        '    "exceptions": ["售后超时未处理则恢复自动确认"],\n'
        '    "confidence": 0.9\n'
        "  }\n"
        "]\n"
        "```\n\n"
        "请严格以 JSON 数组格式输出。每条规则包含 condition/action/exceptions/confidence。"
    )

    IF_THEN_USER = (
        "请将以下知识内容转化为 IF-THEN 规则:\n\n{content}"
    )

    # ── 矛盾检测 ──────────────────────────────────────────────
    # 设计意图: 同一话题下不同来源的回答可能存在逻辑矛盾,
    # 需要在合成前识别并标记,避免最终知识体系自相矛盾。
    CONFLICT_DETECTION_SYSTEM = (
        "你是一位逻辑审计师。对比以下两段知识内容,"
        "判断它们之间是否存在逻辑矛盾。\n\n"
        "判断标准:\n"
        "- 同一条件下得出相反结论 → high severity\n"
        "- 数值/期限不一致 → medium severity\n"
        "- 措辞差异但逻辑兼容 → 无矛盾\n\n"
        "## Few-shot 示例\n\n"
        '内容A: "退货期限为7天"\n'
        '内容B: "退货期限为15天"\n\n'
        "输出:\n"
        "```json\n"
        "{\n"
        '  "has_conflict": true,\n'
        '  "severity": "medium",\n'
        '  "description": "退货期限数值不一致: A为7天, B为15天",\n'
        '  "suggested_resolution": "确认适用的法规版本,以最新规定为准"\n'
        "}\n"
        "```\n\n"
        "如无矛盾,输出:\n"
        "```json\n"
        '{"has_conflict": false}\n'
        "```"
    )

    CONFLICT_DETECTION_USER = (
        "请检测以下两段内容之间是否存在矛盾:\n\n"
        "【内容A (ID: {id_a})】\n{content_a}\n\n"
        "【内容B (ID: {id_b})】\n{content_b}"
    )

    @classmethod
    def format_induction(
        cls, tag: str, units_text: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.LOGICAL_INDUCTION_SYSTEM},
            {"role": "user", "content": cls.LOGICAL_INDUCTION_USER.format(
                tag=tag, units_text=units_text,
            )},
        ]

    @classmethod
    def format_if_then(cls, content: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.IF_THEN_SYSTEM},
            {"role": "user", "content": cls.IF_THEN_USER.format(content=content)},
        ]

    @classmethod
    def format_conflict(
        cls,
        id_a: str, content_a: str,
        id_b: str, content_b: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.CONFLICT_DETECTION_SYSTEM},
            {"role": "user", "content": cls.CONFLICT_DETECTION_USER.format(
                id_a=id_a, content_a=content_a,
                id_b=id_b, content_b=content_b,
            )},
        ]
