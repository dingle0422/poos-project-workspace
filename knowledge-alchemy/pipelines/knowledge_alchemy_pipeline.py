"""QA知识炼金系统 - 流水线编排

四阶段知识炼金流水线：
Stage 1: Deconstruction - 去噪与结构化标签
Stage 2: Abstraction - 逻辑抽象与模式识别
Stage 3: Synthesis - 知识补全与系统合成
Stage 4: Layering - 层级化编排与润色
"""

from typing import Optional, Any
import dataclasses

from schemas.knowledge_unit import KnowledgeUnit, QAPair
from agents.deconstruction_agent import DeconstructionAgent
from agents.abstraction_agent import AbstractionAgent
from agents.synthesis_agent import SynthesisAgent
from agents.editor_agent import EditorAgent
from utils.config import LLMConfig


def _json_safe(obj: Any) -> Any:
    """将Python对象递归转换为JSON安全的格式（set -> list等）"""
    if isinstance(obj, set):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, tuple):
        return [_json_safe(item) for item in obj]
    # 处理 dataclass 实例
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return obj


class KnowledgeAlchemyPipeline:
    """QA知识炼金流水线

    将非结构化问答对通过四个阶段的AI处理，转化为教科书式知识体系。

    架构设计理念：
    - 每个阶段由专门的Agent负责，单一职责
    - 阶段之间通过标准数据结构传递
    - 预留人机协作hook（专家审核点）
    - 完整保留中间结果，便于追溯和迭代

    使用示例：
        pipeline = KnowledgeAlchemyPipeline(llm_config)
        result = pipeline.run(qa_pairs, domain="法律咨询")
        metrics = pipeline.evaluate(result)
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.deconstructor = DeconstructionAgent()
        self.abstractioner = AbstractionAgent()
        self.synthesizer = SynthesisAgent()
        self.editor = EditorAgent()

    def run(self, qa_pairs: list[dict],
            domain: str = "通用",
            stop_at_stage: Optional[int] = None) -> dict:
        """执行完整流水线

        Args:
            qa_pairs: 原始问答对列表 [{"question": "...", "answer": "...", "source": "..."}]
            domain: 知识领域/行业，用于生成教科书大纲
            stop_at_stage: 可选，在指定阶段停止（用于调试）

        Returns:
            包含所有阶段结果的字典
        """
        stages = {}

        # ========== Stage 1: Deconstruction ==========
        print("\n[Stage 1/4] 🔍 去噪与结构化标签...")
        # 只传递 QAPair 支持的字段
        valid_qa_fields = {"question", "answer", "source", "metadata", "id"}
        decon_qa_pairs = [
            QAPair(**{k: v for k, v in qa.items() if k in valid_qa_fields})
            for qa in qa_pairs
        ]

        all_units = []
        for i, qa in enumerate(decon_qa_pairs):
            print(f"  处理问答对 {i+1}/{len(decon_qa_pairs)}...")
            units = self.deconstructor.extract_and_atomize(qa, self.llm_config)
            all_units.extend(units)

        print(f"  → 提取了 {len(all_units)} 个知识单元")
        stages["deconstruction"] = {
            "qa_pairs_processed": len(decon_qa_pairs),
            "knowledge_units": [_json_safe(dict(u)) for u in all_units],
            "units_count": len(all_units)
        }

        if stop_at_stage == 1:
            return stages

        # ========== Stage 2: Abstraction ==========
        print("\n[Stage 2/4] 🧠 逻辑抽象与模式识别...")
        abstractions = []
        if all_units:
            # 逻辑归纳
            print("  执行逻辑归纳...")
            logic_result = self.abstractioner.logical_induction(all_units, self.llm_config)
            abstractions.append(logic_result)

            # IF-THEN转化
            print("  执行IF-THEN逻辑转化...")
            if_then_results = []
            for i, unit in enumerate(all_units[:min(len(all_units), 5)]):
                result = self.abstractioner.if_then_conversion(unit, self.llm_config)
                if_then_results.append(result)
            logic_result["if_then_trees"] = if_then_results

            # 矛盾检测
            print("  检测逻辑矛盾...")
            conflicts = self.abstractioner.conflict_detection(all_units, self.llm_config)
            logic_result["conflicts"] = conflicts

            if conflicts:
                print(f"  ⚠️ 检测到 {len(conflicts)} 处逻辑矛盾，需人工审核")

        stages["abstraction"] = abstractions or [{}]
        stages["conflicts"] = abstractions[0].get("conflicts", []) if abstractions else []

        if stop_at_stage == 2:
            return stages

        # ========== Stage 3: Synthesis ==========
        print("\n[Stage 3/4] 🏗️ 知识补全与系统合成...")
        print("  生成教科书大纲...")
        outline = self.synthesizer.generate_outline(domain, self.llm_config)
        print(f"  → 大纲包含 {len(outline['chapters'])} 章")

        # 知识映射
        units_as_dicts = [_json_safe(dict(u)) for u in all_units]
        mapping = self.synthesizer.knowledge_mapping(outline, units_as_dicts, self.llm_config)

        # 空白识别
        covered_chapters = list(mapping.get("coverage", {}).keys())
        gaps = self.synthesizer.gap_analysis(
            outline, covered_chapters, units_as_dicts, self.llm_config
        )
        print(f"  → 识别到 {len(gaps)} 处知识空白")

        # RAG补全（预留）
        rag_completions = self.synthesizer.rag_completion(gaps, self.llm_config)

        # 构建教科书结构
        textbook_structure = self.synthesizer.build_textbook_structure(
            outline, mapping, units_as_dicts, stages.get("conflicts", [])
        )
        textbook_structure["gaps"] = gaps
        textbook_structure["rag_completions"] = rag_completions

        stages["synthesis"] = {
            "outline": dict(outline),
            "mapping": mapping,
            "gaps": gaps,
            "rag_completions": rag_completions,
            "textbook_structure": textbook_structure
        }
        stages["gaps"] = gaps

        if stop_at_stage == 3:
            return stages

        # ========== Stage 4: Layering ==========
        print("\n[Stage 4/4] ✍️ 层级化编排与润色...")

        # 逐章节润色
        final_chapters = []
        for ch in textbook_structure.get("chapters", []):
            chapter_title = ch.get("title", "")

            if ch.get("knowledge_units"):
                # 有内容，进行三层撰写
                content_dict = {
                    "title": chapter_title,
                    "units": ch["knowledge_units"]
                }
                layered = self.editor.layering(content_dict, self.llm_config)
                cross_ref = self.editor.cross_reference(layered, self.llm_config)
                final = self.editor.edit_final(layered, cross_ref, self.llm_config)

                final_chapters.append({
                    "chapter_number": ch.get("chapter_number"),
                    "title": chapter_title,
                    "l1_intro": layered.get("l1_intro", ""),
                    "l2_sop": layered.get("l2_sop", ""),
                    "l3_cases": layered.get("l3_cases", ""),
                    "glossary": cross_ref.get("glossary", {}),
                    "cross_refs": cross_ref.get("跳转链接", {})
                })
            else:
                # 无内容，标记为待补充
                final_chapters.append({
                    "chapter_number": ch.get("chapter_number"),
                    "title": chapter_title,
                    "status": "pending_content",
                    "gap_note": "该章节尚无足够的知识单元覆盖"
                })

        # 生成教科书封面信息
        textbook_final = {
            "title": outline.get("title", f"{domain}知识手册"),
            "domain": domain,
            "chapters": final_chapters,
            "glossary": {},  # 汇总所有章节的术语表
            "index": [],      # 汇总所有索引词
        }

        # 汇总术语表和索引
        all_glossary = {}
        all_index = set()
        for ch in final_chapters:
            all_glossary.update(ch.get("glossary", {}))
            for ref in ch.get("cross_refs", {}).values():
                all_index.add(ref)
        textbook_final["glossary"] = all_glossary
        textbook_final["index"] = list(all_index)

        stages["layering"] = {
            "textbook": textbook_final
        }
        stages["textbook"] = textbook_final

        print(f"  → 生成 {len(final_chapters)} 个章节")
        print("\n✅ 四阶段流水线执行完成！")

        return _json_safe(stages)

    def evaluate(self, result: dict) -> dict:
        """评价流水线输出质量

        三个核心指标：
        - Coverage（覆盖率）: 原始问答中有多少被吸纳进教科书
        - Redundancy（冗余度）: 教科书中是否有重复/冗余内容
        - Consistency（一致性）: 逻辑是否自洽（无矛盾）

        注意：这些是近似指标，不是精确的学术评估。
        """
        coverage = 0.0
        redundancy = 0.0
        consistency = 0.0

        # Coverage: 有内容的章节数 / 总章节数
        if "synthesis" in result and "textbook_structure" in result["synthesis"]:
            ts = result["synthesis"]["textbook_structure"]
            total_chapters = len(ts.get("chapters", []))
            covered_chapters = sum(
                1 for ch in ts.get("chapters", [])
                if ch.get("knowledge_units")
            )
            if total_chapters > 0:
                coverage = covered_chapters / total_chapters

        # Redundancy: 基于未匹配知识点的比例
        if "synthesis" in result:
            mapping = result["synthesis"].get("mapping", {})
            unmatched = len(mapping.get("unmatched_knowledge_points", []))
            total = (unmatched +
                     sum(len(v) for v in mapping.get("coverage", {}).values()))
            if total > 0:
                redundancy = 1.0 - (unmatched / total)

        # Consistency: 无矛盾的比例
        conflicts = result.get("conflicts", [])
        if "deconstruction" in result:
            total_units = result["deconstruction"].get("units_count", 0)
            if total_units > 0 and conflicts:
                conflict_ratio = len(conflicts) / total_units
                consistency = max(0.0, 1.0 - conflict_ratio * 2)
            else:
                consistency = 1.0

        return {
            "coverage": coverage,
            "redundancy": redundancy,
            "consistency": consistency,
            "coverage_display": f"{coverage:.1%}",
            "redundancy_display": f"{redundancy:.1%}",
            "consistency_display": f"{consistency:.1%}"
        }

    def human_review_hook(self, stage: str, data: dict) -> bool:
        """人机协作审核Hook（预留接口）

        在指定阶段结束后、进入下一阶段前，触发人工审核。

        Args:
            stage: 当前阶段 ("abstraction" | "synthesis")
            data: 该阶段的输出数据

        Returns:
            True: 审核通过，继续流水线
            False: 审核不通过，暂停等待人工介入

        使用示例：
            if not pipeline.human_review_hook("abstraction", abstraction_result):
                print("等待专家审核...")
                return
        """
        # TODO: 实现人机协作审核逻辑
        # 当前版本：自动通过，不阻塞流水线
        return True
