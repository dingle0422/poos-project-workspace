"""知识炼金系统 - 命令行入口"""

import argparse
import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.knowledge_alchemy_pipeline import KnowledgeAlchemyPipeline
from examples.sample_qa import SAMPLE_QA_PAIRS
from utils.config import LLMConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="QA知识炼金系统 - 将非结构化问答转化为教科书式知识体系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 使用内置示例运行
  python main.py --input qa.json --output ./output  # 指定输入文件
  python main.py --domain 金融 --model openai       # 指定领域和模型
        """
    )
    parser.add_argument(
        "--input", "-i",
        help="输入JSON文件路径，格式: [{\"question\": \"...\", \"answer\": \"...\", \"source\": \"...\"}]"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="输出目录路径 (默认: ./output)"
    )
    parser.add_argument(
        "--domain", "-d",
        default="法律咨询",
        help="知识领域/行业 (默认: 法律咨询)"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM后端 (默认: 从环境变量推断)"
    )
    parser.add_argument(
        "--stage", "-s",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="运行到哪个阶段 (默认: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    return parser.parse_args()


def load_input(path: str) -> list[dict]:
    """加载输入文件"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "qa_pairs" in data:
        return data["qa_pairs"]
    return data


def save_output(result: dict, output_dir: str):
    """保存输出结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整结果
    with open(f"{output_dir}/textbook.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存各阶段中间结果
    if "stages" in result:
        for stage_name, stage_data in result["stages"].items():
            stage_file = f"{output_dir}/stage_{stage_name}.json"
            with open(stage_file, "w", encoding="utf-8") as f:
                json.dump(stage_data, f, ensure_ascii=False, indent=2)

    # 保存评价指标
    if "metrics" in result:
        with open(f"{output_dir}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(result["metrics"], f, ensure_ascii=False, indent=2)


def print_result_summary(result: dict, metrics: dict, verbose: bool = False):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("📚 知识炼金完成 - 结果摘要")
    print("=" * 60)

    # 评价指标
    print("\n📊 评价指标:")
    print(f"   覆盖率 (Coverage):  {metrics.get('coverage', 0):.1%}")
    print(f"   冗余度 (Redundancy): {metrics.get('redundancy', 0):.1%}")
    print(f"   一致性 (Consistency): {metrics.get('consistency', 0):.1%}")

    # 教科书结构
    if "textbook" in result:
        textbook = result["textbook"]
        print(f"\n📖 生成教科书章节数: {len(textbook.get('chapters', []))}")

        if verbose:
            print("\n章节预览:")
            for i, ch in enumerate(textbook.get("chapters", [])[:3], 1):
                print(f"  第{i}章: {ch.get('title', '未命名')}")
                print(f"         知识单元数: {len(ch.get('knowledge_units', []))}")

    # 知识空白
    if "gaps" in result and result["gaps"]:
        print(f"\n⚠️  识别到的知识空白: {len(result['gaps'])}处")
        for gap in result["gaps"][:3]:
            print(f"  - {gap}")

    # 逻辑矛盾
    if "conflicts" in result and result["conflicts"]:
        print(f"\n⚠️  检测到的逻辑矛盾: {len(result['conflicts'])}处")
        for conflict in result["conflicts"][:2]:
            print(f"  - {conflict.get('description', '矛盾')}")

    print(f"\n💾 结果已保存至: {result.get('_output_dir', './output')}")
    print("=" * 60)


def main():
    args = parse_args()

    print("🔮 QA知识炼金系统初始化...")
    print(f"   领域: {args.domain}")
    print(f"   阶段: {'全部' if args.stage == 'all' else f'阶段{args.stage}'}")

    # 1. 加载数据
    if args.input:
        print(f"\n📂 从文件加载数据: {args.input}")
        qa_pairs = load_input(args.input)
    else:
        print("\n📂 使用内置示例数据")
        qa_pairs = SAMPLE_QA_PAIRS

    print(f"   加载问答对: {len(qa_pairs)}条")

    # 2. 初始化LLM配置
    llm_config = LLMConfig.from_env(default_model=args.model)

    # 3. 初始化流水线
    pipeline = KnowledgeAlchemyPipeline(llm_config)

    # 4. 执行流水线
    print(f"\n🚀 开始执行知识炼金流水线...")
    try:
        result = pipeline.run(
            qa_pairs,
            domain=args.domain,
            stop_at_stage=None if args.stage == "all" else int(args.stage)
        )
        result["_output_dir"] = args.output
    except Exception as e:
        print(f"\n❌ 流水线执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # 5. 评价
    metrics = pipeline.evaluate(result)
    result["metrics"] = metrics

    # 6. 保存
    save_output(result, args.output)

    # 7. 打印摘要
    print_result_summary(result, metrics, verbose=args.verbose)


if __name__ == "__main__":
    main()
