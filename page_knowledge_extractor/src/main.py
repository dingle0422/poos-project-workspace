"""CLI 入口：extract（知识抽取）和 reason（推理检索）两个子命令。"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="page_knowledge_extractor",
        description="文档知识结构化抽取 + 渐进式披露检索框架",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── extract 子命令 ──
    p_extract = subparsers.add_parser(
        "extract",
        help="从文档中抽取知识结构",
        description="将非结构化的多级标题文档自动抽取为树形知识库",
    )
    p_extract.add_argument(
        "--input", "-i",
        required=True,
        help="输入文件路径（.docx 或 .txt）",
    )
    p_extract.add_argument(
        "--output", "-o",
        default="./page_knowledge",
        help="输出目录（默认: ./page_knowledge）",
    )

    # ── reason 子命令 ──
    p_reason = subparsers.add_parser(
        "reason",
        help="基于知识库进行推理检索",
        description="使用 REACT 模式在知识库中渐进式推理检索",
    )
    p_reason.add_argument(
        "--knowledge-dir", "-k",
        required=True,
        help="知识库目录路径（page_knowledge 下的具体知识目录）",
    )

    q_group = p_reason.add_mutually_exclusive_group(required=True)
    q_group.add_argument(
        "--question", "-q",
        help="单个问题",
    )
    q_group.add_argument(
        "--questions",
        help="批量问题文件路径（.csv 或 .xlsx）",
    )

    p_reason.add_argument(
        "--question-col",
        default="问题",
        help="问题列名称（批量模式，默认: '问题'）",
    )
    p_reason.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="每个子智能体的最大推理轮次（默认: 5）",
    )
    p_reason.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="最大并发子智能体数量（默认: 5）",
    )
    p_reason.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude 模型名称",
    )
    p_reason.add_argument(
        "--output", "-o",
        help="批量模式下的结果输出文件路径（.csv 或 .xlsx）",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        _run_extract(args)
    elif args.command == "reason":
        _run_reason(args)


def _run_extract(args: argparse.Namespace) -> None:
    from .extractor import extract
    try:
        result_dir = extract(args.input, args.output)
        print(f"\n🎉 知识抽取完成: {result_dir}")
    except Exception as e:
        print(f"\n❌ 抽取失败: {e}", file=sys.stderr)
        sys.exit(1)


def _run_reason(args: argparse.Namespace) -> None:
    from .reasoning.reactor import reason_single, reason_batch

    try:
        if args.question:
            result = asyncio.run(reason_single(
                question=args.question,
                knowledge_dir=args.knowledge_dir,
                max_rounds=args.max_rounds,
                max_concurrent=args.max_concurrent,
                model=args.model,
            ))
        else:
            results = asyncio.run(reason_batch(
                questions_file=args.questions,
                question_col=args.question_col,
                knowledge_dir=args.knowledge_dir,
                output_file=args.output,
                max_rounds=args.max_rounds,
                max_concurrent=args.max_concurrent,
                model=args.model,
            ))
    except Exception as e:
        print(f"\n❌ 推理失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
