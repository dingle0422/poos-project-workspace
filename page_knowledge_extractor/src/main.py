#!/usr/bin/env python3
"""
Page Knowledge Extractor - CLI 入口
文档知识结构化抽取 + 渐进式披露检索框架
"""

import argparse
import sys
import os
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import Extractor
from src.reasoning.reactor import Reactor


def cmd_extract(args):
    """抽取命令"""
    print(f"📖 开始抽取文档: {args.input}")
    print(f"📁 输出目录: {args.output}")
    
    extractor = Extractor(output_dir=args.output)
    
    if os.path.isdir(args.input):
        # 批量抽取目录下的所有支持文件
        from src.file_reader import FileReader
        supported = FileReader.get_supported_extensions()
        files = []
        for ext in supported:
            files.extend(Path(args.input).glob(f"*{ext}"))
        
        if not files:
            print(f"❌ 目录中没有找到支持的文件: {', '.join(supported)}")
            return 1
        
        print(f"🔍 发现 {len(files)} 个文件")
        results = extractor.extract_multiple([str(f) for f in files])
        print(f"\n✅ 成功抽取 {len(results)} 个文档")
        
    else:
        # 单文件抽取
        result = extractor.extract(args.input)
        print(f"\n✅ 抽取完成: {result}")
    
    return 0


def cmd_reason(args):
    """推理命令"""
    print(f"🔍 开始推理")
    print(f"📁 知识目录: {args.knowledge_dir}")
    
    # 初始化推理引擎
    reactor = Reactor(
        knowledge_dir=args.knowledge_dir,
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # 加载问题
    questions = []
    if args.questions:
        # 从文件加载
        qfile = Path(args.questions)
        if qfile.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(args.questions)
            if args.question_col not in df.columns:
                print(f"❌ 列 '{args.question_col}' 不存在，可用列: {list(df.columns)}")
                return 1
            questions = df[args.question_col].tolist()
        elif qfile.suffix.lower() in ['.xlsx', '.xls']:
            import pandas as pd
            df = pd.read_excel(args.questions)
            if args.question_col not in df.columns:
                print(f"❌ 列 '{args.question_col}' 不存在，可用列: {list(df.columns)}")
                return 1
            questions = df[args.question_col].tolist()
        else:
            print(f"❌ 不支持的文件格式: {qfile.suffix}")
            return 1
        
        print(f"📋 从文件加载了 {len(questions)} 个问题")
        
        # 批量推理
        results, df = reactor.reason_batch(
            questions, 
            max_rounds=args.max_rounds,
            max_workers=args.max_workers
        )
        
        # 保存结果
        if args.output:
            Reactor.save_results(results, args.output)
    
    else:
        # 单问题推理
        result, evidence = reactor.reason(args.question, args.max_rounds)
        print(f"\n{'='*60}")
        print(f"📝 问题: {args.question}")
        print(f"{'='*60}")
        print(f"💡 答案:\n{result}")
        if evidence:
            print(f"\n📄 证据来源:")
            for i, e in enumerate(evidence, 1):
                preview = e[:300] + "..." if len(e) > 300 else e
                print(f"  [{i}] {preview}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="📚 Page Knowledge Extractor - 文档知识结构化抽取 + 渐进式披露检索框架",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # Extract 子命令
    extract_parser = subparsers.add_parser("extract", help="抽取文档知识结构")
    extract_parser.add_argument("--input", "-i", required=True, help="输入文件或目录")
    extract_parser.add_argument("--output", "-o", default="./page_knowledge", 
                                help="输出目录 (默认: ./page_knowledge)")
    
    # Reason 子命令
    reason_parser = subparsers.add_parser("reason", help="知识推理")
    reason_parser.add_argument("--knowledge-dir", "-k", required=True,
                               help="知识目录 (page_knowledge 下的独立知识目录)")
    reason_parser.add_argument("--question", "-q", help="单个问题")
    reason_parser.add_argument("--questions", help="问题文件 (CSV/XLSX)")
    reason_parser.add_argument("--question-col", help="问题列名 (当使用 questions 参数时)")
    reason_parser.add_argument("--max-rounds", type=int, default=5,
                              help="每轮最大推理次数 (默认: 5)")
    reason_parser.add_argument("--max-workers", type=int, default=4,
                              help="最大并行工作数 (默认: 4)")
    reason_parser.add_argument("--api-key", help="Anthropic API Key")
    reason_parser.add_argument("--output", "-o", help="结果输出文件 (CSV)")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        return cmd_extract(args)
    elif args.command == "reason":
        return cmd_reason(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
