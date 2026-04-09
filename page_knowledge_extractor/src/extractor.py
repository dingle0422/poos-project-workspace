"""知识抽取主逻辑：读取文件 -> 解析标题 -> 构建知识目录。"""

from __future__ import annotations

import time
from pathlib import Path

from .file_reader import read_file
from .parser import parse_headings
from .knowledge_base import build_knowledge_tree


def extract(input_path: str, output_dir: str) -> Path:
    """执行完整的知识抽取流程。

    Args:
        input_path: 输入文件路径（.docx 或 .txt）
        output_dir: 输出目录（page_knowledge 目录）

    Returns:
        创建的知识库根目录路径
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    print(f"📖 读取文件: {input_file.name}")
    text = read_file(input_file)

    if not text.strip():
        raise ValueError(f"文件内容为空: {input_path}")

    print(f"🔍 解析标题结构...")
    nodes = parse_headings(text)

    if not nodes:
        raise ValueError("未识别到任何标准数字序号标题（如 1. / 1.1 / 1.1.1）")

    total_nodes = _count_nodes(nodes)
    max_depth = _max_depth(nodes)
    print(f"   找到 {len(nodes)} 个顶级标题，共 {total_nodes} 个节点，最大深度 {max_depth} 层")

    timestamp_ms = int(time.time() * 1000)
    stem = input_file.stem
    knowledge_dir_name = f"{stem}_{timestamp_ms}"
    knowledge_root = Path(output_dir) / knowledge_dir_name

    print(f"📁 构建知识目录: {knowledge_root}")
    build_knowledge_tree(nodes, knowledge_root)

    print(f"✅ 抽取完成！知识库位于: {knowledge_root}")
    _print_tree_summary(nodes, indent=0)

    return knowledge_root


def _count_nodes(nodes: list) -> int:
    count = 0
    for n in nodes:
        count += 1 + _count_nodes(n.children)
    return count


def _max_depth(nodes: list, current: int = 1) -> int:
    if not nodes:
        return current - 1
    return max(_max_depth(n.children, current + 1) for n in nodes)


def _print_tree_summary(nodes: list, indent: int) -> None:
    for node in nodes:
        prefix = "  " * indent + ("├── " if indent > 0 else "")
        has_content = "📝" if node.content.strip() else "  "
        child_count = f"({len(node.children)} 子节点)" if node.children else ""
        print(f"   {prefix}{has_content} {node.number} {node.title} {child_count}")
        _print_tree_summary(node.children, indent + 1)
