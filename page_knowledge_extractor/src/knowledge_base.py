"""知识库文件管理：根据解析树创建目录结构，写入 knowledge.md。"""

from __future__ import annotations

import os
from pathlib import Path

from .parser import HeadingNode


def build_knowledge_tree(nodes: list[HeadingNode], output_dir: Path) -> None:
    """递归创建知识目录结构并写入 knowledge.md。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for node in nodes:
        _write_node(node, output_dir)


def _write_node(node: HeadingNode, parent_dir: Path) -> None:
    node_dir = parent_dir / node.folder_name
    node_dir.mkdir(parents=True, exist_ok=True)

    for child in node.children:
        _write_node(child, node_dir)

    _write_knowledge_md(node, node_dir)


def _write_knowledge_md(node: HeadingNode, node_dir: Path) -> None:
    abs_path = os.path.abspath(node_dir)

    own_content = _extract_own_content(node)
    child_summaries = _build_child_summaries(node)

    sections: list[str] = []

    sections.append(f"# {node.number} {node.title}\n")
    sections.append(f"**路径:** `{abs_path}`\n")

    if own_content.strip():
        sections.append("## 本节内容\n")
        sections.append(own_content.strip())
        sections.append("")

    if child_summaries:
        sections.append("## 子目录概览\n")
        sections.append("> 以下为子章节摘要，详细内容请打开对应子目录的 knowledge.md 进行渐进式探索。\n")
        for summary in child_summaries:
            sections.append(summary)

    md_path = node_dir / "knowledge.md"
    md_path.write_text("\n".join(sections), encoding="utf-8")


def _extract_own_content(node: HeadingNode) -> str:
    """提取本级标题下除子标题内容外的正文。

    parser 已在 content 字段中保留本级正文（不含子级段落的文本），
    直接返回即可。
    """
    return node.content


def _build_child_summaries(node: HeadingNode) -> list[str]:
    summaries: list[str] = []
    for child in node.children:
        summary_text = _generate_summary(child)
        entry = f"### 📂 `{child.folder_name}/`\n\n{summary_text}\n"
        summaries.append(entry)
    return summaries


def _generate_summary(node: HeadingNode, max_chars: int = 300) -> str:
    """为子节点生成内容摘要。

    策略：取本级 content 的前 max_chars 字符作为摘要，
    并附上其直接子标题列表作为结构提示。
    """
    parts: list[str] = []

    content = node.content.strip()
    if content:
        truncated = content[:max_chars]
        if len(content) > max_chars:
            truncated += "……"
        parts.append(truncated)

    if node.children:
        child_names = [f"`{c.folder_name}/`" for c in node.children]
        parts.append(f"包含子章节: {', '.join(child_names)}")

    if not parts:
        parts.append(f"标题: {node.number} {node.title}")

    return "\n\n".join(parts)
