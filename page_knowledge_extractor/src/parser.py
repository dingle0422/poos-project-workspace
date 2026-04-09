"""解析文档标题结构，识别标准数字序号（如 1. / 1.1 / 1.1.1）并构建树形结构。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# 匹配标准数字序号标题：1. / 1.1 / 1.1.1 等
# 序号后可跟 空格/制表符/全角空格，然后是标题文字
HEADING_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)\s*[.\u3000\s]\s*(.+)$"
)


@dataclass
class HeadingNode:
    """一个标题节点，包含序号、标题名、直属内容和子节点。"""
    number: str          # "1" / "1.1" / "1.1.1"
    title: str           # 标题文字
    content: str = ""    # 本级标题下的直属正文（不含子标题段落）
    children: list[HeadingNode] = field(default_factory=list)

    @property
    def level(self) -> int:
        return self.number.count(".") + 1

    @property
    def folder_name(self) -> str:
        safe_title = self.title.strip().replace("/", "_").replace("\\", "_")
        return f"{self.number}_{safe_title}"


def parse_headings(text: str) -> list[HeadingNode]:
    """将全文解析为标题树的顶层节点列表。"""
    lines = text.split("\n")
    segments = _split_into_segments(lines)
    if not segments:
        return []
    return _build_tree(segments)


@dataclass
class _Segment:
    number: str
    title: str
    body_lines: list[str] = field(default_factory=list)


def _split_into_segments(lines: list[str]) -> list[_Segment]:
    """按标题行切分文本为段落。"""
    segments: list[_Segment] = []
    preamble_lines: list[str] = []

    for line in lines:
        m = HEADING_PATTERN.match(line.strip())
        if m:
            seg = _Segment(number=m.group(1), title=m.group(2).strip())
            segments.append(seg)
        elif segments:
            segments[-1].body_lines.append(line)
        else:
            preamble_lines.append(line)

    # 如果有前言（标题之前的内容），附到一个虚拟的根段落
    if preamble_lines and segments:
        pre_text = "\n".join(preamble_lines).strip()
        if pre_text:
            segments[0].body_lines.insert(0, pre_text)

    return segments


def _build_tree(segments: list[_Segment]) -> list[HeadingNode]:
    """将扁平段落列表构建为嵌套树。"""
    nodes = [
        HeadingNode(
            number=s.number,
            title=s.title,
            content="\n".join(s.body_lines).strip(),
        )
        for s in segments
    ]

    root_nodes: list[HeadingNode] = []
    stack: list[HeadingNode] = []

    for node in nodes:
        # 弹出所有层级 >= 当前节点的栈元素
        while stack and _is_ancestor_or_same_level(stack[-1], node) is False:
            stack.pop()

        if stack and _is_direct_parent(stack[-1], node):
            stack[-1].children.append(node)
        else:
            # 回溯找合适的父节点
            placed = False
            for i in range(len(stack) - 1, -1, -1):
                if _is_direct_parent(stack[i], node):
                    stack[i].children.append(node)
                    stack = stack[: i + 1]
                    placed = True
                    break
            if not placed:
                root_nodes.append(node)
                stack.clear()

        stack.append(node)

    return root_nodes


def _is_direct_parent(parent: HeadingNode, child: HeadingNode) -> bool:
    """判断 parent 是否是 child 的直接父级。如 1 是 1.1 的父级，1.1 是 1.1.2 的父级。"""
    return child.number.startswith(parent.number + ".")


def _is_ancestor_or_same_level(node: HeadingNode, other: HeadingNode) -> bool:
    """node 是 other 的祖先或同级。"""
    if other.number.startswith(node.number + "."):
        return True
    if node.level == other.level:
        return True
    return False
