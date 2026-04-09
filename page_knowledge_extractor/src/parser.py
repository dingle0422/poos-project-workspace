"""
标题结构解析器 - 识别标准多级序号并构建树形结构
标准格式：1 -> 1.1 -> 1.1.1（数字+句点构成）
"""

import re
from typing import Optional
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class HeadingNode:
    """标题节点"""
    number: str          # 序号（如 "1.1.2"）
    title: str           # 标题名称
    content: str         # 本级内容（不含子标题内容）
    children: list['HeadingNode'] = field(default_factory=list)
    parent: Optional['HeadingNode'] = None
    level: int = 1       # 层级（从1开始）

    @property
    def folder_name(self) -> str:
        """生成文件夹名称：序号_标题"""
        safe_title = self.title.replace('/', '_').replace('\\', '_').replace(':', '_')
        return f"{self.number}_{safe_title}"


class HeadingParser:
    """标题结构解析器"""

    # 标准序号正则：1 / 1.1 / 1.1.1 / 1.1.1.1 等
    STANDARD_NUMBER_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+')

    @classmethod
    def parse_heading_line(cls, line: str) -> Optional[tuple[str, str]]:
        """
        解析单行标题
        Returns: (序号, 标题名) 或 None
        """
        line = line.strip()
        if not line:
            return None

        match = cls.STANDARD_NUMBER_PATTERN.match(line)
        if match:
            number = match.group(1)
            title = line[match.end():].strip()
            return number, title
        return None

    @classmethod
    def get_number_level(cls, number: str) -> int:
        """获取序号层级（1 -> 1级, 1.1 -> 2级, 1.1.1 -> 3级）"""
        return number.count('.') + 1

    @classmethod
    def is_direct_child(cls, parent_number: str, child_number: str) -> bool:
        """判断 child_number 是否是 parent_number 的直接子级"""
        return child_number.startswith(parent_number + '.') and child_number.count('.') == parent_number.count('.') + 1

    @classmethod
    def parse_content_blocks(cls, lines: list[str]) -> list[tuple[Optional[tuple[str, str]], str]]:
        """
        将文本解析为 (标题, 内容块) 列表
        非标准格式的行被归入最近标准标题的内容
        """
        blocks = []
        current_heading = None
        current_content = []

        for line in lines:
            parsed = cls.parse_heading_line(line)
            if parsed:
                # 保存上一个块
                if current_heading is not None or current_content:
                    blocks.append((current_heading, '\n'.join(current_content).strip()))
                
                # 开始新块
                current_heading = parsed
                current_content = []
            else:
                if current_heading is not None:
                    current_content.append(line)
                # 非标准行暂存，等遇到标准标题后归入其内容

        # 最后一个块
        if current_heading is not None or current_content:
            blocks.append((current_heading, '\n'.join(current_content).strip()))

        return blocks

    @classmethod
    def build_tree(cls, blocks: list[tuple[Optional[tuple[str, str]], str]]) -> list[HeadingNode]:
        """
        从 blocks 构建树形结构
        非标准标题作为最近标准标题的content处理
        """
        roots = []
        stack = []  # 维护当前路径上的节点栈

        for heading_info, content in blocks:
            if heading_info is None:
                # 非标准格式内容，归入最近父级
                if stack:
                    stack[-1].content += '\n' + content
                continue

            number, title = heading_info
            level = cls.get_number_level(number)
            node = HeadingNode(
                number=number,
                title=title,
                content=content,
                level=level
            )

            # 找到父级
            while stack and not cls.is_direct_child(stack[-1].number, number):
                stack.pop()

            if stack:
                node.parent = stack[-1]
                stack[-1].children.append(node)
            else:
                roots.append(node)

            stack.append(node)

        return roots

    @classmethod
    def parse_text(cls, text: str) -> list[HeadingNode]:
        """解析纯文本内容"""
        lines = text.split('\n')
        blocks = cls.parse_content_blocks(lines)
        return cls.build_tree(blocks)

    @classmethod
    def collect_all_nodes(cls, roots: list[HeadingNode]) -> list[HeadingNode]:
        """收集所有节点（深度优先）"""
        result = []
        def collect(node):
            result.append(node)
            for child in node.children:
                collect(child)
        for root in roots:
            collect(root)
        return result
