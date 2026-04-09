"""
知识库管理器 - 管理 knowledge.md 文件的创建和写入
"""

import os
from pathlib import Path
from typing import Optional
from .parser import HeadingNode, HeadingParser


class KnowledgeBase:
    """知识库管理器"""

    KNOWLEDGE_FILE = "knowledge.md"

    @staticmethod
    def create_directory_structure(root_path: str, nodes: list[HeadingNode]) -> None:
        """
        创建知识目录结构
        """
        os.makedirs(root_path, exist_ok=True)
        for node in nodes:
            KnowledgeBase._create_node_dir(root_path, node)

    @staticmethod
    def _create_node_dir(parent_path: str, node: HeadingNode) -> None:
        """递归创建节点目录"""
        # 创建当前节点目录
        node_dir = os.path.join(parent_path, node.folder_name)
        os.makedirs(node_dir, exist_ok=True)

        # 写入当前节点的 knowledge.md
        KnowledgeBase._write_knowledge_md(node_dir, node)

        # 递归处理子节点
        for child in node.children:
            KnowledgeBase._create_node_dir(node_dir, child)

    @staticmethod
    def _write_knowledge_md(node_dir: str, node: HeadingNode) -> None:
        """写入 knowledge.md 文件"""
        file_path = os.path.join(node_dir, KnowledgeBase.KNOWLEDGE_FILE)
        
        # 构建内容
        lines = [
            f"# {node.number} {node.title}\n",
            f"**文件路径**: {node_dir}\n",
            f"\n## 本章节内容\n",
            f"{node.content if node.content else '(无正文内容)'}\n",
        ]

        # 添加子文件夹摘要
        if node.children:
            lines.append("\n## 子目录摘要\n")
            lines.append("> 以下为子文件夹内容摘要，具体细节可能遗漏，建议按需打开子文件夹内的 knowledge.md 渐进式探索。\n")
            for child in node.children:
                lines.append(f"### {child.folder_name}\n")
                lines.append(f"- **序号**: {child.number}\n")
                lines.append(f"- **标题**: {child.title}\n")
                lines.append(f"- **内容摘要**: {child.content[:200]}..." if len(child.content) > 200 else f"- **内容摘要**: {child.content}\n")
                lines.append(f"- **子节点数**: {len(child.children)}\n")
                lines.append("\n")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    @staticmethod
    def read_knowledge_md(dir_path: str) -> Optional[str]:
        """读取指定目录的 knowledge.md"""
        file_path = os.path.join(dir_path, KnowledgeBase.KNOWLEDGE_FILE)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    @staticmethod
    def get_subdirs(dir_path: str) -> list[str]:
        """获取指定目录下的所有子文件夹（按名称排序）"""
        subdirs = []
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        return sorted(subdirs)

    @staticmethod
    def get_dir_number(dir_name: str) -> Optional[str]:
        """从文件夹名称提取序号部分"""
        # 文件夹格式: "1.1.2_标题名"
        parts = dir_name.split('_', 1)
        if parts:
            return parts[0]
        return None

    @staticmethod
    def get_parent_dirs(knowledge_dir: str) -> list[str]:
        """获取从根目录到当前目录的完整路径列表"""
        current = knowledge_dir
        parents = []
        while True:
            parent = os.path.dirname(current)
            if parent == current:
                break
            parents.insert(0, os.path.basename(current))
            current = parent
        return parents

    @staticmethod
    def get_knowledge_dir_name(file_name: str, timestamp_ms: int) -> str:
        """生成知识目录名称：文件名_毫秒时间戳"""
        name = Path(file_name).stem
        return f"{name}_{timestamp_ms}"
