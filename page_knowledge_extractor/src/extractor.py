"""
知识抽取器 - 将文档抽取为知识目录结构
"""

import os
import time
from pathlib import Path
from typing import Optional
from .file_reader import FileReader
from .parser import HeadingParser, HeadingNode
from .knowledge_base import KnowledgeBase


class Extractor:
    """文档知识抽取器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract(self, input_path: str, file_name: Optional[str] = None) -> str:
        """
        抽取文档知识结构
        Returns: 生成的独立知识目录路径
        """
        # 读取文件内容
        content = FileReader.read(input_path)

        # 生成知识目录名
        if file_name is None:
            file_name = os.path.basename(input_path)
        timestamp_ms = int(time.time() * 1000)
        knowledge_dir_name = KnowledgeBase.get_knowledge_dir_name(file_name, timestamp_ms)
        knowledge_dir = os.path.join(self.output_dir, knowledge_dir_name)

        # 解析文本结构
        roots = HeadingParser.parse_text(content)

        if not roots:
            # 无标题时，创建根目录存放全文
            os.makedirs(knowledge_dir, exist_ok=True)
            root_node = HeadingNode(
                number="1",
                title=Path(file_name).stem,
                content=content,
                level=1
            )
            KnowledgeBase._create_node_dir(self.output_dir, root_node)
            # 重命名
            actual_dir = os.path.join(self.output_dir, root_node.folder_name)
            if os.path.exists(actual_dir):
                import shutil
                new_dir = knowledge_dir
                idx = 1
                while os.path.exists(new_dir):
                    new_dir = f"{knowledge_dir}_{idx}"
                    idx += 1
                shutil.move(actual_dir, new_dir)
                return new_dir
            return actual_dir

        # 创建目录结构
        KnowledgeBase.create_directory_structure(knowledge_dir, roots)

        return knowledge_dir

    def extract_multiple(self, input_paths: list[str]) -> list[str]:
        """批量抽取多个文档"""
        results = []
        for path in input_paths:
            try:
                result = self.extract(path)
                results.append(result)
                print(f"✓ 已抽取: {path} -> {result}")
            except Exception as e:
                print(f"✗ 抽取失败: {path}, 错误: {e}")
        return results
