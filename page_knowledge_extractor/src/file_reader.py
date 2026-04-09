"""
文件读取器 - 支持 docx 和 txt 格式
"""

import os
from pathlib import Path
from typing import Optional


class FileReader:
    """文件读取基类"""

    @staticmethod
    def read(path: str) -> str:
        """根据文件扩展名调用对应读取方法"""
        ext = Path(path).suffix.lower()
        if ext == '.docx':
            return FileReader.read_docx(path)
        elif ext == '.txt':
            return FileReader.read_txt(path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")


    @staticmethod
    def read_docx(path: str) -> str:
        """读取 docx 文件"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")

        doc = Document(path)
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        return '\n'.join(paragraphs)


    @staticmethod
    def read_txt(path: str) -> str:
        """读取 txt 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


    @staticmethod
    def get_supported_extensions() -> list[str]:
        """获取支持的文件扩展名"""
        return ['.docx', '.txt']
