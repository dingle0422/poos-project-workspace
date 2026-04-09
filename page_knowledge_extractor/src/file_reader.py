"""读取 docx 和 txt 文件，返回纯文本内容。"""

from __future__ import annotations

from pathlib import Path

from docx import Document


def read_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return _read_docx(path)
    elif suffix == ".txt":
        return _read_txt(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持 .docx 和 .txt")


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    lines: list[str] = []
    for para in doc.paragraphs:
        lines.append(para.text)
    return "\n".join(lines)


def _read_txt(path: Path) -> str:
    encodings = ["utf-8", "gbk", "gb2312", "utf-16", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码文件 {path}，已尝试编码: {encodings}")
