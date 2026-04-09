# 文档知识结构化抽取 + 渐进式披露检索框架

将非结构化的多级标题文档自动抽取为树形知识库，并支持基于 REACT 模式的渐进式推理检索。

## 安装

```bash
pip install -r requirements.txt
```

需要设置 `ANTHROPIC_API_KEY` 环境变量（推理功能依赖 Claude API）。

## 使用方法

### 1. 知识抽取

从 docx 或 txt 文档中抽取结构化知识：

```bash
python3 -m src.main extract --input ./docs/文档.docx --output ./page_knowledge
```

输入文档中的标准数字序号标题（如 `1.` / `1.1` / `1.1.1`）会被识别为层级结构。其他序号类型（中文序号、字母序号等）视为所在标准标题的正文内容。

抽取后在 `page_knowledge/` 下生成 `{文件名}_{时间戳}/` 目录，内部按标题层级嵌套，每层包含 `knowledge.md`。

### 2. 知识推理

#### 单问题推理

```bash
python3 -m src.main reason \
  --knowledge-dir ./page_knowledge/文档_1234567890 \
  --question "你的问题"
```

#### 批量推理

```bash
python3 -m src.main reason \
  --knowledge-dir ./page_knowledge/文档_1234567890 \
  --questions ./questions.csv \
  --question-col "问题" \
  --max-rounds 5 \
  --output ./results.csv
```

## 推理机制

采用 REACT（Reasoning + Acting）模式的渐进式推理：

1. 从知识库根目录开始，每轮只读取当前层的 `knowledge.md`
2. 大模型判断当前信息的相关性和颗粒度
3. 向下探索只能逐层进行，向上回溯可以跳级
4. 遇到多个相关子目录时，分叉出多个子智能体并行推理
5. 所有子智能体结果递归合并，给出最终答案

## 目录结构

```
page_knowledge_extractor/
├── page_knowledge/           # 知识抽取输出目录
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py               # CLI 入口
│   ├── extractor.py          # 知识抽取主逻辑
│   ├── parser.py             # 标题结构解析
│   ├── file_reader.py        # docx/txt 读取
│   ├── knowledge_base.py     # 知识库文件管理
│   └── reasoning/
│       ├── __init__.py
│       ├── agent.py          # REACT 子智能体
│       ├── graph.py          # AgentGraph 管理
│       └── reactor.py        # 推理引擎
├── requirements.txt
└── README.md
```
