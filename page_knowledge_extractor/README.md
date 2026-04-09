# Page Knowledge Extractor
**文档知识结构化抽取 + 渐进式披露检索框架**

## 功能特性

### 1. 知识抽取 (extract)
- **输入格式**: 支持 docx、txt，标题格式为标准多级序号（1 → 1.1 → 1.1.1）
- **非标准序号处理**: 自动归入最近父级标题的 content
- **输出结构**: 层级目录 + knowledge.md 文件
- **目录命名**: `序号_标题名称`（如 `1.1.2_农产品免税条例`）
- **knowledge.md 内容**:
  - 当前文件绝对路径
  - 本章节内容（不含子标题内容）
  - 子目录摘要（鼓励渐进式探索）

### 2. 知识推理 (reason)
- **REACT 模式**: 每轮只披露一层 knowledge.md
- **渐进式披露**: 从外到里层层深入
- **多路径并行**: 可同时探索多个子路径
- **智能回溯**: 支持跳级回溯到上游任意目录
- **AgentGraph**: 每个问题衍生多个子智能体，形成有向无环图
- **证据汇总**: 各子智能体推理完成后递归汇总

## 安装

```bash
cd page_knowledge_extractor
pip install -r requirements.txt
```

## 使用方法

### 抽取文档

```bash
# 单文件抽取
python3 -m src.main extract -i ./docs/农产品免税条例.docx -o ./page_knowledge

# 批量抽取目录
python3 -m src.main extract -i ./docs/ -o ./page_knowledge
```

### 知识推理

```bash
# 单问题推理
python3 -m src.main reason -k ./page_knowledge/农产品免税条例_1234567890 -q "哪些农产品可以免税？"

# 批量推理（CSV）
python3 -m src.main reason \
  -k ./page_knowledge/农产品免税条例_1234567890 \
  --questions ./questions.csv \
  --question-col "问题" \
  --output ./results.csv

# 批量推理（XLSX）
python3 -m src.main reason \
  -k ./page_knowledge/农产品免税条例_1234567890 \
  --questions ./questions.xlsx \
  --question-col "问题" \
  --max-rounds 8 \
  --output ./results.csv
```

## 项目结构

```
page_knowledge_extractor/
├── page_knowledge/           # 知识抽取输出目录
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI 入口
│   ├── extractor.py          # 抽取主逻辑
│   ├── parser.py             # 标题结构解析
│   ├── file_reader.py        # 文件读取
│   ├── knowledge_base.py     # 知识库管理
│   └── reasoning/
│       ├── __init__.py
│       ├── agent.py          # REACT 子智能体
│       ├── graph.py          # AgentGraph 管理
│       └── reactor.py        # 推理引擎
├── requirements.txt
└── README.md
```

## 依赖

- python-docx >= 1.1.0
- pandas >= 2.1.0
- openpyxl >= 3.1.0
- anthropic >= 0.25.0

## 环境变量

- `ANTHROPIC_API_KEY`: Anthropic API Key（用于推理功能）

## 注意事项

1. 抽取时请确保文档标题格式符合标准（数字+句点）
2. 推理前需设置 ANTHROPIC_API_KEY 环境变量
3. max-rounds 参数控制每个子智能体的最大推理轮次
