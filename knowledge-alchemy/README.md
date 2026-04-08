# 🧪 QA 知识炼金系统

> 将非结构化「问答对」转化为「教科书式知识体系」的 AI 流水线

## 核心理念

**AI 在其中的角色是「高级矿工」和「初级架构师」。**

不是拿着笔在白纸上写教科书，而是像**雕刻家**一样，面对 AI 帮你收集并粗加工好的一块巨大"逻辑大理石"，进行最后的精雕细琢。

---

## 四阶段流水线

```
原始问答对
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1: Deconstruction               │
│  去噪与结构化标签                         │
│  → 提取五要素 / 原子化拆解 / 语义聚类打标   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 2: Abstraction                  │
│  逻辑抽象与模式识别                       │
│  → 逻辑归纳 / IF-THEN转化 / 矛盾检测       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 3: Synthesis                     │
│  知识补全与系统合成                        │
│  → 生成大纲 / 知识映射 / 空白识别 / RAG补全  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 4: Layering                      │
│  层级化编排与润色                          │
│  → 语体转换 / 三层撰写 / 交叉引用生成        │
└────────────────┬────────────────────────┘
                 ▼
           教科书输出
```

---

## 目录结构

```
knowledge-alchemy/
├── agents/                       # 四个核心Agent
│   ├── deconstruction_agent.py   # Stage 1: 去噪与结构化
│   ├── abstraction_agent.py       # Stage 2: 逻辑抽象
│   ├── synthesis_agent.py         # Stage 3: 知识合成
│   └── editor_agent.py            # Stage 4: 层级编排
├── pipelines/
│   └── knowledge_alchemy_pipeline.py  # 流水线编排
├── prompts/                       # 各阶段Prompt模板
│   ├── deconstruction_prompts.py
│   ├── abstraction_prompts.py
│   ├── synthesis_prompts.py
│   └── editor_prompts.py
├── schemas/                      # 数据结构定义
│   └── knowledge_unit.py
├── utils/                        # 工具函数
│   ├── config.py                 # LLM配置管理
│   ├── embedder.py               # 向量嵌入与聚类
│   └── llm_client.py             # 统一LLM调用客户端
├── examples/
│   └── sample_qa.py              # 示例问答数据
├── main.py                       # 命令行入口
└── requirements.txt              # Python依赖
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 选择一个即可
export OPENAI_API_KEY="sk-..."        # OpenAI GPT-4
export ANTHROPIC_API_KEY="sk-ant-..." # Anthropic Claude
```

### 3. 运行

```bash
# 使用内置示例运行
python main.py

# 指定领域和模型
python main.py --domain 金融 --model anthropic

# 指定输入文件
python main.py --input my_qa.json --output ./output

# 在指定阶段停止（用于调试）
python main.py --stage 2
```

---

## 各阶段详解

### Stage 1: Deconstruction（去噪与结构化）

将非结构化问答对转化为带元数据的"知识原子"：

| 提取要素 | 说明 |
|---------|------|
| Scenario | 业务场景（这件事发生在什么背景下）|
| Pain Point | 核心痛点（遇到了什么麻烦）|
| Trigger | 触发条件（什么情况引发这个问题）|
| Actors | 涉及主体（谁参与了这个场景）|
| Authority | 合规依据（法规/制度/行业标准）|

### Stage 2: Abstraction（逻辑抽象）

从个案上升到方法论：

- **逻辑归纳**：同标签下的多个案例 → 通用逻辑骨架
- **IF-THEN 转化**：具体回复 → 逻辑判断树
- **矛盾检测**：扫描潜在逻辑矛盾，提请人工裁决

### Stage 3: Synthesis（知识合成）

以"理想大纲"为参照，逆向补充缺失：

1. **Top-down 大纲生成**：根据行业标准生成教科书目录
2. **知识映射**：将碎片化知识嵌入大纲
3. **空白识别**：对比大纲与已有素材，识别缺失点
4. **RAG 补全**：预留接口，可接入外部知识库

### Stage 4: Layering（层级编排）

调整语体风格，建立分层索引：

| 层级 | 内容 | 说明 |
|------|------|------|
| L1 | 导论 | 核心原理和基础概念 |
| L2 | 标准流程 | 标准业务 SOP |
| L3 | 疑难穿插 | 典型问答对作为案例模块 |

---

## 评价指标

```python
pipeline = KnowledgeAlchemyPipeline(config)
result = pipeline.run(qa_pairs)
metrics = pipeline.evaluate(result)

print(f"覆盖率: {metrics['coverage']:.1%}")     # 原始问答被吸纳的比例
print(f"冗余度: {metrics['redundancy']:.1%}")   # 删除了重复内容后的质量
print(f"一致性: {metrics['consistency']:.1%}")   # 逻辑自洽程度
```

---

## 人机协作

每个阶段都预留了 `human_review_hook` 接口：

```python
# 第二阶段后专家审核逻辑骨架
if not pipeline.human_review_hook("abstraction", data):
    print("等待专家审核...")
    # 暂停，等待人工介入
```

---

## License

MIT
