"""
REACT 子智能体 - 单个知识探索智能体
"""

import os
import json
from typing import Optional, Literal
from dataclasses import dataclass, field
from anthropic import Anthropic


@dataclass
class ReactStep:
    """REACT 推理步骤"""
    thought: str          # 思考过程
    action: str           # 执行的动作 (disclose|reasoning|finish|backtrack)
    target_dir: Optional[str] = None  # 目标目录（disclose/backtrack时）
    evidence: Optional[str] = None    # 证据（finish时）
    answer: Optional[str] = None      # 最终答案（finish时）
    sub_paths: list[str] = field(default_factory=list)  # 并行探索的子路径


@dataclass
class ReactAgent:
    """单个 REACT 推理智能体"""
    
    # 状态
    current_dir: str                    # 当前所在目录
    knowledge_dir: str                  # 知识库根目录
    question: str                       # 当前问题
    max_rounds: int = 5                 # 最大推理轮次
    
    # 内部状态
    _history: list[ReactStep] = field(default_factory=list)
    _client: Optional[Anthropic] = field(default=None, repr=False)
    
    # 回调（由外部设置）
    on_disclose: callable = None        # 披露knowledge.md时的回调
    on_spawn_child: callable = None    # 派生子智能体时的回调
    
    ANTHROPIC_API_KEY = "sk-ant-api03-placeholder"  # 占位，运行时替换
    
    def __post_init__(self):
        self._client = Anthropic(api_key=self.ANthropIC_API_KEY)
    
    def set_api_key(self, api_key: str):
        """设置 API Key"""
        self._client = Anthropic(api_key=api_key)
    
    @property
    def history(self) -> list[ReactStep]:
        return self._history
    
    @property
    def parent_tree(self) -> list[str]:
        """获取父目录树路径"""
        parts = []
        current = self.current_dir
        root = self.knowledge_dir
        while current and current != root:
            parts.insert(0, os.path.basename(current))
            current = os.path.dirname(current)
        return parts
    
    def read_knowledge_md(self, dir_path: str) -> Optional[str]:
        """读取指定目录的 knowledge.md"""
        file_path = os.path.join(dir_path, "knowledge.md")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def get_subdirs(self, dir_path: str) -> list[str]:
        """获取子目录列表"""
        subdirs = []
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        return sorted(subdirs)
    
    def disclose_and_read(self, dir_path: str) -> tuple[Optional[str], list[str]]:
        """披露（读取）指定目录的 knowledge.md 并返回其子目录"""
        content = self.read_knowledge_md(dir_path)
        subdirs = self.get_subdirs(dir_path)
        
        if self.on_disclose:
            self.on_disclose(dir_path, content, subdirs)
        
        return content, subdirs
    
    def build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个专业的知识推理智能体，擅长从文档知识库中抽取证据回答用户问题。

## 核心工作模式：REACT (Reasoning + Acting)

你将逐步推理，每次只披露（读取）一层目录的 knowledge.md 内容，然后判断：
1. 当前层级的知识是否与问题相关？
2. 当前颗粒度是否足够支撑推理？
3. 如果不够，应该探索哪些子目录？

## 向下披露规则
- 每次只向下一层，不能跳跃
- 可以选择多个子路径并行探索
- 每次探索都是独立的 REACT 循环

## 向上回溯规则
- 发现需要补充背景知识时，可以直接回溯到上游任意层级
- 回溯后重新选择向下路径

## 输出格式（JSON）
```json
{
  "thought": "你的推理过程",
  "action": "disclose|parallel_disclose|backtrack|reasoning|finish",
  "target_dir": "目标目录名称（disclose/backtrack时）",
  "sub_paths": ["子路径1", "子路径2"]（parallel_disclose时）,
  "evidence": "支撑问题的证据（finish时）",
  "answer": "最终答案（finish时）"
}
```

## 重要原则
- 鼓励探索，但要有针对性
- 证据要具体，引用原文
- 多路径并行时，分别推理后汇总"""

    def build_user_prompt(self, current_content: Optional[str], subdirs: list[str], 
                         parent_tree: list[str], round_num: int) -> str:
        """构建用户提示"""
        subdirs_info = "\n".join([f"- {d}" for d in subdirs]) if subdirs else "(无子目录)"
        
        prompt = f"""## 当前任务
问题: {self.question}

## 当前状态
- 当前轮次: {round_num}/{self.max_rounds}
- 当前目录: {os.path.basename(self.current_dir)}
- 父目录树: {' -> '.join(parent_tree) if parent_tree else '(根目录)'}"""

        if current_content:
            prompt += f"""
## 当前 knowledge.md 内容
{current_content}"""
        else:
            prompt += """
## 当前 knowledge.md 内容
(未找到内容)"""

        prompt += f"""
## 可探索的子目录
{subdirs_info}

请基于以上信息，输出你的推理结果（JSON格式）。
"""
        return prompt
    
    def run(self) -> ReactStep:
        """
        运行 REACT 推理
        Returns: 最终的 ReactStep（包含答案和证据）
        """
        current_content, subdirs = self.disclose_and_read(self.current_dir)
        
        for round_num in range(1, self.max_rounds + 1):
            # 构建提示
            user_prompt = self.build_user_prompt(
                current_content, subdirs, self.parent_tree, round_num
            )
            
            # 调用 LLM
            response = self._call_llm(user_prompt)
            
            # 解析响应
            step = self._parse_response(response)
            self._history.append(step)
            
            # 执行动作
            if step.action == "finish":
                return step
            
            elif step.action == "disclose":
                if step.target_dir:
                    target_path = os.path.join(self.current_dir, step.target_dir)
                    if os.path.exists(target_path):
                        self.current_dir = target_path
                        current_content, subdirs = self.disclose_and_read(target_path)
                    else:
                        # 子目录不存在，尝试在同一层继续推理
                        step.thought += f"\n[警告] 目标目录不存在: {step.target_dir}"
            
            elif step.action == "parallel_disclose":
                # 返回需要并行探索的子路径，由调用者派生子智能体
                return step
            
            elif step.action == "backtrack":
                if step.target_dir:
                    # 回溯到指定的上游目录
                    target_path = os.path.join(self.knowledge_dir, step.target_dir)
                    if os.path.exists(target_path):
                        self.current_dir = target_path
                        current_content, subdirs = self.disclose_and_read(target_path)
                    else:
                        step.thought += f"\n[警告] 回溯目标不存在: {step.target_dir}"
            
            elif step.action == "reasoning":
                # 仅推理不披露，继续尝试向下
                if not subdirs:
                    # 没有更多子目录可探索，尝试完成
                    step.action = "finish"
                    step.answer = step.thought
                    step.evidence = current_content or ""
                    return step
        
        # 达到最大轮次，强制结束
        final_step = ReactStep(
            thought=f"[达到最大轮次 {self.max_rounds}] " + "\n".join([s.thought for s in self._history]),
            action="finish",
            evidence=current_content or "",
            answer="基于现有知识库内容无法给出完整答案，建议扩大知识库或调整问题。"
        )
        self._history.append(final_step)
        return final_step
    
    def _call_llm(self, user_prompt: str) -> str:
        """调用 LLM"""
        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=self.build_system_prompt(),
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return json.dumps({
                "thought": f"API调用失败: {e}",
                "action": "finish",
                "answer": f"推理过程出错: {e}",
                "evidence": ""
            })
    
    def _parse_response(self, response: str) -> ReactStep:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            return ReactStep(
                thought=data.get("thought", ""),
                action=data.get("action", "reasoning"),
                target_dir=data.get("target_dir"),
                sub_paths=data.get("sub_paths", []),
                evidence=data.get("evidence"),
                answer=data.get("answer")
            )
        except json.JSONDecodeError:
            # 解析失败，当作纯推理处理
            return ReactStep(
                thought=response,
                action="reasoning"
            )
