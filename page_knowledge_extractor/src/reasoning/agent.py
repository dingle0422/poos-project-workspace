"""单个 REACT 子智能体：负责在知识树中渐进式探索，回答问题。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import anthropic


class ActionType(str, Enum):
    EXPLORE = "explore"       # 向下探索子目录
    BACKTRACK = "backtrack"   # 回溯到上游目录
    ANSWER = "answer"         # 给出最终答案
    STUCK = "stuck"           # 无法找到相关信息


@dataclass
class ReactStep:
    """一轮 REACT 循环的记录。"""
    round_num: int
    current_path: str
    thought: str
    action: ActionType
    action_detail: dict[str, Any]
    observation: str = ""


@dataclass
class AgentResult:
    """子智能体的最终结果。"""
    agent_id: str
    question: str
    answer: str
    confidence: float
    evidence: list[str]
    steps: list[ReactStep]
    status: str = "success"


@dataclass
class ReactAgent:
    """REACT 推理子智能体。

    每个 agent 从指定目录开始，逐层读取 knowledge.md，
    判断信息相关性和颗粒度是否足够，决定向下探索或回溯。
    """
    agent_id: str
    question: str
    knowledge_root: Path
    start_dir: Path
    max_rounds: int = 5
    model: str = "claude-sonnet-4-20250514"
    client: anthropic.Anthropic | None = None
    steps: list[ReactStep] = field(default_factory=list)

    def __post_init__(self):
        if self.client is None:
            self.client = anthropic.Anthropic()

    async def run(self) -> AgentResult:
        """执行 REACT 推理循环。"""
        current_dir = self.start_dir

        for round_num in range(1, self.max_rounds + 1):
            knowledge_content = self._read_knowledge(current_dir)
            dir_structure = self._get_dir_structure(current_dir)
            ancestry = self._get_ancestry_path(current_dir)

            prompt = self._build_prompt(
                round_num=round_num,
                question=self.question,
                current_dir=str(current_dir),
                ancestry=ancestry,
                knowledge_content=knowledge_content,
                dir_structure=dir_structure,
                history=self.steps,
            )

            response = self._call_llm(prompt)
            decision = self._parse_decision(response)

            step = ReactStep(
                round_num=round_num,
                current_path=str(current_dir),
                thought=decision.get("thought", ""),
                action=ActionType(decision.get("action", "stuck")),
                action_detail=decision,
            )

            if step.action == ActionType.ANSWER:
                step.observation = "推理完成，给出答案。"
                self.steps.append(step)
                return AgentResult(
                    agent_id=self.agent_id,
                    question=self.question,
                    answer=decision.get("answer", ""),
                    confidence=decision.get("confidence", 0.0),
                    evidence=decision.get("evidence", []),
                    steps=self.steps,
                    status="success",
                )

            if step.action == ActionType.EXPLORE:
                target_dirs = decision.get("target_dirs", [])
                if len(target_dirs) == 1:
                    next_dir = current_dir / target_dirs[0]
                    if next_dir.is_dir():
                        step.observation = f"进入子目录: {target_dirs[0]}"
                        current_dir = next_dir
                    else:
                        step.observation = f"子目录不存在: {target_dirs[0]}，停留在当前层。"
                elif len(target_dirs) > 1:
                    # 多路探索：返回特殊结果，由 graph 层处理分叉
                    step.observation = f"需要分叉探索 {len(target_dirs)} 个子目录"
                    self.steps.append(step)
                    return AgentResult(
                        agent_id=self.agent_id,
                        question=self.question,
                        answer="",
                        confidence=0.0,
                        evidence=[],
                        steps=self.steps,
                        status="fork",
                    )
                else:
                    step.observation = "未指定探索目标，停留在当前层。"

            elif step.action == ActionType.BACKTRACK:
                target_path = decision.get("backtrack_to", "")
                if target_path:
                    bt_dir = Path(target_path)
                    if bt_dir.is_dir() and self._is_ancestor(bt_dir, self.knowledge_root):
                        step.observation = f"回溯到: {target_path}"
                        current_dir = bt_dir
                    else:
                        step.observation = f"回溯目标无效: {target_path}，停留在当前层。"
                else:
                    step.observation = "未指定回溯目标。"

            elif step.action == ActionType.STUCK:
                step.observation = "无法继续推理。"
                self.steps.append(step)
                return AgentResult(
                    agent_id=self.agent_id,
                    question=self.question,
                    answer=decision.get("answer", "未找到相关信息。"),
                    confidence=decision.get("confidence", 0.0),
                    evidence=[],
                    steps=self.steps,
                    status="stuck",
                )

            self.steps.append(step)

        return AgentResult(
            agent_id=self.agent_id,
            question=self.question,
            answer="达到最大推理轮次，未能得出确定答案。",
            confidence=0.0,
            evidence=[s.observation for s in self.steps],
            steps=self.steps,
            status="max_rounds_reached",
        )

    def _read_knowledge(self, dir_path: Path) -> str:
        km = dir_path / "knowledge.md"
        if km.exists():
            return km.read_text(encoding="utf-8")
        return "(当前目录无 knowledge.md)"

    def _get_dir_structure(self, dir_path: Path) -> list[str]:
        if not dir_path.is_dir():
            return []
        return sorted(
            d.name for d in dir_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _get_ancestry_path(self, current: Path) -> list[str]:
        """获取从知识库根到当前目录的路径链。"""
        parts: list[str] = []
        p = current
        while p != self.knowledge_root and p != p.parent:
            parts.append(p.name)
            p = p.parent
        parts.append(self.knowledge_root.name)
        parts.reverse()
        return parts

    def _is_ancestor(self, target: Path, root: Path) -> bool:
        try:
            target.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _build_prompt(
        self,
        round_num: int,
        question: str,
        current_dir: str,
        ancestry: list[str],
        knowledge_content: str,
        dir_structure: list[str],
        history: list[ReactStep],
    ) -> str:
        history_text = ""
        if history:
            entries = []
            for s in history:
                entries.append(
                    f"  第{s.round_num}轮: [{s.action.value}] {s.thought} -> {s.observation}"
                )
            history_text = "\n".join(entries)

        subdirs_text = "\n".join(f"  - {d}/" for d in dir_structure) if dir_structure else "  (无子目录)"
        ancestry_text = " > ".join(ancestry)

        return f"""你是一个知识检索推理智能体。你的任务是在层级知识库中渐进式地查找信息来回答问题。

## 当前状态
- 第 {round_num}/{self.max_rounds} 轮
- 问题: {question}
- 当前目录: {current_dir}
- 路径链: {ancestry_text}

## 当前目录的 knowledge.md 内容
{knowledge_content}

## 当前目录的子目录
{subdirs_text}

## 推理历史
{history_text if history_text else "(首轮)"}

## 规则
1. 向下探索只能一层一层进行（从当前目录的直接子目录中选择）
2. 向上回溯可以跳级（直接回到祖先目录的任一级）
3. 如果当前 knowledge.md 的信息已足够回答问题，直接给出答案
4. 如果需要更细粒度的信息，选择最相关的子目录向下探索
5. 如果发现当前分支不相关，可以回溯到上游重新选择方向

## 请以严格 JSON 格式返回你的决策

如果要**向下探索**:
```json
{{
  "thought": "当前知识不够细致，需要查看子目录 X 的详细内容",
  "action": "explore",
  "target_dirs": ["子目录名称1", "子目录名称2"]
}}
```

如果要**回溯**:
```json
{{
  "thought": "当前分支不相关，需要回到上级重新探索",
  "action": "backtrack",
  "backtrack_to": "目标目录的完整路径"
}}
```

如果**可以回答**:
```json
{{
  "thought": "已收集到足够信息",
  "action": "answer",
  "answer": "完整的回答内容",
  "confidence": 0.85,
  "evidence": ["证据1: ...", "证据2: ..."]
}}
```

如果**无法继续**:
```json
{{
  "thought": "在知识库中未找到相关信息",
  "action": "stuck",
  "answer": "说明为什么无法回答",
  "confidence": 0.0
}}
```

请返回纯 JSON，不要包含其他内容。"""

    def _call_llm(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _parse_decision(self, response: str) -> dict[str, Any]:
        text = response.strip()
        # 提取 JSON 块
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "thought": "无法解析 LLM 返回",
                "action": "stuck",
                "answer": f"LLM 返回格式异常: {response[:200]}",
                "confidence": 0.0,
            }
