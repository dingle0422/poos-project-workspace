"""AgentGraph：管理 REACT 子智能体的有向无环图结构。

当某个子智能体发现需要同时探索多个子目录时，会分叉出多个子智能体，
形成树/DAG 结构。最终各叶节点结果递归合并回根节点。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from .agent import AgentResult, ReactAgent


@dataclass
class GraphNode:
    """图中的一个节点，对应一个子智能体。"""
    agent_id: str
    parent_id: str | None
    explore_dir: Path
    result: AgentResult | None = None
    children: list[GraphNode] = field(default_factory=list)


@dataclass
class AgentGraph:
    """管理智能体推理图的主控结构。"""
    question: str
    knowledge_root: Path
    max_rounds: int = 5
    max_concurrent: int = 5
    model: str = "claude-sonnet-4-20250514"
    client: anthropic.Anthropic | None = None
    root_node: GraphNode | None = None
    all_nodes: dict[str, GraphNode] = field(default_factory=dict)
    _semaphore: asyncio.Semaphore | None = None

    def __post_init__(self):
        if self.client is None:
            self.client = anthropic.Anthropic()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def execute(self) -> dict[str, Any]:
        """执行整个推理图。"""
        root_id = self._gen_id()
        self.root_node = GraphNode(
            agent_id=root_id,
            parent_id=None,
            explore_dir=self.knowledge_root,
        )
        self.all_nodes[root_id] = self.root_node

        await self._run_node(self.root_node)

        return self._synthesize()

    async def _run_node(self, node: GraphNode) -> None:
        async with self._semaphore:
            agent = ReactAgent(
                agent_id=node.agent_id,
                question=self.question,
                knowledge_root=self.knowledge_root,
                start_dir=node.explore_dir,
                max_rounds=self.max_rounds,
                model=self.model,
                client=self.client,
            )
            result = await agent.run()
            node.result = result

        if result.status == "fork":
            last_step = result.steps[-1] if result.steps else None
            if last_step and last_step.action_detail.get("target_dirs"):
                target_dirs = last_step.action_detail["target_dirs"]
                current_dir = Path(last_step.current_path)

                child_tasks = []
                for subdir_name in target_dirs:
                    child_dir = current_dir / subdir_name
                    if not child_dir.is_dir():
                        continue

                    child_id = self._gen_id()
                    child_node = GraphNode(
                        agent_id=child_id,
                        parent_id=node.agent_id,
                        explore_dir=child_dir,
                    )
                    node.children.append(child_node)
                    self.all_nodes[child_id] = child_node
                    child_tasks.append(self._run_node(child_node))

                if child_tasks:
                    await asyncio.gather(*child_tasks)

    def _synthesize(self) -> dict[str, Any]:
        """递归收集所有节点结果，合成最终答案。"""
        all_evidence: list[str] = []
        all_answers: list[dict[str, Any]] = []
        total_steps = 0

        self._collect_results(self.root_node, all_evidence, all_answers)
        for node in self.all_nodes.values():
            if node.result:
                total_steps += len(node.result.steps)

        successful = [a for a in all_answers if a["status"] in ("success",)]
        best_answer = ""
        best_confidence = 0.0

        if successful:
            best = max(successful, key=lambda x: x["confidence"])
            best_answer = best["answer"]
            best_confidence = best["confidence"]
        elif all_answers:
            best = max(all_answers, key=lambda x: x["confidence"])
            best_answer = best["answer"]
            best_confidence = best["confidence"]

        return {
            "question": self.question,
            "answer": best_answer,
            "confidence": best_confidence,
            "evidence": all_evidence,
            "total_agents": len(self.all_nodes),
            "total_steps": total_steps,
            "all_answers": all_answers,
        }

    def _collect_results(
        self,
        node: GraphNode | None,
        evidence: list[str],
        answers: list[dict[str, Any]],
    ) -> None:
        if node is None:
            return
        if node.result and node.result.status != "fork":
            answers.append({
                "agent_id": node.agent_id,
                "answer": node.result.answer,
                "confidence": node.result.confidence,
                "evidence": node.result.evidence,
                "status": node.result.status,
            })
            evidence.extend(node.result.evidence)
        for child in node.children:
            self._collect_results(child, evidence, answers)

    @staticmethod
    def _gen_id() -> str:
        return f"agent-{uuid.uuid4().hex[:8]}"
