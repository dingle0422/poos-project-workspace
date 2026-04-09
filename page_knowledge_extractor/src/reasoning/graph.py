"""
AgentGraph - 智能体图管理器
负责管理多智能体的派生、调度和结果汇总
"""

import os
import asyncio
import concurrent.futures
from typing import Optional, Literal
from dataclasses import dataclass, field
from collections import defaultdict
from .agent import ReactAgent, ReactStep


@dataclass
class AgentNode:
    """智能体节点"""
    agent_id: str
    parent_id: Optional[str]
    question: str
    current_dir: str
    knowledge_dir: str
    max_rounds: int
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: Optional[ReactStep] = None
    children_ids: list[str] = field(default_factory=list)
    evidence: str = ""           # 汇总的证据
    sub_agent_answers: dict[str, str] = field(default_factory=dict)  # 子智能体回答


class AgentGraph:
    """智能体图管理器"""

    def __init__(self, knowledge_dir: str, question: str, max_rounds: int = 5, 
                 api_key: Optional[str] = None):
        self.knowledge_dir = knowledge_dir
        self.question = question
        self.max_rounds = max_rounds
        self.api_key = api_key
        
        # 节点存储
        self.nodes: dict[str, AgentNode] = {}
        self.next_id = 1
        
        # 根节点
        self.root_id = self._create_node(None, knowledge_dir)

    def _create_node(self, parent_id: Optional[str], start_dir: str) -> str:
        """创建新节点"""
        node_id = f"agent_{self.next_id}"
        self.next_id += 1
        
        node = AgentNode(
            agent_id=node_id,
            parent_id=parent_id,
            question=self.question,
            current_dir=start_dir,
            knowledge_dir=self.knowledge_dir,
            max_rounds=self.max_rounds
        )
        
        self.nodes[node_id] = node
        
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        
        return node_id

    def run(self) -> tuple[str, list[str]]:
        """
        运行智能体图
        Returns: (最终答案, 证据列表)
        """
        # 从根节点开始
        root_result = self._run_node(self.root_id)
        
        # 汇总所有证据
        all_evidence = self._collect_evidence(self.root_id)
        
        # 生成最终答案
        final_answer = self._generate_final_answer(root_result, all_evidence)
        
        return final_answer, all_evidence

    def _run_node(self, node_id: str) -> ReactStep:
        """运行单个节点（及其子节点）"""
        node = self.nodes[node_id]
        node.status = "running"
        
        # 创建智能体
        agent = ReactAgent(
            current_dir=node.current_dir,
            knowledge_dir=node.knowledge_dir,
            question=node.question,
            max_rounds=node.max_rounds
        )
        
        if self.api_key:
            agent.set_api_key(self.api_key)
        
        # 设置披露回调
        def on_disclose(dir_path, content, subdirs):
            print(f"  [{node_id}] 披露: {os.path.basename(dir_path)}")
        
        agent.on_disclose = on_disclose
        
        # 运行推理
        result = agent.run()
        node.result = result
        node.status = "done"
        
        # 处理并行探索
        if result.action == "parallel_disclose" and result.sub_paths:
            for sub_path in result.sub_paths:
                target_path = os.path.join(node.current_dir, sub_path)
                if os.path.exists(target_path):
                    child_id = self._create_node(node_id, target_path)
                    child_result = self._run_node(child_id)
                    node.sub_agent_answers[child_id] = child_result.answer or ""
        
        return result

    def _collect_evidence(self, node_id: str) -> list[str]:
        """收集所有子节点的证据"""
        evidence = []
        
        node = self.nodes[node_id]
        if node.result and node.result.evidence:
            evidence.append(node.result.evidence)
        
        for child_id in node.children_ids:
            evidence.extend(self._collect_evidence(child_id))
        
        return evidence

    def _generate_final_answer(self, root_result: ReactStep, 
                               all_evidence: list[str]) -> str:
        """生成最终答案"""
        if root_result.answer:
            return root_result.answer
        
        if all_evidence:
            evidence_text = "\n\n".join([f"证据{i+1}: {e[:500]}..." if len(e) > 500 else f"证据{i+1}: {e}" 
                                        for i, e in enumerate(all_evidence)])
            return f"基于知识库分析：\n\n{evidence_text}\n\n结论：需要进一步调查以给出准确答案。"
        
        return "知识库中未找到相关信息。"


class ParallelAgentGraph(AgentGraph):
    """并行版本的 AgentGraph"""

    def __init__(self, knowledge_dir: str, question: str, max_rounds: int = 5,
                 api_key: Optional[str] = None, max_workers: int = 4):
        super().__init__(knowledge_dir, question, max_rounds, api_key)
        self.max_workers = max_workers

    def _run_node(self, node_id: str) -> ReactStep:
        """并行运行节点"""
        node = self.nodes[node_id]
        node.status = "running"
        
        # 创建智能体
        agent = ReactAgent(
            current_dir=node.current_dir,
            knowledge_dir=node.knowledge_dir,
            question=node.question,
            max_rounds=node.max_rounds
        )
        
        if self.api_key:
            agent.set_api_key(self.api_key)
        
        def on_disclose(dir_path, content, subdirs):
            print(f"  [{node_id}] 披露: {os.path.basename(dir_path)}")
        
        agent.on_disclose = on_disclose
        
        # 运行推理
        result = agent.run()
        node.result = result
        node.status = "done"
        
        # 处理并行探索
        if result.action == "parallel_disclose" and result.sub_paths:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for sub_path in result.sub_paths:
                    target_path = os.path.join(node.current_dir, sub_path)
                    if os.path.exists(target_path):
                        child_id = self._create_node(node_id, target_path)
                        future = executor.submit(self._run_node, child_id)
                        futures[future] = child_id
                
                for future in concurrent.futures.as_completed(futures):
                    child_id = futures[future]
                    try:
                        child_result = future.result()
                        node.sub_agent_answers[child_id] = child_result.answer or ""
                    except Exception as e:
                        print(f"子智能体 {child_id} 执行失败: {e}")
        
        return result
