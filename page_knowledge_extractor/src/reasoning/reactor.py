"""
REACTOR - 主推理引擎
负责加载问题集、调度 AgentGraph、输出结果
"""

import os
import csv
from pathlib import Path
from typing import Optional
import pandas as pd

from .graph import AgentGraph, ParallelAgentGraph


class Reactor:
    """推理引擎主类"""

    def __init__(self, knowledge_dir: str, api_key: Optional[str] = None):
        self.knowledge_dir = knowledge_dir
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("必须提供 API Key，可通过参数或环境变量 ANTHROPIC_API_KEY")
        
        # 验证知识目录
        if not os.path.exists(knowledge_dir):
            raise ValueError(f"知识目录不存在: {knowledge_dir}")

    def reason(self, question: str, max_rounds: int = 5) -> tuple[str, list[str]]:
        """
        对单个问题进行推理
        Returns: (答案, 证据列表)
        """
        # 找到根目录（第一个子文件夹）
        subdirs = sorted([d for d in os.listdir(self.knowledge_dir) 
                         if os.path.isdir(os.path.join(self.knowledge_dir, d))])
        
        if not subdirs:
            return "知识目录为空，无法推理。", []
        
        # 从根目录开始
        root_dir = os.path.join(self.knowledge_dir, subdirs[0])
        
        # 创建并运行 AgentGraph
        graph = AgentGraph(
            knowledge_dir=self.knowledge_dir,
            question=question,
            max_rounds=max_rounds,
            api_key=self.api_key
        )
        
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        answer, evidence = graph.run()
        
        return answer, evidence

    def reason_batch(self, questions: list[str], max_rounds: int = 5,
                     max_workers: int = 4) -> list[dict]:
        """
        批量推理
        Returns: [{question, answer, evidence}, ...]
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n[进度] 处理问题 {i}/{len(questions)}")
            
            try:
                answer, evidence = self.reason(question, max_rounds)
                results.append({
                    "question": question,
                    "answer": answer,
                    "evidence": evidence,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "answer": f"推理失败: {e}",
                    "evidence": [],
                    "status": "failed"
                })
        
        return results

    def reason_from_csv(self, csv_path: str, question_col: str,
                        max_rounds: int = 5, max_workers: int = 4) -> list[dict]:
        """
        从 CSV 文件加载问题集进行推理
        """
        df = pd.read_csv(csv_path)
        
        if question_col not in df.columns:
            raise ValueError(f"CSV 中不存在列 '{question_col}'，可用列: {list(df.columns)}")
        
        questions = df[question_col].tolist()
        results = self.reason_batch(questions, max_rounds, max_workers)
        
        # 添加到 DataFrame
        df['answer'] = [r['answer'] for r in results]
        df['status'] = [r['status'] for r in results]
        
        return results, df

    def reason_from_xlsx(self, xlsx_path: str, question_col: str,
                        max_rounds: int = 5, max_workers: int = 4) -> list[dict]:
        """
        从 XLSX 文件加载问题集进行推理
        """
        df = pd.read_excel(xlsx_path)
        
        if question_col not in df.columns:
            raise ValueError(f"XLSX 中不存在列 '{question_col}'，可用列: {list(df.columns)}")
        
        questions = df[question_col].tolist()
        results = self.reason_batch(questions, max_rounds, max_workers)
        
        # 添加到 DataFrame
        df['answer'] = [r['answer'] for r in results]
        df['status'] = [r['status'] for r in results]
        
        return results, df

    @staticmethod
    def save_results(results: list[dict], output_path: str, 
                     include_evidence: bool = True) -> None:
        """保存结果到 CSV """
        rows = []
        for r in results:
            row = {
                "问题": r["question"],
                "答案": r["answer"],
                "状态": r["status"]
            }
            if include_evidence:
                row["证据"] = " | ".join([e[:200] for e in r.get("evidence", [])])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {output_path}")
