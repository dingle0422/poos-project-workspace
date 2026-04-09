"""推理引擎：对接 AgentGraph，支持单问题和批量问题推理。"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from .graph import AgentGraph


async def reason_single(
    question: str,
    knowledge_dir: str,
    max_rounds: int = 5,
    max_concurrent: int = 5,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """对单个问题执行推理。"""
    kdir = Path(knowledge_dir)
    if not kdir.is_dir():
        raise FileNotFoundError(f"知识目录不存在: {knowledge_dir}")

    client = anthropic.Anthropic()

    print(f"🤔 问题: {question}")
    print(f"📚 知识库: {kdir}")
    print(f"⚙️  最大轮次: {max_rounds}, 并发上限: {max_concurrent}")
    print()

    graph = AgentGraph(
        question=question,
        knowledge_root=kdir,
        max_rounds=max_rounds,
        max_concurrent=max_concurrent,
        model=model,
        client=client,
    )

    result = await graph.execute()

    _print_result(result)
    return result


async def reason_batch(
    questions_file: str,
    question_col: str,
    knowledge_dir: str,
    output_file: str | None = None,
    max_rounds: int = 5,
    max_concurrent: int = 5,
    model: str = "claude-sonnet-4-20250514",
) -> list[dict[str, Any]]:
    """批量处理问题文件。"""
    qpath = Path(questions_file)
    if not qpath.exists():
        raise FileNotFoundError(f"问题文件不存在: {questions_file}")

    suffix = qpath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(qpath)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(qpath)
    else:
        raise ValueError(f"不支持的问题文件格式: {suffix}，仅支持 .csv / .xlsx")

    if question_col not in df.columns:
        raise ValueError(
            f"列 '{question_col}' 不存在。可用列: {list(df.columns)}"
        )

    questions = df[question_col].dropna().tolist()
    print(f"📋 共 {len(questions)} 个问题待处理\n")

    results: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        print(f"{'='*60}")
        print(f"📌 [{i}/{len(questions)}]")
        try:
            r = await reason_single(
                question=str(q),
                knowledge_dir=knowledge_dir,
                max_rounds=max_rounds,
                max_concurrent=max_concurrent,
                model=model,
            )
            results.append(r)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            results.append({
                "question": str(q),
                "answer": f"处理失败: {e}",
                "confidence": 0.0,
                "evidence": [],
                "total_agents": 0,
                "total_steps": 0,
                "error": str(e),
            })
        print()

    if output_file:
        _save_results(results, output_file, df, question_col)

    return results


def _print_result(result: dict[str, Any]) -> None:
    print(f"{'─'*50}")
    print(f"💡 答案: {result.get('answer', '未知')}")
    print(f"🎯 置信度: {result.get('confidence', 0):.0%}")
    print(f"🤖 使用智能体: {result.get('total_agents', 0)} 个")
    print(f"🔄 总推理步数: {result.get('total_steps', 0)}")

    evidence = result.get("evidence", [])
    if evidence:
        print(f"📎 证据:")
        for e in evidence[:5]:
            print(f"   • {e}")
    print(f"{'─'*50}")


def _save_results(
    results: list[dict[str, Any]],
    output_file: str,
    original_df: pd.DataFrame,
    question_col: str,
) -> None:
    out_path = Path(output_file)
    suffix = out_path.suffix.lower()

    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append({
            "问题": r.get("question", ""),
            "答案": r.get("answer", ""),
            "置信度": r.get("confidence", 0.0),
            "证据": json.dumps(r.get("evidence", []), ensure_ascii=False),
            "智能体数": r.get("total_agents", 0),
            "推理步数": r.get("total_steps", 0),
        })

    out_df = pd.DataFrame(rows)

    if suffix == ".csv":
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    elif suffix in (".xlsx", ".xls"):
        out_df.to_excel(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n📊 结果已保存到: {out_path}")
