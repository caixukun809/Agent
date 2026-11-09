"""Run the LangGraph research workflow from the project root."""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agent import format_report, run_research  # noqa: E402


def show_progress(node: str, update: dict, state: dict) -> None:
    if node == "plan":
        print("搜索计划：")
        for query in update.get("search_plan", []):
            print(f"  - {query}")
    elif node == "search":
        queries = update.get("searched_queries", [])
        sources = update.get("sources", [])
        print(f"第 {state.get('round', 1)} 轮搜索完成：累计执行 {len(queries)} 条搜索词，收集 {len(sources)} 个来源。")
    elif node == "critique":
        print(f"第 {state.get('round', 1)} 轮评审：{update.get('critique', '')}")
        for topic in update.get("missing_topics", []):
            print(f"  资料缺口：{topic}")
        if not update.get("sufficient"):
            for query in update.get("pending_queries", []):
                print(f"  补充搜索：{query}")
    elif node == "write":
        print("资料评审通过，开始撰写报告。")


def main() -> None:
    parser = argparse.ArgumentParser(description="用 LangGraph 规划、搜索、评审并撰写调研报告")
    parser.add_argument("question", nargs="*", help="研究问题；省略时进入交互输入")
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    if not question:
        try:
            question = input("请输入研究问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if not question:
        parser.error("研究问题不能为空")

    try:
        print("正在规划搜索并收集资料……")
        result = run_research(question, on_step=show_progress)
        print(format_report(result))
    except Exception as exc:
        print(f"调研失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
