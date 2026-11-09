"""LangGraph multi-agent workflow using the model configured in project .env.

Install: pip install -r langgraph/requirements.txt
Run from the project root: python Multi-Agent/Langgraph/multi_agent.py "研究主题"
"""

import argparse
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class ResearchState(TypedDict, total=False):
    topic: str
    plan: str
    analysis: str
    review: str
    answer: str


def create_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("缺少 OPENAI_API_KEY，请在项目根目录 .env 中配置")

    model = os.getenv("OPENAI_MODEL", "")
    if not model:
        raise ValueError("缺少 OPENAI_MODEL，请在项目根目录 .env 中配置")

    options = {
        "model": model,
        "api_key": api_key,
        "max_tokens": max(512, int(os.getenv("MAX_COMPLETION_TOKENS", "12000"))),
        "temperature": 0.2,
        "streaming": False,
        "max_retries": 1,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        options["base_url"] = base_url
    return ChatOpenAI(**options)


def create_workflow(model: ChatOpenAI):
    def ask_agent(role: str, instruction: str, topic: str, context: str = "") -> str:
        response = model.invoke([
            SystemMessage(content=role),
            HumanMessage(content=f"主题：{topic}\n\n{instruction}\n\n已有材料：\n{context}"),
        ])
        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
        return str(content or "").strip()

    def planner(state: ResearchState) -> dict:
        return {"plan": ask_agent(
            "你是研究规划员。",
            "拆解成 4 至 6 个互不重复的问题和分析维度，简洁列出。",
            state["topic"],
        )}

    def researcher(state: ResearchState) -> dict:
        return {"analysis": ask_agent(
            "你是研究分析员。区分事实、推断和未知，不得编造数据或来源。",
            "依据提纲分析主题，并标明证据不足的部分。",
            state["topic"],
            state["plan"],
        )}

    def reviewer(state: ResearchState) -> dict:
        return {"review": ask_agent(
            "你是独立审稿人。只检查现有材料，不要引入未经支持的新事实。",
            "指出关键遗漏、无依据结论和需要保留的限制，给出具体修正建议。",
            state["topic"],
            f"研究提纲：\n{state['plan']}\n\n初步分析：\n{state['analysis']}",
        )}

    def writer(state: ResearchState) -> dict:
        return {"answer": ask_agent(
            "你是报告撰写员。用清楚、准确的中文组织最终答复。",
            "综合提纲、分析和审稿意见回答主题。保留不确定性，不声称做过未执行的检索或验证。",
            state["topic"],
            f"研究提纲：\n{state['plan']}\n\n分析：\n{state['analysis']}\n\n审稿意见：\n{state['review']}",
        )}

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("reviewer", reviewer)
    graph.add_node("writer", writer)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "reviewer")
    graph.add_edge("reviewer", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def run(topic: str) -> ResearchState:
    if not topic.strip():
        raise ValueError("研究主题不能为空")
    workflow = create_workflow(create_model())
    return workflow.invoke(
        {"topic": topic.strip()},
        config={"recursion_limit": 10},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 多 agent 研究协作示例")
    parser.add_argument("topic", nargs="?", help="需要协作研究的主题")
    args = parser.parse_args()
    topic = args.topic or input("请输入研究主题：").strip()
    if not topic:
        parser.error("研究主题不能为空")
    result = run(topic)
    print(result["answer"])


if __name__ == "__main__":
    main()
