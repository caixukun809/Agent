"""CrewAI sequential multi-agent example using the project .env model.

Install: pip install "crewai" python-dotenv
Run from the project root: python Multi-Agent/CrewAI/multi_agent.py "研究主题"
"""

import argparse
import os
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def create_model() -> LLM:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("缺少 OPENAI_API_KEY，请在项目根目录 .env 中配置")

    model = os.getenv("OPENAI_MODEL", "")
    if not model:
        raise ValueError("缺少 OPENAI_MODEL，请在项目根目录 .env 中配置")

    options = {
        "model": f"openai/{model}",
        "api_key": api_key,
        "max_tokens": max(512, int(os.getenv("MAX_COMPLETION_TOKENS", "12000"))),
        "temperature": 0.2,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        options["base_url"] = base_url
    return LLM(**options)


def run(topic: str) -> str:
    model = create_model()
    planner = Agent(
        role="研究规划员",
        goal="把主题拆成可回答的问题和分析维度",
        backstory="你擅长建立清楚、完整且不重复的研究框架。",
        llm=model,
        verbose=False,
    )
    researcher = Agent(
        role="研究分析员",
        goal="按研究框架形成有边界的分析",
        backstory="你区分已知事实、推断和不确定内容，不编造数据或来源。",
        llm=model,
        verbose=False,
    )
    reviewer = Agent(
        role="独立审稿人",
        goal="检查分析是否遗漏关键问题或夸大证据",
        backstory="你会提出具体的缺口和修正建议。",
        llm=model,
        verbose=False,
    )
    writer = Agent(
        role="报告撰写员",
        goal="将研究和审稿意见整理成准确、清楚的中文答复",
        backstory="你重视结论、依据和限制之间的一致性。",
        llm=model,
        verbose=False,
    )

    plan_task = Task(
        description="为主题“{topic}”列出 4 至 6 个互不重复的研究问题和分析维度。",
        expected_output="一份简洁的研究提纲。",
        agent=planner,
    )
    research_task = Task(
        description=(
            "依据研究提纲分析主题“{topic}”。明确区分事实、推断与未知；"
            "没有证据支持的细节不要补造。"
        ),
        expected_output="按提纲组织的初步分析，并标出不确定处。",
        agent=researcher,
        context=[plan_task],
    )
    review_task = Task(
        description="审阅前述提纲和分析，指出重要遗漏、无依据结论及应补充的限制。",
        expected_output="简明的审稿意见和可执行的修正建议。",
        agent=reviewer,
        context=[plan_task, research_task],
    )
    write_task = Task(
        description=(
            "综合提纲、研究分析和审稿意见，回答主题“{topic}”。"
            "保留证据限制，不声称做过未执行的检索或验证。"
        ),
        expected_output="结构清楚的最终中文答复。",
        agent=writer,
        context=[plan_task, research_task, review_task],
    )

    crew = Crew(
        agents=[planner, researcher, reviewer, writer],
        tasks=[plan_task, research_task, review_task, write_task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff(inputs={"topic": topic}))


def main() -> None:
    parser = argparse.ArgumentParser(description="CrewAI 多 agent 研究协作示例")
    parser.add_argument("topic", nargs="?", help="需要协作研究的主题")
    args = parser.parse_args()
    topic = args.topic or input("请输入研究主题：").strip()
    if not topic:
        parser.error("研究主题不能为空")
    print(run(topic))


if __name__ == "__main__":
    main()
