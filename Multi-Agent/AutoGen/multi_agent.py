"""AutoGen group-chat example using the model configured in the project .env.

Install: pip install "autogen-agentchat" "autogen-ext[openai]" python-dotenv
Run from the project root: python Multi-Agent/AutoGen/multi_agent.py "研究主题"
"""

import argparse
import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def create_model_client() -> OpenAIChatCompletionClient:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("缺少 OPENAI_API_KEY，请在项目根目录 .env 中配置")

    model = os.getenv("OPENAI_MODEL", "")
    if not model:
        raise ValueError("缺少 OPENAI_MODEL，请在项目根目录 .env 中配置")

    client_options = {
        "model": model,
        "api_key": api_key,
        "max_tokens": max(512, int(os.getenv("MAX_COMPLETION_TOKENS", "12000"))),
        # The .env model can be a custom name unknown to AutoGen's model catalog.
        "model_info": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
        },
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_options["base_url"] = base_url
    return OpenAIChatCompletionClient(**client_options)


async def run(topic: str) -> None:
    model_client = create_model_client()
    agents = [
        AssistantAgent(
            "planner",
            model_client=model_client,
            description="拆解研究任务并列出需要回答的问题。",
            system_message="你是研究规划员。为主题拆解清晰的问题和分析维度，简洁输出。",
        ),
        AssistantAgent(
            "researcher",
            model_client=model_client,
            description="根据现有讨论整理分析和证据边界。",
            system_message=(
                "你是研究员。基于对话中的主题和计划给出有条理的分析；"
                "无法核实的事实要标明不确定，不要编造来源或数据。"
            ),
        ),
        AssistantAgent(
            "reviewer",
            model_client=model_client,
            description="检查分析中的遗漏、推断和不确定性。",
            system_message=(
                "你是独立审稿人。指出关键遗漏、证据不足和可能的过度推断，"
                "并给出具体修正建议。"
            ),
        ),
        AssistantAgent(
            "writer",
            model_client=model_client,
            description="综合讨论并给出最终答复。",
            system_message=(
                "你是报告撰写员。结合规划、分析和审稿意见，给出结构清楚的中文结论；"
                "保留重要限制，不要声称未执行的检索或验证。"
            ),
        ),
    ]
    team = RoundRobinGroupChat(agents, termination_condition=MaxMessageTermination(4))
    try:
        result = await team.run(task=f"请多个 agent 协作完成以下主题：{topic}")
        for message in result.messages:
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                print(f"\n[{getattr(message, 'source', 'agent')}]\n{content.strip()}")
    finally:
        await model_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGen 多 agent 研究协作示例")
    parser.add_argument("topic", nargs="?", help="需要协作研究的主题")
    args = parser.parse_args()
    topic = args.topic or input("请输入研究主题：").strip()
    if not topic:
        parser.error("研究主题不能为空")
    asyncio.run(run(topic))


if __name__ == "__main__":
    main()
