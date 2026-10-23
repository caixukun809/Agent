"""一个最小的命令行 AI Agent：模型选择工具，Python 执行工具。"""

import os
from datetime import datetime

from openai import OpenAI


MODEL = os.getenv("OPENAI_MODEL", "gpt-6-Luna")
MAX_STEPS = 5
SYSTEM_PROMPT = "你是一个简洁、诚实的中文 AI 助手。需要知道当前时间时，调用 get_current_time 工具。"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取运行本程序的电脑当前的本地日期和时间。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


def run_agent(client: OpenAI, messages: list[dict]) -> str:
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        answer = response.choices[0].message

        if not answer.tool_calls:
            text = answer.content or "（模型没有返回文本）"
            messages.append({"role": "assistant", "content": text})
            return text

        messages.append(
            {
                "role": "assistant",
                "content": answer.content,
                "tool_calls": [call.model_dump() for call in answer.tool_calls],
            }
        )
        for call in answer.tool_calls:
            if call.function.name == "get_current_time":
                result = datetime.now().astimezone().isoformat(timespec="seconds")
            else:
                result = f"未知工具：{call.function.name}"

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    text = "已达到本轮最大执行次数，请换一种问法再试。"
    messages.append({"role": "assistant", "content": text})
    return text


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置环境变量 OPENAI_API_KEY。")
        return

    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("AI 助手已启动。输入“退出”结束。")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if user_input in {"退出", "exit", "quit"}:
            print("再见。")
            break
        if not user_input:
            continue

        history_length = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            print("助手：" + run_agent(client, messages))
        except Exception as exc:
            del messages[history_length:]
            print(f"调用失败：{exc}")


if __name__ == "__main__":
    main()
