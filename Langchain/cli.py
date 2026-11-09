"""[新增-LangChain] 便于和原 main.py 对照的命令行入口。"""

from Langchain.agent import LangChainAgent, Settings


def main() -> None:
    agent = LangChainAgent(Settings.from_env())
    session_id = agent.default_session_id
    print(f"LangChain Agent 已启动。session：{session_id}。输入 退出 结束，/clear 新建会话。")
    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in {"退出", "exit", "quit"}:
            break
        if text == "/clear":
            session_id = agent.new_session()
            print(f"新 session：{session_id}")
            continue
        if not text:
            continue

        def show(event: dict) -> None:
            if event["type"] == "tool_search":
                print(f"[lc:tools] {', '.join(event['available'])}")
            elif event["type"] == "model_tool_calls":
                print(f"[lc:model] 请求工具：{[call['name'] for call in event['calls']]}")
            elif event["type"] == "tool_result":
                print(f"[lc:tool] {event['name']} -> {event['result']}")
            elif event["type"] == "answer":
                print(f"Agent：{event['text']}")

        agent.chat(session_id, text, show)


if __name__ == "__main__":
    main()
