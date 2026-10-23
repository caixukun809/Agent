"""一个最小的命令行 AI Agent：模型选择工具，Python 执行工具。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

from openai import OpenAI, APIConnectionError, RateLimitError


# [优化1] 默认模型改为真实存在的模型，否则不设环境变量一对话就 404
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

MAX_STEPS = 5            # 单轮对话内最多工具调用轮次
MAX_OUTPUT = 4000        # 工具输出截断长度
MAX_HISTORY_ROUNDS = 20  # [优化3] 最多保留的对话轮数
TEMPERATURE = 0.3        # [优化4] 显式温度
MAX_RETRIES = 2          # [优化5] 网络/限流重试次数
WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE", "./workspace")).resolve()
MAX_FILE_SIZE = 100_000  # [新增] 单个文本文件最大写入字节数
# [新增] 流式开关：中转站不支持时可在终端设置 AGENT_STREAM=false
STREAM_OUTPUT = os.getenv("AGENT_STREAM", "true").lower() not in {"0", "false", "no"}

SYSTEM_PROMPT = (
    "你是一个简洁、诚实的中文 AI 助手。"
    "需要知道当前时间时，调用 get_current_time 工具。"
    "需要执行 Python 代码时，调用 run_python 工具。"
    "需要创建新文件时，调用 create_file 工具。"
    "需要覆盖或写入已有文件时，调用 write_file 工具。"
    "工具调用时不要伪造执行结果。"
    "不要在回答中伪造工具执行结果。"
)

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
    },
    # [优化6] 新增 run_python；strict 模式下 code 必须出现在 required 里，否则请求 400
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "在本地子进程中执行一段 Python 代码，返回 stdout/stderr。"
            "仅适用于本地可信环境，不具备沙箱隔离。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 源代码。"}
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    # [新增] 工作空间工具：模型只能通过相对路径操作 workspace 目录
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "在工作空间中创建一个不存在的 UTF-8 文本文件。只能使用相对路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作空间的文件路径"},
                    "content": {"type": "string", "description": "写入文件的完整文本"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    # [新增] 与 create_file 分开，明确表示这是允许覆盖已有内容的操作
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "在工作空间中创建或覆盖一个 UTF-8 文本文件。只能使用相对路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作空间的文件路径"},
                    "content": {"type": "string", "description": "写入文件的完整文本"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def _truncate(text: str) -> str:
    # [优化8] 工具输出统一截断，防止撑爆上下文
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + "\n...(输出已截断)"
    return text


def _workspace_path(relative_path: str) -> Path:
    """[新增] 将相对路径解析到工作空间，并拒绝越界路径。"""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("文件路径不能为空")

    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("只能使用工作空间内的相对路径")

    candidate = (WORKSPACE_ROOT / requested).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("路径不能离开工作空间") from exc
    return candidate


def create_file(relative_path: str, content: str) -> str:
    """[新增] 创建新文件；如果文件已经存在则拒绝覆盖。"""
    if not isinstance(content, str):
        return "错误：content 必须是字符串。"
    if len(content.encode("utf-8")) > MAX_FILE_SIZE:
        return f"错误：文件不能超过 {MAX_FILE_SIZE} 字节。"

    path = _workspace_path(relative_path)
    if path.exists():
        return f"错误：文件已存在，未覆盖：{relative_path}"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return f"已创建文件：{relative_path}（{len(content.encode('utf-8'))} 字节）"


def write_file(relative_path: str, content: str) -> str:
    """[新增] 创建或覆盖工作空间中的 UTF-8 文本文件。"""
    if not isinstance(content, str):
        return "错误：content 必须是字符串。"
    if len(content.encode("utf-8")) > MAX_FILE_SIZE:
        return f"错误：文件不能超过 {MAX_FILE_SIZE} 字节。"

    path = _workspace_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return f"已写入文件：{relative_path}（{len(content.encode('utf-8'))} 字节）"


def run_python(code: str) -> str:
    # [优化7] 清空子进程环境变量（避免把 OPENAI_API_KEY 等泄漏给模型代码）、超时、隔离模式
    if not isinstance(code, str) or not code.strip():
        return "错误：code 为空。"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired:
        return "执行超时（超过 5 秒）。"
    output = (completed.stdout or "") + (completed.stderr or "")
    if not output:
        return f"（无输出，返回码 {completed.returncode}）"
    return _truncate(output)


# [优化2] 工具注册表：定义与执行集中一处，加工具只改这里
TOOL_HANDLERS = {
    "get_current_time": lambda args: datetime.now().astimezone().isoformat(timespec="seconds"),
    "run_python": lambda args: run_python(args.get("code", "")),
    # [新增] 工具 schema 和实际 Python 函数通过这个注册表连接
    "create_file": lambda args: create_file(args.get("path", ""), args.get("content", "")),
    "write_file": lambda args: write_file(args.get("path", ""), args.get("content", "")),
}


def dispatch_tool(name: str, raw_arguments: str) -> str:
    # [优化9] 解析 arguments（字符串形式的 JSON）+ 未知工具兜底 + 执行异常兜底
    print(f"\n[tool:start] {name}")  # [新增] 展示模型请求执行了哪个工具
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        result = f"错误：工具参数不是合法 JSON（{exc}）。"
        print(f"[tool:error] {result}")
        return result

    # [新增] 展示完整工具参数，便于观察模型如何选择工具和构造参数
    print(f"[tool:args] {json.dumps(args, ensure_ascii=False)}")

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        result = f"未知工具：{name}"
        print(f"[tool:error] {result}")
        return result

    try:
        result = _truncate(str(handler(args)))
    except Exception as exc:
        result = f"工具执行出错：{exc}"

    # [新增] 工具结果也显示给开发者；结果同时会回传给模型
    print(f"[tool:result] {result}")
    print(f"[tool:end] {name}")
    return result


def call_model_stream(client: OpenAI, messages: list[dict]) -> dict:
    """[新增] 流式调用模型，同时收集文本和可能的工具调用。"""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        finish_reason = None
        printed_text_header = False  # [新增] 只打印一次模型文本事件标签
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=TEMPERATURE,
                stream=STREAM_OUTPUT,  # [新增] 可用环境变量关闭中转站不支持的流式模式
            )

            # [新增] 兼容只支持普通 Chat Completions 的中转站
            if not STREAM_OUTPUT:
                choice = stream.choices[0]
                answer = choice.message
                text = answer.content or ""
                if text:
                    print(f"[model:text] {text}")
                return {
                    "text": text,
                    "tool_calls": [call.model_dump() for call in (answer.tool_calls or [])],
                    "finish_reason": choice.finish_reason,
                }

            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta

                # [新增] 普通回答逐片显示；这不是隐藏思维链，只是模型实际输出的文本
                if delta.content:
                    if not printed_text_header:
                        print("[model:text] ", end="", flush=True)
                        printed_text_header = True
                    text_parts.append(delta.content)
                    print(delta.content, end="", flush=True)

                # [新增] 工具名和 JSON 参数也会分片返回，这里按 index 拼接
                for partial in delta.tool_calls or []:
                    index = partial.index
                    while len(tool_calls) <= index:
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    current = tool_calls[index]
                    if partial.id:
                        current["id"] = partial.id
                    if partial.type:
                        current["type"] = partial.type
                    if partial.function:
                        if partial.function.name:
                            current["function"]["name"] += partial.function.name
                        if partial.function.arguments:
                            current["function"]["arguments"] += partial.function.arguments

            return {
                "text": "".join(text_parts),
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            }
        except (APIConnectionError, RateLimitError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(
                    f"\n  [重试 {attempt + 1}/{MAX_RETRIES}] "
                    f"{type(exc).__name__}，{wait} 秒后重连…"
                )
                time.sleep(wait)
    print(f"\n  [重试失败] 已重试 {MAX_RETRIES} 次，放弃。")
    raise last_exc


def run_agent(client: OpenAI, messages: list[dict]) -> str:
    for step in range(1, MAX_STEPS + 1):
        print(f"\n\n[agent:step {step}/{MAX_STEPS}] 请求模型")  # [新增] 展示 Agent 调度轮次
        streamed = call_model_stream(client, messages)
        text = streamed["text"]
        tool_calls = streamed["tool_calls"]
        finish_reason = streamed["finish_reason"]

        if text:
            print()  # [新增] 流式输出结束后换行

        # [优化10] 检查 finish_reason，区分长度截断和正常结束
        if finish_reason == "length":
            final_text = "（回答因长度限制被截断，请换个更短的问题）"
            messages.append({"role": "assistant", "content": final_text})
            print(f"[agent:end] {final_text}")
            return final_text

        if not tool_calls:
            final_text = text or "（模型没有返回文本）"
            messages.append({"role": "assistant", "content": final_text})
            print("[agent:end] 本轮没有工具调用")  # [新增] 展示停止原因
            return final_text

        # [新增] 这是可观察的模型输出和执行轨迹，不是模型隐藏思维链
        print(f"[model:tool_calls] 共请求 {len(tool_calls)} 个工具")
        messages.append(
            {
                "role": "assistant",
                "content": text or None,
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            # [新增] 将流式拼接的工具调用转换为原 dispatch_tool 使用的形状
            result = dispatch_tool(
                call["function"]["name"],
                call["function"]["arguments"],
            )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    final_text = "已达到本轮最大执行次数，请换一种问法再试。"
    messages.append({"role": "assistant", "content": final_text})
    print(f"[agent:end] {final_text}")
    return final_text


def trim_history(messages: list[dict], max_rounds: int) -> None:
    # [优化3] 按完整轮次截断：从某个 user 消息处切开，
    # 保证 assistant 的 tool_calls 和对应的 tool 结果不会被拆散
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_indices) <= max_rounds:
        return
    cut = user_indices[len(user_indices) - max_rounds]
    del messages[1:cut]  # 保留 messages[0] 的系统提示词


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置环境变量 OPENAI_API_KEY。")
        return

    # [新增] 启动时创建工作空间目录；工具只能在这里读写
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(
        "AI 助手已启动。输入“退出”结束。"
        f"工作空间：{WORKSPACE_ROOT}"
        "命令：/clear 清空历史，/help 帮助。"
    )

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

        # [优化12] 斜杠命令
        if user_input == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("已清空对话历史。")
            continue
        if user_input == "/help":
            print("命令：/clear 清空历史，/help 帮助，退出 结束。")
            continue

        history_length = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            # [新增] run_agent 自己负责流式打印，避免这里再次拼接导致答案重复
            run_agent(client, messages)
        except Exception as exc:
            # [优化13] 保留 user 消息，只回滚 assistant/tool，再补一条 assistant 错误消息
            del messages[history_length + 1:]
            error_text = f"（调用失败：{exc}）"
            messages.append({"role": "assistant", "content": error_text})
            print(error_text)

        # [优化3] 每轮结束后裁剪历史
        trim_history(messages, MAX_HISTORY_ROUNDS)


if __name__ == "__main__":
    main()
