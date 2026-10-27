"""一个最小的命令行 AI Agent：模型选择工具，Python 执行工具。"""

import os
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError

from memory import MemoryStore
from skills import SkillCatalog
from tool_index import ToolIndex
from tools.registry import ToolRegistry
from tools.runtime import ToolRuntime
from vector_store import HashEmbedding, LocalEmbedding, OpenAIEmbedding, VectorStore


# [新增-配置] 启动时自动加载项目根目录 .env，终端不再需要重复设置变量。
load_dotenv(Path(__file__).resolve().with_name(".env"))


# [优化1] 默认模型改为真实存在的模型，否则不设环境变量一对话就 404
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

MAX_STEPS = 5            # 单轮对话内最多工具调用轮次
MAX_OUTPUT = 4000        # 工具输出截断长度
TEMPERATURE = 0.3        # [优化4] 显式温度
MAX_RETRIES = 2          # [优化5] 网络/限流重试次数
WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE", "./workspace")).resolve()
MAX_FILE_SIZE = 100_000  # [新增] 单个文本文件最大写入字节数
# [新增] 流式开关：中转站不支持时可在终端设置 AGENT_STREAM=false
STREAM_OUTPUT = os.getenv("AGENT_STREAM", "true").lower() not in {"0", "false", "no"}
# [新增-记忆] 短期上下文只读取当前 session；达到阈值才压缩旧原文。
SHORT_TERM_ROUNDS = int(os.getenv("SHORT_TERM_ROUNDS", "8"))
SUMMARY_TRIGGER_ROUNDS = int(os.getenv("SUMMARY_TRIGGER_ROUNDS", "12"))
SUMMARY_KEEP_ROUNDS = max(
    1,
    min(int(os.getenv("SUMMARY_KEEP_ROUNDS", "6")), SUMMARY_TRIGGER_ROUNDS - 1),
)
MEMORY_DB = Path(os.getenv("AGENT_MEMORY_DB", str(WORKSPACE_ROOT / "agent_memory.sqlite3")))
OWNER_ID = os.getenv("AGENT_OWNER_ID", "local-user")
VECTOR_EMBEDDING_MODE = os.getenv("VECTOR_EMBEDDING_MODE", "local").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    str(Path(__file__).resolve().parent / "models" / "bge-small-zh-v1.5"),
)
# [新增-工具索引] 当前工具少于默认上限时全部提供；工具增多后再检索筛选。
TOOL_SEARCH_LIMIT = max(1, int(os.getenv("TOOL_SEARCH_LIMIT", "8")))

SYSTEM_PROMPT = (
    "你是一个简洁、诚实的中文 AI 助手。"
    "需要当前时间、执行代码或操作文件时，使用本轮提供的相关工具。"
    "工具调用时不要伪造执行结果。"
    "不要在回答中伪造工具执行结果。"
)

def _truncate(text: str) -> str:
    # [优化8] 工具输出统一截断，防止撑爆上下文
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + "\n...(输出已截断)"
    return text


def call_model_stream(client: OpenAI, messages: list[dict], tools: list[dict]) -> dict:
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
                tools=tools,
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


def summarize_session(client: OpenAI, payload: dict) -> str:
    """[新增-记忆] 用模型压缩旧原文；原始消息仍保存在 SQLite 中。"""
    existing = payload.get("existing_summary") or "（暂无历史摘要）"
    transcript = _truncate(payload["transcript"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是会话摘要器。只总结事实、用户目标、已完成事项、未完成事项、"
                    "关键决定和约束。不要添加原文没有的信息，不要输出推理过程。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"已有摘要：\n{existing}\n\n"
                    f"需要归档的原始消息：\n{transcript}\n\n"
                    "请输出一份简洁的中文增量摘要。"
                ),
            },
        ],
        temperature=0.1,
    )
    return (response.choices[0].message.content or "").strip()


def finish_memory_lifecycle(
    client: OpenAI,
    memory: MemoryStore,
    session_id: str,
    user_message_id: int,
    user_text: str,
) -> None:
    """[新增-记忆] 一轮结束后的完整生命周期入口。"""
    print(f"[memory:lifecycle:start] session={session_id!r} user_message_id={user_message_id}")
    kind = memory.capture_explicit_memory(session_id, user_message_id, user_text)
    if kind:
        print(f"[memory:write] 已接受一条长期记忆，类型：{kind}")

    payload = memory.compaction_payload(
        session_id,
        trigger_rounds=SUMMARY_TRIGGER_ROUNDS,
        keep_rounds=SUMMARY_KEEP_ROUNDS,
    )
    if not payload:
        print(f"[memory:lifecycle:end] session={session_id!r} summary_triggered=False")
        return

    print("[memory:summary] 达到阈值，开始总结旧 session 原文")
    try:
        summary = summarize_session(client, payload)
        if summary:
            memory.save_summary_and_archive(
                session_id,
                payload["cutoff_message_id"],
                summary,
            )
            print("[memory:summary] 摘要已保存，旧原文已归档")
        else:
            print("[memory:summary] 模型没有返回摘要，原文保持未归档")
    except Exception as exc:
        # [新增-记忆] 摘要失败不影响当前回答，也不归档原文，下一轮继续重试。
        print(f"[memory:summary:error] {exc}")
    print(f"[memory:lifecycle:end] session={session_id!r} summary_triggered=True")


def run_agent(
    client: OpenAI,
    messages: list[dict],
    user_query: str,
    tool_index: ToolIndex | None,
    registry: ToolRegistry,
) -> str:
    last_tool_result = ""
    for step in range(1, MAX_STEPS + 1):
        # [新增-工具目录] 当前任务和上一步结果决定本步提供的工具 schema。
        query = (user_query + "\n" + last_tool_result)[:1500]
        available_tools = registry.select(query, tool_index, TOOL_SEARCH_LIMIT)
        print(f"\n\n[agent:step {step}/{MAX_STEPS}] 请求模型")  # [新增] 展示 Agent 调度轮次
        streamed = call_model_stream(client, messages, available_tools)
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
            # [新增-工具目录] 将模型请求交给独立注册表校验并分发。
            result = registry.dispatch(
                call["function"]["name"],
                call["function"]["arguments"],
                available_tools,
            )
            last_tool_result = result
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    final_text = "已达到本轮最大执行次数，请换一种问法再试。"
    messages.append({"role": "assistant", "content": final_text})
    print(f"[agent:end] {final_text}")
    return final_text


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置环境变量 OPENAI_API_KEY。")
        return

    # [新增] 启动时创建工作空间目录；工具只能在这里读写
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    # [新增-向量] 默认加载本地语义模型；hash/openai 仍作为备用模式。
    if VECTOR_EMBEDDING_MODE == "openai":
        embedder = OpenAIEmbedding(client, EMBEDDING_MODEL)
    elif VECTOR_EMBEDDING_MODE == "local":
        model_path = Path(LOCAL_EMBEDDING_MODEL)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        embedder = LocalEmbedding(model_path)
    else:
        embedder = HashEmbedding()
    vector_store = VectorStore(WORKSPACE_ROOT, embedder)
    skill_catalog = SkillCatalog(Path(__file__).resolve().parent / "skill")
    # [新增-工具目录] 清单和处理函数独立于主循环，启动时核对名称。
    runtime = ToolRuntime(WORKSPACE_ROOT, skill_catalog, MAX_FILE_SIZE, MAX_OUTPUT)
    registry = ToolRegistry(Path(__file__).resolve().parent / "tools" / "catalog.json", runtime)
    # [新增-工具索引] 工具向量与长期记忆向量各自存盘、使用同一 embedding。
    embedding_id = f"{VECTOR_EMBEDDING_MODE}:{LOCAL_EMBEDDING_MODEL if VECTOR_EMBEDDING_MODE == 'local' else EMBEDDING_MODEL}"
    try:
        tool_index = ToolIndex(WORKSPACE_ROOT, embedder, registry.entries, embedding_id)
    except Exception as exc:
        tool_index = None
        print(f"[tool:index:error] {exc}；暂时提供全部工具")
    # [新增-记忆] 每个窗口默认生成独立 session；设置 AGENT_SESSION_ID 才会恢复旧会话。
    memory = MemoryStore(
        MEMORY_DB,
        OWNER_ID,
        os.getenv("AGENT_SESSION_ID") or None,
        vector_store=vector_store,
    )
    session_id = memory.session_id
    print(
        "AI 助手已启动。输入“退出”结束。"
        f"工作空间：{WORKSPACE_ROOT}，session：{session_id}。"
        f" 向量：{VECTOR_EMBEDDING_MODE}。"
        " 命令：/clear、/session、/reindex、/reindex-tools、/skills、/help。"
    )
    if vector_store.load_error:
        print(f"[memory:vector:error] {vector_store.load_error}")

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
            session_id = memory.create_session()
            print(f"已创建新 session：{session_id}。旧 session 原文仍保留。")
            continue
        if user_input == "/session":
            print(f"[memory:session] {memory.stats(session_id)}")
            if vector_store.load_error:
                print(f"[memory:vector:error] {vector_store.load_error}")
            if memory.last_vector_error:
                print(f"[memory:vector:error] {memory.last_vector_error}")
            continue
        if user_input == "/reindex":
            try:
                count = memory.rebuild_vectors()
                print(f"[memory:vector] 已重建 {count} 条长期记忆向量")
            except Exception as exc:
                print(f"[memory:vector:error] {exc}")
            continue
        if user_input == "/reindex-tools":
            try:
                if tool_index is None:
                    tool_index = ToolIndex(WORKSPACE_ROOT, embedder, registry.entries, embedding_id)
                else:
                    tool_index.rebuild()
            except Exception as exc:
                tool_index = None
                print(f"[tool:index:error] {exc}")
            continue
        if user_input == "/skills":
            print(skill_catalog.descriptions() or "没有可用的 skill")
            continue
        if user_input == "/help":
            print("命令：/clear 新建 session，/session 查看状态，/reindex 重建记忆向量，/reindex-tools 重建工具向量，/skills 查看 skill，/help 帮助，退出 结束。")
            continue

        # [新增-记忆] 每轮按当前 query 重新构建上下文：摘要、长期记忆和当前 session 的短期原文。
        messages = memory.build_context_messages(
            SYSTEM_PROMPT + (
                "\n可用 skill（相关任务先调用 read_skill）：\n" + skill_catalog.descriptions()
                if skill_catalog.entries else ""
            ),
            user_input,
            max_rounds=SHORT_TERM_ROUNDS,
        )
        turn_start = len(messages)
        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        user_message_id = memory.save_message(session_id, user_message)
        try:
            # [新增] run_agent 自己负责流式打印，避免这里再次拼接导致答案重复
            run_agent(client, messages, user_input, tool_index, registry)
        except Exception as exc:
            # [优化13] 保留当前用户消息，并把错误作为本轮可追踪结果保存。
            error_text = f"（调用失败：{exc}）"
            messages.append({"role": "assistant", "content": error_text})
            print(error_text)

        # [新增-记忆] 保存本轮 assistant/tool 事件，形成不可变原始事件记录。
        for message in messages[turn_start + 1:]:
            memory.save_message(session_id, message)

        # [新增-记忆] 执行“候选记忆 -> 去重更新 -> 摘要归档”的生命周期。
        finish_memory_lifecycle(
            client,
            memory,
            session_id,
            user_message_id,
            user_input,
        )


if __name__ == "__main__":
    main()
