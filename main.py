"""一个最小的命令行 AI Agent：模型选择工具，Python 执行工具。"""

import json
import os
import copy
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

try:
    import tiktoken
except ImportError:  # [新增-上下文预算] 依赖缺失时使用保守字符回退。
    tiktoken = None


# [新增-配置] 启动时自动加载项目根目录 .env，终端不再需要重复设置变量。
load_dotenv(Path(__file__).resolve().with_name(".env"))


# [优化1] 默认模型改为真实存在的模型，否则不设环境变量一对话就 404
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

MAX_STEPS = 5            # 单轮对话内最多工具调用轮次
MAX_OUTPUT = 4000        # 工具输出截断长度
TOOL_OUTPUT_TOKEN_BUDGET = int(os.getenv("TOOL_OUTPUT_TOKEN_BUDGET", "1800"))
CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "8000"))
SUMMARY_TOKEN_BUDGET = int(os.getenv("SUMMARY_TOKEN_BUDGET", "3000"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "12000"))
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
MEMORY_CONFIRM_MODE = os.getenv("MEMORY_CONFIRM_MODE", "always").lower()
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

_TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base") if tiktoken else None


def _token_count(text: str) -> int:
    if _TOKEN_ENCODER:
        return len(_TOKEN_ENCODER.encode(text or ""))
    return max(1, len(text or "") // 2)


# [新增-上下文预算] 所有送入模型的长文本按 token 截断，避免中文字符数与 token 数失真。
def _truncate_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    if _TOKEN_ENCODER:
        tokens = _TOKEN_ENCODER.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _TOKEN_ENCODER.decode(tokens[:max_tokens]) + "\n...(token 预算已截断)"
    if len(text) <= max_tokens * 2:
        return text
    return text[: max_tokens * 2] + "\n...(token 预算已截断)"


def _truncate(text: str) -> str:
    # [新增-上下文预算] 保留旧调用接口，但由 token 预算控制工具输出。
    return _truncate_tokens(text, TOOL_OUTPUT_TOKEN_BUDGET)


def _mark_truncated(text: str) -> str:
    """保留模型已生成内容，标记达到输出上限，避免上层重复请求。"""
    return text.rstrip() + "\n\n（内容达到模型单次输出上限，以上为已生成部分。）"


# [新增-上下文预算-工具循环] 工具调用参数可能比工具结果更长，
# 每次请求模型前都重新计算预算，避免多轮工具链把 messages 撑爆。
def _message_tokens(message: dict) -> int:
    parts = [str(message.get("content") or ""), str(message.get("tool_call_id") or "")]
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return _token_count("\n".join(parts))


def _fit_runtime_messages(
    messages: list[dict],
    user_query: str,
    tools: list[dict] | None = None,
) -> list[dict]:
    """返回本次模型请求用的副本，不修改持久化用的完整事件链。"""
    if not messages or not CONTEXT_TOKEN_BUDGET:
        return messages

    # 保留 system 和当前轮 user；为工具链、回答和协议开销预留空间。
    tool_tokens = _token_count(json.dumps(tools or [], ensure_ascii=False))
    budget = max(512, CONTEXT_TOKEN_BUDGET - _token_count(user_query) - tool_tokens - 128)
    # [新增-上下文预算-工具循环] 只修改请求副本，SQLite 要保存的原始 tool_calls 不被截断。
    fitted = [copy.deepcopy(messages[0])]
    body = copy.deepcopy(messages[1:])

    # [新增-上下文预算-工具循环] 旧回合按完整 user 回合删除，避免留下孤立 tool 消息。
    while body and sum(_message_tokens(item) for item in fitted + body) > budget:
        user_positions = [
            i for i, item in enumerate(body) if item.get("role") == "user"
        ]
        if len(user_positions) <= 1:
            # 只有当前轮时，可以删除它之前残留的 assistant/tool 记录，
            # 但绝不能删除当前 user，否则模型会失去本轮任务。
            if user_positions and user_positions[0] > 0:
                del body[:user_positions[0]]
                continue
            break
        # 不删除最后一个 user 以及它之后的工具链；只清理更早完整回合。
        del body[user_positions[0]:user_positions[1]]

    # 超大工具参数不能按 token 直接裁 JSON，否则参数将不可解析。
    # 当前轮中尚未执行的调用保留完整；已经执行完的历史调用可改写成简短摘要。
    total = sum(_message_tokens(item) for item in fitted + body)
    if total > budget:
        # 先缩短旧 tool 输出；若仍超限，下面再压缩对应的已完成工具协议组。
        for item in body:
            if item.get("role") == "tool" and total > budget:
                original = str(item.get("content") or "")
                clipped = _truncate_tokens(original, max(64, TOOL_OUTPUT_TOKEN_BUDGET // 4))
                if clipped != original:
                    item["content"] = clipped
                    total = sum(_message_tokens(entry) for entry in fitted + body)

    # [新增-上下文预算-工具循环] 大文件写入可能留下超长参数。
    # 只压缩已有 tool result 的 assistant/tool 组；当前未完成调用没有结果，始终保留。
    while total > budget:
        compacted = False
        for index, item in enumerate(body):
            if item.get("role") != "assistant" or not item.get("tool_calls"):
                continue
            end = index + 1
            tool_results = []
            while end < len(body) and body[end].get("role") == "tool":
                tool_results.append(str(body[end].get("content") or ""))
                end += 1
            if not tool_results:
                continue
            # 协议组内部可有多个并行调用；只有都找到结果才视为已完成。
            expected = len(item.get("tool_calls") or [])
            if len(tool_results) < expected:
                continue
            names = [call.get("function", {}).get("name", "tool")
                     for call in item.get("tool_calls") or []]
            compact = "[已完成工具调用：" + ", ".join(names) + "]"
            body[index:end] = [{"role": "assistant", "content": compact}]
            compacted = True
            total = sum(_message_tokens(entry) for entry in fitted + body)
            break
        if not compacted:
            break

    return fitted + body


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
                max_tokens=MAX_COMPLETION_TOKENS,
                stream=STREAM_OUTPUT,  # [新增] 可用环境变量关闭中转站不支持的流式模式
            )

            # [新增] 兼容只支持普通 Chat Completions 的中转站
            if not STREAM_OUTPUT:
                choice = stream.choices[0]
                answer = choice.message
                text = answer.content or ""
                if text:
                    print(f"[model:text] {text}")
                if choice.finish_reason == "length":
                    text = _mark_truncated(text)
                    print("[model:truncated] 输出达到 max_tokens，上层不会重试相同请求")
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

            text = "".join(text_parts)
            if finish_reason == "length":
                text = _mark_truncated(text)
                print("\n[model:truncated] 输出达到 max_tokens，上层不会重试相同请求")
            return {
                "text": text,
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
    # [新增-上下文预算] 摘要旧内容与已有摘要共享输入预算，避免摘要请求本身再次超限。
    existing = _truncate_tokens(existing, max(256, SUMMARY_TOKEN_BUDGET // 3))
    transcript_budget = max(256, SUMMARY_TOKEN_BUDGET - _token_count(existing) - 256)
    transcript = _truncate_tokens(payload["transcript"], transcript_budget)
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
    emit=None,
) -> None:
    """[新增-记忆] 一轮结束后的完整生命周期入口。"""
    print(f"[memory:lifecycle:start] session={session_id!r} user_message_id={user_message_id}")
    kind = memory.capture_explicit_memory(session_id, user_message_id, user_text)
    if emit:
        emit({"type": "memory_write", "written": bool(kind), "kind": kind})
    if kind == "pending_confirmation":
        print("[memory:confirm] 检测到长期记忆候选，请回复“确认”或“取消”。")
        # [新增-长期记忆-用户确认] 候选未确认前不触发摘要归档，避免把待确认事件混入压缩结果。
        return
    elif kind:
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
    if emit:
        emit({"type": "summary_start"})
    try:
        summary = summarize_session(client, payload)
        if summary:
            memory.save_summary_and_archive(
                session_id,
                payload["cutoff_message_id"],
                summary,
            )
            print("[memory:summary] 摘要已保存，旧原文已归档")
            if emit:
                emit({"type": "summary_saved", "length": len(summary)})
        else:
            print("[memory:summary] 模型没有返回摘要，原文保持未归档")
    except Exception as exc:
        # [新增-记忆] 摘要失败不影响当前回答，也不归档原文，下一轮继续重试。
        print(f"[memory:summary:error] {exc}")
        if emit:
            emit({"type": "summary_error", "message": str(exc)})
    print(f"[memory:lifecycle:end] session={session_id!r} summary_triggered=True")


def run_agent(
    client: OpenAI,
    messages: list[dict],
    user_query: str,
    tool_index: ToolIndex | None,
    registry: ToolRegistry,
    emit=None,
    model_call=None,
) -> str:
    model_call = model_call or call_model_stream
    last_tool_result = ""
    for step in range(1, MAX_STEPS + 1):
        # [新增-工具目录] 当前任务和上一步结果决定本步提供的工具 schema。
        query = (user_query + "\n" + last_tool_result)[:1500]
        selection = {}
        available_tools = registry.select(query, tool_index, TOOL_SEARCH_LIMIT, selection)
        if emit:
            emit({"type": "tool_search", "step": step, **selection})
            emit({"type": "model_start", "step": step})
        print(f"\n\n[agent:step {step}/{MAX_STEPS}] 请求模型")  # [新增] 展示 Agent 调度轮次
        request_messages = _fit_runtime_messages(messages, user_query, available_tools)
        if len(request_messages) != len(messages):
            print(
                f"[agent:context] 本次请求使用 {len(request_messages)}/{len(messages)} 条消息"
            )
        streamed = model_call(client, request_messages, available_tools)
        text = streamed["text"]
        tool_calls = streamed["tool_calls"]
        finish_reason = streamed["finish_reason"]
        if text and emit:
            emit({"type": "model_text", "step": step, "text": text})

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
        if emit:
            emit({
                "type": "model_tool_calls", "step": step,
                "calls": [
                    {"name": call["function"]["name"], "args": call["function"]["arguments"]}
                    for call in tool_calls
                ],
            })
        messages.append(
            {
                "role": "assistant",
                "content": text or None,
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            # [新增-工具目录] 将模型请求交给独立注册表校验并分发。
            if emit:
                emit({"type": "tool_start", "name": call["function"]["name"],
                      "args": call["function"]["arguments"]})
            result = registry.dispatch(
                call["function"]["name"],
                call["function"]["arguments"],
                available_tools,
            )
            last_tool_result = result
            if emit:
                tool_name = call["function"]["name"]
                if tool_name == "read_skill":
                    try:
                        skill_name = json.loads(call["function"]["arguments"]).get("name", "")
                    except (json.JSONDecodeError, AttributeError):
                        skill_name = ""
                    emit({
                        "type": "skill_loaded", "name": skill_name,
                        "content": result, "loaded": not result.startswith(("未知 skill：", "skill 文件不存在：")),
                    })
                else:
                    emit({"type": "tool_result", "name": tool_name, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    final_text = "已达到本轮最大执行次数，请换一种问法再试。"
    # 保留最近工具结果摘要，供下一轮对话继续处理，而不是只返回循环上限。
    if last_tool_result:
        final_text = "已达到本轮最大工具调用轮次，最近一次工具结果如下：\n" + _truncate_tokens(
            last_tool_result, max(128, TOOL_OUTPUT_TOKEN_BUDGET // 2)
        )
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
    runtime = ToolRuntime(
        WORKSPACE_ROOT,
        skill_catalog,
        MAX_FILE_SIZE,
        max_output=MAX_OUTPUT,
        max_output_tokens=TOOL_OUTPUT_TOKEN_BUDGET,
    )
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
        confirmation_mode=MEMORY_CONFIRM_MODE,
    )
    session_id = memory.session_id
    print(
        "AI 助手已启动。输入“退出”结束。"
        f"工作空间：{WORKSPACE_ROOT}，session：{session_id}。"
        f" 向量：{VECTOR_EMBEDDING_MODE}。"
        " 命令：/clear、/session、/memories、/forget、/reindex、/reindex-tools、/skills、/help。"
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

        # [新增-长期记忆-用户确认] 确认消息不进入模型上下文，避免“确认”被误当成新问题。
        pending = memory.pending_memory()
        if pending and not user_input.startswith("/"):
            answer = user_input.strip().lower()
            if answer in {"确认", "记住", "是", "同意", "yes", "y"}:
                result = memory.accept_pending_memory(pending["candidate_id"])
                print(f"已保存长期记忆：{result['kind']}（{result['action']}）。")
                continue
            if answer in {"取消", "拒绝", "不要", "否", "no", "n"}:
                memory.reject_pending_memory(pending["candidate_id"])
                print("已取消保存这条长期记忆。")
                continue
            print("当前有一条待确认的长期记忆，请回复“确认”或“取消”。")
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
        if user_input == "/memories":
            memories = memory.list_memories()
            if not memories:
                print("当前没有 active 长期记忆。")
            else:
                for item in memories:
                    print(f"[{item['memory_id']}] {item['kind']}: {item['content']}")
            continue
        if user_input.startswith("/forget"):
            identifier = user_input[len("/forget"):].strip()
            if not identifier:
                print("用法：/forget <memory_id|fact_key|all>")
            elif memory.forget_memory(identifier):
                print(f"已遗忘长期记忆：{identifier}")
            else:
                print(f"未找到可遗忘的长期记忆：{identifier}")
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
            print("命令：/clear 新建 session，/session 查看状态，/memories 查看长期记忆，/forget <id|fact_key|all> 遗忘记忆，/reindex 重建记忆向量，/reindex-tools 重建工具向量，/skills 查看 skill，/help 帮助，退出 结束。")
            continue

        # [新增-记忆] 每轮按当前 query 重新构建上下文：摘要、长期记忆和当前 session 的短期原文。
        messages = memory.build_context_messages(
            SYSTEM_PROMPT + (
                "\n可用 skill（相关任务先调用 read_skill）：\n" + skill_catalog.descriptions()
                if skill_catalog.entries else ""
            ),
            user_input,
            max_rounds=SHORT_TERM_ROUNDS,
            max_tokens=CONTEXT_TOKEN_BUDGET,
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
