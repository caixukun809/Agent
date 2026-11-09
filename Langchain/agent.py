"""[新增-LangChain] LangChain Agent loop with the existing memory and tool stores."""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, OpenAI, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory import MemoryStore  # noqa: E402
from skills import SkillCatalog  # noqa: E402
from tool_index import ToolIndex  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402
from tools.runtime import ToolRuntime  # noqa: E402
from vector_store import HashEmbedding, LocalEmbedding, OpenAIEmbedding, VectorStore  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

Emit = Callable[[dict], None]

SYSTEM_PROMPT = (
    "你是一个简洁、诚实的中文 AI 助手。需要当前时间、执行代码或操作文件时，"
    "使用本轮提供的工具。工具和文件操作只能报告实际执行结果。"
)


def _path_from_env(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default)))
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


@dataclass(frozen=True)
class Settings:
    workspace: Path
    memory_db: Path
    owner_id: str
    model_name: str
    api_key: str
    base_url: str | None
    embedding_mode: str
    embedding_model: str
    local_embedding_model: Path
    short_term_rounds: int
    summary_trigger_rounds: int
    summary_keep_rounds: int
    tool_search_limit: int
    max_steps: int
    model_timeout: float
    max_completion_tokens: int

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = _path_from_env("AGENT_WORKSPACE", PROJECT_ROOT / "workspace")
        trigger = max(2, int(os.getenv("SUMMARY_TRIGGER_ROUNDS", "12")))
        return cls(
            workspace=workspace,
            memory_db=_path_from_env("AGENT_MEMORY_DB", workspace / "agent_memory.sqlite3"),
            owner_id=os.getenv("AGENT_OWNER_ID", "local-user"),
            model_name=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            embedding_mode=os.getenv("VECTOR_EMBEDDING_MODE", "local").lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            local_embedding_model=_path_from_env(
                "LOCAL_EMBEDDING_MODEL", PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
            ),
            short_term_rounds=max(1, int(os.getenv("SHORT_TERM_ROUNDS", "8"))),
            summary_trigger_rounds=trigger,
            summary_keep_rounds=max(1, min(int(os.getenv("SUMMARY_KEEP_ROUNDS", "6")), trigger - 1)),
            tool_search_limit=max(1, int(os.getenv("TOOL_SEARCH_LIMIT", "8"))),
            max_steps=max(1, int(os.getenv("MAX_STEPS", "5"))),
            model_timeout=max(5.0, float(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))),
            max_completion_tokens=max(512, int(os.getenv("MAX_COMPLETION_TOKENS", "12000"))),
        )


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "")


def _to_langchain(message: dict):
    role = message["role"]
    if role == "user":
        return HumanMessage(content=message.get("content") or "")
    if role == "tool":
        return ToolMessage(
            content=message.get("content") or "",
            tool_call_id=message["tool_call_id"],
        )
    if role == "assistant":
        calls = []
        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except (KeyError, json.JSONDecodeError):
                args = {}
            calls.append({"id": call["id"], "name": call["function"]["name"], "args": args})
        return AIMessage(content=message.get("content") or "", tool_calls=calls)
    raise ValueError(f"不支持的历史消息角色：{role}")


def _to_stored_message(message) -> dict | None:
    if isinstance(message, AIMessage):
        calls = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["args"], ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
        return {"role": "assistant", "content": _text(message.content), "tool_calls": calls}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _text(message.content),
        }
    return None


class LangChainAgent:
    def __init__(self, settings: Settings, model=None, embedder=None):
        self.settings = settings
        settings.workspace.mkdir(parents=True, exist_ok=True)
        if model is None:
            if not settings.api_key:
                raise ValueError("缺少 OPENAI_API_KEY，请在项目 .env 中配置")
            # [LangChain] ChatOpenAI 替代直接调用 OpenAI Chat Completions。
            # Reuse the SDK HTTP client that works with the configured relay.
            sdk_client = OpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.model_timeout,
                max_retries=0,
            )
            model = ChatOpenAI(
                model=settings.model_name,
                api_key=settings.api_key,
                base_url=settings.base_url,
                temperature=0.3,
                streaming=False,
                stream_usage=False,
                use_responses_api=False,
                max_tokens=settings.max_completion_tokens,
                timeout=settings.model_timeout,
                max_retries=0,
                client=sdk_client.chat.completions,
                root_client=sdk_client,
            )
        elif hasattr(model, "max_tokens") and getattr(model, "max_tokens", None) is None:
            # Fake/custom chat models remain usable; apply the configured cap when supported.
            try:
                model = model.bind(max_tokens=settings.max_completion_tokens)
            except (AttributeError, TypeError, ValueError):
                pass
        self.model = model
        if embedder is None:
            if settings.embedding_mode == "local":
                embedder = LocalEmbedding(settings.local_embedding_model)
            elif settings.embedding_mode == "openai":
                embedder = OpenAIEmbedding(
                    OpenAI(api_key=settings.api_key, base_url=settings.base_url),
                    settings.embedding_model,
                )
            elif settings.embedding_mode == "hash":
                embedder = HashEmbedding()
            else:
                raise ValueError(f"未知 embedding 模式：{settings.embedding_mode}")
        self.vector_store = VectorStore(settings.workspace, embedder)
        self.skill_catalog = SkillCatalog(PROJECT_ROOT / "skill")
        self.runtime = ToolRuntime(settings.workspace, self.skill_catalog)
        self.registry = ToolRegistry(PROJECT_ROOT / "tools" / "catalog.json", self.runtime)
        embedding_id = (
            f"{settings.embedding_mode}:"
            f"{settings.local_embedding_model if settings.embedding_mode == 'local' else settings.embedding_model}"
        )
        try:
            self.tool_index = ToolIndex(settings.workspace, embedder, self.registry.entries, embedding_id)
        except Exception as exc:
            print(f"[tool:index:error] {exc}；提供全部工具")
            self.tool_index = None
        self.default_session_id = self._store().session_id

    def _store(self, session_id: str | None = None) -> MemoryStore:
        return MemoryStore(
            self.settings.memory_db,
            self.settings.owner_id,
            session_id,
            vector_store=self.vector_store,
        )

    def new_session(self) -> str:
        return self._store().session_id

    def state(self, session_id: str | None = None) -> dict:
        store = self._store(session_id or self.default_session_id)
        selected = store.session_id
        return {
            "session_id": selected,
            "sessions": store.list_sessions(),
            "messages": store.list_messages(selected),
            "summary": store.get_summary(selected),
            "memories": store.list_memories(),
            "stats": store.stats(selected),
            "tools": [
                {"id": entry["id"], "description": entry["schema"]["function"]["description"]}
                for entry in self.registry.entries
            ],
            "skills": self.skill_catalog.entries,
            "model": self.settings.model_name,
            "embedding": self.settings.embedding_mode,
        }

    def _build_agent(self, system_prompt: str, emit: Emit):
        # [LangChain] 从现有清单创建工具，新增工具不需要在此处再写包装函数。
        available_schemas = [entry["schema"] for entry in self.registry.entries]

        def execute(name: str, args: dict) -> str:
            emit({"type": "tool_start", "name": name, "args": args})
            result = self.registry.dispatch(
                name, json.dumps(args, ensure_ascii=False), available_schemas
            )
            if name == "read_skill":
                emit({
                    "type": "skill_loaded", "name": args.get("name", ""),
                    "content": result,
                    "loaded": not result.startswith(("未知 skill：", "skill 文件不存在：")),
                })
            else:
                emit({"type": "tool_result", "name": name, "result": result})
            return result

        tool_map = {}
        for entry in self.registry.entries:
            schema = entry["schema"]["function"]
            name = entry["id"]
            tool_map[name] = StructuredTool.from_function(
                func=lambda _name=name, **kwargs: execute(_name, kwargs),
                name=name,
                description=schema["description"],
                args_schema=schema["parameters"],
            )
        model_step = 0

        @wrap_model_call
        def select_tools(request, handler):
            nonlocal model_step
            model_step += 1
            user_text = next(
                (_text(msg.content) for msg in reversed(request.messages) if isinstance(msg, HumanMessage)),
                "",
            )
            last_result = next(
                (_text(msg.content) for msg in reversed(request.messages) if isinstance(msg, ToolMessage)),
                "",
            )
            query = (user_text + "\n" + last_result)[:1500]
            selection = {}
            available_schemas[:] = self.registry.select(
                query, self.tool_index, self.settings.tool_search_limit, selection
            )
            emit({"type": "tool_search", "step": model_step, **selection})
            emit({"type": "model_start", "step": model_step})
            selected = [tool_map[schema["function"]["name"]] for schema in available_schemas]
            # The relay accepts the strict tool schema used by the SDK agent.
            bound_request = request.override(
                tools=selected,
                model_settings={**request.model_settings, "strict": True},
            )
            for attempt in range(3):
                try:
                    response = handler(bound_request)
                    break
                except (APIConnectionError, RateLimitError) as exc:
                    if attempt == 2:
                        raise
                    delay = 2 ** attempt
                    emit({
                        "type": "model_retry", "step": model_step,
                        "attempt": attempt + 1, "delay": delay, "message": str(exc),
                    })
                    time.sleep(delay)
            message = response.result[0] if hasattr(response, "result") else response
            if message.tool_calls:
                emit({
                    "type": "model_tool_calls",
                    "step": model_step,
                    "calls": [
                        {"name": call["name"], "args": call["args"]}
                        for call in message.tool_calls
                    ],
                })
            elif _text(message.content):
                emit({"type": "model_text", "step": model_step, "text": _text(message.content)})
            return response

        return create_agent(
            self.model,
            tools=list(tool_map.values()),
            system_prompt=system_prompt,
            middleware=[select_tools],
        )

    def _summarize(self, payload: dict) -> str:
        old = payload.get("existing_summary") or "（暂无历史摘要）"
        answer = self.model.invoke([
            SystemMessage(content=(
                "你是会话摘要器。只总结事实、用户目标、已完成事项、未完成事项、"
                "关键决定和约束。不要添加原文没有的信息。"
            )),
            HumanMessage(content=(
                f"已有摘要：\n{old}\n\n"
                f"需要归档的原始消息：\n{payload['transcript'][:4000]}\n\n"
                "请输出一份简洁的中文增量摘要。"
            )),
        ])
        return _text(answer.content).strip()

    def chat(self, session_id: str, user_text: str, emit: Emit) -> str:
        store = self._store(session_id)
        context_trace = {}
        context = store.build_context_messages(
            SYSTEM_PROMPT + (
                "\n可用 skill（相关任务先调用 read_skill）：\n"
                + self.skill_catalog.descriptions()
                if self.skill_catalog.entries else ""
            ),
            user_text,
            self.settings.short_term_rounds,
            trace=context_trace,
        )
        emit({"type": "context", **context_trace})
        history = [_to_langchain(message) for message in context[1:]]
        history.append(HumanMessage(content=user_text))
        user_message_id = store.save_message(session_id, {"role": "user", "content": user_text})
        answer = ""
        try:
            agent = self._build_agent(context[0]["content"], emit)
            # [LangChain] invoke 会循环模型与工具，取代原版的手写 for/if 调度。
            # LangGraph 的 recursion_limit 统计模型节点和工具节点；每轮最多一次
            # 模型请求和一次工具批次，因此按 max_steps 留出协议节点余量，避免长工具链
            # 在上下文压缩后仍因过小的递归上限提前卡在运行中。
            result = agent.invoke(
                {"messages": history},
                config={"recursion_limit": max(8, self.settings.max_steps * 3 + 2)},
            )
            new_messages = result["messages"][len(history):]
            for message in new_messages:
                stored = _to_stored_message(message)
                if stored:
                    store.save_message(session_id, stored)
            answer = next(
                (
                    _text(message.content)
                    for message in reversed(new_messages)
                    if isinstance(message, AIMessage) and not message.tool_calls
                ),
                "（模型没有返回文本）",
            )
            emit({"type": "answer", "text": answer})
        except Exception as exc:
            answer = f"（调用失败：{exc}）"
            store.save_message(session_id, {"role": "assistant", "content": answer})
            emit({"type": "error", "message": str(exc)})
            emit({"type": "answer", "text": answer})

        kind = store.capture_explicit_memory(session_id, user_message_id, user_text)
        emit({"type": "memory_write", "written": bool(kind), "kind": kind})
        payload = store.compaction_payload(
            session_id,
            self.settings.summary_trigger_rounds,
            self.settings.summary_keep_rounds,
        )
        if payload:
            emit({"type": "summary_start"})
            try:
                summary = self._summarize(payload)
                if summary:
                    store.save_summary_and_archive(
                        session_id, payload["cutoff_message_id"], summary
                    )
                    emit({"type": "summary_saved", "length": len(summary)})
            except Exception as exc:
                emit({"type": "summary_error", "message": str(exc)})
        emit({"type": "done", "session_id": session_id, "stats": store.stats(session_id)})
        return answer
