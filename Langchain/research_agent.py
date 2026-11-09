"""Adapt the LangGraph research workflow to the existing workbench API."""

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_MODULE_PATH = PROJECT_ROOT / "langgraph" / "agent.py"
_spec = importlib.util.spec_from_file_location(
    "agent_workbench_research_graph", RESEARCH_MODULE_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载调研流程：{RESEARCH_MODULE_PATH}")
_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_research)


class ResearchAgent:
    def __init__(self, shared, runner=None):
        self.shared = shared
        self.runner = runner or _research.run_research

    def chat(self, session_id: str, user_text: str, emit) -> str:
        store = self.shared._store(session_id)
        user_message_id = store.save_message(
            session_id, {"role": "user", "content": user_text}
        )

        def on_step(node: str, update: dict, state: dict) -> None:
            if node == "plan":
                emit({"type": "research_plan", "queries": update.get("search_plan", [])})
            elif node == "search":
                emit({
                    "type": "research_search",
                    "round": state.get("round", 1),
                    "queries": update.get("searched_queries", []),
                    "source_count": len(update.get("sources", [])),
                    "errors": update.get("search_errors", []),
                })
            elif node == "critique":
                emit({
                    "type": "research_critique",
                    "round": state.get("round", 1),
                    "critique": update.get("critique", ""),
                    "missing_topics": update.get("missing_topics", []),
                    "sufficient": update.get("sufficient", False),
                    "follow_up_queries": update.get("pending_queries", []),
                })
            elif node == "write":
                emit({"type": "research_write_start"})

        try:
            result = self.runner(
                user_text,
                model=self.shared.model,
                on_step=on_step,
            )
            answer = _research.format_report(result)
            store.save_message(session_id, {"role": "assistant", "content": answer})
            emit({"type": "answer", "text": answer})
        except Exception as exc:
            answer = f"（调研失败：{exc}）"
            store.save_message(session_id, {"role": "assistant", "content": answer})
            emit({"type": "error", "message": str(exc)})
            emit({"type": "answer", "text": answer})

        emit({
            "type": "done",
            "session_id": session_id,
            "stats": store.stats(session_id),
            "user_message_id": user_message_id,
        })
        return answer
