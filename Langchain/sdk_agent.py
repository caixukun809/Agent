"""Expose the original OpenAI SDK agent loop to the shared web workbench."""

from openai import OpenAI

import main as sdk


class SDKAgent:
    def __init__(self, shared, client=None, model_call=None):
        self.shared = shared
        self.client = client or OpenAI(
            api_key=shared.settings.api_key,
            base_url=shared.settings.base_url,
        )
        self.model_call = model_call

    def chat(self, session_id: str, user_text: str, emit) -> str:
        store = self.shared._store(session_id)
        context_trace = {}
        prompt = sdk.SYSTEM_PROMPT
        if self.shared.skill_catalog.entries:
            prompt += "\n可用 skill（相关任务先调用 read_skill）：\n" + self.shared.skill_catalog.descriptions()
        messages = store.build_context_messages(
            prompt, user_text, self.shared.settings.short_term_rounds, trace=context_trace
        )
        emit({"type": "context", **context_trace})
        turn_start = len(messages)
        messages.append({"role": "user", "content": user_text})
        user_message_id = store.save_message(session_id, messages[-1])
        try:
            answer = sdk.run_agent(
                self.client, messages, user_text, self.shared.tool_index,
                self.shared.registry, emit=emit, model_call=self.model_call,
            )
            emit({"type": "answer", "text": answer})
        except Exception as exc:
            answer = f"（调用失败：{exc}）"
            messages.append({"role": "assistant", "content": answer})
            emit({"type": "error", "message": str(exc)})
            emit({"type": "answer", "text": answer})
        for message in messages[turn_start + 1:]:
            store.save_message(session_id, message)
        sdk.finish_memory_lifecycle(
            self.client, store, session_id, user_message_id, user_text, emit=emit
        )
        emit({"type": "done", "session_id": session_id, "stats": store.stats(session_id)})
        return answer
