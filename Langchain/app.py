"""Local workbench for the SDK and LangChain agents. Run: python -m Langchain.app."""

import json
import os
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from Langchain.agent import LangChainAgent, Settings
from Langchain.research_agent import ResearchAgent
from Langchain.sdk_agent import SDKAgent
from workspace_manager import Workspace, WorkspaceManager

FRONTEND = Path(__file__).resolve().parent / "frontend"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HIDDEN_FILES = {
    "agent_memory.sqlite3", "memory_vectors.faiss", "memory_vectors.meta.json",
    "tool_vectors.faiss", "tool_vectors.meta.json",
}


def model_profiles() -> list[dict]:
    profiles = [{
        "id": "default",
        "label": f"默认 · {os.getenv('OPENAI_MODEL', 'gpt-6-luna')}",
        "model": os.getenv("OPENAI_MODEL", "gpt-6-luna"),
    }]
    local_model = os.getenv("OPENAI_LOCAL_MODEL", "").strip()
    local_api_key = os.getenv("OPENAI_LOCAL_API_KEY", "").strip()
    local_base_url = os.getenv("OPENAI_LOCAL_BASE_URL", "").strip()
    if local_model and local_api_key and local_base_url:
        profiles.append({
            "id": "local",
            "label": f"本地 · {local_model}",
            "model": local_model,
        })
    return profiles


class Workbench(ThreadingHTTPServer):
    def __init__(
        self,
        address,
        agent,
        sdk_agent=None,
        workspace_manager=None,
        agent_factory=None,
        model_factory=None,
    ):
        super().__init__(address, WorkbenchHandler)
        self.agent = agent
        self.workspace_manager = workspace_manager
        self.agent_factory = agent_factory
        self.model_factory = model_factory
        self._bundles = {}
        default_id = workspace_manager.get(None).workspace_id if workspace_manager else "default"
        self._bundles[default_id] = {
            "langchain": agent,
            "sdk": sdk_agent or SDKAgent(agent),
            "langgraph": ResearchAgent(agent),
        }
        self._profile_bundles = {}
        self._profile_bundles_guard = threading.Lock()
        self._session_locks = {}
        self._session_locks_guard = threading.Lock()

    def session_lock(self, workspace_id: str | None, session_id: str) -> threading.Lock:
        key = (workspace_id or "default", session_id)
        with self._session_locks_guard:
            return self._session_locks.setdefault(key, threading.Lock())

    def bundle(self, workspace_id: str | None = None, model_profile: str = "default") -> dict:
        if model_profile not in {item["id"] for item in model_profiles()}:
            raise ValueError("未知模型配置")
        if model_profile != "default":
            workspace = self.workspace_manager.get(workspace_id) if self.workspace_manager else None
            workspace_key = workspace.workspace_id if workspace else "default"
            key = (workspace_key, model_profile)
            with self._profile_bundles_guard:
                if key not in self._profile_bundles:
                    if not self.model_factory:
                        raise RuntimeError("未配置本地模型工厂")
                    self._profile_bundles[key] = self.model_factory(workspace, model_profile)
            return self._profile_bundles[key]

        if not self.workspace_manager:
            return self._bundles["default"]
        workspace = self.workspace_manager.get(workspace_id)
        if workspace.workspace_id not in self._bundles:
            if not self.agent_factory:
                raise RuntimeError("未配置工作区 Agent 工厂")
            self._bundles[workspace.workspace_id] = self.agent_factory(workspace)
        return self._bundles[workspace.workspace_id]

    def _hidden_runtime_file(self, path: Path, agent=None) -> bool:
        database = (agent or self.agent).settings.memory_db.resolve()
        return (
            path.resolve() == database
            or path.name.startswith(database.name + "-")
            or path.name in HIDDEN_FILES
        )

    def files(self, agent=None) -> list[dict]:
        root = (agent or self.agent).settings.workspace
        items = []
        for path in root.rglob("*"):
            if len(items) >= 200:
                break
            relative = path.relative_to(root)
            if (
                not path.is_file() or path.is_symlink()
                or any(part.startswith(".") for part in relative.parts)
                or self._hidden_runtime_file(path, agent)
            ):
                continue
            items.append({"path": relative.as_posix(), "size": path.stat().st_size})
        return sorted(items, key=lambda item: item["path"].lower())

    def read_file(self, relative_path: str, agent=None) -> dict:
        selected_agent = agent or self.agent
        root = selected_agent.settings.workspace.resolve()
        requested = Path(relative_path)
        if requested.is_absolute() or not relative_path:
            raise ValueError("无效文件路径")
        path = (root / requested).resolve()
        if (
            not path.is_relative_to(root) or not path.is_file()
            or path.is_symlink() or self._hidden_runtime_file(path, selected_agent)
            or any(part.startswith(".") for part in requested.parts)
        ):
            raise ValueError("文件不存在或不可预览")
        if path.stat().st_size > 200_000:
            raise ValueError("文件过大，无法预览")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("只能预览 UTF-8 文本文件") from exc
        return {"path": relative_path, "content": content}


class WorkbenchHandler(BaseHTTPRequestHandler):
    server: Workbench
    server_version = "AgentWorkbench/1.0"

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 1_000_000:
            raise ValueError("请求内容过大")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("请求必须是 JSON 对象")
        return payload

    def _serve_static(self, name: str, mime: str) -> None:
        path = FRONTEND / name
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                return self._serve_static("index.html", "text/html; charset=utf-8")
            if parsed.path == "/styles.css":
                return self._serve_static("styles.css", "text/css; charset=utf-8")
            if parsed.path == "/app.js":
                return self._serve_static("app.js", "text/javascript; charset=utf-8")
            if re.fullmatch(r"/icons/[a-z-]+\.svg", parsed.path):
                name = parsed.path.removeprefix("/icons/")
                if name in {"plus.svg", "panel-left.svg", "panel-right.svg", "message-square.svg", "file-text.svg", "arrow-up.svg", "x.svg", "copy.svg", "archive.svg", "archive-restore.svg"}:
                    return self._serve_static("icons/" + name, "image/svg+xml")
            if parsed.path == "/api/health":
                return self._json(200, {"ok": True, "model": self.server.agent.settings.model_name})
            if parsed.path == "/api/models":
                return self._json(200, {"models": model_profiles()})
            if parsed.path == "/api/workspaces":
                if not self.server.workspace_manager:
                    return self._json(200, {"workspaces": [{
                        "workspace_id": "default", "name": "默认工作区",
                        "path": str(self.server.agent.settings.workspace),
                        "is_default": True,
                    }]})
                return self._json(200, {"workspaces": self.server.workspace_manager.public()})
            if parsed.path == "/api/state":
                session_id = parse_qs(parsed.query).get("session", [None])[0]
                workspace_id = parse_qs(parsed.query).get("workspace", [None])[0]
                model_profile = parse_qs(parsed.query).get("model_profile", ["default"])[0]
                if session_id and not SESSION_ID.fullmatch(session_id):
                    raise ValueError("无效 session ID")
                bundle = self.server.bundle(workspace_id, model_profile)
                state = bundle["langchain"].state(session_id)
                state["model_profile"] = model_profile
                state["models"] = model_profiles()
                workspace = self.server.workspace_manager.get(workspace_id) if self.server.workspace_manager else None
                state["workspace"] = workspace.public() if workspace else {
                    "workspace_id": "default", "name": "默认工作区",
                    "path": str(bundle["langchain"].settings.workspace), "is_default": True,
                }
                state["workspaces"] = self.server.workspace_manager.public() if self.server.workspace_manager else [state["workspace"]]
                state["files"] = self.server.files(bundle["langchain"])
                return self._json(200, state)
            if parsed.path == "/api/file":
                relative = parse_qs(parsed.query).get("path", [""])[0]
                workspace_id = parse_qs(parsed.query).get("workspace", [None])[0]
                return self._json(200, self.server.read_file(relative, self.server.bundle(workspace_id)["langchain"]))
            self._json(404, {"error": "未找到页面"})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self):
        try:
            if self.path == "/api/session":
                payload = self._body()
                workspace_id = payload.get("workspace_id")
                bundle = self.server.bundle(workspace_id)
                session_id = bundle["langchain"].new_session()
                return self._json(201, {"session_id": session_id, "workspace_id": workspace_id or "default"})
            if self.path == "/api/workspace":
                payload = self._body()
                if not self.server.workspace_manager:
                    raise ValueError("当前运行模式不支持新建工作区")
                workspace = self.server.workspace_manager.create(payload.get("name", ""))
                return self._json(201, {"workspace": workspace.public()})
            if self.path == "/api/session/archive":
                payload = self._body()
                session_id = payload.get("session_id")
                workspace_id = payload.get("workspace_id")
                archived = payload.get("archived")
                if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
                    raise ValueError("无效 session ID")
                if not isinstance(archived, bool):
                    raise ValueError("archived 必须是布尔值")
                bundle = self.server.bundle(workspace_id)
                with self.server.session_lock(workspace_id, session_id):
                    store = bundle["langchain"]._store(bundle["langchain"].default_session_id)
                    value = store.set_session_archived(session_id, archived)
                return self._json(200, {"session_id": session_id, "archived_at": value})
            if self.path != "/api/chat":
                return self._json(404, {"error": "未找到接口"})
            payload = self._body()
            session_id = payload.get("session_id")
            text = payload.get("message")
            workspace_id = payload.get("workspace_id")
            backend = payload.get("backend", "langchain")
            model_profile = payload.get("model_profile", "default")
            bundle = self.server.bundle(workspace_id, model_profile)
            if backend not in bundle:
                raise ValueError("未知 Agent")
            if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
                raise ValueError("无效 session ID")
            if not isinstance(text, str) or not text.strip() or len(text) > 20_000:
                raise ValueError("消息为空或过长")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            return self._json(503, {"error": str(exc)})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()
        output_lock = threading.Lock()
        disconnected = False

        def emit(event: dict) -> None:
            nonlocal disconnected
            with output_lock:
                if disconnected:
                    return
                try:
                    line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    disconnected = True

        try:
            with self.server.session_lock(workspace_id, session_id):
                bundle[backend].chat(session_id, text.strip(), emit)
        except Exception as exc:
            emit({"type": "error", "message": str(exc)})
            emit({"type": "done", "session_id": session_id})


def main() -> None:
    base_settings = Settings.from_env()
    workspace_manager = WorkspaceManager(PROJECT_ROOT, base_settings.workspace)
    default_agent = LangChainAgent(base_settings)
    shared_model = default_agent.model
    shared_embedder = default_agent.vector_store.embedder

    def agent_factory(workspace: Workspace):
        settings = replace(base_settings, workspace=workspace.path, memory_db=workspace.memory_db)
        langchain_agent = LangChainAgent(settings, model=shared_model, embedder=shared_embedder)
        return {
            "langchain": langchain_agent,
            "sdk": SDKAgent(langchain_agent),
            "langgraph": ResearchAgent(langchain_agent),
        }

    def model_factory(workspace: Workspace | None, profile: str):
        if profile != "local":
            raise ValueError("未知模型配置")
        workspace_path = workspace.path if workspace else base_settings.workspace
        memory_db = workspace.memory_db if workspace else base_settings.memory_db
        settings = replace(
            base_settings,
            workspace=workspace_path,
            memory_db=memory_db,
            model_name=os.environ["OPENAI_LOCAL_MODEL"],
            api_key=os.environ["OPENAI_LOCAL_API_KEY"],
            base_url=os.environ["OPENAI_LOCAL_BASE_URL"],
        )
        local_agent = LangChainAgent(settings, embedder=shared_embedder)
        return {
            "langchain": local_agent,
            "sdk": SDKAgent(local_agent),
            "langgraph": ResearchAgent(local_agent),
        }

    agent = default_agent
    preferred = int(os.getenv("AGENT_WEB_PORT", "8765"))
    for port in range(preferred, preferred + 20):
        try:
            server = Workbench(("127.0.0.1", port), agent,
                               workspace_manager=workspace_manager,
                               agent_factory=agent_factory,
                               model_factory=model_factory)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("没有可用的本地端口")
    print(f"Agent 工作台：http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
