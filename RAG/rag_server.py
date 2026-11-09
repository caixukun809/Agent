"""企业知识助手本地 Web 服务。"""
from __future__ import annotations

import argparse
import html
import hmac
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rag_pipeline import (
    DEFAULT_DOCUMENTS,
    DEFAULT_EMBEDDING,
    DEFAULT_RERANKER,
    DEFAULT_RUNTIME,
    RAGPipeline,
    SUPPORTED_SUFFIXES,
    _extract_file,
    _project_path,
    configure_logging,
)

UI_PATH = Path(__file__).with_name("rag_ui.html")
MAX_REQUEST_BYTES = 1_000_000


def _model_display_name(path: Path, fallback: str) -> str:
    name = path.name
    if name.lower() in {"reranker", "bge-reranker-base"}:
        return "BGE Reranker Base · 中文/多语种"
    return name or fallback


def _read_source(path_value: str, line_start: int, line_end: int, documents: Path) -> tuple[str, str, int, int, int]:
    root = documents.resolve()
    source = Path(path_value).expanduser().resolve(strict=True)
    if not source.is_file() or not source.is_relative_to(root):
        raise ValueError("只能预览知识库目录中的文档")
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("该文档类型不支持原文预览")
    content = _extract_file(source)
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    line_start = max(1, min(line_start, max(1, len(lines))))
    line_end = max(line_start, min(line_end, max(1, len(lines))))
    return content, source.name, line_start, line_end, len(lines)


class Handler(BaseHTTPRequestHandler):
    pipeline: RAGPipeline
    documents: Path
    tenant_pipelines: dict[str, RAGPipeline]
    tenant_pipeline_lock: threading.RLock
    tenant_ingest_lock: threading.RLock
    internal_api_key: str

    @classmethod
    def _tenant_id(cls, value: object) -> str:
        tenant_id = str(value or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tenant_id):
            raise ValueError("租户标识无效")
        return tenant_id

    def _is_internal_request(self) -> bool:
        supplied = self.headers.get("X-RAG-Internal-Key", "")
        return bool(self.internal_api_key) and hmac.compare_digest(supplied, self.internal_api_key)

    @classmethod
    def _new_tenant_pipeline(cls, tenant_id: str) -> RAGPipeline:
        return RAGPipeline(
            runtime=cls.pipeline.runtime / "tenants" / tenant_id,
            embedding_path=cls.pipeline.embedding_path,
            reranker_path=DEFAULT_RERANKER,
            shared_models=(cls.pipeline.embedder, cls.pipeline.reranker, cls.pipeline.generator),
        )

    @classmethod
    def _get_tenant_pipeline(cls, tenant_id: str) -> RAGPipeline:
        with cls.tenant_pipeline_lock:
            pipeline = cls.tenant_pipelines.get(tenant_id)
            if pipeline is None:
                pipeline = cls._new_tenant_pipeline(tenant_id)
                cls.tenant_pipelines[tenant_id] = pipeline
            return pipeline

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", os.getenv("RAG_LOGIN_URL", "http://127.0.0.1:8080/"))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        if parsed.path == "/api/health":
            pointer = self.pipeline.runtime / "active.json"
            active_build = json.loads(pointer.read_text(encoding="utf-8")).get("active_build") if pointer.exists() else None
            generator = self.pipeline.generator
            reranker = self.pipeline.reranker
            reranker_path = Path(getattr(reranker, "model_path", DEFAULT_RERANKER))
            body = {
                "status": "ready" if self.pipeline.chunks else "index_empty",
                "chunk_count": len(self.pipeline.chunks),
                "active_build": active_build,
                "embedding_model": self.pipeline.embedding_path.name,
                "reranker_model": _model_display_name(reranker_path, getattr(reranker, "name", "Cross-Encoder")),
                "generator_model": getattr(generator, "model", getattr(generator, "name", type(generator).__name__)),
                "min_evidence_score": self.pipeline.min_evidence_score,
            }
            self._send_json(200, body)
            return

        if parsed.path == "/source/view":
            if not self._is_internal_request():
                self._send(403, b"Forbidden", "text/plain; charset=utf-8")
                return
            query = parse_qs(parsed.query)
            try:
                source = query.get("path", [""])[0]
                if (self.documents / "tenants").resolve() in Path(source).expanduser().resolve(strict=True).parents:
                    self._send(403, "租户文档需要通过登录后的文档预览接口访问".encode("utf-8"), "text/plain; charset=utf-8")
                    return
                line_start = int(query.get("line_start", ["1"])[0])
                line_end = int(query.get("line_end", [str(line_start)])[0])
                content, filename, line_start, line_end, line_count = _read_source(
                    source, line_start, line_end, self.documents
                )
                lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                first = max(1, line_start - 60)
                last = min(line_count, line_end + 60)
                rendered = []
                if first > 1:
                    rendered.append(f'<p class="notice">前面省略 {first - 1} 行</p>')
                for number in range(first, last + 1):
                    active = line_start <= number <= line_end
                    rendered.append(
                        f'<div class="line{" highlight" if active else ""}" id="line-{number}">'
                        f'<span class="number">{number}</span><code>{html.escape(lines[number - 1]) or " "}</code></div>'
                    )
                if last < line_count:
                    rendered.append(f'<p class="notice">后面省略 {line_count - last} 行</p>')
                title = html.escape(filename)
                page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · 原文</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;color:#202b35;background:#f2f5f5;font:13px/1.65 "Segoe UI","Microsoft YaHei UI",sans-serif}}header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #dce3e7;padding:13px max(18px,calc((100% - 1000px)/2))}}h1{{font-size:16px;margin:0}}header p{{font-size:12px;color:#687782;margin:4px 0 0;overflow-wrap:anywhere}}main{{max-width:1000px;margin:0 auto;padding:14px 0 50px;font:12px/1.7 Consolas,"Cascadia Code",monospace}}.line{{display:grid;grid-template-columns:58px minmax(0,1fr);padding:0 14px}}.number{{color:#93a0a8;text-align:right;padding-right:14px;user-select:none}}code{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}.highlight{{background:#fff1d8}}.notice{{padding:4px 14px;color:#687782;font:12px/1.5 "Segoe UI","Microsoft YaHei UI",sans-serif}}@media(max-width:520px){{main{{padding-top:8px}}header{{padding-left:14px;padding-right:14px}}}}
</style></head><body><header><h1>{title}</h1><p>引用原文 · 第 {line_start}-{line_end} 行 · 共 {line_count} 行</p></header><main>{''.join(rendered)}</main><script>document.getElementById('line-{line_start}')?.scrollIntoView({{block:'center'}})</script></body></html>"""
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            except (ValueError, OSError, RuntimeError) as exc:
                self._send(400, html.escape(str(exc)).encode("utf-8"), "text/plain; charset=utf-8")
            except Exception:
                logging.getLogger("rag.http").exception("event=source.preview.failed")
                self._send(400, "原文解析失败".encode("utf-8"), "text/plain; charset=utf-8")
            return

        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route not in {"/api/query", "/api/tenant/ingest", "/api/tenant/preview"}:
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求内容为空或超过大小限制")
            payload = json.loads(self.rfile.read(length))
            if route == "/api/tenant/ingest":
                if not self._is_internal_request():
                    self._send_json(403, {"error": "forbidden"})
                    return
                tenant_id = self._tenant_id(payload.get("tenant_id"))
                tenant_documents = (self.documents / "tenants" / tenant_id).resolve()
                if not tenant_documents.is_relative_to((self.documents / "tenants").resolve()):
                    raise ValueError("租户文档目录无效")
                with self.tenant_ingest_lock:
                    tenant_pipeline = self._new_tenant_pipeline(tenant_id)
                    manifest = tenant_pipeline.ensure_index(tenant_documents)
                    with self.tenant_pipeline_lock:
                        self.tenant_pipelines[tenant_id] = tenant_pipeline
                self._send_json(200, {
                    "status": "indexed" if manifest else "unchanged",
                    "tenant_id": tenant_id,
                    "chunk_count": len(tenant_pipeline.chunks),
                    "build_id": (manifest or {}).get("build_id"),
                })
                return
            if route == "/api/tenant/preview":
                if not self._is_internal_request():
                    self._send_json(403, {"error": "forbidden"})
                    return
                tenant_id = self._tenant_id(payload.get("tenant_id"))
                tenant_root = (self.documents / "tenants" / tenant_id).resolve()
                source = Path(str(payload.get("source", ""))).expanduser().resolve(strict=True)
                if not source.is_file() or not source.is_relative_to(tenant_root):
                    raise ValueError("无权预览该文档")
                if source.suffix.lower() not in SUPPORTED_SUFFIXES:
                    raise ValueError("该文档类型不支持原文预览")
                content = _extract_file(source)
                lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                start = max(1, min(int(payload.get("line_start", 1)), max(1, len(lines))))
                end = max(start, min(int(payload.get("line_end", start)), max(1, len(lines))))
                self._send_json(200, {
                    "title": source.name,
                    "content": content,
                    "line_start": start,
                    "line_end": end,
                    "line_count": len(lines),
                })
                return
            if route == "/api/query" and not self._is_internal_request():
                self._send_json(403, {"error": "forbidden"})
                return
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("问题不能为空")
            top_k = max(1, min(20, int(payload.get("top_k", os.getenv("RAG_TOP_K", "5")))))
            raw_history = payload.get("history", [])
            history = []
            if isinstance(raw_history, list):
                for item in raw_history[-12:]:
                    if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                        continue
                    content = str(item.get("content", ""))[:6000]
                    if content:
                        history.append({"role": item["role"], "content": content})
            started = time.perf_counter()
            tenant_id = payload.get("tenant_id")
            if tenant_id is None:
                raise ValueError("必须指定租户")
            tenant_id = self._tenant_id(tenant_id)
            tenant_pipeline = self._get_tenant_pipeline(tenant_id)
            tenant_root = (self.documents / "tenants" / tenant_id).resolve()
            raw_sources = payload.get("allowed_sources", [])
            if not isinstance(raw_sources, list):
                raise ValueError("授权文档列表无效")
            allowed_sources = set()
            for raw_source in raw_sources:
                source = Path(str(raw_source)).expanduser().resolve(strict=False)
                if not source.is_relative_to(tenant_root) or not source.is_file():
                    continue
                allowed_sources.add(str(source))
            result = tenant_pipeline.answer(
                question,
                top_k=top_k,
                role="admin" if payload.get("role") == "ADMIN" else None,
                history=history,
                allowed_sources=allowed_sources,
            )
            logging.getLogger("rag.http").info(
                "trace_id=%s event=http.request.completed method=POST path=/api/query status=200 elapsed_ms=%.1f",
                result["trace_id"],
                (time.perf_counter() - started) * 1000,
            )
            self._send_json(200, result)
        except Exception as exc:
            logging.getLogger("rag.http").exception("event=http.request.failed method=POST path=/api/query")
            self._send_json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="企业知识助手")
    parser.add_argument("--host", default=os.getenv("RAG_HOST", "127.0.0.1"), help="默认仅绑定本机回环地址")
    parser.add_argument("--port", type=int, default=int(os.getenv("RAG_PORT", "8765")))
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--reranker", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--generator", choices=["auto", "extractive", "local", "openai", "openai-compatible"], default=os.getenv("RAG_GENERATOR", "openai-compatible"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=os.getenv("RAG_LOG_LEVEL", "INFO").upper())
    parser.add_argument(
        "--log-file",
        type=Path,
        default=_project_path(os.getenv("RAG_LOG_FILE"), DEFAULT_RUNTIME / "logs" / "server.log"),
    )
    parser.add_argument("--no-auto-ingest", action="store_true", help="不在启动时检查/重建文档索引")
    args = parser.parse_args()
    configure_logging(args.log_level, args.log_file)
    Handler.documents = args.documents.resolve()
    Handler.tenant_pipelines = {}
    Handler.tenant_pipeline_lock = threading.RLock()
    Handler.tenant_ingest_lock = threading.RLock()
    Handler.internal_api_key = os.getenv("RAG_INTERNAL_API_KEY", "local-rag-development-key-change-me")
    Handler.pipeline = RAGPipeline(
        runtime=args.runtime,
        embedding_path=args.embedding,
        reranker_path=args.reranker,
        generator=args.generator,
    )
    if not args.no_auto_ingest and os.getenv("RAG_AUTO_INGEST", "true").lower() not in {"0", "false", "no"}:
        Handler.pipeline.ensure_index(args.documents)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logging.getLogger("rag").info(
        "event=server.started url=http://%s:%d documents=%s",
        args.host,
        args.port,
        args.documents.resolve(),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
