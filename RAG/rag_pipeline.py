"""企业级 RAG 最小可运行链路。

配置从项目根目录的 .env 读取；服务启动入口为 rag_server.py。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from dotenv import load_dotenv

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - the project requirements include faiss-cpu
    faiss = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(value: str | None, fallback: Path) -> Path:
    path = Path(value).expanduser() if value else fallback
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


DEFAULT_RUNTIME = _project_path(os.getenv("RAG_RUNTIME_DIR"), Path(__file__).resolve().parent / ".runtime")
DEFAULT_DOCUMENTS = _project_path(os.getenv("RAG_DOCUMENT_DIR"), Path(__file__).resolve().parent / "documents")
DEFAULT_EMBEDDING = _project_path(os.getenv("RAG_EMBEDDING_MODEL"), PROJECT_ROOT / "models" / "bge-small-zh-v1.5")
DEFAULT_RERANKER = _project_path(os.getenv("RAG_RERANKER_MODEL"), DEFAULT_RUNTIME / "reranker")
DEFAULT_RERANKER_ID = "BAAI/bge-reranker-base"
DEFAULT_GENERATOR_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".json", ".pdf", ".docx", ".pptx", ".xlsx"}
LOGGER = logging.getLogger("rag")


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure readable console logs and optional persistent UTF-8 log output."""
    logger = logging.getLogger("rag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        log_file = Path(log_file).resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _append_pipeline_step(
    steps: list[dict[str, Any]] | None,
    stage: str,
    label: str,
    started: float,
    detail: str,
    status: str = "completed",
) -> None:
    if steps is not None:
        steps.append({
            "stage": stage,
            "label": label,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "detail": detail,
        })


def _sha(text: str, size: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]


def _stable_id(*parts: str) -> str:
    return _sha("\x1f".join(parts))


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """中文使用字/双字混合 token，英文和编号使用词 token。"""
    text = unicodedata.normalize("NFKC", text.lower())
    tokens: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9_./:-]{1,}", text))
    return tokens


def _strip_html(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|h[1-6]|li|tr|table)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def _extract_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt", ".json"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return _strip_html(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc
        pages = []
        for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
            pages.append(f"\n[第 {page_no} 页]\n{page.extract_text() or ''}")
        return "\n".join(pages)
    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore
        except ImportError as exc:
            raise RuntimeError("解析 DOCX 需要安装 python-docx：pip install python-docx") from exc
        document = Document(str(path))
        blocks = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(blocks)
    if suffix == ".pptx":
        try:
            from pptx import Presentation  # type: ignore
        except ImportError as exc:
            raise RuntimeError("解析 PPTX 需要安装 python-pptx：pip install python-pptx") from exc
        presentation = Presentation(str(path))
        slides = []
        for slide_no, slide in enumerate(presentation.slides, start=1):
            parts = [shape.text for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()]
            slides.append(f"\n[第 {slide_no} 页]\n" + "\n".join(parts))
        return "\n".join(slides)
    if suffix == ".xlsx":
        try:
            from openpyxl import load_workbook  # type: ignore
        except ImportError as exc:
            raise RuntimeError("解析 XLSX 需要安装 openpyxl：pip install openpyxl") from exc
        workbook = load_workbook(str(path), read_only=True, data_only=True)
        sheets = []
        for sheet in workbook.worksheets:
            sheets.append(f"\n## 工作表：{sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if values:
                    sheets.append(" | ".join(values))
        return "\n".join(sheets)
    raise ValueError(f"不支持的文件类型：{path.suffix}")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    version_id: str
    source_path: str
    title: str
    section_path: list[str]
    text: str
    line_start: int
    line_end: int
    security_level: str = "internal"
    allowed_roles: list[str] | None = None
    effective_from: str | None = None
    effective_to: str | None = None


@dataclass
class Evidence:
    chunk_id: str
    text: str
    title: str
    source_path: str
    section_path: list[str]
    line_start: int
    line_end: int
    sparse_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    coarse_score: float = 0.0
    rerank_score: float = 0.0
    reranker_logit: float = 0.0
    final_score: float = 0.0
    retrieval_channels: list[str] | None = None


class HashEmbedding:
    """不依赖模型的离线 fallback，保证缺少 torch 时仍能跑通整条链路。"""

    dimension = 384

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype="float32")
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.embed(text) for text in texts], dtype="float32")


class LocalEmbedding:
    def __init__(self, model_path: Path):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_path = str(model_path)
        self.model = SentenceTransformer(self.model_path, local_files_only=True)
        self.dimension = int(self.model.get_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype="float32")


class BM25:
    """小型内置 BM25，避免额外依赖；索引可 JSON 持久化。"""

    def __init__(self, documents: Sequence[Sequence[str]] | None = None):
        self.documents = [list(d) for d in (documents or [])]
        self.doc_len = [len(d) for d in self.documents]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.idf: dict[str, float] = {}
        self.k1 = 1.5
        self.b = 0.75
        for index, tokens in enumerate(self.documents):
            counts = Counter(tokens)
            for token, count in counts.items():
                self.postings[token][index] = count
        n = len(self.documents)
        for token, posting in self.postings.items():
            df = len(posting)
            self.idf[token] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        scores = defaultdict(float)
        for token in tokenize(query):
            posting = self.postings.get(token, {})
            idf = self.idf.get(token, 0.0)
            for index, tf in posting.items():
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[index] / max(self.avgdl, 1e-6))
                scores[index] += idf * tf * (self.k1 + 1) / max(denom, 1e-6)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {"documents": self.documents}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BM25":
        return cls(payload.get("documents", []))


class LexicalReranker:
    """Cross-Encoder 不可用时的可解释精排：覆盖度、短语命中和长度惩罚。"""

    name = "lexical-fallback"

    def score(self, query: str, text: str) -> float:
        q = tokenize(query)
        d = set(tokenize(text))
        if not q:
            return 0.0
        coverage = sum(token in d for token in q) / len(q)
        phrase = 1.0 if query.strip().lower() in text.lower() else 0.0
        return min(1.0, 0.75 * coverage + 0.25 * phrase)


class EmbeddingReranker:
    """没有 Cross-Encoder 权重时复用本地 embedding 做语义精排。

    这不是 Cross-Encoder，只是比纯关键词 fallback 更接近语义排序；运行信息
    会明确标记为 embedding-reranker-fallback，避免把降级路径伪装成精排模型。
    """

    name = "embedding-reranker-fallback"

    def __init__(self, embedder: Any):
        self.embedder = embedder

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        query_vector = np.asarray(self.embedder.embed(query), dtype="float32")
        vectors = self.embedder.embed_many(texts)
        query_norm = np.linalg.norm(query_vector) or 1.0
        vector_norms = np.linalg.norm(vectors, axis=1)
        return (vectors @ query_vector / np.maximum(vector_norms * query_norm, 1e-8)).tolist()


class CrossEncoderReranker:
    def __init__(self, model_path: str | Path):
        from sentence_transformers import CrossEncoder  # type: ignore
        import torch  # type: ignore

        self.model_path = str(model_path)
        self.name = "BAAI/bge-reranker-base" if Path(model_path).name.lower() == "reranker" else f"cross-encoder:{Path(model_path).name}"
        self.model = CrossEncoder(self.model_path, local_files_only=True)
        self.activation = torch.nn.Identity()

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self.model.predict(
            [(query, text) for text in texts],
            show_progress_bar=False,
            activation_fn=self.activation,
        )
        return [float(score) for score in np.asarray(scores).reshape(-1)]


class ExtractiveGenerator:
    name = "extractive-local"

    def generate(self, query: str, evidence: Sequence[Evidence], history: Sequence[dict[str, str]] = ()) -> str:
        if not evidence:
            return "未找到足以支持该问题的授权证据，暂时无法确认。"
        lines = ["基于检索到的知识，相关信息如下："]
        for index, item in enumerate(evidence, start=1):
            snippet = re.sub(r"\s+", " ", item.text).strip()
            if len(snippet) > 260:
                snippet = snippet[:260].rstrip() + "…"
            lines.append(f"[{index}] {snippet}")
        lines.append("以上内容来自检索证据；如需正式结论，请以引用文档的最新有效版本为准。")
        return "\n".join(lines)


class OpenAIGenerator:
    name = "openai-compatible"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI  # type: ignore

        kwargs: dict[str, Any] = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        self.model = model
        self.client = OpenAI(**kwargs)

    def generate(self, query: str, evidence: Sequence[Evidence], history: Sequence[dict[str, str]] = ()) -> str:
        context = "\n\n".join(
            f"[{index}] {item.title} / {' > '.join(item.section_path)}\n{item.text}"
            for index, item in enumerate(evidence, start=1)
        )
        if evidence:
            system_prompt = "你是企业知识库助手。只能依据给出的企业文档证据回答，证据不足时明确说明无法确认；关键结论必须带 [编号] 引用。"
            user_prompt = f"问题：{query}\n\n证据：\n{context}"
        else:
            system_prompt = "你是企业知识库助手。本次检索没有找到达到相关性阈值的企业文档证据。请基于你的通用知识尽力回答；如果问题依赖企业内部信息或你不确定，请明确说明无法确认。不要编造引用，也不要声称答案来自知识库。"
            user_prompt = f"问题：{query}\n\n当前没有可引用的企业知识库证据。"
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in history[-12:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=messages,
        )
        return response.choices[0].message.content or "模型未返回答案。"


class LocalCausalGenerator:
    name = "local-causal"

    def __init__(self, model_path: str | Path):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
        self.model.eval()

    def generate(self, query: str, evidence: Sequence[Evidence], history: Sequence[dict[str, str]] = ()) -> str:
        context = "\n\n".join(f"[{i}] {e.text}" for i, e in enumerate(evidence, start=1))
        previous = "\n".join(
            f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
            for item in history[-12:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )
        if evidence:
            prompt = f"你是企业知识库助手，只能依据证据回答并引用编号。\n对话历史：\n{previous}\n问题：{query}\n证据：\n{context}\n答案："
        else:
            prompt = f"你是企业知识库助手。本次没有检索到达到相关性阈值的企业文档，请基于通用知识尽力回答；如果依赖企业内部信息或不确定，请明确说明。不得编造引用。\n对话历史：\n{previous}\n问题：{query}\n答案："
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def _split_structured(text: str, max_chars: int = 700, overlap: int = 100) -> list[tuple[str, list[str], int, int]]:
    lines = text.splitlines()
    title = "未命名文档"
    current_section: list[str] = []
    blocks: list[tuple[str, list[str], int, int]] = []
    buffer: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line
        content = _norm_text("\n".join(buffer))
        if not content:
            buffer = []
            start_line = end_line + 1
            return
        if len(content) <= max_chars:
            blocks.append((content, list(current_section), start_line, end_line))
        else:
            cursor = 0
            while cursor < len(content):
                end = min(len(content), cursor + max_chars)
                if end < len(content):
                    boundary = max(content.rfind("。", cursor, end), content.rfind("；", cursor, end), content.rfind(" ", cursor, end))
                    if boundary > cursor + max_chars // 2:
                        end = boundary + 1
                part = content[cursor:end].strip()
                if part:
                    part_start = start_line + content[:cursor].count("\n")
                    part_end = start_line + content[:end].count("\n")
                    blocks.append((part, list(current_section), part_start, max(part_start, part_end)))
                if end >= len(content):
                    break
                cursor = max(cursor + 1, end - overlap)
        buffer = []
        start_line = end_line + 1

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if heading:
            flush(line_no - 1)
            level = len(heading.group(1))
            value = heading.group(2).strip()
            if level == 1 and title == "未命名文档":
                title = value
            current_section = current_section[: level - 1] + [value]
            start_line = line_no + 1
            continue
        if not stripped:
            flush(line_no - 1)
            continue
        if not buffer:
            start_line = line_no
        buffer.append(line)
    flush(len(lines))
    return blocks, title


class RAGPipeline:
    def __init__(
        self,
        runtime: Path = DEFAULT_RUNTIME,
        embedding_path: Path = DEFAULT_EMBEDDING,
        reranker_path: Path | None = None,
        generator: str | None = None,
        generator_model: Path | None = None,
        openai_model: str | None = None,
        min_evidence_score: float | None = None,
        shared_models: tuple[Any, Any, Any] | None = None,
    ):
        self.runtime = Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.embedding_path = Path(embedding_path).expanduser().resolve()
        self.min_evidence_score = float(
            os.getenv("RAG_MIN_EVIDENCE_SCORE", "0.5")
            if min_evidence_score is None
            else min_evidence_score
        )
        if not 0.0 <= self.min_evidence_score <= 1.0:
            raise ValueError("RAG_MIN_EVIDENCE_SCORE 必须在 0 到 1 之间")
        self.chunks: list[Chunk] = []
        self.bm25 = BM25()
        self.dense_index: Any = None
        self.embedding_name = "hash-fallback"
        if shared_models:
            self.embedder, self.reranker, self.generator = shared_models
            self.embedding_name = getattr(self.embedder, "name", self.embedding_name)
            if self.embedding_name == "hash-fallback" and isinstance(self.embedder, LocalEmbedding):
                self.embedding_name = f"local:{self.embedding_path}"
        else:
            self.embedder = self._load_embedder()
            self.reranker = self._load_reranker(reranker_path or DEFAULT_RERANKER)
            self.generator = self._load_generator(
                generator or os.getenv("RAG_GENERATOR", "openai-compatible"),
                generator_model or (Path(os.environ["RAG_GENERATOR_MODEL"]) if os.getenv("RAG_GENERATOR_MODEL") else None),
                openai_model or os.getenv("RAG_LLM_MODEL") or os.getenv("OPENAI_MODEL"),
            )
        self.load()
        LOGGER.info(
            "event=models.ready embedding=%s reranker=%s generator=%s indexed_chunks=%d",
            self.embedding_name,
            getattr(self.reranker, "name", type(self.reranker).__name__),
            getattr(self.generator, "name", type(self.generator).__name__),
            len(self.chunks),
        )

    def _load_embedder(self):
        if self.embedding_path.exists():
            try:
                embedder = LocalEmbedding(self.embedding_path)
                self.embedding_name = f"local:{self.embedding_path}"
                return embedder
            except Exception as exc:
                print(f"[WARN] 本地 embedding 加载失败，使用 HashEmbedding：{exc}", file=sys.stderr)
        return HashEmbedding()

    def _load_reranker(self, reranker_path: Path | None):
        path = reranker_path or (self.runtime / "reranker")
        if path.exists() and _has_model_weights(path):
            try:
                return CrossEncoderReranker(path)
            except Exception as exc:
                print(f"[WARN] Cross-Encoder 权重不可用，使用本地 embedding 重排：{exc}", file=sys.stderr)
        if self.embedding_name.startswith("local:"):
            return EmbeddingReranker(self.embedder)
        return LexicalReranker()

    def _load_generator(self, kind: str, generator_model: Path | None, openai_model: str | None):
        if kind in {"local", "auto"} and generator_model and Path(generator_model).exists():
            try:
                return LocalCausalGenerator(generator_model)
            except Exception as exc:
                print(f"[WARN] 本地生成模型加载失败，使用抽取式生成：{exc}", file=sys.stderr)
        if kind in {"openai", "openai-compatible", "auto"} and os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIGenerator(
                    openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    os.getenv("RAG_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
                )
            except Exception as exc:
                print(f"[WARN] OpenAI 兼容模型不可用，使用抽取式生成：{exc}", file=sys.stderr)
        return ExtractiveGenerator()

    def _active_index_dir(self) -> Path:
        pointer = self.runtime / "active.json"
        if pointer.exists():
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            build_id = payload.get("active_build")
            if build_id:
                candidate = self.runtime / "builds" / build_id
                if candidate.is_dir():
                    return candidate
        return self.runtime

    def load(self) -> None:
        index_dir = self._active_index_dir()
        chunks_path = index_dir / "chunks.json"
        sparse_path = index_dir / "sparse.json"
        dense_path = index_dir / "dense.faiss"
        if not (chunks_path.exists() and sparse_path.exists() and dense_path.exists() and faiss):
            return
        self.chunks = [Chunk(**item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
        self.bm25 = BM25.from_dict(json.loads(sparse_path.read_text(encoding="utf-8")))
        index = faiss.read_index(str(dense_path))
        if index.d != int(self.embedder.dimension):
            print(f"[WARN] 已有向量索引维度 {index.d} 与当前 embedding {self.embedder.dimension} 不同，请重新 ingest。", file=sys.stderr)
            return
        self.dense_index = index

    def ingest(self, source: Path, activate: bool = True) -> dict[str, Any]:
        started = time.perf_counter()
        source = Path(source).resolve()
        if source.is_file():
            files = [source] if source.suffix.lower() in SUPPORTED_SUFFIXES else []
        else:
            files = _document_files(source)
        chunks: list[Chunk] = []
        errors: list[dict[str, str]] = []
        for path in sorted(files):
            try:
                raw = _norm_text(_extract_file(path))
                if not raw:
                    continue
                title = path.stem
                blocks, parsed_title = _split_structured(raw)
                if parsed_title != "未命名文档":
                    title = parsed_title
                document_id = _stable_id(str(path), str(path.stat().st_size))
                version_id = f"{document_id}:{_sha(raw)}"
                for ordinal, (text, section, line_start, line_end) in enumerate(blocks):
                    chunk_id = f"{document_id}:c{ordinal:04d}"
                    chunks.append(Chunk(chunk_id, document_id, version_id, str(path), title, section, text, line_start, line_end))
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        if not chunks:
            raise RuntimeError(f"没有解析出可索引内容：{source}")
        source_fingerprint = _fingerprint_files(source, files)
        LOGGER.info("event=ingest.parsed source=%s files=%d chunks=%d errors=%d", source, len(files), len(chunks), len(errors))
        self.chunks = chunks
        self.bm25 = BM25([tokenize(item.text + " " + item.title + " " + " ".join(item.section_path)) for item in chunks])
        if not faiss:
            raise RuntimeError("缺少 faiss-cpu，无法创建向量索引")
        vectors = self.embedder.embed_many([item.text for item in chunks])
        faiss.normalize_L2(vectors)
        self.dense_index = faiss.IndexFlatIP(vectors.shape[1])
        self.dense_index.add(vectors)
        build_id = time.strftime("build_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}_" + _sha(str(source) + str(len(chunks)), 6)
        build_dir = self.runtime / "builds" / build_id
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "chunks.json").write_text(json.dumps([asdict(item) for item in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        (build_dir / "sparse.json").write_text(json.dumps(self.bm25.to_dict(), ensure_ascii=False), encoding="utf-8")
        faiss.write_index(self.dense_index, str(build_dir / "dense.faiss"))
        manifest = {
            "build_id": build_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": str(source),
            "source_fingerprint": source_fingerprint,
            "chunk_count": len(chunks),
            "file_count": len(files),
            "errors": errors,
            "embedding": self.embedding_name,
            "dimension": int(self.embedder.dimension),
            "reranker": getattr(self.reranker, "name", type(self.reranker).__name__),
            "generator": getattr(self.generator, "name", type(self.generator).__name__),
            "status": "active" if activate else "pending_approval",
        }
        (build_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if activate:
            self._activate_build(build_id)
        LOGGER.info(
            "event=ingest.completed build_id=%s status=%s chunks=%d dimension=%d elapsed_ms=%.1f",
            build_id,
            manifest["status"],
            len(chunks),
            int(self.embedder.dimension),
            (time.perf_counter() - started) * 1000,
        )
        return manifest

    def ensure_index(self, source: Path) -> dict[str, Any] | None:
        """Build or refresh the active index only when the configured documents change."""
        source = Path(source).expanduser().resolve()
        source.mkdir(parents=True, exist_ok=True)
        files = _document_files(source)
        if not files:
            if self.chunks:
                LOGGER.warning("event=ingest.skipped reason=no_supported_documents source=%s keeping_existing_chunks=%d", source, len(self.chunks))
                return None
            raise RuntimeError(f"文档目录没有可索引文件：{source}（支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}）")
        current_fingerprint = _fingerprint_files(source, files)
        manifest_path = self._active_index_dir() / "manifest.json"
        if self.chunks and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_fingerprint") == current_fingerprint and manifest.get("embedding") == self.embedding_name:
                LOGGER.info("event=ingest.skipped reason=index_up_to_date build_id=%s files=%d chunks=%d", manifest.get("build_id"), len(files), len(self.chunks))
                return None
        LOGGER.info("event=ingest.refresh source=%s files=%d reason=missing_or_changed_index", source, len(files))
        return self.ingest(source, activate=True)

    def _activate_build(self, build_id: str, approver: str | None = None) -> dict[str, Any]:
        build_dir = self.runtime / "builds" / build_id
        manifest_path = build_dir / "manifest.json"
        if not manifest_path.exists() or not (build_dir / "dense.faiss").exists():
            raise RuntimeError(f"找不到完整的候选索引版本：{build_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"pending_approval", "active", "approved"}:
            raise RuntimeError(f"索引版本状态不允许发布：{manifest.get('status')}")
        manifest["status"] = "active"
        manifest["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["approved_by"] = approver or "local-operator"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pointer_path = self.runtime / "active.json"
        temp_path = self.runtime / "active.json.tmp"
        temp_path.write_text(json.dumps({"active_build": build_id, "activated_at": manifest["approved_at"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(pointer_path)
        self.load()
        LOGGER.info("event=index.activated build_id=%s approver=%s chunks=%d", build_id, manifest["approved_by"], len(self.chunks))
        return manifest

    def _allowed(self, item: Chunk, role: str | None) -> bool:
        return not item.allowed_roles or (role is not None and role in item.allowed_roles)

    @staticmethod
    def _norm_scores(values: Sequence[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high - low < 1e-8:
            return [1.0 if high > 0 else 0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    def _normalize_rerank_scores(self, values: Sequence[float]) -> list[float]:
        if isinstance(self.reranker, CrossEncoderReranker):
            clipped = np.clip(np.asarray(values, dtype="float32"), -20.0, 20.0)
            return (1.0 / (1.0 + np.exp(-clipped))).tolist()
        if isinstance(self.reranker, EmbeddingReranker):
            return [min(1.0, max(0.0, (float(value) + 1.0) / 2.0)) for value in values]
        return [min(1.0, max(0.0, float(value))) for value in values]

    def search(
        self,
        query: str,
        top_k: int = 5,
        role: str | None = None,
        recall_k: int = 50,
        coarse_k: int = 20,
        trace_id: str | None = None,
        pipeline_steps: list[dict[str, Any]] | None = None,
        allowed_sources: set[str] | None = None,
    ) -> list[Evidence]:
        trace_id = trace_id or uuid.uuid4().hex
        if not self.chunks or self.dense_index is None:
            _append_pipeline_step(pipeline_steps, "retrieval", "知识库检索", time.perf_counter(), "当前租户暂无可用索引", "empty")
            _append_pipeline_step(pipeline_steps, "threshold", "引用阈值校验", time.perf_counter(), "无候选片段 · 不附加知识库引用", "empty")
            return []
        stage_started = time.perf_counter()
        # ACL filtering must happen before recall truncation; otherwise private
        # chunks can crowd authorized chunks out of a user's top candidates.
        sparse = self.bm25.search(query, len(self.chunks) if allowed_sources is not None else recall_k * 2)
        sparse_elapsed = (time.perf_counter() - stage_started) * 1000
        _append_pipeline_step(pipeline_steps, "keyword", "关键词召回", stage_started, f"BM25 返回 {len(sparse)} 个候选")
        LOGGER.info("trace_id=%s stage=keyword_retrieval candidates=%d elapsed_ms=%.1f", trace_id, len(sparse), sparse_elapsed)
        stage_started = time.perf_counter()
        query_vector = np.asarray([self.embedder.embed(query)], dtype="float32")
        faiss.normalize_L2(query_vector)
        dense_limit = len(self.chunks) if allowed_sources is not None else min(recall_k * 2, len(self.chunks))
        dense_scores, dense_ids = self.dense_index.search(query_vector, dense_limit)
        dense = [(int(i), float(s)) for i, s in zip(dense_ids[0], dense_scores[0]) if i >= 0]
        _append_pipeline_step(pipeline_steps, "vector", "向量召回", stage_started, f"BGE 向量检索返回 {len(dense)} 个候选")
        LOGGER.info("trace_id=%s stage=vector_retrieval candidates=%d elapsed_ms=%.1f", trace_id, len(dense), (time.perf_counter() - stage_started) * 1000)
        stage_started = time.perf_counter()
        candidates: dict[int, Evidence] = {}
        sparse_rank = {index: rank for rank, (index, _) in enumerate(sparse, start=1)}
        dense_rank = {index: rank for rank, (index, _) in enumerate(dense, start=1)}
        sparse_score = dict(sparse)
        dense_score = dict(dense)
        for index in set(sparse_rank) | set(dense_rank):
            item = self.chunks[index]
            if not self._allowed(item, role):
                continue
            if allowed_sources is not None and str(Path(item.source_path).resolve()) not in allowed_sources:
                continue
            channels: list[str] = []
            if index in sparse_rank:
                channels.append("sparse")
            if index in dense_rank:
                channels.append("dense")
            evidence = Evidence(item.chunk_id, item.text, item.title, item.source_path, item.section_path, item.line_start, item.line_end, sparse_score.get(index, 0.0), dense_score.get(index, 0.0), sum(1 / (60 + rank) for rank in (sparse_rank.get(index, 10**6), dense_rank.get(index, 10**6))), retrieval_channels=channels)
            candidates[index] = evidence
        ordered = sorted(candidates.items(), key=lambda pair: pair[1].rrf_score, reverse=True)
        ordered = ordered[:coarse_k]
        LOGGER.info("trace_id=%s stage=hybrid_fusion candidates=%d elapsed_ms=%.1f", trace_id, len(ordered), (time.perf_counter() - stage_started) * 1000)
        _append_pipeline_step(pipeline_steps, "fusion", "混合融合", stage_started, f"BM25 {len(sparse)} · 向量 {len(dense)} · 合并后 {len(ordered)}")
        stage_started = time.perf_counter()
        sparse_norm = self._norm_scores([item.sparse_score for _, item in ordered])
        dense_norm = self._norm_scores([item.dense_score for _, item in ordered])
        for (index, item), sn, dn in zip(ordered, sparse_norm, dense_norm):
            coverage = LexicalReranker().score(query, item.text)
            item.coarse_score = 0.45 * item.rrf_score + 0.25 * sn + 0.2 * dn + 0.1 * coverage
        ordered.sort(key=lambda pair: pair[1].coarse_score, reverse=True)
        LOGGER.info("trace_id=%s stage=coarse_ranking candidates=%d elapsed_ms=%.1f", trace_id, len(ordered), (time.perf_counter() - stage_started) * 1000)
        _append_pipeline_step(pipeline_steps, "coarse", "候选粗排", stage_started, f"综合排序后保留 {len(ordered)} 个候选")
        rerank_items = ordered[: min(coarse_k, 30)]
        stage_started = time.perf_counter()
        if not rerank_items:
            reranker_logits = []
        elif isinstance(self.reranker, CrossEncoderReranker) or isinstance(self.reranker, EmbeddingReranker):
            rerank_scores = self.reranker.score_pairs(query, [item.text for _, item in rerank_items])
            reranker_logits = [float(score) for score in rerank_scores]
        else:
            reranker_logits = [float(self.reranker.score(query, item.text)) for _, item in rerank_items]
        rerank_scores = self._normalize_rerank_scores(reranker_logits)
        above_threshold = sum(score >= self.min_evidence_score for score in rerank_scores)
        LOGGER.info(
            "trace_id=%s stage=reranking model=%s candidates=%d above_threshold=%d threshold=%.4f max_relevance=%.4f elapsed_ms=%.1f",
            trace_id,
            getattr(self.reranker, "name", type(self.reranker).__name__),
            len(rerank_items),
            above_threshold,
            self.min_evidence_score,
            max(rerank_scores, default=0.0),
            (time.perf_counter() - stage_started) * 1000,
        )
        _append_pipeline_step(
            pipeline_steps,
            "rerank",
            "Cross-Encoder 精排",
            stage_started,
            f"{getattr(self.reranker, 'name', type(self.reranker).__name__)} 评分 {len(rerank_items)} 条，最高相关度 {max(rerank_scores, default=0.0):.4f}",
        )
        stage_started = time.perf_counter()
        final: list[Evidence] = []
        for (_, item), raw_score, score in zip(rerank_items, reranker_logits, rerank_scores):
            item.reranker_logit = raw_score
            item.rerank_score = float(score)
            item.final_score = float(score)
            if score >= self.min_evidence_score:
                final.append(item)
        final.sort(key=lambda item: item.final_score, reverse=True)
        selected = final[:top_k]
        LOGGER.info(
            "trace_id=%s stage=threshold_gate selected=%d rejected=%d threshold=%.4f top_relevance=%.4f elapsed_ms=%.1f",
            trace_id,
            len(selected),
            len(rerank_items) - above_threshold,
            self.min_evidence_score,
            selected[0].final_score if selected else 0.0,
            (time.perf_counter() - stage_started) * 1000,
        )
        _append_pipeline_step(
            pipeline_steps,
            "threshold",
            "引用阈值校验",
            stage_started,
            f"达标 {above_threshold}/{len(rerank_items)} · 阈值 {self.min_evidence_score:.2f} · 最终引用 {len(selected)} 条",
            "completed" if selected else "empty",
        )
        return selected

    def answer(
        self,
        query: str,
        top_k: int = 5,
        role: str | None = None,
        history: Sequence[dict[str, str]] = (),
        allowed_sources: set[str] | None = None,
    ) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex
        request_started = time.perf_counter()
        LOGGER.info("trace_id=%s event=request.started query_chars=%d top_k=%d", trace_id, len(query), top_k)
        pipeline_steps: list[dict[str, Any]] = []
        evidence = self.search(
            query,
            top_k=top_k,
            role=role,
            trace_id=trace_id,
            pipeline_steps=pipeline_steps,
            allowed_sources=allowed_sources,
        )
        confidence = 0.0 if not evidence else round(evidence[0].final_score, 4)
        generation_started = time.perf_counter()
        generation_fallback = False
        try:
            answer = self.generator.generate(query, evidence, history=history)
        except Exception as exc:
            LOGGER.exception("trace_id=%s event=generation.failed fallback=extractive", trace_id)
            answer = ExtractiveGenerator().generate(query, evidence)
            generation_fallback = True
        if not evidence:
            answer = "未检索到可引用的相关资料。以下回答基于大模型已有知识，未经知识库验证。\n\n" + answer
        LOGGER.info(
            "trace_id=%s stage=generation model=%s elapsed_ms=%.1f",
            trace_id,
            getattr(self.generator, "name", type(self.generator).__name__),
            (time.perf_counter() - generation_started) * 1000,
        )
        generator_name = getattr(self.generator, "model", getattr(self.generator, "name", type(self.generator).__name__))
        if generation_fallback:
            generation_detail = f"模型调用失败，已切换抽取式回答 · 检索证据 {len(evidence)} 条"
        elif evidence:
            generation_detail = f"{generator_name} · 使用证据 {len(evidence)} 条"
        else:
            generation_detail = f"{generator_name} · 无证据，按通用知识回答"
        _append_pipeline_step(
            pipeline_steps,
            "generation",
            "大模型生成",
            generation_started,
            generation_detail,
            "warning" if generation_fallback else "completed",
        )
        citations = [
            {
                "evidence_id": f"E{index}",
                "chunk_id": item.chunk_id,
                "title": item.title,
                "source": item.source_path,
                "section": item.section_path,
                "lines": [item.line_start, item.line_end],
                "quote": item.text,
                "score": round(item.final_score, 4),
                "scores": {
                    "bm25": round(item.sparse_score, 4),
                    "vector": round(item.dense_score, 4),
                    "rrf": round(item.rrf_score, 6),
                    "coarse": round(item.coarse_score, 4),
                    "reranker_raw_logit": round(item.reranker_logit, 4),
                    "reranker": round(item.rerank_score, 4),
                    "final": round(item.final_score, 4),
                },
                "channels": item.retrieval_channels,
            }
            for index, item in enumerate(evidence, start=1)
        ]
        pointer = self.runtime / "active.json"
        index_version = json.loads(pointer.read_text(encoding="utf-8")).get("active_build") if pointer.exists() else "legacy"
        LOGGER.info(
            "trace_id=%s event=request.completed index_version=%s evidence=%d confidence=%.4f elapsed_ms=%.1f",
            trace_id,
            index_version,
            len(evidence),
            confidence,
            (time.perf_counter() - request_started) * 1000,
        )
        return {
            "trace_id": trace_id,
            "index_version": index_version,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "confidence_type": "uncalibrated_relevance_score",
            "confidence_note": "Cross-Encoder 原始 logit 经 sigmoid 映射后的相关性分数；不是经业务数据校准的概率。",
            "citation_threshold": self.min_evidence_score,
            "evidence_status": "qualified" if evidence else "below_threshold",
            "need_human_review": not evidence,
            "pipeline_steps": pipeline_steps,
            "retrieval": {
                "top_k": top_k,
                "reranker": getattr(self.reranker, "name", type(self.reranker).__name__),
                "generator": getattr(self.generator, "name", type(self.generator).__name__),
                "embedding": self.embedding_name,
                "min_evidence_score": self.min_evidence_score,
            },
        }


def _download_model(model_id: str, target: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError("下载模型需要 huggingface-hub，请先安装项目 requirements") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(repo_id=model_id, local_dir=str(target))
    return Path(downloaded)


def _has_model_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "model.onnx")) or any(path.glob("*.safetensors"))


def _document_files(source: Path) -> list[Path]:
    source = source.resolve()
    tenant_root = (DEFAULT_DOCUMENTS / "tenants").resolve()
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and DEFAULT_RUNTIME not in path.parents
        and not (source == DEFAULT_DOCUMENTS and path.resolve().is_relative_to(tenant_root))
    )


def _fingerprint_files(source: Path, files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        try:
            name = path.relative_to(source).as_posix()
        except ValueError:
            name = path.name
        digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="企业级 RAG 最小可运行链路")
    parser.add_argument("command", choices=["ingest", "ask", "run", "approve", "eval"], help="执行建库、问答、审批或评测")
    parser.add_argument("--source", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--question", default="企业级 RAG 的基本链路是什么？")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--role", default=None)
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--reranker", type=Path, default=None)
    parser.add_argument("--generator", choices=["auto", "extractive", "local", "openai", "openai-compatible"], default=os.getenv("RAG_GENERATOR", "openai-compatible"))
    parser.add_argument("--generator-model", type=Path, default=None)
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--download-models", action="store_true", help="显式下载缺少的 reranker/生成模型；离线环境不要开启")
    parser.add_argument("--candidate", action="store_true", help="仅构建候选索引，待审批后发布")
    parser.add_argument("--build-id", default=None, help="approve 命令要发布的候选索引版本")
    parser.add_argument("--approver", default="local-operator", help="本地审批人标识")
    parser.add_argument("--eval-set", type=Path, default=None, help="JSONL 评测集，字段包含 question 和 expected_chunk_id(s)")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=os.getenv("RAG_LOG_LEVEL", "INFO").upper())
    parser.add_argument("--log-file", type=Path, default=_project_path(os.getenv("RAG_LOG_FILE"), DEFAULT_RUNTIME / "logs" / "rag.log") if os.getenv("RAG_LOG_FILE") else None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    runtime = args.runtime.resolve()
    reranker = args.reranker
    if args.download_models and reranker is None:
        reranker = runtime / "reranker"
        if not _has_model_weights(reranker):
            print(f"[INFO] 下载 Cross-Encoder：{DEFAULT_RERANKER_ID}")
            _download_model(DEFAULT_RERANKER_ID, reranker)
    generator_model = args.generator_model
    if args.download_models and args.generator == "local" and generator_model is None:
        generator_model = runtime / "generator"
        if not _has_model_weights(generator_model):
            print(f"[INFO] 下载本地生成模型：{DEFAULT_GENERATOR_ID}")
            _download_model(DEFAULT_GENERATOR_ID, generator_model)
    pipeline = RAGPipeline(runtime, args.embedding, reranker, args.generator, generator_model, args.openai_model)
    if args.command == "approve":
        if not args.build_id:
            raise SystemExit("approve 命令需要 --build-id")
        print(json.dumps(pipeline._activate_build(args.build_id, args.approver), ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval":
        if not args.eval_set:
            raise SystemExit("eval 命令需要 --eval-set")
        rows = [json.loads(line) for line in args.eval_set.read_text(encoding="utf-8").splitlines() if line.strip()]
        hit_count = 0
        reciprocal_rank_sum = 0.0
        evaluated = 0
        for row in rows:
            expected = row.get("expected_chunk_ids", row.get("expected_chunk_id", []))
            expected_ids = {expected} if isinstance(expected, str) else set(expected)
            if not expected_ids:
                continue
            results = pipeline.search(row["question"], top_k=args.top_k, role=row.get("role"))
            ranks = [rank for rank, item in enumerate(results, start=1) if item.chunk_id in expected_ids]
            evaluated += 1
            if ranks:
                hit_count += 1
                reciprocal_rank_sum += 1 / ranks[0]
        report = {
            "sample_count": len(rows),
            "evaluated_count": evaluated,
            "recall_at_k": round(hit_count / evaluated, 4) if evaluated else 0.0,
            "mrr_at_k": round(reciprocal_rank_sum / evaluated, 4) if evaluated else 0.0,
            "top_k": args.top_k,
            "index": str(pipeline._active_index_dir()),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"ingest", "run"}:
        manifest = pipeline.ingest(args.source, activate=not args.candidate)
        if args.command == "ingest":
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
    result = pipeline.answer(args.question, top_k=args.top_k, role=args.role)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        print("\n引用：")
        for citation in result["citations"]:
            print(f"- {citation['evidence_id']} {citation['title']} {citation['source']} 行 {citation['lines'][0]}-{citation['lines'][1]}")
        print(f"\n置信度：{result['confidence']}，重排：{result['retrieval']['reranker']}，生成：{result['retrieval']['generator']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
