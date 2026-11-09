"""[新增-向量] FAISS 本地向量索引和两种 embedding 实现。"""

import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np


class HashEmbedding:
    """[新增-向量] 无需网络的确定性向量，适合本地测试和中转站环境。"""

    dimension = 384

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype="float32")
        lowered = text.lower()
        tokens = re.findall(r"[A-Za-z0-9_]{2,}", lowered)
        for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
            tokens.extend(run)
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()


class OpenAIEmbedding:
    """[新增-向量] 使用 OpenAI 兼容 embeddings 接口，模型由环境变量指定。"""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.dimension = None

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        vector = response.data[0].embedding
        self.dimension = len(vector)
        return vector


class LocalEmbedding:
    """[新增-向量] 使用本地 SentenceTransformer 模型生成语义向量。"""

    def __init__(self, model_path: str | Path):
        from sentence_transformers import SentenceTransformer

        self.model_path = str(model_path)
        self.model = SentenceTransformer(self.model_path)
        self.dimension = self.model.get_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class VectorStore:
    """[新增-向量] FAISS 只做候选召回，SQLite 负责正文、权限和状态过滤。"""

    def __init__(self, root: Path, embedder):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "memory_vectors.faiss"
        self.meta_path = self.root / "memory_vectors.meta.json"
        self.embedder = embedder
        self.index = None
        self.load_error = None
        self._load()

    @property
    def ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def _load(self) -> None:
        """[新增-向量] 启动时恢复磁盘索引；维度不匹配时拒绝混用。"""
        if not self.index_path.exists():
            return
        index = faiss.read_index(str(self.index_path))
        expected = getattr(self.embedder, "dimension", None)
        if expected is not None and index.d != expected:
            self.load_error = f"向量维度不匹配：索引为 {index.d}，当前 embedding 为 {expected}，请执行 /reindex。"
            return
        self.index = index

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None:
            base = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIDMap2(base)
        elif self.index.d != dimension:
            raise ValueError(f"向量维度不匹配：索引为 {self.index.d}，新向量为 {dimension}。")

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(
                {
                    "dimension": self.index.d,
                    "count": self.index.ntotal,
                    "embedding": type(self.embedder).__name__,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def upsert(self, memory_id: int, text: str) -> None:
        """[新增-向量] 新增或替换一个长期记忆的向量。"""
        vector = np.asarray([self.embedder.embed(text)], dtype="float32")
        faiss.normalize_L2(vector)
        self._ensure_index(vector.shape[1])
        if self.index.ntotal:
            self.index.remove_ids(np.asarray([memory_id], dtype="int64"))
        self.index.add_with_ids(vector, np.asarray([memory_id], dtype="int64"))
        self._save()

    def delete(self, memory_id: int) -> None:
        """[新增-长期记忆-删除机制] 从 FAISS 中删除已遗忘记忆。"""
        if self.index is None or not self.index.ntotal:
            return
        self.index.remove_ids(np.asarray([memory_id], dtype="int64"))
        if self.index.ntotal:
            self._save()
        else:
            self.index = None
            for path in (self.index_path, self.meta_path):
                if path.exists():
                    path.unlink()

    def search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """[新增-向量] 返回 memory_id 和 cosine 相似度候选。"""
        if not self.ready:
            return []
        vector = np.asarray([self.embedder.embed(query)], dtype="float32")
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, min(limit, self.index.ntotal))
        return [
            (int(memory_id), float(score))
            for memory_id, score in zip(ids[0], scores[0])
            if memory_id != -1
        ]

    def rebuild(self, items: list[tuple[int, str]]) -> int:
        """[新增-向量] 从 SQLite active 记忆重建索引，用于升级 embedding 模型。"""
        self.index = None
        if not items:
            for path in (self.index_path, self.meta_path):
                if path.exists():
                    path.unlink()
            return 0
        for memory_id, content in items:
            self.upsert(memory_id, content)
        return len(items)
