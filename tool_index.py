"""[新增-工具索引] FAISS 保存工具向量，JSON 保存可核对的工具 ID 映射。"""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np


class ToolIndex:
    def __init__(self, root: Path, embedder, entries: list[dict], embedding_id: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "tool_vectors.faiss"
        self.meta_path = self.root / "tool_vectors.meta.json"
        self.embedder = embedder
        self.entries = entries
        self.embedding_id = embedding_id
        self.by_vector_id = {}
        for entry in entries:
            vector_id = self._vector_id(entry["id"])
            if vector_id in self.by_vector_id:
                raise ValueError(f"工具向量 ID 冲突：{entry['id']}")
            self.by_vector_id[vector_id] = entry
        self.items = [
            {
                "id": entry["id"],
                "vector_id": self._vector_id(entry["id"]),
                "description": entry["schema"]["function"]["description"],
                "search_text": entry["search_text"],
            }
            for entry in entries
        ]
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {"embedding_id": embedding_id, "entries": entries},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.index = None
        self._load_or_build()

    @staticmethod
    def _vector_id(name: str) -> int:
        return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)

    @staticmethod
    def _search_text(entry: dict) -> str:
        function = entry["schema"]["function"]
        parameters = function["parameters"].get("properties", {})
        fields = " ".join(
            f"{name} {spec.get('description', '')}" for name, spec in parameters.items()
        )
        return f"{entry['id']} {function['description']} {entry['search_text']} {fields}"

    def _load_or_build(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                index = faiss.read_index(str(self.index_path))
                expected_dimension = getattr(self.embedder, "dimension", None)
                if (
                    meta.get("fingerprint") == self.fingerprint
                    and meta.get("tools") == self.items
                    and index.ntotal == len(self.entries)
                    and (expected_dimension is None or index.d == expected_dimension)
                    and set(faiss.vector_to_array(index.id_map).tolist()) == set(self.by_vector_id)
                ):
                    self.index = index
                    print(f"[tool:index] 已加载 {index.ntotal} 个工具向量")
                    return
            except (AttributeError, OSError, ValueError, RuntimeError) as exc:
                print(f"[tool:index] 旧索引读取失败，准备重建：{exc}")
        self.rebuild()

    def rebuild(self) -> int:
        vectors = np.asarray(
            [self.embedder.embed(self._search_text(entry)) for entry in self.entries],
            dtype="float32",
        )
        if vectors.ndim != 2 or len(vectors) != len(self.entries):
            raise ValueError("工具 embedding 返回了无效的向量")
        faiss.normalize_L2(vectors)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        ids = np.asarray([item["vector_id"] for item in self.items], dtype="int64")
        index.add_with_ids(vectors, ids)
        faiss.write_index(index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "embedding_id": self.embedding_id,
                    "dimension": index.d,
                    "count": index.ntotal,
                    "tools": self.items,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.index = index
        print(f"[tool:index] 已建立 {index.ntotal} 个工具向量")
        return index.ntotal

    def search(self, query: str, limit: int) -> list[tuple[dict, float]]:
        vector = np.asarray([self.embedder.embed(query)], dtype="float32")
        if vector.shape[1] != self.index.d:
            raise ValueError("工具索引维度与当前 embedding 不一致，请重建索引")
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, min(max(1, limit), len(self.entries)))
        return [
            (self.by_vector_id[int(vector_id)], float(score))
            for vector_id, score in zip(ids[0], scores[0])
            if int(vector_id) in self.by_vector_id
        ]
