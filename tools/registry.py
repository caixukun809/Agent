"""[新增-工具目录] 校验工具清单，选择 schema 并分发调用。"""

import json
from pathlib import Path


class ToolRegistry:
    def __init__(self, catalog_path: Path, runtime):
        self.entries = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.runtime = runtime
        if not isinstance(self.entries, list) or not self.entries:
            raise ValueError("工具清单不能为空")
        ids = [entry["id"] for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("工具 ID 重复")
        for entry in self.entries:
            if entry["schema"]["function"]["name"] != entry["id"]:
                raise ValueError(f"工具 ID 与 schema 名称不一致：{entry['id']}")
            if entry["id"] not in runtime.handlers:
                raise ValueError(f"工具缺少执行函数：{entry['id']}")

    def select(self, query: str, index, limit: int) -> list[dict]:
        searchable = [entry for entry in self.entries if not entry.get("always_available")]
        pinned = [entry for entry in self.entries if entry.get("always_available")]
        if index is None:
            print("[tool:search] 索引不可用，提供全部工具")
            selected = searchable
        else:
            try:
                candidate_count = min(len(self.entries), max(1, limit) + len(pinned))
                matches = index.search(query, candidate_count)
                if len(matches) != candidate_count:
                    raise ValueError("索引返回的工具数量与清单不一致")
                print("[tool:search] " + ", ".join(
                    f"{entry['id']}={score:.3f}" for entry, score in matches
                ))
                ranked = [entry for entry, _ in matches if not entry.get("always_available")]
                selected = ranked[:max(1, limit)]
                if len(selected) != min(max(1, limit), len(searchable)):
                    raise ValueError("索引没有返回足够的普通工具")
            except Exception as exc:
                print(f"[tool:search:error] {exc}；提供全部工具")
                selected = searchable
        schemas = [entry["schema"] for entry in [*selected, *pinned]]
        print("[tool:available] " + ", ".join(
            schema["function"]["name"] for schema in schemas
        ))
        return schemas

    def dispatch(self, name: str, raw_arguments: str, available: list[dict]) -> str:
        allowed = {schema["function"]["name"] for schema in available}
        if name not in allowed:
            result = f"错误：工具本轮未提供：{name}"
            print(f"[tool:error] {result}")
            return result
        return self.runtime.dispatch(name, raw_arguments)
