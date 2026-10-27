"""[新增-Skill] 从本地清单发现 skill，按需读取正文。"""

import json
from pathlib import Path


class SkillCatalog:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        manifest = self.root / "manifest.json"
        self.entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
        names = [entry["name"] for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("skill 名称重复")

    def descriptions(self) -> str:
        return "\n".join(
            f"- {entry['name']}: {entry['description']}" for entry in self.entries
        )

    def read(self, name: str) -> str:
        entry = next((item for item in self.entries if item["name"] == name), None)
        if entry is None:
            return f"未知 skill：{name}"
        path = (self.root / name / "SKILL.md").resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            return f"skill 文件不存在：{name}"
        return path.read_text(encoding="utf-8")
