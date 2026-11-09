"""工作区/项目注册表：一个工作区包含文件、记忆和多个独立 session。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Workspace:
    workspace_id: str
    name: str
    path: Path
    created_at: str
    is_default: bool = False

    @property
    def memory_db(self) -> Path:
        return self.path / "agent_memory.sqlite3"

    def public(self) -> dict:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "path": str(self.path),
            "created_at": self.created_at,
            "is_default": self.is_default,
        }


class WorkspaceManager:
    """管理项目工作区；注册表只保存元数据，工作区内容留在各自目录。"""

    def __init__(self, project_root: Path, default_path: Path):
        self.project_root = Path(project_root).resolve()
        self.registry_path = self.project_root / ".agent_workspaces.json"
        self.projects_root = self.project_root / "workspaces"
        self.default_path = Path(default_path).resolve()
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._records = self._load()
        self._ensure_default()

    def _load(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self) -> None:
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._records, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.registry_path)

    def _ensure_default(self) -> None:
        self.default_path.mkdir(parents=True, exist_ok=True)
        if not any(item.get("workspace_id") == "default" for item in self._records):
            self._records.insert(0, {
                "workspace_id": "default",
                "name": "默认工作区",
                "path": str(self.default_path),
                "created_at": _now(),
                "is_default": True,
            })
            self._save()

    def _item(self, item: dict) -> Workspace:
        path = Path(item["path"]).resolve()
        return Workspace(
            workspace_id=item["workspace_id"],
            name=item["name"],
            path=path,
            created_at=item.get("created_at", ""),
            is_default=bool(item.get("is_default")),
        )

    def list_workspaces(self) -> list[Workspace]:
        return [self._item(item) for item in self._records]

    def public(self) -> list[dict]:
        return [item.public() for item in self.list_workspaces()]

    def get(self, workspace_id: str | None) -> Workspace:
        selected = workspace_id or "default"
        for item in self._records:
            if item.get("workspace_id") == selected:
                workspace = self._item(item)
                workspace.path.mkdir(parents=True, exist_ok=True)
                return workspace
        raise ValueError("工作区不存在")

    def create(self, name: str) -> Workspace:
        clean_name = re.sub(r"\s+", " ", str(name or "")).strip()
        if not clean_name or len(clean_name) > 80:
            raise ValueError("工作区名称不能为空且不能超过 80 个字符")
        workspace_id = "ws_" + uuid.uuid4().hex[:12]
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", clean_name).strip("-")[:40] or "project"
        path = (self.projects_root / f"{slug}-{workspace_id[3:]}").resolve()
        if not path.is_relative_to(self.projects_root.resolve()):
            raise ValueError("工作区路径无效")
        path.mkdir(parents=True, exist_ok=False)
        record = {
            "workspace_id": workspace_id,
            "name": clean_name,
            "path": str(path),
            "created_at": _now(),
            "is_default": False,
        }
        self._records.append(record)
        self._save()
        return self._item(record)
