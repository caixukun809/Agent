"""[新增-工具目录] 工具的实际 Python 执行逻辑。"""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


class ToolRuntime:
    def __init__(self, workspace_root: Path, skill_catalog, max_file_size=100_000, max_output=4000):
        self.workspace_root = Path(workspace_root).resolve()
        self.skill_catalog = skill_catalog
        self.max_file_size = max_file_size
        self.max_output = max_output
        self.handlers = {
            "get_current_time": lambda args: datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_python": lambda args: self.run_python(args.get("code", "")),
            "create_file": lambda args: self.create_file(args.get("path", ""), args.get("content", "")),
            "write_file": lambda args: self.write_file(args.get("path", ""), args.get("content", "")),
            "read_skill": lambda args: self.skill_catalog.read(args.get("name", "")),
        }

    def _truncate(self, text: str) -> str:
        return text[:self.max_output] + "\n...(输出已截断)" if len(text) > self.max_output else text

    def _workspace_path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("文件路径不能为空")
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("只能使用工作空间内的相对路径")
        candidate = (self.workspace_root / requested).resolve()
        if not candidate.is_relative_to(self.workspace_root):
            raise ValueError("路径不能离开工作空间")
        return candidate

    def _check_content(self, content: str) -> int:
        if not isinstance(content, str):
            raise ValueError("content 必须是字符串")
        size = len(content.encode("utf-8"))
        if size > self.max_file_size:
            raise ValueError(f"文件不能超过 {self.max_file_size} 字节")
        return size

    def create_file(self, relative_path: str, content: str) -> str:
        size = self._check_content(content)
        path = self._workspace_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="") as file:
                file.write(content)
        except FileExistsError:
            return f"错误：文件已存在，未覆盖：{relative_path}"
        return f"已创建文件：{relative_path}（{size} 字节）"

    def write_file(self, relative_path: str, content: str) -> str:
        size = self._check_content(content)
        path = self._workspace_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return f"已写入文件：{relative_path}（{size} 字节）"

    def run_python(self, code: str) -> str:
        if not isinstance(code, str) or not code.strip():
            return "错误：code 为空。"
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return "执行超时（超过 5 秒）。"
        output = (completed.stdout or "") + (completed.stderr or "")
        return self._truncate(output) if output else f"（无输出，返回码 {completed.returncode}）"

    def dispatch(self, name: str, raw_arguments: str) -> str:
        print(f"\n[tool:start] {name}")
        try:
            args = json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("工具参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            result = f"错误：工具参数无效（{exc}）。"
            print(f"[tool:error] {result}")
            return result
        print(f"[tool:args] {json.dumps(args, ensure_ascii=False)}")
        handler = self.handlers.get(name)
        if handler is None:
            result = f"未知工具：{name}"
            print(f"[tool:error] {result}")
            return result
        try:
            result = self._truncate(str(handler(args)))
        except Exception as exc:
            result = f"工具执行出错：{exc}"
        print(f"[tool:result] {result}")
        print(f"[tool:end] {name}")
        return result
