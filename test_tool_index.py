"""[新增-测试] 工具向量索引与 skill 读取的离线验证。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from skills import SkillCatalog
from tool_index import ToolIndex
from tools.registry import ToolRegistry
from tools.runtime import ToolRuntime
from vector_store import HashEmbedding


class ToolIndexTests(unittest.TestCase):
    def make_registry(self, workspace: Path) -> ToolRegistry:
        project = Path(__file__).resolve().parent
        skills = SkillCatalog(project / "skill")
        runtime = ToolRuntime(workspace, skills)
        return ToolRegistry(project / "tools" / "catalog.json", runtime)

    def test_build_reload_and_schema_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_registry(root)
            embedder = HashEmbedding()
            index = ToolIndex(root, embedder, registry.entries, "hash:test")
            self.assertEqual(index.index.ntotal, len(registry.entries))
            self.assertIn(
                "create_file",
                [entry["id"] for entry, _ in index.search("创建 Python 文件", 4)],
            )
            metadata = json.loads((root / "tool_vectors.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {item["id"] for item in metadata["tools"]},
                {entry["id"] for entry in registry.entries},
            )
            self.assertEqual(len({item["vector_id"] for item in metadata["tools"]}), len(registry.entries))

            cached = ToolIndex(root, embedder, registry.entries, "hash:test")
            self.assertEqual(cached.fingerprint, index.fingerprint)
            self.assertEqual(cached.index.ntotal, len(registry.entries))

            changed_tools = copy.deepcopy(registry.entries)
            changed_tools[0]["schema"]["function"]["description"] += " 测试变更"
            rebuilt = ToolIndex(root, embedder, changed_tools, "hash:test")
            self.assertNotEqual(rebuilt.fingerprint, index.fingerprint)
            self.assertEqual(rebuilt.index.ntotal, len(registry.entries))

    def test_select_and_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_registry(root)
            index = ToolIndex(root, HashEmbedding(), registry.entries, "hash:test")
            available = registry.select("创建一个 Python 文件", index, 8)
            self.assertEqual(
                {schema["function"]["name"] for schema in available},
                {entry["id"] for entry in registry.entries},
            )
            narrow = registry.select("创建一个 Python 文件", index, 1)
            self.assertEqual(len(narrow), 2)
            self.assertEqual(narrow[-1]["function"]["name"], "read_skill")
            self.assertIn("已创建", registry.dispatch(
                "create_file", json.dumps({"path": "hello.py", "content": "print('hello')\n"}), available
            ))
            self.assertEqual((root / "hello.py").read_text(encoding="utf-8"), "print('hello')\n")
            self.assertIn("未覆盖", registry.dispatch(
                "create_file", json.dumps({"path": "hello.py", "content": "other"}), available
            ))
            self.assertIn("已写入", registry.dispatch(
                "write_file", json.dumps({"path": "hello.py", "content": "updated"}), available
            ))
            self.assertEqual((root / "hello.py").read_text(encoding="utf-8"), "updated")
            self.assertIn("create_file", registry.dispatch(
                "read_skill", '{"name":"python-hello-world"}', available
            ))
            self.assertIn("本轮未提供", registry.dispatch("run_python", "{}", []))

    def test_skill_catalog(self):
        catalog = SkillCatalog(Path(__file__).resolve().parent / "skill")
        self.assertIn("python-hello-world", catalog.descriptions())
        self.assertIn("create_file", catalog.read("python-hello-world"))
        self.assertIn("未知 skill", catalog.read("missing"))

    def test_agent_loop_reads_skill_then_creates_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_registry(root)
            index = ToolIndex(root, HashEmbedding(), registry.entries, "hash:test")
            responses = iter([
                {
                    "text": "", "finish_reason": "tool_calls",
                    "tool_calls": [{
                        "id": "call-skill", "type": "function",
                        "function": {"name": "read_skill", "arguments": '{"name":"python-hello-world"}'},
                    }],
                },
                {
                    "text": "", "finish_reason": "tool_calls",
                    "tool_calls": [{
                        "id": "call-file", "type": "function",
                        "function": {
                            "name": "create_file",
                            "arguments": json.dumps({"path": "hello.py", "content": 'print("Hello, world!")\n'}),
                        },
                    }],
                },
                {"text": "文件已创建", "finish_reason": "stop", "tool_calls": []},
            ])
            offered = []

            def fake_model(client, messages, schemas):
                offered.append({schema["function"]["name"] for schema in schemas})
                return next(responses)

            messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "创建 Hello World 文件"}]
            with patch.object(main, "call_model_stream", side_effect=fake_model):
                result = main.run_agent(None, messages, "创建 Hello World 文件", index, registry)
            self.assertEqual(result, "文件已创建")
            self.assertTrue(all({"read_skill", "create_file"} <= names for names in offered))
            self.assertEqual((root / "hello.py").read_text(encoding="utf-8"), 'print("Hello, world!")\n')
            self.assertEqual([message["role"] for message in messages[-5:]],
                             ["assistant", "tool", "assistant", "tool", "assistant"])


if __name__ == "__main__":
    unittest.main()
