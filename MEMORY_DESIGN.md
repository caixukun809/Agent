# [新增-记忆] 上下文与记忆生命周期

## 选择的数据模型

第一版使用 SQLite，数据库默认位于 `workspace/agent_memory.sqlite3`。

- `sessions`：每个窗口的一条会话记录，保存 `session_id` 和当前摘要。
- `messages`：不可变的原始消息、工具调用和工具结果。
- `memory_candidates`：从用户明确表达中提取出的长期记忆候选及处理状态。
- `memories`：已经接受的用户级长期记忆，带类型、来源、置信度、重要性和更新时间。

FAISS 作为本地向量索引已经加入。SQLite 仍然是正文、权限和状态的主存储，FAISS 只保存 `memory_id -> vector`。当前默认使用本地 `BAAI/bge-small-zh-v1.5`，不依赖中转站的 embeddings 接口；设置 `VECTOR_EMBEDDING_MODE=openai` 后可使用 `EMBEDDING_MODEL` 指定的 embeddings 接口。

## 一轮对话的生命周期

```text
启动窗口
  -> 创建或恢复 session_id
  -> 只读取当前 session 的未归档消息

用户输入
  -> 按 query 召回长期记忆
  -> 组装系统提示词、session 摘要、长期记忆、短期原文
  -> 执行 Agent 工具循环
  -> 保存 user / assistant / tool 原始事件

一轮结束
  -> 检查“记住 / 以后 / 我的偏好”等明确表达
  -> 写入 memory_candidates
  -> 去重或更新 memories
  -> 未归档回合达到阈值时生成摘要
  -> 摘要成功后才归档旧原文
```

## Session 隔离

每次启动默认生成新的 `session_id`，因此多个窗口不会共享短期对话。要恢复某个窗口的短期历史，可以设置：

```powershell
$env:AGENT_SESSION_ID="已有的 session id"
python .\main.py
```

长期记忆按 `owner_id` 共享。设置不同的 `AGENT_OWNER_ID` 可以隔离不同用户：

```powershell
$env:AGENT_OWNER_ID="user-a"
```

## 当前生命周期参数

- `SHORT_TERM_ROUNDS=8`：本轮上下文最多读取最近 8 个用户回合。
- `SUMMARY_TRIGGER_ROUNDS=12`：未归档回合超过此值时触发摘要。
- `SUMMARY_KEEP_ROUNDS=6`：摘要后保留最近 6 个回合，旧消息只做归档，不删除。
- `VECTOR_EMBEDDING_MODE=local`：使用本地 `models/bge-small-zh-v1.5`；也可改为 `hash` 或 `openai`。
- `LOCAL_EMBEDDING_MODEL=./models/bge-small-zh-v1.5`：本地 SentenceTransformer 模型路径。
- `EMBEDDING_MODEL=text-embedding-3-small`：OpenAI 兼容 embeddings 模型名。

可以通过环境变量调整这些值。

输入 `/reindex` 可以从 SQLite 中的 active 记忆重建 FAISS 索引，适合更换 embedding 模型或修复索引。

## [新增-工具索引与 Skill]

`tools/catalog.json` 是工具元数据的主存储，包含稳定 ID、检索文本和完整的模型调用 schema；`tools/runtime.py` 实现 Python 执行函数；`tools/registry.py` 在启动时核对定义与函数，并负责选择、校验和分发。四个原有工具和 `read_skill` 都在清单中，`main.py` 只负责 Agent 调度。

`tool_index.py` 将清单转成向量，存入独立的 `workspace/tool_vectors.faiss`，使用稳定数字 ID；`workspace/tool_vectors.meta.json` 保存 ID 对应的名称、描述和检索文本，便于检查索引。完整 schema 仍以清单为准。工具索引不与长期记忆索引混用。清单或 embedding 配置变动时自动重建旧索引，也可输入 `/reindex-tools` 手动重建。

每步先按当前任务查询索引，再提供最多 `TOOL_SEARCH_LIMIT` 个普通工具及常驻的 `read_skill`，并打印 `[tool:search]` 分数和 `[tool:available]` 列表。默认上限为 8，因此目前四个普通工具都会提供，避免小工具集误召回；工具增多后才会筛选。模型返回的工具名必须是本步提供的工具，注册表才会执行。索引失败时提供全部工具。

`skill/manifest.json` 记录可用 skill 的名称与用途；`skill/<名称>/SKILL.md` 保存具体步骤。模型只先看到目录描述，需要时调用常驻的 `read_skill` 工具读取正文。输入 `/skills` 可查看目录。示例 `python-hello-world` 演示“读说明 -> 调用 create_file -> 返回实际路径”的过程。skill 是操作说明，不是可执行 Python 函数；新增 skill 后重启程序即可加载。

## [新增-日志] 记忆调用日志

程序会在终端打印这些事件：

- `memory:context:start/end`：开始和完成上下文构建。
- `memory:context:summary`：本轮是否使用 session 摘要。
- `memory:recall:start/end`：是否执行长期记忆召回、命中数量和分数。
- `memory:recall:vector`：FAISS 返回的候选数量。
- `memory:write:skip/accepted`：本轮是否写入长期记忆，以及 insert/update 结果。
- `memory:summary:check/saved`：是否达到摘要阈值，以及摘要和归档结果。
- `memory:vector:rebuild`：是否重建向量索引。
