"""[新增-记忆] Agent 的会话、摘要和长期记忆存储。"""

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _memory_log(event: str, **fields) -> None:
    """[新增-日志] 统一打印记忆生命周期事件。"""
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    print(f"[memory:{event}]" + (f" {details}" if details else ""))


class MemoryStore:
    """[新增-记忆] 使用 SQLite 实现 session 隔离、短期上下文和长期记忆。"""

    def __init__(
        self,
        db_path: Path,
        owner_id: str,
        session_id: str | None = None,
        vector_store=None,
    ):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.owner_id = owner_id
        self.vector_store = vector_store
        self.last_vector_error = None
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._init_schema()
        self.create_session(self.session_id)
        _memory_log("init", owner=self.owner_id, session=self.session_id, vector=bool(self.vector_store))

    @contextmanager
    def _connect(self):
        """[新增-记忆] 提交并关闭连接，避免 Windows 锁住 SQLite 文件。"""
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_schema(self) -> None:
        """[新增-记忆] 数据模型：原文、session 摘要、候选记忆、长期记忆分表。"""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_compacted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    tool_calls_json TEXT,
                    tool_call_id TEXT,
                    created_at TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_active
                    ON messages(session_id, archived, message_id);

                CREATE TABLE IF NOT EXISTS memory_candidates (
                    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source_message_id INTEGER,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'owner',
                    session_id TEXT,
                    kind TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_message_id INTEGER,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    importance REAL NOT NULL DEFAULT 0.5,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    valid_until TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_owner_key
                    ON memories(owner_id, scope, memory_key);
                CREATE INDEX IF NOT EXISTS idx_memories_owner_status
                    ON memories(owner_id, status, updated_at);
                """
            )

    def create_session(self, session_id: str | None = None) -> str:
        """[新增-记忆] 新窗口默认创建新 session；显式传入 ID 时可恢复旧 session。"""
        selected = session_id or uuid.uuid4().hex[:12]
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT owner_id FROM sessions WHERE session_id = ?",
                (selected,),
            ).fetchone()
            if existing and existing["owner_id"] != self.owner_id:
                raise ValueError("session 不属于当前 owner，拒绝访问")
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions
                    (session_id, owner_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (selected, self.owner_id, now, now),
            )
        self.session_id = selected
        _memory_log("session", action="ready", session=selected, owner=self.owner_id)
        return selected

    def save_message(self, session_id: str, message: dict) -> int:
        """[新增-记忆] 保存不可变原始消息，工具调用单独保存 JSON。"""
        role = message.get("role", "unknown")
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        with self._connect() as connection:
            session = connection.execute(
                "SELECT owner_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not session or session["owner_id"] != self.owner_id:
                raise ValueError("session 不属于当前 owner，拒绝写入")
            cursor = connection.execute(
                """
                INSERT INTO messages
                    (session_id, owner_id, role, content, tool_calls_json,
                     tool_call_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    self.owner_id,
                    role,
                    str(content),
                    tool_calls_json,
                    message.get("tool_call_id"),
                    _now(),
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (_now(), session_id),
            )
            return int(cursor.lastrowid)

    def _recent_rows(self, session_id: str, max_rounds: int) -> list[sqlite3.Row]:
        """[新增-记忆] 只读取当前 session 未归档的消息，避免窗口互相污染。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND owner_id = ? AND archived = 0
                ORDER BY message_id
                """,
                (session_id, self.owner_id),
            ).fetchall()

        user_positions = [index for index, row in enumerate(rows) if row["role"] == "user"]
        if len(user_positions) > max_rounds:
            rows = rows[user_positions[-max_rounds]:]
        return rows

    def get_summary(self, session_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary FROM sessions WHERE session_id = ? AND owner_id = ?",
                (session_id, self.owner_id),
            ).fetchone()
        return row["summary"] if row else ""

    def _query_terms(self, text: str) -> list[str]:
        """[新增-记忆] 简单关键词切分；后续可替换为 embedding/vector 检索。"""
        terms = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text.lower())
        return list(dict.fromkeys(terms))

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """[新增-记忆] 先向量召回候选，再按 owner/status 过滤和综合打分。"""
        _memory_log(
            "recall:start",
            session=self.session_id,
            query=query[:120],
            limit=limit,
            vector_enabled=bool(self.vector_store),
        )
        terms = self._query_terms(query)
        vector_scores = {}
        if self.vector_store:
            try:
                vector_scores = dict(self.vector_store.search(query, limit=limit * 5))
                self.last_vector_error = None
                _memory_log("recall:vector", candidates=len(vector_scores))
            except Exception as exc:
                # [新增-向量] embedding 服务失败时降级关键词召回，不影响主对话。
                self.last_vector_error = str(exc)
                _memory_log("recall:vector_error", error=str(exc)[:200], fallback="keyword")
        if not terms and not vector_scores:
            _memory_log("recall:end", hit_count=0, reason="no_query_terms")
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE owner_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                """,
                (self.owner_id,),
            ).fetchall()
        _memory_log("recall:filter", active_memories=len(rows), terms=terms[:8])

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if vector_scores:
                if row["memory_id"] not in vector_scores:
                    continue
                score = max(vector_scores[row["memory_id"]], 0.0)
            else:
                haystack = row["content"].lower()
                matches = sum(1 for term in terms if term in haystack)
                if matches == 0:
                    continue
                score = matches / len(terms)
            score += row["importance"] * 0.2 + row["confidence"] * 0.1
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        with self._connect() as connection:
            for score, row in scored[:limit]:
                connection.execute(
                    "UPDATE memories SET last_accessed_at = ? WHERE memory_id = ?",
                    (_now(), row["memory_id"]),
                )
                results.append(
                    {
                        "memory_id": row["memory_id"],
                        "kind": row["kind"],
                        "content": row["content"],
                        "score": round(score, 3),
                    }
                )
        _memory_log(
            "recall:end",
            hit_count=len(results),
            memories=[{"id": item["memory_id"], "score": item["score"]} for item in results],
        )
        return results

    def build_context_messages(
        self, system_prompt: str, query: str, max_rounds: int
    ) -> list[dict]:
        """[新增-记忆] 组装本轮上下文：系统规则、摘要、召回记忆、当前 session 原文。"""
        _memory_log("context:start", session=self.session_id, query=query[:120], short_term_rounds=max_rounds)
        sections = [system_prompt]
        summary = self.get_summary(self.session_id)
        if summary:
            sections.append(f"\n[本 session 历史摘要]\n{summary}")
        _memory_log("context:summary", used=bool(summary), chars=len(summary))

        memories = self.recall(query)
        if memories:
            memory_text = "\n".join(
                f"- ({item['kind']}, score={item['score']}) {item['content']}"
                for item in memories
            )
            sections.append(f"\n[可用长期记忆]\n{memory_text}")
        _memory_log("context:long_term", used=bool(memories), count=len(memories))

        context = [{"role": "system", "content": "\n".join(sections)}]
        for row in self._recent_rows(self.session_id, max_rounds):
            # [新增-记忆] assistant 工具调用的 content 保持 null，避免恢复 session 后破坏 API 消息格式。
            message = {
                "role": row["role"],
                "content": None if row["tool_calls_json"] else row["content"],
            }
            if row["tool_calls_json"]:
                message["tool_calls"] = json.loads(row["tool_calls_json"])
            if row["tool_call_id"]:
                message["tool_call_id"] = row["tool_call_id"]
            context.append(message)
        _memory_log("context:end", message_count=len(context) - 1, system_chars=len(context[0]["content"]))
        return context

    def _memory_candidate(self, text: str) -> tuple[str, str] | None:
        """[新增-记忆] 第一版只捕获明确表达的长期信息，减少错误写入。"""
        explicit_signals = ("记住", "记下来", "以后", "我喜欢", "我偏好", "请保存")
        fact_signal = "我的" in text and any(word in text for word in ("是", "叫", "使用", "技术栈"))
        if not any(signal in text for signal in explicit_signals) and not fact_signal:
            return None

        kind = "preference" if any(word in text for word in ("喜欢", "偏好", "以后", "习惯")) else "fact"
        content = text.strip()
        normalized = re.sub(r"\s+", "", content).lower()
        key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return kind, key

    def capture_explicit_memory(self, session_id: str, source_message_id: int, text: str) -> str | None:
        """[新增-记忆] 生命周期：候选写入 -> 接受 -> 去重合并 -> active。"""
        candidate = self._memory_candidate(text)
        if not candidate:
            _memory_log("write:skip", session=session_id, reason="not_explicit", chars=len(text))
            return None

        kind, key = candidate
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memory_candidates
                    (owner_id, session_id, source_message_id, kind, content, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (self.owner_id, session_id, source_message_id, kind, text.strip(), now),
            )
            candidate_id = int(cursor.lastrowid)

            existing = connection.execute(
                """
                SELECT memory_id FROM memories
                WHERE owner_id = ? AND scope = 'owner' AND memory_key = ?
                """,
                (self.owner_id, key),
            ).fetchone()
            if existing:
                memory_id = int(existing["memory_id"])
                write_action = "update"
                connection.execute(
                    """
                    UPDATE memories
                    SET content = ?, kind = ?, session_id = ?, source_message_id = ?,
                        confidence = 0.9, importance = 0.8, status = 'active', updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (text.strip(), kind, session_id, source_message_id, now, existing["memory_id"]),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO memories
                        (owner_id, scope, session_id, kind, memory_key, content,
                         source_message_id, confidence, importance, status, created_at, updated_at)
                    VALUES (?, 'owner', ?, ?, ?, ?, ?, 0.9, 0.8, 'active', ?, ?)
                    """,
                    (self.owner_id, session_id, kind, key, text.strip(), source_message_id, now, now),
                )
                memory_id = int(cursor.lastrowid)
                write_action = "insert"
            connection.execute(
                "UPDATE memory_candidates SET status = 'accepted', reviewed_at = ? WHERE candidate_id = ?",
                (now, candidate_id),
            )
        if self.vector_store and memory_id is not None:
            try:
                self.vector_store.upsert(memory_id, text.strip())
                self.last_vector_error = None
            except Exception as exc:
                # [新增-向量] SQLite 记忆已经成功，向量失败可通过 /reindex 修复。
                self.last_vector_error = str(exc)
                _memory_log("write:vector_error", memory_id=memory_id, error=str(exc)[:200])
        _memory_log(
            "write:accepted",
            action=write_action,
            memory_id=memory_id,
            kind=kind,
            session=session_id,
            vector_synced=bool(self.vector_store and not self.last_vector_error),
        )
        return kind

    def rebuild_vectors(self) -> int:
        """[新增-向量] 从 SQLite active 记忆重建当前 owner 的 FAISS 索引。"""
        if not self.vector_store:
            _memory_log("vector:rebuild_skip", reason="vector_store_disabled")
            return 0
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id, content FROM memories
                WHERE owner_id = ? AND status = 'active'
                ORDER BY memory_id
                """,
                (self.owner_id,),
            ).fetchall()
        count = self.vector_store.rebuild(
            [(int(row["memory_id"]), row["content"]) for row in rows]
        )
        self.vector_store.load_error = None
        self.last_vector_error = None
        _memory_log("vector:rebuild", owner=self.owner_id, count=count)
        return count

    def compaction_payload(
        self, session_id: str, trigger_rounds: int, keep_rounds: int
    ) -> dict | None:
        """[新增-记忆] 到达阈值时取旧原文，保留最近 keep_rounds 个 user 回合。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND owner_id = ? AND archived = 0
                ORDER BY message_id
                """,
                (session_id, self.owner_id),
            ).fetchall()

        user_positions = [index for index, row in enumerate(rows) if row["role"] == "user"]
        if len(user_positions) <= trigger_rounds or len(user_positions) <= keep_rounds:
            _memory_log(
                "summary:check",
                session=session_id,
                rounds=len(user_positions),
                threshold=trigger_rounds,
                triggered=False,
            )
            return None

        cutoff_index = user_positions[-keep_rounds]
        archived_rows = rows[:cutoff_index]
        if not archived_rows:
            return None

        _memory_log(
            "summary:check",
            session=session_id,
            rounds=len(user_positions),
            threshold=trigger_rounds,
            triggered=True,
            archive_messages=len(archived_rows),
            keep_rounds=keep_rounds,
        )

        lines = []
        for row in archived_rows:
            label = row["role"]
            content = row["content"]
            if row["tool_calls_json"]:
                content += f"\n工具调用：{row['tool_calls_json']}"
            lines.append(f"{label}: {content}")
        return {
            "cutoff_message_id": archived_rows[-1]["message_id"],
            "transcript": "\n".join(lines),
            "existing_summary": self.get_summary(session_id),
        }

    def save_summary_and_archive(self, session_id: str, cutoff_message_id: int, summary: str) -> None:
        """[新增-记忆] 摘要成功后才归档原文，失败时原文仍可重试。"""
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_version FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            version = (row["summary_version"] if row else 0) + 1
            connection.execute(
                """
                UPDATE sessions
                SET summary = ?, summary_version = ?, updated_at = ?, last_compacted_at = ?
                WHERE session_id = ? AND owner_id = ?
                """,
                (summary.strip(), version, now, now, session_id, self.owner_id),
            )
            connection.execute(
                """
                UPDATE messages SET archived = 1
                WHERE session_id = ? AND owner_id = ? AND message_id <= ?
                """,
                (session_id, self.owner_id, cutoff_message_id),
            )
        _memory_log(
            "summary:saved",
            session=session_id,
            version=version,
            cutoff_message_id=cutoff_message_id,
            summary_chars=len(summary),
        )

    def stats(self, session_id: str) -> dict:
        """[新增-记忆] 用于调试 session 隔离和生命周期状态。"""
        with self._connect() as connection:
            messages = connection.execute(
                """
                SELECT COUNT(*) AS n FROM messages
                WHERE session_id = ? AND owner_id = ? AND archived = 0
                """,
                (session_id, self.owner_id),
            ).fetchone()["n"]
            memories = connection.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE owner_id = ? AND status = 'active'",
                (self.owner_id,),
            ).fetchone()["n"]
        return {"session_id": session_id, "active_messages": messages, "active_memories": memories}
