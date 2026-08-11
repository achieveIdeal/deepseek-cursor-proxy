from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": tool_call.get("id"),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": function.get("name") or "",
            "arguments": arguments,
        },
    }
    return normalized


def tool_call_signature(tool_call: dict[str, Any]) -> str:
    normalized = normalize_tool_call(tool_call)
    normalized.pop("id", None)
    canonical = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.append(str(tool_call["id"]))
    return ids


def tool_call_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names


def message_signature(message: dict[str, Any]) -> str:
    tool_calls = [
        normalize_tool_call(tool_call)
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    ]
    payload = {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_scope_message(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {"role": message.get("role")}
    for key in ("content", "name", "tool_call_id", "prefix"):
        if key in message:
            canonical[key] = message[key]
    if message.get("tool_calls"):
        canonical["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in message.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
    return canonical


def conversation_scope(messages: list[dict[str, Any]], namespace: str = "") -> str:
    scope_messages = [canonical_scope_message(message) for message in messages]
    payload: Any = scope_messages
    if namespace:
        payload = {"namespace": namespace, "messages": scope_messages}
    return _sha256_json(payload)


def turn_context_signature(prior_messages: list[dict[str, Any]]) -> str:
    last_user_index = next(
        (
            index
            for index in range(len(prior_messages) - 1, -1, -1)
            if prior_messages[index].get("role") == "user"
        ),
        -1,
    )
    start_index = 0
    if last_user_index != -1:
        start_index = last_user_index
        while start_index > 0 and prior_messages[start_index - 1].get("role") == "user":
            start_index -= 1

    context_messages = [
        canonical_scope_message(message)
        for message in prior_messages[start_index:]
        if message.get("role") != "system"
    ]
    return _sha256_json(context_messages)


def scoped_reasoning_keys(message: dict[str, Any], scope: str) -> list[str]:
    keys = [f"scope:{scope}:signature:{message_signature(message)}"]
    keys.extend(
        f"scope:{scope}:tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"scope:{scope}:tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    # 最后手段的恢复键。处理流式响应在用户按停止后、tool_call.id 块到达前
    # 被中断的情况，此时 tool_call_id 和 tool_call_signature（规范化参数）
    # 都无法在 Cursor 对话记录往返中保留。
    keys.extend(
        f"scope:{scope}:tool_name:{tool_name}" for tool_name in tool_call_names(message)
    )
    return keys


def portable_reasoning_keys(
    message: dict[str, Any],
    cache_namespace: str,
    prior_messages: list[dict[str, Any]],
) -> list[str]:
    if not cache_namespace:
        return []

    turn_signature = turn_context_signature(prior_messages)
    keys = [
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"signature:{message_signature(message)}"
    ]
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:" f"tool_name:{tool_name}"
        for tool_name in tool_call_names(message)
    )
    return keys


class ReasoningStore:
    def __init__(
        self,
        reasoning_content_path: str | Path,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.max_age_seconds = max_age_seconds
        self.max_rows = max_rows
        if str(reasoning_content_path) == ":memory:":
            self.reasoning_content_path: str | Path = ":memory:"
        else:
            self.reasoning_content_path = Path(reasoning_content_path).expanduser()
            self.reasoning_content_path.parent.mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.reasoning_content_path, check_same_thread=False
        )

        # ── 性能优化 ──
        # WAL 模式：写入不阻塞读取，适合代理的多线程场景
        self._conn.execute("PRAGMA journal_mode=WAL")
        # 降低同步级别：NORMAL 在 WAL 模式下足够安全
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # 增大缓存到 16MB，减少磁盘 I/O
        self._conn.execute("PRAGMA cache_size=-16000")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                key TEXT PRIMARY KEY,
                reasoning TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )

        # 为 key_reversed 列做兼容迁移（旧版本数据库无此列）
        columns = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(reasoning_cache)"
            ).fetchall()
        }
        if "key_reversed" not in columns:
            self._conn.execute(
                "ALTER TABLE reasoning_cache ADD COLUMN key_reversed TEXT"
            )
            # 回填已有数据
            existing = self._conn.execute(
                "SELECT key FROM reasoning_cache"
            ).fetchall()
            if existing:
                for (key,) in existing:
                    self._conn.execute(
                        "UPDATE reasoning_cache SET key_reversed = ? WHERE key = ?",
                        (key[::-1], key),
                    )

        # 索引：加速按时间排序的淘汰和反向键前缀搜索
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rc_created_at "
            "ON reasoning_cache(created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rc_key_reversed "
            "ON reasoning_cache(key_reversed)"
        )
        self._conn.commit()

        # ── 运行时状态 ──
        # 近似行数计数器，避免每次写入都 COUNT(*) 全表扫描
        row = self._conn.execute(
            "SELECT COUNT(*) FROM reasoning_cache"
        ).fetchone()
        self._approx_row_count: int = int(row[0]) if row else 0

        # 延迟 LRU touch：get() 收集被访问的 key，批量更新
        self._pending_touches: set[str] = set()

        # 概率性 prune 计数器：每 N 次写入才真正执行一次 prune
        self._prune_counter: int = 0
        self._prune_interval: int = 16

        # 初始 prune
        self.prune()

        if isinstance(self.reasoning_content_path, Path):
            self.reasoning_content_path.chmod(0o600)

    def close(self) -> None:
        with self._lock:
            self._flush_touches_locked()
            self._conn.commit()
            self._conn.close()

    # ── 内部辅助方法 ──

    def _flush_touches_locked(self) -> None:
        """将累积的 LRU touch 批量写入（需持有锁）。

        使用 time.time() + 0.1 作为基础时间戳，确保被 touch 的行
        始终比同 tick 内未被 touch 的行排序靠后。0.1s 偏移对
        max_age_seconds（默认 30 天）的淘汰影响可忽略。
        """
        if not self._pending_touches:
            return
        base = time.time() + 0.1
        for i, key in enumerate(self._pending_touches):
            self._conn.execute(
                "UPDATE reasoning_cache SET created_at = ? WHERE key = ?",
                (base + i * 0.000001, key),
            )
        self._pending_touches.clear()

    def _should_prune(self) -> bool:
        """判断当前是否应该执行 prune。

        策略：每 self._prune_interval 次写入执行一次，或当行数
        超过 max_rows 20% 时强制 prune。
        """
        self._prune_counter += 1
        if self._prune_counter >= self._prune_interval:
            return True
        if (
            self.max_rows is not None
            and self.max_rows > 0
            and self._approx_row_count > self.max_rows + self.max_rows // 5
        ):
            return True
        return False

    def _maybe_prune_and_commit_locked(self, new_rows: int = 0) -> None:
        """写入后检查是否需要 prune 并提交（需持有锁）。"""
        self._approx_row_count += new_rows
        if self._should_prune():
            self._prune_counter = 0
            # 先刷新延迟的 LRU touch，确保 created_at 反映最近访问时间
            self._flush_touches_locked()
            self._prune_locked()
        self._flush_touches_locked()
        self._conn.commit()

    # ── 公共 API ──

    def put(self, key: str, reasoning: str, message: dict[str, Any]) -> None:
        if not isinstance(reasoning, str):
            return
        message_json = json.dumps(message, ensure_ascii=False, sort_keys=True)
        with self._lock:
            # 使用 INSERT OR REPLACE 语义判断是否为新行
            existing = self._conn.execute(
                "SELECT 1 FROM reasoning_cache WHERE key = ?", (key,)
            ).fetchone()
            is_new = existing is None

            self._conn.execute(
                """
                INSERT INTO reasoning_cache
                    (key, reasoning, message_json, created_at, key_reversed)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    reasoning = excluded.reasoning,
                    message_json = excluded.message_json,
                    created_at = excluded.created_at
                """,
                (key, reasoning, message_json, time.time(), key[::-1]),
            )
            self._maybe_prune_and_commit_locked(new_rows=1 if is_new else 0)

    def _batch_put(
        self,
        items: list[tuple[str, str, str]],
    ) -> None:
        """批量写入多条记录，单次 commit。

        items 中每个元素为 (key, reasoning, message_json)。
        """
        if not items:
            return
        now = time.time()
        with self._lock:
            # 统计新增行数
            new_count = 0
            for key, _reasoning, _message_json in items:
                existing = self._conn.execute(
                    "SELECT 1 FROM reasoning_cache WHERE key = ?", (key,)
                ).fetchone()
                if existing is None:
                    new_count += 1

            self._conn.executemany(
                """
                INSERT INTO reasoning_cache
                    (key, reasoning, message_json, created_at, key_reversed)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    reasoning = excluded.reasoning,
                    message_json = excluded.message_json,
                    created_at = excluded.created_at
                """,
                [
                    (key, reasoning, message_json, now, key[::-1])
                    for key, reasoning, message_json in items
                ],
            )
            self._maybe_prune_and_commit_locked(new_rows=new_count)

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT reasoning FROM reasoning_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            # 延迟 LRU touch，避免每次读取都触发 commit
            self._pending_touches.add(key)
            # 当累积的 touch 达到阈值时刷新
            if len(self._pending_touches) >= 32:
                self._flush_touches_locked()
                self._conn.commit()
            return str(row[0])

    def get_by_tool_call_id(self, tool_call_id: str) -> str | None:
        """Last-resort lookup when conversation scope/turn hashes no longer match.

        Cursor may rewrite earlier system/user prefixes (compaction, mode switch,
        rule injection) while keeping the same tool_call ids. Scoped and portable
        keys then miss even though the reasoning row is still in SQLite under the
        old hash. Match on the stable `:tool_call:{id}` suffix instead.

        Only returns a value when every matching row shares the same reasoning
        text. Concurrent chats that reuse a tool_call_id with different reasoning
        stay ambiguous and are not cross-wired.
        """
        if not tool_call_id:
            return None
        return self._get_unique_by_key_suffix(f":tool_call:{tool_call_id}")

    def get_by_message_signature(self, signature: str) -> str | None:
        """Like get_by_tool_call_id, but for assistants without tool_call ids."""
        if not signature:
            return None
        return self._get_unique_by_key_suffix(f":signature:{signature}")

    def _get_unique_by_key_suffix(self, suffix: str) -> str | None:
        """通过 key 后缀查找唯一 reasoning。

        使用 key_reversed 列将后缀搜索转为前缀搜索，
        从而利用 idx_rc_key_reversed 索引，避免全表扫描。
        """
        reversed_suffix = suffix[::-1]
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, reasoning FROM reasoning_cache
                WHERE key_reversed LIKE ?
                ORDER BY created_at DESC
                """,
                (reversed_suffix + "%",),
            ).fetchall()
            # key_reversed LIKE 'reversed_suffix%' 精确等价于
            # key LIKE '%suffix'，无需额外的 Python endswith 过滤
            matched: list[tuple[str, str]] = [
                (str(key), str(reasoning)) for key, reasoning in rows
            ]
            if not matched:
                return None
            unique_reasonings = {reasoning for _key, reasoning in matched}
            if len(unique_reasonings) != 1:
                return None
            reasoning = unique_reasonings.pop()
            # Touch all matching keys so active entries survive max_rows prune.
            now = time.time()
            for key, _reasoning in matched:
                self._conn.execute(
                    "UPDATE reasoning_cache SET created_at = ? WHERE key = ?",
                    (now, key),
                )
            self._conn.commit()
            return reasoning

    def store_assistant_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        if message.get("role") != "assistant":
            return 0
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            return 0

        keys = scoped_reasoning_keys(message, scope)
        if prior_messages is not None:
            keys.extend(
                portable_reasoning_keys(message, cache_namespace, prior_messages)
            )
        keys = list(dict.fromkeys(keys))

        # 批量写入：单次 commit 代替每个 key 各一次 commit
        message_json = json.dumps(message, ensure_ascii=False, sort_keys=True)
        items = [(key, reasoning, message_json) for key in keys]
        self._batch_put(items)
        return len(keys)

    def lookup_for_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> str | None:
        keys = scoped_reasoning_keys(message, scope)
        if prior_messages is not None:
            keys.extend(
                portable_reasoning_keys(message, cache_namespace, prior_messages)
            )
        for key in keys:
            reasoning = self.get(key)
            if reasoning is not None:
                return reasoning
        return None

    def backfill_portable_aliases(
        self,
        message: dict[str, Any],
        reasoning: str,
        cache_namespace: str,
        prior_messages: list[dict[str, Any]],
    ) -> int:
        if not isinstance(reasoning, str):
            return 0
        keys = portable_reasoning_keys(message, cache_namespace, prior_messages)
        if not keys:
            return 0
        message_with_reasoning = dict(message)
        message_with_reasoning["reasoning_content"] = reasoning
        message_json = json.dumps(
            message_with_reasoning, ensure_ascii=False, sort_keys=True
        )
        items = [(key, reasoning, message_json) for key in dict.fromkeys(keys)]
        self._batch_put(items)
        return len(keys)

    def clear(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reasoning_cache"
            ).fetchone()
            count = int(row[0] if row else 0)
            self._conn.execute("DELETE FROM reasoning_cache")
            self._approx_row_count = 0
            self._pending_touches.clear()
            self._conn.commit()
        return count

    def clear_namespace(self, namespace: str) -> int:
        with self._lock:
            pattern = f"%namespace:{namespace}:%"
            cursor = self._conn.execute(
                "DELETE FROM reasoning_cache WHERE key LIKE ?",
                (pattern,),
            )
            deleted = cursor.rowcount if cursor.rowcount != -1 else 0
            self._approx_row_count = max(0, self._approx_row_count - deleted)
            self._conn.commit()
        return deleted

    def prune(self) -> int:
        with self._lock:
            deleted = self._prune_locked()
            self._conn.commit()
        return deleted

    def _prune_locked(self) -> int:
        deleted = 0
        if self.max_age_seconds is not None and self.max_age_seconds > 0:
            cutoff = time.time() - self.max_age_seconds
            cursor = self._conn.execute(
                "DELETE FROM reasoning_cache WHERE created_at < ?",
                (cutoff,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        if self.max_rows is not None and self.max_rows > 0:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reasoning_cache"
            ).fetchone()
            count = int(row[0] if row else 0)
            overflow = count - self.max_rows
            if overflow > 0:
                # idx_rc_created_at 索引使 ORDER BY + LIMIT 子查询
                # 无需全表扫描和排序。rowid 确保 created_at 相同时
                # 按插入顺序确定性淘汰（Windows time.time() 精度仅 ~15ms）
                cursor = self._conn.execute(
                    """
                    DELETE FROM reasoning_cache
                    WHERE key IN (
                        SELECT key
                        FROM reasoning_cache
                        ORDER BY created_at ASC, rowid ASC
                        LIMIT ?
                    )
                    """,
                    (overflow,),
                )
                deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        # 更新近似计数器
        self._approx_row_count = max(0, self._approx_row_count - deleted)
        # 定期用精确 COUNT 校正计数器
        row = self._conn.execute(
            "SELECT COUNT(*) FROM reasoning_cache"
        ).fetchone()
        self._approx_row_count = int(row[0]) if row else 0

        return deleted
