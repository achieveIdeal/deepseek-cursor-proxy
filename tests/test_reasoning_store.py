from __future__ import annotations

from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory
import unittest

from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope


class ReasoningStoreTests(unittest.TestCase):
    def test_file_store_creates_private_database_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            reasoning_content_path = (
                Path(temp_dir) / "nested" / "reasoning_content.sqlite3"
            )

            store = ReasoningStore(reasoning_content_path)
            store.close()

            self.assertTrue(reasoning_content_path.exists())
            # Windows 上 os.chmod 不支持 Unix 权限位，仅验证文件存在且非只读
            if sys.platform == "win32":
                self.assertFalse(
                    reasoning_content_path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY,
                    "数据库文件不应该是只读的",
                )
            else:
                self.assertEqual(
                    stat.S_IMODE(reasoning_content_path.stat().st_mode), 0o600
                )

    def test_store_prunes_to_max_rows_and_can_clear(self) -> None:
        store = ReasoningStore(":memory:", max_rows=2)
        try:
            store.put("a", "reasoning a", {"role": "assistant"})
            store.put("b", "reasoning b", {"role": "assistant"})
            store.put("c", "reasoning c", {"role": "assistant"})

            self.assertIsNone(store.get("a"))
            self.assertEqual(store.get("b"), "reasoning b")
            self.assertEqual(store.get("c"), "reasoning c")
            self.assertEqual(store.clear(), 2)
            self.assertIsNone(store.get("b"))
            self.assertIsNone(store.get("c"))
        finally:
            store.close()

    def test_lru_touch_keeps_recently_read_keys_during_prune(self) -> None:
        store = ReasoningStore(":memory:", max_rows=2)
        try:
            store.put("a", "reasoning a", {"role": "assistant"})
            store.put("b", "reasoning b", {"role": "assistant"})
            # Touch the oldest key so prune prefers evicting untouched "b".
            self.assertEqual(store.get("a"), "reasoning a")
            store.put("c", "reasoning c", {"role": "assistant"})

            self.assertEqual(store.get("a"), "reasoning a")
            self.assertIsNone(store.get("b"))
            self.assertEqual(store.get("c"), "reasoning c")
        finally:
            store.close()

    def test_get_by_tool_call_id_finds_orphaned_scope_keys(self) -> None:
        store = ReasoningStore(":memory:")
        try:
            old_scope = "deadbeef"
            tool_call_id = "call_orphan"
            store.put(
                f"scope:{old_scope}:tool_call:{tool_call_id}",
                "orphaned reasoning",
                {"role": "assistant"},
            )

            self.assertEqual(
                store.get_by_tool_call_id(tool_call_id),
                "orphaned reasoning",
            )
            self.assertIsNone(store.get_by_tool_call_id("call_missing"))
            self.assertIsNone(store.get_by_tool_call_id("call_orph"))
        finally:
            store.close()

    def test_get_by_tool_call_id_ignores_ambiguous_collisions(self) -> None:
        store = ReasoningStore(":memory:")
        try:
            tool_call_id = "call_shared"
            store.put(
                f"scope:aaaa:tool_call:{tool_call_id}",
                "reasoning from thread A",
                {"role": "assistant"},
            )
            store.put(
                f"scope:bbbb:tool_call:{tool_call_id}",
                "reasoning from thread B",
                {"role": "assistant"},
            )
            self.assertIsNone(store.get_by_tool_call_id(tool_call_id))
        finally:
            store.close()

    def test_empty_reasoning_content_is_stored_as_present_value(self) -> None:
        store = ReasoningStore(":memory:")
        try:
            scope = conversation_scope([{"role": "user", "content": "lookup"}])
            tool_call = {
                "id": "call_empty",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
            message = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [tool_call],
            }

            self.assertGreater(store.store_assistant_message(message, scope), 0)
            self.assertEqual(store.get(f"scope:{scope}:tool_call:call_empty"), "")
            self.assertEqual(
                store.lookup_for_message(
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    scope,
                ),
                "",
            )
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
