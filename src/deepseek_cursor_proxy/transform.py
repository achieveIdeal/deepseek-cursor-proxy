from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any
import urllib.request
from urllib.error import HTTPError, URLError

from .config import ProxyConfig
from .logging import LOG
from .reasoning_store import (
    ReasoningStore,
    conversation_scope,
    message_signature,
    tool_call_ids,
    tool_call_names,
    tool_call_signature,
    turn_context_signature,
)
from .streaming import fold_reasoning_into_content


SUPPORTED_REQUEST_FIELDS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "response_format",
    "stop",
    "tools",
    "tool_choice",
    "thinking",
    "reasoning_effort",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    # DeepSeek 会遵守或安全忽略的标准 OpenAI Chat Completions 字段。
    # Cursor 和大多数 OpenAI SDK 会无条件发送这些字段，转发可保持客户端兼容并减少日志噪音。
    "user",
    "seed",
    "n",
    "logit_bias",
}

MESSAGE_FIELDS = {
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "reasoning_content",
    "prefix",
}

ROLE_MESSAGE_FIELDS = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {
        "role",
        "content",
        "name",
        "tool_calls",
        "reasoning_content",
        "prefix",
    },
    "tool": {"role", "content", "tool_call_id"},
}

EFFORT_ALIASES = {
    "low": "high",
    "medium": "high",
    "high": "high",
    "max": "max",
    "xhigh": "max",
}

CURSOR_THINKING_BLOCK_RE = re.compile(
    r"""
    (?:
        <(?:think|thinking)\b[^>]*>[\s\S]*?(?:</(?:think|thinking)>|\Z)
        |
        <details\b[^>]*>\s*
        <summary\b[^>]*>\s*Thinking\s*</summary>
        [\s\S]*?(?:</details>|\Z)
    )\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

RECOVERY_NOTICE_TEXT = "[deepseek-cursor-proxy] 已刷新 reasoning_content 历史记录。"
RECOVERY_NOTICE_CONTENT = f"{RECOVERY_NOTICE_TEXT}\n\n"
RECOVERY_SYSTEM_CONTENT = (
    "deepseek-cursor-proxy 已恢复此请求，因为较早的 DeepSeek 思考模式工具调用 "
    "reasoning_content 不可用。已省略无法恢复的较早工具调用历史；请仅使用剩余已恢复的上下文继续。"
)

SUMMARY_SYSTEM_PREFIX = (
    "[deepseek-cursor-proxy] 对话摘要（已保留先前上下文）：\n\n"
)

RESPONSE_LANGUAGE_INSTRUCTIONS = {
    "zh": (
        "请始终使用中文进行思考和回答。"
        "你的 reasoning_content（思考过程）和最终回复内容都必须使用中文。"
    ),
    "en": (
        "Always think and respond in English. "
        "Both your reasoning_content and final reply must be in English."
    ),
}


def _format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = extract_text_content(msg.get("content")) or ""
        if role == "system":
            lines.append(f"[系统]: {content}")
        elif role == "user":
            lines.append(f"[用户]: {content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                tc_desc = ", ".join(
                    str((tc.get("function") or {}).get("name", "?"))
                    for tc in tool_calls
                    if isinstance(tc, dict)
                )
                lines.append(f"[助手 — 调用了 {tc_desc}]: {content}")
            else:
                lines.append(f"[助手]: {content}")
        elif role == "tool":
            tool_id = msg.get("tool_call_id", "?")
            truncated = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"[工具 {tool_id} 的结果]: {truncated}")
    return "\n\n".join(lines)


def summarize_conversation(
    messages_to_summarize: list[dict[str, Any]],
    config: ProxyConfig,
    authorization: str,
    upstream_model: str,
) -> str | None:
    if not messages_to_summarize:
        return None

    formatted = _format_messages_for_summary(messages_to_summarize)
    summary_payload = {
        "model": upstream_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是对话摘要助手。请对以下对话历史生成全面但简洁的摘要。"
                    "保留所有关键信息：已做决策、编写或修改的代码、发现的错误及修复、"
                    "技术上下文、用户偏好以及进行中的任务。摘要必须包含足够细节，"
                    "以便在不丢失上下文的情况下继续对话。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请摘要以下对话。"
                    f"务必全面但简洁：\n\n{formatted}"
                ),
            },
        ],
        "stream": False,
        "max_tokens": 2000,
        "thinking": {"type": "disabled"},
    }

    upstream_url = f"{config.upstream_base_url}/chat/completions"
    body = json.dumps(
        summary_payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    request = urllib.request.Request(
        upstream_url,
        data=body,
        method="POST",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout) as resp:
            response_body = resp.read()
            result = json.loads(response_body.decode("utf-8"))
            choices = result.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                summary = message.get("content", "")
                if summary:
                    LOG.info(
                        "已将 %s 条消息摘要为 %s 个字符",
                        len(messages_to_summarize),
                        len(summary),
                    )
                    return summary.strip()
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        LOG.warning("对话摘要失败: %s", exc)

    return None


def rebuild_messages_with_summary(
    messages: list[dict[str, Any]],
    summary: str,
    keep_recent: int,
) -> list[dict[str, Any]]:
    system_msgs = leading_system_messages(messages)
    non_system_msgs = messages[len(system_msgs):]

    split_index = max(0, len(non_system_msgs) - keep_recent)
    recent_msgs = non_system_msgs[split_index:]

    summary_msg = {
        "role": "system",
        "content": SUMMARY_SYSTEM_PREFIX + summary,
    }

    return [*system_msgs, summary_msg, *recent_msgs]


@dataclass(frozen=True)
class PreparedRequest:
    payload: dict[str, Any]
    original_model: str
    upstream_model: str
    cache_namespace: str
    patched_reasoning_messages: int
    missing_reasoning_messages: int
    recovered_reasoning_messages: int = 0
    recovery_dropped_messages: int = 0
    recovery_notice: str | None = None
    record_response_scope: str | None = None
    record_response_messages: list[dict[str, Any]] = field(default_factory=list)
    record_response_contexts: list[tuple[str, list[dict[str, Any]]]] = field(
        default_factory=list
    )
    reasoning_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    recovery_steps: list[dict[str, Any]] = field(default_factory=list)
    continued_recovery_boundary: bool = False
    retired_prefix_messages: int = 0


def normalize_reasoning_effort(value: Any) -> str:
    if not isinstance(value, str):
        return "high"
    return EFFORT_ALIASES.get(value.strip().lower(), "high")


def extract_text_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            text = item.get("text") or item.get("content")
            if item_type in {"text", "input_text"} and isinstance(text, str):
                parts.append(text)
            elif isinstance(text, str):
                parts.append(text)
            elif item_type:
                parts.append(f"[{item_type} 已由 DeepSeek 文本代理省略]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, (dict, tuple)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def strip_cursor_thinking_blocks(content: str) -> str:
    return CURSOR_THINKING_BLOCK_RE.sub("", content).lstrip("\r\n")


def normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        tool_call = {}
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": str(tool_call.get("id") or ""),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": arguments,
        },
    }
    if not normalized["id"]:
        normalized.pop("id")
    return normalized


def normalize_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {
            "type": "function",
            "function": {"name": "", "description": "", "parameters": {}},
        }
    normalized = dict(tool)
    normalized["type"] = normalized.get("type") or "function"
    function = normalized.get("function")
    if isinstance(function, dict):
        normalized["function"] = function
    return normalized


def legacy_function_to_tool(function: Any) -> dict[str, Any]:
    if not isinstance(function, dict):
        function = {}
    return {"type": "function", "function": function}


def convert_function_call(function_call: Any) -> Any:
    if isinstance(function_call, str):
        if function_call in {"auto", "none", "required"}:
            return function_call
        return None
    if isinstance(function_call, dict) and function_call.get("name"):
        return {
            "type": "function",
            "function": {"name": str(function_call["name"])},
        }
    return None


def normalize_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            function = tool_choice.get("function")
            if isinstance(function, dict) and function.get("name"):
                return {
                    "type": "function",
                    "function": {"name": str(function["name"])},
                }
        return tool_choice
    return tool_choice


def normalize_message(
    message: Any,
    store: ReasoningStore | None,
    prior_messages: list[dict[str, Any]],
    cache_namespace: str,
    repair_reasoning: bool,
    keep_reasoning: bool,
) -> tuple[dict[str, Any], bool, bool, dict[str, Any] | None]:
    if not isinstance(message, dict):
        message = {"role": "user", "content": str(message)}
    normalized = {key: value for key, value in message.items() if key in MESSAGE_FIELDS}
    role = normalized.get("role") or "user"
    normalized["role"] = role

    if role == "function":
        normalized["role"] = "tool"

    if "content" in normalized:
        normalized["content"] = extract_text_content(normalized["content"]) or ""
    elif normalized["role"] in {"assistant", "tool", "system", "user"}:
        normalized["content"] = ""
    if normalized["role"] == "assistant" and isinstance(normalized.get("content"), str):
        normalized["content"] = strip_cursor_thinking_blocks(normalized["content"])

    if normalized.get("tool_calls"):
        normalized["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in normalized.get("tool_calls") or []
        ]

    patched = False
    missing = False
    diagnostic: dict[str, Any] | None = None
    if normalized["role"] == "assistant":
        if not keep_reasoning:
            normalized.pop("reasoning_content", None)
        elif repair_reasoning:
            reasoning = normalized.get("reasoning_content")
            if not isinstance(reasoning, str):
                normalized.pop("reasoning_content", None)
                needs_reasoning = assistant_needs_reasoning_for_tool_context(
                    normalized, prior_messages
                )
                lookup_scope = conversation_scope(prior_messages, cache_namespace)
                lookup_keys = (
                    reasoning_lookup_keys(
                        normalized,
                        lookup_scope,
                        cache_namespace,
                        prior_messages,
                    )
                    if needs_reasoning
                    else []
                )
                hit_kind = None
                if needs_reasoning and store is not None:
                    for lookup_key in lookup_keys:
                        restored = store.get(str(lookup_key["key"]))
                        if restored is not None:
                            lookup_key["hit"] = True
                            hit_kind = lookup_key["kind"]
                            normalized["reasoning_content"] = restored
                            patched = True
                            if not lookup_key.get("portable"):
                                store.backfill_portable_aliases(
                                    normalized,
                                    restored,
                                    cache_namespace,
                                    prior_messages,
                                )
                            break
                if needs_reasoning and not patched:
                    missing = True
                if needs_reasoning:
                    diagnostic = {
                        "message_index": len(prior_messages),
                        "role": "assistant",
                        "needs_reasoning": True,
                        "had_reasoning_content": False,
                        "patched": patched,
                        "missing": missing,
                        "lookup_scope": lookup_scope,
                        "message_signature": message_signature(normalized),
                        "tool_call_ids": tool_call_ids(normalized),
                        "lookup_keys": lookup_keys,
                        "hit_kind": hit_kind,
                    }
            elif assistant_needs_reasoning_for_tool_context(normalized, prior_messages):
                diagnostic = {
                    "message_index": len(prior_messages),
                    "role": "assistant",
                    "needs_reasoning": True,
                    "had_reasoning_content": True,
                    "patched": False,
                    "missing": False,
                    "lookup_scope": conversation_scope(prior_messages, cache_namespace),
                    "message_signature": message_signature(normalized),
                    "tool_call_ids": tool_call_ids(normalized),
                    "lookup_keys": [],
                    "hit_kind": "request",
                }

    allowed_fields = ROLE_MESSAGE_FIELDS.get(str(normalized["role"]), MESSAGE_FIELDS)
    normalized = {
        key: value for key, value in normalized.items() if key in allowed_fields
    }
    return normalized, patched, missing, diagnostic


def reasoning_lookup_keys(
    message: dict[str, Any],
    scope: str,
    cache_namespace: str = "",
    prior_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    keys = [
        {
            "kind": "message_signature",
            "key": f"scope:{scope}:signature:{message_signature(message)}",
            "portable": False,
            "hit": False,
        }
    ]
    keys.extend(
        {
            "kind": "tool_call_id",
            "tool_call_id": tool_call_id,
            "key": f"scope:{scope}:tool_call:{tool_call_id}",
            "portable": False,
            "hit": False,
        }
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        {
            "kind": "tool_call_signature",
            "function_name": str((tool_call.get("function") or {}).get("name") or ""),
            "key": (
                f"scope:{scope}:tool_call_signature:"
                f"{tool_call_signature(tool_call)}"
            ),
            "portable": False,
            "hit": False,
        }
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    keys.extend(
        {
            "kind": "tool_name",
            "function_name": tool_name,
            "key": f"scope:{scope}:tool_name:{tool_name}",
            "portable": False,
            "hit": False,
        }
        for tool_name in tool_call_names(message)
    )
    if cache_namespace and prior_messages is not None:
        turn_signature = turn_context_signature(prior_messages)
        keys.append(
            {
                "kind": "portable_message_signature",
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"signature:{message_signature(message)}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
        )
        keys.extend(
            {
                "kind": "portable_tool_call_id",
                "tool_call_id": tool_call_id,
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"tool_call:{tool_call_id}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_call_id in tool_call_ids(message)
        )
        keys.extend(
            {
                "kind": "portable_tool_call_signature",
                "function_name": str(
                    (tool_call.get("function") or {}).get("name") or ""
                ),
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"tool_call_signature:{tool_call_signature(tool_call)}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_call in (message.get("tool_calls") or [])
            if isinstance(tool_call, dict)
        )
        keys.extend(
            {
                "kind": "portable_tool_name",
                "function_name": tool_name,
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"tool_name:{tool_name}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_name in tool_call_names(message)
        )
    return keys


def normalize_messages(
    messages: Any,
    store: ReasoningStore | None,
    cache_namespace: str,
    repair_reasoning: bool,
    keep_reasoning: bool,
) -> tuple[list[dict[str, Any]], int, list[int], list[dict[str, Any]]]:
    if not isinstance(messages, list):
        return [], 0, [], []
    normalized_messages: list[dict[str, Any]] = []
    patched_count = 0
    missing_indexes: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for message in messages:
        normalized, patched, missing, diagnostic = normalize_message(
            message,
            store,
            normalized_messages,
            cache_namespace,
            repair_reasoning,
            keep_reasoning,
        )
        normalized_messages.append(normalized)
        if patched:
            patched_count += 1
        if missing:
            missing_indexes.append(len(normalized_messages) - 1)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return normalized_messages, patched_count, missing_indexes, diagnostics


def has_recovery_notice(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "assistant"
        and isinstance(content, str)
        and content.startswith(RECOVERY_NOTICE_TEXT)
    )


def strip_recovery_notice_for_upstream(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cursor 在后续轮次会将代理的恢复通知回传给我们。
    该通知作为代理的边界标记，但 DeepSeek 不应看到代理生成的文本。
    返回剥离了助手消息前缀的副本；保持输入不变，以便缓存作用域/记录上下文
    继续匹配 Cursor 下次将发送的带前缀历史。"""
    stripped: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            stripped.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(RECOVERY_NOTICE_TEXT):
            stripped.append(message)
            continue
        cleaned = dict(message)
        cleaned["content"] = content[len(RECOVERY_NOTICE_TEXT) :].lstrip("\r\n")
        stripped.append(cleaned)
    return stripped


def inject_response_language_instruction(
    messages: list[dict[str, Any]],
    language: str | None,
) -> list[dict[str, Any]]:
    if not language or language not in RESPONSE_LANGUAGE_INSTRUCTIONS:
        return messages
    instruction = RESPONSE_LANGUAGE_INSTRUCTIONS[language]
    leading = leading_system_messages(messages)
    for message in leading:
        if message.get("content") == instruction:
            return messages
    rest = messages[len(leading):]
    return [*leading, {"role": "system", "content": instruction}, *rest]


def leading_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leading_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            leading_messages.append(message)
            continue
        break
    return leading_messages


def active_messages_from_recovery_boundary(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]] | None:
    recovery_boundary_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if has_recovery_notice(messages[index])
        ),
        -1,
    )
    if recovery_boundary_index == -1:
        return None

    context_user_index = next(
        (
            index
            for index in range(recovery_boundary_index - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    leading_messages = leading_system_messages(messages)
    recovered_tail = []
    if context_user_index != -1:
        recovered_tail.append(messages[context_user_index])
    recovered_tail.extend(messages[recovery_boundary_index:])
    active_messages = [
        *leading_messages,
        {"role": "system", "content": RECOVERY_SYSTEM_CONTENT},
        *recovered_tail,
    ]
    kept_context_messages = 1 if context_user_index != -1 else 0
    retired_messages = (
        recovery_boundary_index - len(leading_messages) - kept_context_messages
    )
    retired_messages = max(retired_messages, 0)
    step = {
        "strategy": "continued_recovery_boundary",
        "recovery_boundary_index": recovery_boundary_index,
        "context_user_index": context_user_index,
        "retired_prefix_messages": retired_messages,
    }
    return active_messages, retired_messages, step


def recover_messages_from_missing_reasoning(
    messages: list[dict[str, Any]],
    missing_indexes: list[int],
) -> tuple[list[dict[str, Any]], int, str | None, dict[str, Any]]:
    recovery_boundary_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if has_recovery_notice(messages[index])
            and any(missing_index < index for missing_index in missing_indexes)
        ),
        -1,
    )
    if recovery_boundary_index != -1:
        context_user_index = next(
            (
                index
                for index in range(recovery_boundary_index - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            -1,
        )
        leading_messages = leading_system_messages(messages)
        recovered_tail = []
        if context_user_index != -1:
            recovered_tail.append(messages[context_user_index])
        recovered_tail.extend(messages[recovery_boundary_index:])
        recovered = [
            *leading_messages,
            {"role": "system", "content": RECOVERY_SYSTEM_CONTENT},
            *recovered_tail,
        ]
        kept_context_messages = 1 if context_user_index != -1 else 0
        omitted_messages = (
            recovery_boundary_index - len(leading_messages) - kept_context_messages
        )
        return (
            recovered,
            omitted_messages,
            None,
            {
                "strategy": "recovery_boundary",
                "missing_indexes": missing_indexes,
                "recovery_boundary_index": recovery_boundary_index,
                "context_user_index": context_user_index,
                "dropped_messages": omitted_messages,
                "notice": None,
            },
        )

    last_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    if last_user_index == -1:
        return (
            messages,
            0,
            None,
            {
                "strategy": "none",
                "missing_indexes": missing_indexes,
                "last_user_index": None,
                "dropped_messages": 0,
                "notice": None,
            },
        )

    recovered = leading_system_messages(messages)
    omitted_messages = len(messages) - len(recovered) - 1
    recovered.append({"role": "system", "content": RECOVERY_SYSTEM_CONTENT})
    recovered.append(messages[last_user_index])
    return (
        recovered,
        omitted_messages,
        RECOVERY_NOTICE_CONTENT,
        {
            "strategy": "latest_user",
            "missing_indexes": missing_indexes,
            "last_user_index": last_user_index,
            "dropped_messages": omitted_messages,
            "notice": RECOVERY_NOTICE_CONTENT,
        },
    )


def assistant_needs_reasoning_for_tool_context(
    message: dict[str, Any],
    prior_messages: list[dict[str, Any]],
) -> bool:
    if message.get("tool_calls"):
        return True
    for prior_message in reversed(prior_messages):
        role = prior_message.get("role")
        if role == "tool":
            return True
        if role in {"user", "system"}:
            return False
    return False


def upstream_model_for(original_model: str, config: ProxyConfig) -> str:
    if original_model.startswith("deepseek-"):
        return original_model
    LOG.warning(
        "正在将非 DeepSeek 模型 %r 重写为配置的回退模型 %r",
        original_model,
        config.upstream_model,
    )
    return config.upstream_model


def reasoning_model_family(upstream_model: str) -> str:
    if upstream_model in {"deepseek-v4-pro", "deepseek-v4-flash"}:
        return "deepseek-v4"
    return upstream_model


def reasoning_cache_namespace(
    config: ProxyConfig,
    upstream_model: str,
    thinking: Any,
    reasoning_effort: Any,
    authorization: str | None = None,
) -> str:
    auth_hash = ""
    if authorization:
        auth_hash = hashlib.sha256(authorization.encode("utf-8")).hexdigest()
    payload = {
        "base_url": config.upstream_base_url,
        "model": reasoning_model_family(upstream_model),
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "authorization_hash": auth_hash,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def response_recording_contexts(
    *items: tuple[str, list[dict[str, Any]]] | None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    contexts: list[tuple[str, list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        scope, messages = item
        if scope in seen:
            continue
        seen.add(scope)
        contexts.append((scope, messages))
    return contexts


def prepare_upstream_request(
    payload: dict[str, Any],
    config: ProxyConfig,
    store: ReasoningStore | None,
    authorization: str | None = None,
) -> PreparedRequest:
    original_model = str(payload.get("model") or config.upstream_model)
    upstream_model = upstream_model_for(original_model, config)

    prepared = {
        key: value for key, value in payload.items() if key in SUPPORTED_REQUEST_FIELDS
    }
    dropped_fields = sorted(
        key
        for key in payload.keys()
        if key not in SUPPORTED_REQUEST_FIELDS
        and key not in {"max_completion_tokens", "functions", "function_call"}
    )
    if dropped_fields:
        LOG.warning(
            "正在丢弃不支持的请求字段: %s", ", ".join(dropped_fields)
        )
    if "max_tokens" not in prepared and "max_completion_tokens" in payload:
        prepared["max_tokens"] = payload["max_completion_tokens"]

    prepared["model"] = upstream_model
    if prepared.get("stream"):
        stream_options = prepared.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        else:
            stream_options = dict(stream_options)
        stream_options["include_usage"] = True
        prepared["stream_options"] = stream_options

    if "tools" in prepared and isinstance(prepared["tools"], list):
        prepared["tools"] = [normalize_tool(tool) for tool in prepared["tools"]]
    elif isinstance(payload.get("functions"), list):
        prepared["tools"] = [
            legacy_function_to_tool(function) for function in payload["functions"]
        ]

    if "tool_choice" in prepared:
        tool_choice = normalize_tool_choice(prepared["tool_choice"])
        if tool_choice is None:
            prepared.pop("tool_choice", None)
        else:
            prepared["tool_choice"] = tool_choice
    elif "function_call" in payload:
        tool_choice = convert_function_call(payload.get("function_call"))
        if tool_choice is not None:
            prepared["tool_choice"] = tool_choice

    prepared["thinking"] = {"type": config.thinking}
    thinking_enabled = config.thinking == "enabled"
    thinking_disabled = config.thinking == "disabled"
    if thinking_enabled:
        prepared["reasoning_effort"] = normalize_reasoning_effort(
            config.reasoning_effort
        )

    cache_namespace = reasoning_cache_namespace(
        config,
        upstream_model,
        prepared.get("thinking"),
        prepared.get("reasoning_effort"),
        authorization,
    )

    raw_messages = payload.get("messages")
    if (
        config.summary_enabled
        and isinstance(raw_messages, list)
        and len(raw_messages) > config.summary_max_messages
        and authorization
    ):
        keep_recent = max(1, config.summary_keep_recent)
        messages_to_summarize = raw_messages[: -keep_recent] if keep_recent < len(raw_messages) else raw_messages
        summary = summarize_conversation(
            messages_to_summarize,
            config,
            authorization,
            upstream_model,
        )
        if summary:
            if store is not None:
                cleared = store.clear_namespace(cache_namespace)
                if cleared:
                    LOG.info(
                        "已清除命名空间 %s 的 %s 条 reasoning 缓存条目",
                        cache_namespace[:12],
                        cleared,
                    )
            payload["messages"] = rebuild_messages_with_summary(
                raw_messages, summary, keep_recent,
            )
            LOG.info(
                "已重建对话: %s 条消息 -> %s 条消息（摘要 %s 个字符）",
                len(raw_messages),
                len(payload["messages"]),
                len(summary),
            )

    pre_repair_messages, _, _, _ = normalize_messages(
        payload.get("messages"),
        None,
        cache_namespace,
        repair_reasoning=False,
        keep_reasoning=not thinking_disabled,
    )
    record_response_messages = pre_repair_messages
    record_response_scope = conversation_scope(
        record_response_messages, cache_namespace
    )
    messages_for_repair = pre_repair_messages
    continued_recovery_boundary = False
    retired_prefix_messages = 0
    recovered_count = 0
    recovery_dropped_messages = 0
    recovery_notice = None
    recovery_steps: list[dict[str, Any]] = []
    if thinking_enabled and config.missing_reasoning_strategy == "recover":
        boundary = active_messages_from_recovery_boundary(pre_repair_messages)
        if boundary is not None:
            messages_for_repair, retired_prefix_messages, boundary_step = boundary
            continued_recovery_boundary = True
            recovery_steps.append(boundary_step)

    messages, patched_count, missing_indexes, reasoning_diagnostics = (
        normalize_messages(
            messages_for_repair,
            store,
            cache_namespace,
            repair_reasoning=thinking_enabled,
            keep_reasoning=not thinking_disabled,
        )
    )
    while missing_indexes and config.missing_reasoning_strategy == "recover":
        recovered_messages, dropped_messages, notice, recovery_step = (
            recover_messages_from_missing_reasoning(messages, missing_indexes)
        )
        recovery_steps.append(recovery_step)
        if not dropped_messages:
            break
        recovered_count += len(missing_indexes)
        recovery_dropped_messages += dropped_messages
        if notice:
            recovery_notice = notice
        (
            messages,
            patched_count,
            missing_indexes,
            latest_diagnostics,
        ) = normalize_messages(
            recovered_messages,
            store,
            cache_namespace,
            repair_reasoning=thinking_enabled,
            keep_reasoning=not thinking_disabled,
        )
        reasoning_diagnostics.extend(latest_diagnostics)
    active_record_response_scope = conversation_scope(messages, cache_namespace)
    record_response_contexts = response_recording_contexts(
        (record_response_scope, record_response_messages),
        (active_record_response_scope, messages),
    )
    messages = inject_response_language_instruction(
        messages,
        config.response_language,
    )
    prepared["messages"] = strip_recovery_notice_for_upstream(messages)

    return PreparedRequest(
        payload=prepared,
        original_model=original_model,
        upstream_model=upstream_model,
        cache_namespace=cache_namespace,
        patched_reasoning_messages=patched_count,
        missing_reasoning_messages=len(missing_indexes),
        recovered_reasoning_messages=recovered_count,
        recovery_dropped_messages=recovery_dropped_messages,
        recovery_notice=recovery_notice,
        record_response_scope=record_response_scope,
        record_response_messages=record_response_messages,
        record_response_contexts=record_response_contexts,
        reasoning_diagnostics=reasoning_diagnostics,
        recovery_steps=recovery_steps,
        continued_recovery_boundary=continued_recovery_boundary,
        retired_prefix_messages=retired_prefix_messages,
    )


def record_response_reasoning(
    response_payload: dict[str, Any],
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
    cache_namespace: str = "",
    scope: str | None = None,
    prior_messages: list[dict[str, Any]] | None = None,
    recording_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> int:
    if store is None:
        return 0
    stored = 0
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return stored
    if recording_contexts is None:
        response_scope = (
            scope
            if scope is not None
            else conversation_scope(request_messages, cache_namespace)
        )
        response_prior_messages = (
            prior_messages if prior_messages is not None else request_messages
        )
        recording_contexts = [(response_scope, response_prior_messages)]
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            for response_scope, response_prior_messages in recording_contexts:
                stored += store.store_assistant_message(
                    message,
                    response_scope,
                    cache_namespace,
                    response_prior_messages,
                )
    return stored


def rewrite_response_body(
    body: bytes,
    original_model: str,
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
    cache_namespace: str = "",
    content_prefix: str | None = None,
    scope: str | None = None,
    prior_messages: list[dict[str, Any]] | None = None,
    recording_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
    display_reasoning: bool = False,
    collapsible_reasoning: bool = True,
) -> bytes:
    response_payload = json.loads(body.decode("utf-8"))
    if isinstance(response_payload, dict):
        if content_prefix:
            prefix_response_content(response_payload, content_prefix)
        record_response_reasoning(
            response_payload,
            store,
            request_messages,
            cache_namespace,
            scope=scope,
            prior_messages=prior_messages,
            recording_contexts=recording_contexts,
        )
        if display_reasoning:
            fold_reasoning_into_content(response_payload, collapsible_reasoning)
        if "model" in response_payload:
            response_payload["model"] = original_model
    return json.dumps(
        response_payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def prefix_response_content(response_payload: dict[str, Any], prefix: str) -> bool:
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        message["content"] = prefix + (content if isinstance(content, str) else "")
        return True
    return False
