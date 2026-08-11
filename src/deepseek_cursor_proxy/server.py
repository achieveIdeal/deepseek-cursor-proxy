from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gzip
from http.client import HTTPException
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request
import zlib

from .config import (
    ProxyConfig,
    default_config_path,
    default_reasoning_content_path,
)
from .http_util import upstream_urlopen
from .logging import (
    LOG,
    TerminalSpinner,
    configure_logging,
)
from .reasoning_store import ReasoningStore, conversation_scope
from .streaming import CursorReasoningDisplayAdapter, StreamAccumulator
from .trace import TraceRequest, TraceWriter
from .tunnel import NgrokTunnel, local_tunnel_target
from .transform import (
    RECOVERY_NOTICE_CONTENT,
    build_second_pass_payload,
    extract_assistant_text_from_response,
    prepare_upstream_request,
    rewrite_response_body,
)


class RequestBodyTooLarge(ValueError):
    pass


@dataclass
class ProxyResponseResult:
    sent: bool
    usage: dict[str, Any] | None = None


class DeepSeekProxyServer(ThreadingHTTPServer):
    config: ProxyConfig
    reasoning_store: ReasoningStore
    trace_writer: TraceWriter | None


class DeepSeekProxyHandler(BaseHTTPRequestHandler):
    server_version = "DeepSeekPythonProxy/0.1"

    @property
    def config(self) -> ProxyConfig:
        return self.server.config  # type: ignore[return-value]

    @property
    def reasoning_store(self) -> ReasoningStore:
        return self.server.reasoning_store  # type: ignore[return-value]

    @property
    def trace_writer(self) -> TraceWriter | None:
        return getattr(self.server, "trace_writer", None)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        request_path = urlparse(self.path).path
        if self.config.verbose:
            LOG.info(
                "收到 OPTIONS %s，来自 %s",
                request_path,
                self.client_address[0],
            )
        self._send_response_headers(204, [], "发送 CORS 预检响应")

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if self.config.verbose:
            LOG.info("收到 GET %s，来自 %s", request_path, self.client_address[0])
        if request_path in {"/healthz", "/v1/healthz"}:
            self._send_json(200, {"ok": True})
            return
        if request_path in {"/models", "/v1/models"}:
            self._send_models()
            return
        self._send_json(404, {"error": {"message": "未找到"}})

    def do_POST(self) -> None:
        started = time.monotonic()
        request_path = urlparse(self.path).path
        trace = self._start_trace(request_path)
        if self.config.verbose:
            LOG.info(
                "收到 POST %s，来自 %s content_length=%s user_agent=%s",
                request_path,
                self.client_address[0],
                self.headers.get("Content-Length", "0"),
                self.headers.get("User-Agent", ""),
            )
        if request_path not in {"/chat/completions", "/v1/chat/completions"}:
            LOG.warning("拒绝不支持的 POST path=%s status=404", request_path)
            self._record_request_body_for_trace(trace)
            self._send_json(
                404,
                {"error": {"message": "仅支持 /v1/chat/completions"}},
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=404)
            return
        cursor_authorization = self._cursor_authorization()
        if cursor_authorization is None:
            LOG.warning(
                "拒绝请求 path=%s status=401 reason=missing_bearer_token",
                request_path,
            )
            self._record_request_body_for_trace(trace)
            self._send_json(
                401,
                {"error": {"message": "缺少 Authorization bearer 令牌"}},
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=401)
            return

        try:
            payload = self._read_json_body()
        except RequestBodyTooLarge as exc:
            LOG.warning(
                "拒绝请求 path=%s status=413 reason=%s", request_path, exc
            )
            self._send_json(413, {"error": {"message": str(exc)}}, trace=trace)
            self._finish_trace(trace, "rejected", http_status=413, reason=str(exc))
            return
        except ValueError as exc:
            LOG.warning(
                "拒绝请求 path=%s status=400 reason=%s", request_path, exc
            )
            self._send_json(400, {"error": {"message": str(exc)}}, trace=trace)
            self._finish_trace(trace, "rejected", http_status=400, reason=str(exc))
            return

        if trace is not None:
            trace.record_cursor_body(payload)

        if self.config.verbose:
            log_json("Cursor 请求体", payload)

        log_cursor_request(payload, self.config)

        prepared = prepare_upstream_request(
            payload,
            self.config,
            self.reasoning_store,
            authorization=cursor_authorization,
        )
        if trace is not None:
            trace.record_transform(prepared)
        log_context_summary(prepared)
        if (
            prepared.missing_reasoning_messages
            and self.config.missing_reasoning_strategy == "reject"
        ):
            LOG.warning(
                (
                    "严格缺失-reasoning 模式拒绝请求 path=%s "
                    "status=409 reason=missing_reasoning_content count=%s"
                ),
                request_path,
                prepared.missing_reasoning_messages,
            )
            self._send_json(
                409,
                {
                    "error": {
                        "message": (
                            "deepseek-cursor-proxy 正在严格缺失-reasoning 模式下运行，"
                            "无法自动恢复此思考模式工具调用历史，因为 "
                            f"{prepared.missing_reasoning_messages} 条助手消息"
                            "缺少缓存的 DeepSeek reasoning_content。"
                            "请在不使用 `--missing-reasoning-strategy reject` 的情况下重启，"
                            "或传入 `--missing-reasoning-strategy recover`，"
                            "以便代理自动从部分对话历史中恢复。"
                        ),
                        "type": "missing_reasoning_content",
                        "code": "missing_reasoning_content",
                        "missing_reasoning_messages": prepared.missing_reasoning_messages,
                    }
                },
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=409)
            return

        if self.config.verbose:
            LOG.info(
                (
                    "上游请求元数据: original_model=%s upstream_model=%s "
                    "patched_reasoning=%s missing_reasoning=%s %s"
                ),
                prepared.original_model,
                prepared.upstream_model,
                prepared.patched_reasoning_messages,
                prepared.missing_reasoning_messages,
                summarize_chat_payload(prepared.payload),
            )

        if self.config.verbose:
            log_json("上游请求体", prepared.payload)

        # 构建上游请求体
        upward_stream = bool(prepared.payload.get("stream"))
        if prepared.multimodal_routing and prepared.vision_payload:
            # 多模态第一遍：发送给视觉 API（如智谱 GLM）
            first_pass_payload = dict(prepared.vision_payload)
            first_pass_payload["stream"] = False
            upstream_body = json.dumps(
                first_pass_payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            upstream_url = (
                f"{self.config.vision_base_url}/chat/completions"
            )
            first_pass_stream = False
            # 视觉 API 使用自己的 API Key
            vision_auth = f"Bearer {self.config.vision_api_key}"
            upstream_headers = self._upstream_headers(
                stream=False,
                authorization=vision_auth,
            )
            if self.config.verbose:
                LOG.info(
                    "多模态第一遍 → 视觉 API vision_url=%s model=%s",
                    upstream_url,
                    first_pass_payload.get("model"),
                )
        else:
            first_pass_payload = dict(prepared.payload)
            upstream_body = json.dumps(
                first_pass_payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            upstream_url = f"{self.config.upstream_base_url}/chat/completions"
            first_pass_stream = upward_stream
            upstream_headers = self._upstream_headers(
                stream=first_pass_stream,
                authorization=cursor_authorization,
            )
        if trace is not None:
            trace.record_upstream_request(
                url=upstream_url,
                headers=upstream_headers,
                body_bytes=upstream_body,
            )
        request = Request(
            upstream_url,
            data=upstream_body,
            method="POST",
            headers=upstream_headers,
        )

        if self.config.verbose:
            log_send_summary(prepared)
        spinner = TerminalSpinner(
            enabled=upward_stream and not self.config.verbose,
            text="└ {frame}",
        ).start()

        try:
            if self.config.verbose:
                LOG.info("正在转发到 %s", upstream_url)
            response = upstream_urlopen(
                request, timeout=self.config.request_timeout
            )
        except HTTPError as exc:
            spinner.stop()
            LOG.warning(
                "请求失败 upstream_status=%s stream=%s elapsed_ms=%s",
                exc.code,
                upward_stream,
                elapsed_ms(started),
            )
            self._send_upstream_error(exc, trace=trace)
            self._finish_trace(
                trace,
                "upstream_error",
                http_status=exc.code,
                stream=upward_stream,
            )
            return
        except URLError as exc:
            spinner.stop()
            LOG.warning(
                "上游请求失败 elapsed_ms=%s reason=%s",
                elapsed_ms(started),
                exc.reason,
            )
            self._send_json(
                502,
                {"error": {"message": f"上游请求失败: {exc.reason}"}},
                trace=trace,
            )
            self._finish_trace(trace, "upstream_error", http_status=502)
            return
        except Exception:
            spinner.stop()
            raise

        try:
            with response:
                upstream_status = getattr(response, "status", 200)
                if self.config.verbose:
                    LOG.info(
                        "上游响应 status=%s stream=%s elapsed_ms=%s",
                        upstream_status,
                        upward_stream,
                        elapsed_ms(started),
                    )

                # ── 多模态两步管道 ──
                if prepared.multimodal_routing:
                    sent_response = self._handle_multimodal_two_pass(
                        response,
                        prepared,
                        cursor_authorization,
                        started,
                        trace,
                    )
                elif upward_stream:
                    sent_response = self._proxy_streaming_response(
                        response,
                        prepared.original_model,
                        prepared.payload["messages"],
                        prepared.cache_namespace,
                        prepared.recovery_notice,
                        trace=trace,
                        record_response_scope=prepared.record_response_scope,
                        record_response_messages=prepared.record_response_messages,
                        record_response_contexts=prepared.record_response_contexts,
                    )
                else:
                    sent_response = self._proxy_regular_response(
                        response,
                        prepared.original_model,
                        prepared.payload["messages"],
                        prepared.cache_namespace,
                        prepared.recovery_notice,
                        trace=trace,
                        record_response_scope=prepared.record_response_scope,
                        record_response_messages=prepared.record_response_messages,
                        record_response_contexts=prepared.record_response_contexts,
                    )
                if not sent_response.sent:
                    spinner.stop()
                    self._finish_trace(
                        trace,
                        "client_disconnected",
                        http_status=upstream_status,
                        stream=upward_stream,
                    )
                    return
                spinner.stop()
                log_stats_summary(sent_response.usage)
                self._finish_trace(
                    trace,
                    "completed",
                    http_status=upstream_status,
                    stream=upward_stream,
                )
        finally:
            spinner.stop()

    def _start_trace(self, request_path: str) -> TraceRequest | None:
        writer = self.trace_writer
        if writer is None:
            return None
        try:
            return writer.start_request(
                method=self.command,
                path=request_path,
                client_address=self.client_address[0],
                headers={name: value for name, value in self.headers.items()},
            )
        except OSError as exc:
            LOG.warning("启动请求追踪失败: %s", exc)
            return None

    def _finish_trace(
        self,
        trace: TraceRequest | None,
        status: str,
        **extra: Any,
    ) -> None:
        if trace is None:
            return
        try:
            trace.finish(status, **extra)
        except OSError as exc:
            LOG.warning("写入请求追踪失败: %s", exc)

    def _cursor_authorization(self) -> str | None:
        auth_header = self.headers.get("Authorization", "")
        scheme, separator, token = auth_header.strip().partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token.strip():
            return None
        return f"Bearer {token.strip()}"

    def _send_cors_headers(self) -> None:
        if not self.config.cors:
            return
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Origin, Content-Type, Accept, Authorization",
        )
        self.send_header("Access-Control-Expose-Headers", "Content-Length")
        self.send_header("Access-Control-Allow-Credentials", "true")

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        trace: TraceRequest | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if trace is not None:
            trace.record_cursor_response(
                status=status,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
                body=body,
            )
        sent_headers = self._send_response_headers(
            status,
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ],
            "发送 JSON 响应头",
        )
        if sent_headers:
            self._write_to_client(body, "发送 JSON 响应体")

    def _send_response_headers(
        self,
        status: int,
        headers: list[tuple[str, str]],
        disconnect_context: str,
    ) -> bool:
        try:
            self.send_response(status)
            self._send_cors_headers()
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
        except (BrokenPipeError, ConnectionError) as exc:
            LOG.warning("客户端断开连接（%s）: %s", disconnect_context, exc)
            return False
        return True

    def _write_to_client(
        self,
        body: bytes,
        disconnect_context: str,
        *,
        flush: bool = False,
    ) -> bool:
        try:
            self.wfile.write(body)
            if flush:
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError) as exc:
            LOG.warning("客户端断开连接（%s）: %s", disconnect_context, exc)
            return False
        return True

    def _send_models(self) -> None:
        created = int(time.time())
        model_ids = list(
            dict.fromkeys(
                [
                    self.config.upstream_model,
                    "deepseek-v4-pro",
                    "deepseek-v4-flash",
                ]
            )
        )
        models = [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
            }
            for model_id in model_ids
        ]
        self._send_json(200, {"object": "list", "data": models})

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("无效的 Content-Length") from exc
        if length < 0:
            raise ValueError("无效的 Content-Length")
        if length > self.config.max_request_body_bytes:
            raise RequestBodyTooLarge(
                f"请求体过大；限制为 {self.config.max_request_body_bytes} 字节"
            )
        raw_body = self.rfile.read(length)
        if not raw_body:
            raise ValueError("请求体为空")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"无效的 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def _record_request_body_for_trace(self, trace: TraceRequest | None) -> None:
        if trace is None:
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            trace.record_cursor_body_omitted(reason="invalid_content_length")
            return
        if length < 0:
            trace.record_cursor_body_omitted(
                reason="invalid_content_length", body_bytes=length
            )
            return
        if length > self.config.max_request_body_bytes:
            trace.record_cursor_body_omitted(reason="body_too_large", body_bytes=length)
            self.close_connection = True
            return
        try:
            raw_body = self.rfile.read(length)
        except OSError as exc:
            trace.record_cursor_body_omitted(
                reason=f"read_failed:{exc}", body_bytes=length
            )
            return
        trace.record_cursor_body_bytes(raw_body)

    def _upstream_headers(self, stream: bool, authorization: str) -> dict[str, str]:
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": self.server_version,
        }
        accept_language = self.headers.get("Accept-Language")
        if accept_language:
            headers["Accept-Language"] = accept_language
        return headers

    def _send_upstream_error(
        self,
        exc: HTTPError,
        *,
        trace: TraceRequest | None = None,
    ) -> None:
        body = read_response_body(exc)
        if self.config.verbose:
            log_bytes("上游错误响应体", body)
        headers = {
            "Content-Type": exc.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_upstream_response(
                status=exc.code,
                headers={name: value for name, value in exc.headers.items()},
                body=body,
            )
            trace.record_cursor_response(status=exc.code, headers=headers, body=body)
        sent_headers = self._send_response_headers(
            exc.code,
            [
                ("Content-Type", headers["Content-Type"]),
                ("Content-Length", headers["Content-Length"]),
            ],
            "发送上游错误响应头",
        )
        if sent_headers:
            self._write_to_client(body, "发送上游错误响应体")

    # ── 多模态两步管道 ──

    def _handle_multimodal_two_pass(
        self,
        first_response: Any,
        prepared: Any,
        cursor_authorization: str,
        started: float,
        trace: TraceRequest | None = None,
    ) -> ProxyResponseResult:
        """两步管道：先视觉 API 识别图片 → 文本，再回传文本给 DeepSeek 推理。

        第一遍始终以非流式模式请求，确保拿到完整文本。
        第二遍使用原始请求的 stream 设置。
        """
        # 读取第一遍响应（视觉 API）
        first_body = read_response_body(first_response)
        if self.config.verbose:
            LOG.info(
                "多模态第一遍响应（%s）已收到",
                self.config.vision_model,
            )
            log_bytes("第一遍响应体", first_body)

        try:
            first_payload = json.loads(first_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOG.warning("解析第一遍 JSON 响应失败: %s", exc)
            self._send_json(
                502,
                {"error": {"message": f"多模态第一遍响应解析失败: {exc}"}},
                trace=trace,
            )
            self._finish_trace(trace, "upstream_error", http_status=502)
            return ProxyResponseResult(False)

        # 提取视觉 API 返回的文本内容
        first_pass_text = extract_assistant_text_from_response(first_payload)
        if not first_pass_text:
            LOG.warning("多模态第一遍响应中没有文本内容")
            # 如果第一遍没有文本（异常情况），直接返回第一遍响应
            body = json.dumps(
                first_payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            }
            sent_headers = self._send_response_headers(
                200,
                [
                    ("Content-Type", headers["Content-Type"]),
                    ("Content-Length", headers["Content-Length"]),
                ],
                "发送多模态响应头（无文本回退）",
            )
            if sent_headers:
                self._write_to_client(body, "发送多模态响应体（无文本回退）")
            usage = first_payload.get("usage") if isinstance(first_payload, dict) else None
            return ProxyResponseResult(sent_headers, usage if isinstance(usage, dict) else None)

        if self.config.verbose:
            LOG.info(
                "多模态第一遍提取文本长度=%s",
                len(first_pass_text),
            )

        # 构建第二遍请求（DeepSeek 推理）
        # 使用 vision_payload（含原始多模态消息）作为模板，替换图片为视觉 API 返回的文本描述
        second_pass_payload = build_second_pass_payload(
            prepared.vision_payload,
            first_pass_text,
            prepared.second_pass_model,
        )

        # 保留原始 stream 设置
        original_stream = bool(prepared.payload.get("stream"))
        second_pass_payload["stream"] = original_stream
        if original_stream:
            stream_options = second_pass_payload.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
            else:
                stream_options = dict(stream_options)
            stream_options["include_usage"] = True
            second_pass_payload["stream_options"] = stream_options

        # 恢复 DeepSeek 专用字段（第一遍 vision_payload 中已剥离）
        for _deepseek_field in ("thinking", "reasoning_effort"):
            if _deepseek_field in prepared.payload:
                second_pass_payload[_deepseek_field] = prepared.payload[_deepseek_field]

        if self.config.verbose:
            LOG.info(
                "多模态第二遍请求 model=%s stream=%s",
                prepared.second_pass_model,
                original_stream,
            )

        # 发送第二遍请求
        second_body = json.dumps(
            second_pass_payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        second_url = f"{self.config.upstream_base_url}/chat/completions"
        second_headers = self._upstream_headers(
            stream=original_stream,
            authorization=cursor_authorization,
        )

        if trace is not None:
            trace.record_upstream_request(
                url=second_url,
                headers=second_headers,
                body_bytes=second_body,
            )

        second_request = Request(
            second_url,
            data=second_body,
            method="POST",
            headers=second_headers,
        )

        try:
            second_response = upstream_urlopen(
                second_request, timeout=self.config.request_timeout
            )
        except HTTPError as exc:
            LOG.warning(
                "多模态第二遍请求失败 upstream_status=%s",
                exc.code,
            )
            self._send_upstream_error(exc, trace=trace)
            self._finish_trace(
                trace,
                "upstream_error",
                http_status=exc.code,
                stream=original_stream,
            )
            return ProxyResponseResult(False)
        except URLError as exc:
            LOG.warning(
                "多模态第二遍上游请求失败 reason=%s",
                exc.reason,
            )
            self._send_json(
                502,
                {"error": {"message": f"上游请求失败: {exc.reason}"}},
                trace=trace,
            )
            self._finish_trace(trace, "upstream_error", http_status=502)
            return ProxyResponseResult(False)

        with second_response:
            second_status = getattr(second_response, "status", 200)
            if self.config.verbose:
                LOG.info(
                    "多模态第二遍响应 status=%s stream=%s elapsed_ms=%s",
                    second_status,
                    original_stream,
                    elapsed_ms(started),
                )

            if original_stream:
                return self._proxy_streaming_response(
                    second_response,
                    prepared.original_model,
                    second_pass_payload["messages"],
                    prepared.cache_namespace,
                    prepared.recovery_notice,
                    trace=trace,
                    record_response_scope=prepared.record_response_scope,
                    record_response_messages=prepared.record_response_messages,
                    record_response_contexts=prepared.record_response_contexts,
                )
            else:
                return self._proxy_regular_response(
                    second_response,
                    prepared.original_model,
                    second_pass_payload["messages"],
                    prepared.cache_namespace,
                    prepared.recovery_notice,
                    trace=trace,
                    record_response_scope=prepared.record_response_scope,
                    record_response_messages=prepared.record_response_messages,
                    record_response_contexts=prepared.record_response_contexts,
                )

    def _proxy_regular_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
        record_response_scope: str | None = None,
        record_response_messages: list[dict[str, Any]] | None = None,
        record_response_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
    ) -> ProxyResponseResult:
        body = read_response_body(response)
        upstream_body = body
        usage = usage_from_body(upstream_body)
        try:
            body = rewrite_response_body(
                body,
                original_model,
                self.reasoning_store,
                request_messages,
                cache_namespace,
                content_prefix=recovery_notice,
                scope=record_response_scope,
                prior_messages=record_response_messages,
                recording_contexts=record_response_contexts,
                display_reasoning=self.config.display_reasoning,
                collapsible_reasoning=self.config.collapsible_reasoning,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOG.warning("重写上游 JSON 响应失败: %s", exc)

        if self.config.verbose:
            log_bytes("Cursor 响应体", body)

        headers = {
            "Content-Type": response.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_upstream_response(
                status=getattr(response, "status", 200),
                headers=response_headers(response),
                body=upstream_body,
                stream=False,
            )
            try:
                upstream_payload = json.loads(upstream_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                upstream_payload = None
            if isinstance(upstream_payload, dict):
                trace.record_usage(upstream_payload.get("usage"))
            trace.record_cursor_response(
                status=getattr(response, "status", 200),
                headers=headers,
                body=body,
            )

        sent_headers = self._send_response_headers(
            getattr(response, "status", 200),
            [
                ("Content-Type", headers["Content-Type"]),
                ("Content-Length", headers["Content-Length"]),
            ],
            "发送上游响应头",
        )
        if not sent_headers:
            return ProxyResponseResult(False, usage)
        sent = self._write_to_client(body, "发送上游响应体")
        return ProxyResponseResult(sent, usage)

    def _proxy_streaming_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
        record_response_scope: str | None = None,
        record_response_messages: list[dict[str, Any]] | None = None,
        record_response_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
    ) -> ProxyResponseResult:
        if trace is not None:
            trace.record_upstream_response(
                status=getattr(response, "status", 200),
                headers=response_headers(response),
                stream=True,
            )
            trace.record_cursor_response(
                status=getattr(response, "status", 200),
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                },
            )
        sent_headers = self._send_response_headers(
            getattr(response, "status", 200),
            [
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "close"),
            ],
            "发送流式响应头",
        )
        if not sent_headers:
            return ProxyResponseResult(False)
        self.close_connection = True

        accumulator = StreamAccumulator()
        usage: dict[str, Any] | None = None
        display_adapter = (
            CursorReasoningDisplayAdapter(self.config.collapsible_reasoning)
            if self.config.display_reasoning
            else None
        )
        scope = (
            record_response_scope
            if record_response_scope is not None
            else conversation_scope(request_messages, cache_namespace)
        )
        response_prior_messages = (
            record_response_messages
            if record_response_messages is not None
            else request_messages
        )
        response_contexts = (
            record_response_contexts
            if record_response_contexts is not None
            else [(scope, response_prior_messages)]
        )
        finalized = False
        pending_recovery_notice = recovery_notice
        try:
            while True:
                try:
                    line = response.readline()
                except (HTTPException, OSError) as exc:
                    LOG.warning("读取上游流式响应失败: %s", exc)
                    return ProxyResponseResult(False, usage)
                if not line:
                    break
                (
                    rewritten,
                    finalized,
                    pending_recovery_notice,
                    chunk_usage,
                ) = self._rewrite_sse_line(
                    line,
                    original_model,
                    accumulator,
                    cache_namespace,
                    response_contexts,
                    display_adapter,
                    pending_recovery_notice,
                    trace,
                )
                if chunk_usage is not None:
                    usage = chunk_usage
                if trace is not None:
                    trace.record_stream_chunk(line, rewritten)
                if not self._write_to_client(
                    rewritten, "发送流式响应块", flush=True
                ):
                    return ProxyResponseResult(False, usage)
                if finalized:
                    break
        finally:
            # 当流在上游 [DONE] 终止符之前退出时（客户端断开、上游读取失败、
            # 异常），存储部分 reasoning。否则，中途按停止会丢弃代理已收到但未缓存的 reasoning。
            if not finalized:
                if self.config.verbose:
                    log_json(
                        "模型流式助手消息", accumulator.messages()
                    )
                stored = sum(
                    accumulator.store_reasoning(
                        self.reasoning_store,
                        ctx_scope,
                        cache_namespace,
                        prior_messages,
                    )
                    for ctx_scope, prior_messages in response_contexts
                )
                if self.config.verbose and stored:
                    LOG.info(
                        "退出前已存储 %s 个流式 reasoning 缓存键",
                        stored,
                    )
        return ProxyResponseResult(True, usage)

    def _rewrite_sse_line(
        self,
        line: bytes,
        original_model: str,
        accumulator: StreamAccumulator,
        cache_namespace: str,
        response_contexts: list[tuple[str, list[dict[str, Any]]]],
        display_adapter: CursorReasoningDisplayAdapter | None,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
    ) -> tuple[bytes, bool, str | None, dict[str, Any] | None]:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line, False, recovery_notice, None

        data = stripped[len(b"data:") :].strip()
        if data == b"[DONE]":
            if self.config.verbose:
                log_json("模型流式助手消息", accumulator.messages())
            stored = sum(
                accumulator.store_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("已存储 %s 个流式 reasoning 缓存键", stored)
            prefix = b""
            if display_adapter is None:
                if recovery_notice:
                    prefix += sse_data(
                        recovery_notice_chunk(original_model, recovery_notice)
                    )
                return prefix + b"data: [DONE]\n\n", True, None, None
            closing_chunk = display_adapter.flush_chunk(original_model)
            if closing_chunk is not None:
                prefix += sse_data(closing_chunk)
            if recovery_notice:
                prefix += sse_data(
                    recovery_notice_chunk(original_model, recovery_notice)
                )
            return prefix + b"data: [DONE]\n\n", True, None, None

        try:
            chunk = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line, False, recovery_notice, None

        if isinstance(chunk, dict):
            if recovery_notice and inject_recovery_notice(chunk, recovery_notice):
                recovery_notice = None
            accumulator.ingest_chunk(chunk)
            stored = sum(
                accumulator.store_ready_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("已存储 %s 个流式 reasoning 缓存键", stored)
            chunk_usage = chunk.get("usage")
            if trace is not None:
                trace.record_usage(chunk_usage)
            if display_adapter is not None:
                display_adapter.rewrite_chunk(chunk)
            if "model" in chunk:
                chunk["model"] = original_model
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
            return (
                (
                    b"data: "
                    + json.dumps(
                        chunk, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    + ending
                ),
                False,
                recovery_notice,
                chunk_usage if isinstance(chunk_usage, dict) else None,
            )
        return line, False, recovery_notice, None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行本地 DeepSeek Cursor 代理")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=f"YAML 配置文件，默认 {default_config_path()}",
    )
    parser.add_argument("--host", help="绑定主机，默认来自配置或 127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        help="绑定端口，默认来自配置或 9000",
    )
    parser.add_argument(
        "--model",
        help=(
            "请求未指定模型时的 DeepSeek 回退模型，"
            "默认来自配置或 deepseek-v4-pro"
        ),
    )
    parser.add_argument(
        "--base-url",
        help=("DeepSeek 基础 URL，默认来自配置或 https://api.deepseek.com"),
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        help="DeepSeek 思考模式，默认来自配置或 enabled",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max", "xhigh"],
        help="DeepSeek reasoning 力度，默认来自配置或 max",
    )
    parser.add_argument(
        "--reasoning-content-path",
        type=Path,
        help=(
            "SQLite reasoning_content 缓存路径，"
            f"默认 {default_reasoning_content_path()}"
        ),
    )
    parser.add_argument(
        "--ngrok",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启动 ngrok 隧道并打印 Cursor 基础 URL",
    )
    parser.add_argument(
        "--ngrok-url",
        metavar="URL",
        help=(
            "向 ngrok 传递 --url=URL（保留端点/自定义域名）；"
            "参见 `ngrok http --help`"
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="记录详细请求元数据和完整载荷",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="将完整结构化请求追踪写入此目录",
    )
    parser.add_argument(
        "--display-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="将 reasoning_content 镜像到 Cursor 可见内容",
    )
    parser.add_argument(
        "--collapsible-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用显示时使用 Markdown details 展示镜像的 reasoning",
    )
    parser.add_argument(
        "--collasible-reasoning",
        "--collasible-resoning",
        dest="collapsible_reasoning",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-collasible-reasoning",
        "--no-collasible-resoning",
        dest="collapsible_reasoning",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="发送宽松的 CORS 响应头",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="上游请求超时（秒），默认来自配置或 300",
    )
    parser.add_argument(
        "--max-request-body-bytes",
        type=int,
        help="最大可接受请求体大小，默认来自配置",
    )
    parser.add_argument(
        "--reasoning-cache-max-age-seconds",
        type=int,
        help="reasoning 缓存行最大存活时间（秒），默认来自配置",
    )
    parser.add_argument(
        "--reasoning-cache-max-rows",
        type=int,
        help="reasoning 缓存最大行数，默认来自配置",
    )
    parser.add_argument(
        "--missing-reasoning-strategy",
        choices=["recover", "reject"],
        help=(
            "缺少必需 reasoning_content 时的处理方式："
            "recover（友好默认）或 reject（严格调试模式）"
        ),
    )
    parser.add_argument(
        "--clear-reasoning-cache",
        action="store_true",
        help="清除本地 reasoning_content SQLite 缓存并退出",
    )
    parser.add_argument(
        "--response-language",
        choices=["zh", "en", "off"],
        help="注入语言指令：zh（中文）、en（英文）或 off（关闭），默认来自配置或 zh",
    )
    return parser


def elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def log_json(label: str, payload: Any) -> None:
    LOG.info(
        "%s:\n%s",
        label,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )


def log_bytes(label: str, body: bytes) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        LOG.info("%s:\n%s", label, body.decode("utf-8", errors="replace"))
        return
    log_json(label, payload)


def usage_from_body(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            return usage
    return None


def log_cursor_request(
    payload: dict[str, Any],
    config: ProxyConfig,
) -> None:
    model = str(payload.get("model") or config.upstream_model)
    LOG.info(
        "┌ 请求 model=%s effort=%s messages=%s",
        model,
        config.reasoning_effort,
        format_count(message_count(payload)),
    )


def log_context_summary(prepared: Any) -> None:
    status = context_status(prepared)
    if status == "ok":
        LOG.info(
            "├ 上下文 status=ok reasoning_context=%s",
            format_count(prepared.patched_reasoning_messages),
        )
        return
    LOG.info(
        "├ 上下文 status=%s missing=%s recovered=%s dropped=%s",
        status,
        format_count(prepared.missing_reasoning_messages),
        format_count(prepared.recovered_reasoning_messages),
        format_count(prepared.recovery_dropped_messages),
    )


def log_send_summary(prepared: Any) -> None:
    LOG.info(
        "├ 发送    user_msgs=%s messages=%s tools=%s reasoning_content=%s",
        format_count(user_message_count(prepared.payload)),
        format_count(message_count(prepared.payload)),
        format_count(tool_count(prepared.payload)),
        format_count(reasoning_content_count(prepared.payload)),
    )


def log_stats_summary(usage: dict[str, Any] | None) -> None:
    LOG.info(
        "└ 统计    prompt=%s output=%s reasoning=%s cache_hit=%s",
        format_usage_count(usage, "prompt_tokens"),
        format_usage_count(usage, "completion_tokens"),
        format_count(reasoning_token_count(usage)),
        cache_hit_rate(usage),
    )


def context_status(prepared: Any) -> str:
    if prepared.recovered_reasoning_messages:
        return "recovered"
    if prepared.missing_reasoning_messages:
        return "missing"
    return "ok"


def message_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def tool_count(payload: dict[str, Any]) -> int:
    tools = payload.get("tools")
    return len(tools) if isinstance(tools, list) else 0


def user_message_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )


def reasoning_content_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("reasoning_content"), str)
    )


def format_usage_count(usage: dict[str, Any] | None, key: str) -> str:
    if not isinstance(usage, dict):
        return "?"
    return format_count(usage.get(key))


def reasoning_token_count(usage: dict[str, Any] | None) -> Any:
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    return details.get("reasoning_tokens")


def cache_hit_rate(usage: dict[str, Any] | None) -> str:
    if not isinstance(usage, dict):
        return "?"
    hit_tokens = usage.get("prompt_cache_hit_tokens")
    miss_tokens = usage.get("prompt_cache_miss_tokens")
    if hit_tokens is None and miss_tokens is None:
        return "?"
    hit = int_or_zero(hit_tokens)
    miss = int_or_zero(miss_tokens)
    total = hit + miss
    if not total:
        return "?"
    return f"{hit / total:.1%}"


def format_count(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def sse_data(payload: dict[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n\n"
    )


def inject_recovery_notice(chunk: dict[str, Any], notice: str) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if "content" not in delta and not delta.get("tool_calls"):
            continue
        existing_content = delta.get("content")
        delta["content"] = notice + (
            existing_content if isinstance(existing_content, str) else ""
        )
        return True
    return False


def recovery_notice_chunk(
    model: str,
    notice: str = RECOVERY_NOTICE_CONTENT,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-deepseek-cursor-proxy-recovery",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": notice},
                "finish_reason": None,
            }
        ],
    }


def summarize_chat_payload(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    tools = payload.get("tools")
    functions = payload.get("functions")
    return (
        f"model={payload.get('model')!r} "
        f"stream={bool(payload.get('stream'))} "
        f"messages={len(messages) if isinstance(messages, list) else 0} "
        f"tools={len(tools) if isinstance(tools, list) else 0} "
        f"functions={len(functions) if isinstance(functions, list) else 0} "
        f"tool_choice={payload.get('tool_choice')!r}"
    )


def read_response_body(response: Any) -> bytes:
    body = response.read()
    encoding = (response.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "deflate":
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)
    return body


def response_headers(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", {})
    if hasattr(headers, "items"):
        return {str(name): str(value) for name, value in headers.items()}
    return {}


def warn_if_insecure_upstream(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return
    host = parsed.hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        return
    LOG.warning("上游 base_url 使用明文 HTTP；bearer 令牌可能暴露")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        config = ProxyConfig.from_file(config_path=args.config_path)
    except ValueError as exc:
        configure_logging(verbose=bool(args.verbose))
        LOG.error("%s", exc)
        return 2
    updates: dict[str, Any] = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.model is not None:
        updates["upstream_model"] = args.model
    if args.base_url is not None:
        updates["upstream_base_url"] = args.base_url.rstrip("/")
    if args.thinking is not None:
        updates["thinking"] = args.thinking
    if args.reasoning_effort is not None:
        updates["reasoning_effort"] = args.reasoning_effort
    if args.reasoning_content_path is not None:
        updates["reasoning_content_path"] = args.reasoning_content_path
    if args.ngrok is not None:
        updates["ngrok"] = args.ngrok
    if args.ngrok_url is not None:
        stripped = str(args.ngrok_url).strip()
        updates["ngrok_url"] = stripped if stripped else None
    if args.verbose is not None:
        updates["verbose"] = args.verbose
    if args.trace_dir is not None:
        updates["trace_dir"] = args.trace_dir
    if args.display_reasoning is not None:
        updates["display_reasoning"] = args.display_reasoning
    if args.collapsible_reasoning is not None:
        updates["collapsible_reasoning"] = args.collapsible_reasoning
    if args.cors is not None:
        updates["cors"] = args.cors
    if args.request_timeout is not None:
        updates["request_timeout"] = args.request_timeout
    if args.max_request_body_bytes is not None:
        updates["max_request_body_bytes"] = args.max_request_body_bytes
    if args.reasoning_cache_max_age_seconds is not None:
        updates["reasoning_cache_max_age_seconds"] = (
            args.reasoning_cache_max_age_seconds
        )
    if args.reasoning_cache_max_rows is not None:
        updates["reasoning_cache_max_rows"] = args.reasoning_cache_max_rows
    if args.missing_reasoning_strategy is not None:
        updates["missing_reasoning_strategy"] = args.missing_reasoning_strategy
    if args.response_language is not None:
        updates["response_language"] = (
            None if args.response_language == "off" else args.response_language
        )
    if updates:
        config = replace(config, **updates)

    configure_logging(verbose=config.verbose)
    warn_if_insecure_upstream(config.upstream_base_url)
    store = ReasoningStore(
        config.reasoning_content_path,
        max_age_seconds=config.reasoning_cache_max_age_seconds,
        max_rows=config.reasoning_cache_max_rows,
    )
    if args.clear_reasoning_cache:
        deleted = store.clear()
        LOG.info("已清除 %s 条 reasoning 缓存行", deleted)
        store.close()
        return 0
    trace_writer: TraceWriter | None = None
    if config.trace_dir is not None:
        try:
            trace_writer = TraceWriter(config.trace_dir)
        except OSError as exc:
            LOG.error("初始化追踪目录失败: %s", exc)
            store.close()
            return 2
    server = DeepSeekProxyServer((config.host, config.port), DeepSeekProxyHandler)
    server.config = config
    server.reasoning_store = store
    server.trace_writer = trace_writer

    tunnel: NgrokTunnel | None = None
    public_url: str | None = None
    if config.ngrok:
        target_url = local_tunnel_target(config.host, config.port)
        tunnel = NgrokTunnel(target_url, ngrok_url=config.ngrok_url)
        try:
            public_url = tunnel.start()
        except RuntimeError as exc:
            LOG.error("%s", exc)
            server.server_close()
            store.close()
            return 2
    local_base_url = f"http://{config.host}:{config.port}/v1"
    api_base_url = (
        f"{public_url.rstrip('/')}/v1" if public_url is not None else local_base_url
    )

    LOG.info(
        "默认模型: %s（%s，%s）",
        config.upstream_model,
        "思考模式" if config.thinking == "enabled" else "无思考模式",
        config.reasoning_effort,
    )

    if config.verbose:
        display_reasoning = "关闭"
        if config.display_reasoning:
            display_reasoning = (
                "开启（可折叠）" if config.collapsible_reasoning else "开启"
            )
        LOG.info("显示 reasoning: %s", display_reasoning)
        LOG.info("缺失 reasoning 策略: %s", config.missing_reasoning_strategy)
        LOG.info("reasoning 缓存: %s", config.reasoning_content_path)
        LOG.warning(
            "已启用详细日志；提示词和代码可能写入 stdout"
        )
    if trace_writer is not None:
        LOG.info("追踪目录: %s", trace_writer.session_dir)
        LOG.warning("已启用追踪日志；提示词和代码将写入磁盘")
    if public_url is None and not config.ngrok:
        LOG.info("公网隧道: 关闭")
    if config.verbose:
        LOG.info("上游 URL: %s/chat/completions", config.upstream_base_url)
    LOG.info("本地基础 URL: %s", local_base_url)
    LOG.info("API 基础 URL: %s", api_base_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("正在关闭")
    finally:
        if tunnel is not None:
            tunnel.stop()
        server.server_close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
