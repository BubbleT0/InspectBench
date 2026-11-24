from __future__ import annotations

import asyncio
import base64
import copy
import json
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from critic_workflows import ModelRequest, ModelResponse
from critic_workflows.utils import parse_response


SENSITIVE_KEY_PARTS = (
    "authorization",
    "api_key",
    "image_url",
    "source_path",
    "filename",
    "reference_label",
)

NORMALIZED_BBOX_SCHEMA = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}

FINDING_SCHEMA = {
    "type": "object",
    "required": [
        "region_id",
        "bbox_norm",
        "artifact_type",
        "observation",
        "abnormality",
        "confidence",
        "evidence_ids",
    ],
    "properties": {
        "region_id": {"type": "string"},
        "bbox_norm": NORMALIZED_BBOX_SCHEMA,
        "artifact_type": {"type": "string"},
        "observation": {"type": "string"},
        "abnormality": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}


def _image_data(image) -> bytes:
    if image.data is not None:
        return image.data
    if image.path is None:
        raise ValueError(f"Image {image.id!r} has no readable source.")
    return Path(image.path).read_bytes()


class ImageDataUrlCache:
    """Thread-safe run-local cache for immutable image payloads."""

    def __init__(self) -> None:
        self._values: dict[tuple[Any, ...], str] = {}
        self._inflight: dict[tuple[Any, ...], threading.Event] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(image: Any) -> tuple[Any, ...]:
        if image.data is not None:
            return ("data", id(image.data), len(image.data), image.mime_type)
        return ("path", str(image.path), image.mime_type)

    def resolve(self, image: Any) -> tuple[str, bool]:
        key = self._key(image)
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                return cached, True
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False
        if not owner:
            event.wait()
            with self._lock:
                cached = self._values.get(key)
            if cached is None:
                return self.resolve(image)
            return cached, True
        try:
            encoded = base64.b64encode(_image_data(image)).decode("ascii")
            value = f"data:{image.mime_type};base64,{encoded}"
            with self._lock:
                self._values[key] = value
            return value, False
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()


def build_payload(
    model: str,
    request: ModelRequest,
    max_tokens: int,
    *,
    image_data_url_resolver: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    requested_max_tokens = request.metadata.get("max_tokens_override")
    if (
        isinstance(requested_max_tokens, int)
        and not isinstance(requested_max_tokens, bool)
        and requested_max_tokens > 0
    ):
        max_tokens = requested_max_tokens
    schema = copy.deepcopy(
        request.response_schema
        or {"type": "object", "additionalProperties": True}
    )
    _enrich_schema(schema)
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                request.user_prompt
                + "\n\n请使用 submit_result 工具提交最终结构化结果。"
                "所有 observation、abnormality、reason、rationale、"
                "counterargument、missing_evidence 等自然语言字段必须使用简体中文；"
                "JSON 字段名、枚举值、ID 和坐标保持 schema 要求不变。"
                "如果工具调用不可用，只返回一个合法 JSON 对象。"
            ),
        }
    ]
    for image in request.images:
        if image_data_url_resolver is None:
            encoded = base64.b64encode(_image_data(image)).decode("ascii")
            image_url = f"data:{image.mime_type};base64,{encoded}"
        else:
            image_url = image_data_url_resolver(image)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    request.system_prompt
                    + "\n所有面向用户的自然语言说明必须使用简体中文。"
                    "结构化字段名、枚举值和标识符保持原样。"
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "submit_result",
                    "description": "Submit the final structured result for this stage.",
                    "parameters": schema,
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "submit_result"},
        },
    }


def _enrich_schema(schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    properties = schema.get("properties")
    if isinstance(properties, dict):
        if "bbox_norm" in properties:
            properties["bbox_norm"] = copy.deepcopy(NORMALIZED_BBOX_SCHEMA)
        findings = properties.get("findings")
        if isinstance(findings, dict) and findings.get("type") == "array":
            item_schema = findings.get("items")
            if not isinstance(item_schema, dict) or not item_schema.get("properties"):
                findings["items"] = copy.deepcopy(FINDING_SCHEMA)
        for child in properties.values():
            _enrich_schema(child)
    if "items" in schema:
        _enrich_schema(schema["items"])


def parse_assistant_message(
    message: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "submit_result":
            continue
        arguments = function.get("arguments")
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(value, Mapping):
            raise ValueError("submit_result arguments must decode to an object.")
        return dict(value), {
            "transport": "result-tool",
            "content": message.get("content") or "",
            "tool_arguments": arguments,
        }
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Assistant returned neither submit_result nor JSON content.")
    value = parse_response(ModelResponse(text=content))
    return value, {
        "transport": "prompt",
        "content": content,
        "tool_arguments": None,
    }


class StructuredOutputError(ValueError):
    """The model response arrived, but its structured payload was malformed."""

    def __init__(self, cause: Exception):
        super().__init__(f"Structured output parse failed: {cause}")
        self.cause = cause


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in SENSITIVE_KEY_PARTS)
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes omitted>"
    return value


class TraceWriter:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._sequence = 0

    def append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._sequence += 1
            value = {"sequence": self._sequence, **_redact(dict(record))}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(value, ensure_ascii=False, default=str) + "\n"
                )
                handle.flush()


class UrllibJsonTransport:
    async def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        def send() -> dict[str, Any]:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=dict(headers),
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError("Model response must be a JSON object.")
            return value

        return await asyncio.to_thread(send)


def _retryable(
    exc: Exception,
    *,
    retry_structured_output: bool = False,
) -> bool:
    if isinstance(exc, StructuredOutputError):
        return retry_structured_output
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429} or exc.code >= 500
    return isinstance(
        exc,
        (
            TimeoutError,
            urllib.error.URLError,
            ConnectionError,
            asyncio.TimeoutError,
        ),
    )


class OpenAICompatibleModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        sample_id: str,
        workflow: str,
        trace_writer: TraceWriter,
        limiter: asyncio.Semaphore,
        transport: Any | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = 2400,
        max_retries: int = 2,
        retry_delay: float = 1.5,
        retry_structured_output: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sample_id = sample_id
        self.workflow = workflow
        self.trace_writer = trace_writer
        self.limiter = limiter
        self.transport = transport or UrllibJsonTransport()
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_structured_output = retry_structured_output
        self._image_data_url_cache = ImageDataUrlCache()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        image_cache_hits = 0
        image_cache_misses = 0

        def resolve_image_data_url(image: Any) -> str:
            nonlocal image_cache_hits, image_cache_misses
            value, cache_hit = self._image_data_url_cache.resolve(image)
            if cache_hit:
                image_cache_hits += 1
            else:
                image_cache_misses += 1
            return value

        payload_started = time.monotonic()
        payload = await asyncio.to_thread(
            build_payload,
            self.model,
            request,
            self.max_tokens,
            image_data_url_resolver=resolve_image_data_url,
        )
        payload_build_seconds = time.monotonic() - payload_started
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            recorded_at_unix = time.time()
            queue_wait_seconds = 0.0
            request_latency_seconds = 0.0
            request_started: float | None = None
            try:
                queue_started = time.monotonic()
                async with self.limiter:
                    queue_wait_seconds = time.monotonic() - queue_started
                    request_started = time.monotonic()
                    body = await self.transport.post(
                        self.base_url + "/v1/chat/completions",
                        payload,
                        headers,
                        self.timeout,
                    )
                    request_latency_seconds = (
                        time.monotonic() - request_started
                    )
                message = body["choices"][0]["message"]
                try:
                    parsed, raw = parse_assistant_message(message)
                except (ValueError, json.JSONDecodeError) as exc:
                    raise StructuredOutputError(exc) from exc
                self.trace_writer.append(
                    {
                        "sample_id": self.sample_id,
                        "workflow": self.workflow,
                        "stage": request.stage,
                        "attempt": attempt + 1,
                        "status": "ok",
                        "recorded_at_unix": recorded_at_unix,
                        "latency_seconds": round(time.monotonic() - started, 4),
                        "queue_wait_seconds": round(queue_wait_seconds, 4),
                        "request_latency_seconds": round(
                            request_latency_seconds,
                            4,
                        ),
                        "payload_build_seconds": round(
                            payload_build_seconds,
                            4,
                        ),
                        "image_cache_hits": image_cache_hits,
                        "image_cache_misses": image_cache_misses,
                        "request": {
                            "system_prompt": request.system_prompt,
                            "user_prompt": request.user_prompt,
                            "images": [
                                {
                                    "id": image.id,
                                    "parent_id": image.parent_id,
                                    "bbox_norm": (
                                        image.bbox.as_list() if image.bbox else None
                                    ),
                                    "mime_type": image.mime_type,
                                }
                                for image in request.images
                            ],
                            "response_schema": request.response_schema,
                            "metadata": request.metadata,
                        },
                        "response": raw,
                        "parsed": parsed,
                        "usage": body.get("usage", {}),
                    }
                )
                return ModelResponse(
                    parsed=parsed,
                    text=str(message.get("content") or ""),
                    raw=body,
                    usage=body.get("usage", {}),
                )
            except Exception as exc:
                last_error = exc
                if request_started is not None:
                    request_latency_seconds = time.monotonic() - request_started
                self.trace_writer.append(
                    {
                        "sample_id": self.sample_id,
                        "workflow": self.workflow,
                        "stage": request.stage,
                        "attempt": attempt + 1,
                        "status": "error",
                        "recorded_at_unix": recorded_at_unix,
                        "latency_seconds": round(time.monotonic() - started, 4),
                        "queue_wait_seconds": round(queue_wait_seconds, 4),
                        "request_latency_seconds": round(
                            request_latency_seconds,
                            4,
                        ),
                        "payload_build_seconds": round(
                            payload_build_seconds,
                            4,
                        ),
                        "image_cache_hits": image_cache_hits,
                        "image_cache_misses": image_cache_misses,
                        "request": {
                            "system_prompt": request.system_prompt,
                            "user_prompt": request.user_prompt,
                            "images": [
                                {
                                    "id": image.id,
                                    "parent_id": image.parent_id,
                                    "bbox_norm": (
                                        image.bbox.as_list() if image.bbox else None
                                    ),
                                    "mime_type": image.mime_type,
                                }
                                for image in request.images
                            ],
                            "response_schema": request.response_schema,
                            "metadata": request.metadata,
                        },
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "structured_output_retry": (
                            isinstance(exc, StructuredOutputError)
                            and self.retry_structured_output
                        ),
                    }
                )
                if attempt >= self.max_retries or not _retryable(
                    exc,
                    retry_structured_output=self.retry_structured_output,
                ):
                    raise
                await asyncio.sleep(
                    self.retry_delay * (2**attempt) + random.random() * 0.2
                )
        assert last_error is not None
        raise last_error
