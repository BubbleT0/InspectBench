from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from PIL import Image


@dataclass(frozen=True)
class ModelConfig:
    name: str
    adapter: str
    endpoint: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: float = 180.0
    max_retries: int = 3
    max_tokens: int = 2048
    temperature: float = 0.0
    rps: float = 1.0
    image_max_side: int | None = None
    image_jpeg_quality: int = 95

    def public_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AdapterResponse:
    text: str
    raw: dict
    usage: dict | None


class AdapterError(RuntimeError):
    pass


@contextmanager
def _wall_clock_deadline(seconds: float):
    """Enforce a whole-request deadline; HTTP read timeouts reset per received chunk."""
    if not hasattr(signal, "SIGALRM") or seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError(f"request exceeded wall-clock deadline of {seconds:g}s")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("models config must contain a 'models' mapping")
    result: dict[str, ModelConfig] = {}
    for name, config in models.items():
        if not isinstance(config, dict):
            raise ValueError(f"model {name!r} config must be an object")
        result[name] = ModelConfig(name=name, **config)
    return result


def _image_payload(image_path: str, max_side: int | None = None, jpeg_quality: int = 95) -> tuple[str, str]:
    if max_side:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, subsampling=0)
        return "image/jpeg", base64.b64encode(buffer.getvalue()).decode("ascii")
    data = Path(image_path).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif data.startswith(b"BM"):
        mime = "image/bmp"
    else:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return mime, encoded


class BaseAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        if config.api_key_env and not self.api_key:
            raise AdapterError(f"required environment variable {config.api_key_env} is not set")
        self.client = httpx.Client(timeout=config.timeout_seconds)

    def generate_once(self, prompt: str, image_path: str) -> AdapterResponse:
        raise NotImplementedError

    def close(self) -> None:
        self.client.close()

    def generate(self, prompt: str, image_path: str) -> tuple[AdapterResponse, int]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            retry_delay = 2 ** (attempt - 1)
            try:
                with _wall_clock_deadline(self.config.timeout_seconds):
                    return self.generate_once(prompt, image_path), attempt
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in {408, 409, 429} and exc.response.status_code < 500:
                    raise AdapterError(f"HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
                if exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("retry-after")
                    try:
                        retry_delay = max(5.0, float(retry_after)) if retry_after else 5.0 * 2 ** (attempt - 1)
                    except ValueError:
                        retry_delay = 5.0 * 2 ** (attempt - 1)
            if attempt < self.config.max_retries:
                time.sleep(retry_delay)
        raise AdapterError(f"request failed after {self.config.max_retries} attempts: {last_error}")


class OpenAIChatAdapter(BaseAdapter):
    def generate_once(self, prompt: str, image_path: str) -> AdapterResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        candidates = [None, 2048, 1536, 1024, 768]
        if self.config.image_max_side is not None:
            candidates = [value for value in candidates if value is not None and value <= self.config.image_max_side]
            if self.config.image_max_side not in candidates:
                candidates.insert(0, self.config.image_max_side)
        response = None
        used_max_side = None
        for index, max_side in enumerate(candidates):
            mime, encoded = _image_payload(image_path, max_side, self.config.image_jpeg_quality)
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ]}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            response = self.client.post(self.config.endpoint, headers=headers, json=payload)
            used_max_side = max_side
            if response.status_code == 413 and index + 1 < len(candidates):
                continue
            response.raise_for_status()
            break
        assert response is not None
        raw = response.json()
        raw["_benchmark_request"] = {"image_max_side": used_max_side, "adaptive_resize": used_max_side is not None}
        try:
            content = raw["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError(f"unexpected OpenAI response: {json.dumps(raw)[:500]}") from exc
        return AdapterResponse(str(content), raw, raw.get("usage"))


class GeminiGenerateContentAdapter(BaseAdapter):
    def generate_once(self, prompt: str, image_path: str) -> AdapterResponse:
        mime, encoded = _image_payload(image_path, self.config.image_max_side, self.config.image_jpeg_quality)
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": encoded}},
            ]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        response = self.client.post(
            self.config.endpoint,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            json=payload,
        )
        response.raise_for_status()
        raw = response.json()
        try:
            parts = raw["candidates"][0]["content"]["parts"]
            content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError(f"unexpected Gemini response: {json.dumps(raw)[:500]}") from exc
        return AdapterResponse(content, raw, raw.get("usageMetadata"))


class AnthropicMessagesAdapter(BaseAdapter):
    def _normalize_pixel_boxes(self, value: object, image_path: str) -> bool:
        if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
            return False
        candidates = []
        for box in value["boxes"]:
            bbox = box.get("bbox_1000_xyxy") if isinstance(box, dict) else None
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    candidates.append((box, [float(item) for item in bbox]))
                except (TypeError, ValueError):
                    pass
        if not candidates or not any(item > 1000 for _, bbox in candidates for item in bbox):
            return False
        with Image.open(image_path) as image:
            width, height = image.size
            if self.config.image_max_side:
                scale = min(1.0, self.config.image_max_side / max(width, height))
                width = max(1, round(width * scale))
                height = max(1, round(height * scale))
        for box, bbox in candidates:
            converted = [
                round(bbox[0] * 1000 / width),
                round(bbox[1] * 1000 / height),
                round(bbox[2] * 1000 / width),
                round(bbox[3] * 1000 / height),
            ]
            box["bbox_1000_xyxy"] = [max(0, min(1000, item)) for item in converted]
        return True

    def generate_once(self, prompt: str, image_path: str) -> AdapterResponse:
        mime, encoded = _image_payload(
            image_path,
            self.config.image_max_side,
            self.config.image_jpeg_quality,
        )
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": encoded,
                            },
                        },
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "tools": [
                {
                    "name": "submit_result",
                    "description": "Submit the final visual anomaly judgment.",
                    "input_schema": {
                        "type": "object",
                        "required": ["no_error", "boxes"],
                        "properties": {
                            "no_error": {"type": "boolean"},
                            "boxes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": [
                                        "bbox_1000_xyxy",
                                        "family",
                                        "description",
                                    ],
                                    "properties": {
                                        "bbox_1000_xyxy": {
                                            "type": "array",
                                            "items": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "maximum": 1000,
                                            },
                                            "minItems": 4,
                                            "maxItems": 4,
                                        },
                                        "family": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "submit_result"},
        }
        response = self.client.post(
            self.config.endpoint,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )
        response.raise_for_status()
        raw = response.json()
        try:
            blocks = raw["content"]
            tool_inputs = [
                block.get("input")
                for block in blocks
                if isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == "submit_result"
            ]
            if tool_inputs:
                normalized = self._normalize_pixel_boxes(tool_inputs[0], image_path)
                content = json.dumps(tool_inputs[0], ensure_ascii=False)
                raw["_benchmark_request"] = {
                    "pixel_bbox_normalized_to_1000": normalized,
                    "image_max_side": self.config.image_max_side,
                }
            else:
                content = "".join(
                    str(block.get("text", ""))
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        except (KeyError, TypeError) as exc:
            raise AdapterError(
                f"unexpected Anthropic response: {json.dumps(raw)[:500]}"
            ) from exc
        if not content.strip():
            raise AdapterError(
                f"Anthropic response contains no text: {json.dumps(raw)[:500]}"
            )
        return AdapterResponse(content, raw, raw.get("usage"))


def create_adapter(config: ModelConfig) -> BaseAdapter:
    adapters = {
        "openai_chat": OpenAIChatAdapter,
        "gemini_generate_content": GeminiGenerateContentAdapter,
        "anthropic_messages": AnthropicMessagesAdapter,
    }
    try:
        return adapters[config.adapter](config)
    except KeyError as exc:
        raise ValueError(f"unsupported adapter {config.adapter!r}; available: {sorted(adapters)}") from exc
