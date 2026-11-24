from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from critic_workflows import ModelRequest

from .model_adapter import OpenAICompatibleModelClient, TraceWriter


TEXT_KEYS = {
    "observation",
    "abnormality",
    "reason",
    "message",
    "summary",
    "rationale",
    "counterargument",
    "missing_evidence",
    "planner_reason",
    "verifier_reason",
    "description",
}
ASCII_LETTER = re.compile(r"[A-Za-z]")


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_translatable_texts(value: Any, parent_key: str = "") -> set[str]:
    output: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            output.update(collect_translatable_texts(item, str(key)))
    elif isinstance(value, list):
        for item in value:
            output.update(collect_translatable_texts(item, parent_key))
    elif (
        isinstance(value, str)
        and parent_key in TEXT_KEYS
        and value.strip()
        and ASCII_LETTER.search(value)
    ):
        output.add(value.strip())
    return output


def localize_value(
    value: Any,
    translations: Mapping[str, str],
    parent_key: str = "",
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: localize_value(item, translations, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            localize_value(item, translations, parent_key)
            for item in value
        ]
    if isinstance(value, str) and parent_key in TEXT_KEYS:
        return translations.get(text_id(value.strip()), value)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def collect_output_texts(output_root: Path) -> set[str]:
    texts: set[str] = set()
    for filename in ("results.jsonl", "traces.jsonl"):
        for row in _jsonl(output_root / filename):
            texts.update(collect_translatable_texts(row))
    return texts


def _batches(
    values: list[str],
    *,
    max_items: int,
    max_characters: int,
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for value in values:
        if current and (
            len(current) >= max_items
            or size + len(value) > max_characters
        ):
            batches.append(current)
            current = []
            size = 0
        current.append(value)
        size += len(value)
    if current:
        batches.append(current)
    return batches


def _waves(values: list[Any], width: int) -> list[list[Any]]:
    if width < 1:
        raise ValueError("Parallel width must be at least 1.")
    return [values[index:index + width] for index in range(0, len(values), width)]


def _write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


async def translate_output(args) -> None:
    output_path = args.output_root / "translations.json"
    if output_path.exists():
        saved = json.loads(output_path.read_text("utf-8"))
        translations = dict(saved.get("translations", {}))
    else:
        translations = {}
    all_texts = collect_output_texts(args.output_root)
    pending = sorted(
        (text for text in all_texts if text_id(text) not in translations),
        key=lambda value: (len(value), value),
    )
    batches = _batches(
        pending,
        max_items=args.batch_items,
        max_characters=args.batch_characters,
    )
    client = OpenAICompatibleModelClient(
        base_url=args.base_url,
        model=args.model,
        sample_id="translation",
        workflow="translation",
        trace_writer=TraceWriter(args.output_root / "translation_traces.jsonl"),
        limiter=asyncio.Semaphore(args.max_in_flight),
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_retries=args.retries,
    )
    schema = {
        "type": "object",
        "required": ["translations"],
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "zh"],
                    "properties": {
                        "id": {"type": "string"},
                        "zh": {"type": "string"},
                    },
                },
            }
        },
    }
    async def translate_batch(index: int, batch: list[str]) -> dict[str, str]:
        items = [{"id": text_id(text), "text": text} for text in batch]
        response = await client.complete(
            ModelRequest(
                stage="translation.batch",
                system_prompt=(
                    "你是视觉质量评估报告翻译器。将每条英文完整翻译为准确、自然、"
                    "简洁的简体中文，不增删判断，不修改数字、坐标、ID、专有名词和分类代码。"
                ),
                user_prompt=(
                    "逐条翻译下面的 JSON 数组。返回相同 id 和对应中文 zh，"
                    "不得遗漏、合并或解释：\n"
                    + json.dumps(items, ensure_ascii=False)
                ),
                response_schema=schema,
            )
        )
        returned = response.parsed.get("translations", [])
        expected = {item["id"] for item in items}
        received = {
            str(item.get("id")): str(item.get("zh", "")).strip()
            for item in returned
            if isinstance(item, Mapping) and str(item.get("zh", "")).strip()
        }
        missing = expected - set(received)
        if missing:
            raise ValueError(
                f"Translation batch {index} omitted {len(missing)} ids."
            )
        return {key: received[key] for key in expected}

    indexed_batches = list(enumerate(batches, start=1))
    for wave in _waves(indexed_batches, args.max_in_flight):
        completed = await asyncio.gather(
            *(translate_batch(index, batch) for index, batch in wave)
        )
        for translated in completed:
            translations.update(translated)
        _write_atomic(
            output_path,
            {
                "version": 1,
                "source_texts": len(all_texts),
                "translated_texts": len(translations),
                "translations": translations,
            },
        )
        print(
            json.dumps(
                {
                    "completed_batches": wave[-1][0],
                    "batches": len(batches),
                    "translated": len(translations),
                    "total": len(all_texts),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-items", type=int, default=60)
    parser.add_argument("--batch-characters", type=int, default=12000)
    parser.add_argument("--max-in-flight", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    asyncio.run(translate_output(parse_args(argv)))


if __name__ == "__main__":
    main()
