from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .types import ModelResponse, Verdict


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_response(response: ModelResponse) -> dict[str, Any]:
    if response.parsed is not None:
        return dict(response.parsed)
    text = response.text.strip()
    if not text:
        raise ValueError("Model returned neither parsed data nor text.")

    fenced = _FENCED_JSON.search(text)
    if fenced:
        text = fenced.group(1)

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("{")
        if start < 0:
            raise ValueError(f"Model response does not contain a JSON object: {text[:200]!r}")
        value, _ = decoder.raw_decode(text[start:])

    if not isinstance(value, Mapping):
        raise ValueError("Expected model response to decode to a JSON object.")
    return dict(value)


def safe_verdict(value: Any, *, default: Verdict = Verdict.UNCERTAIN) -> Verdict:
    normalized = str(value or "").strip().lower()
    aliases = {
        "present": Verdict.ARTIFACT_PRESENT,
        "artifact": Verdict.ARTIFACT_PRESENT,
        "artifact_present": Verdict.ARTIFACT_PRESENT,
        "yes": Verdict.ARTIFACT_PRESENT,
        "absent": Verdict.NO_ARTIFACT,
        "none": Verdict.NO_ARTIFACT,
        "no_artifact": Verdict.NO_ARTIFACT,
        "no": Verdict.NO_ARTIFACT,
        "uncertain": Verdict.UNCERTAIN,
        "abstain": Verdict.UNCERTAIN,
    }
    return aliases.get(normalized, default)


def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)

