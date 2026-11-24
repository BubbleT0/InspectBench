from __future__ import annotations

from typing import Any, Mapping


CORE_CONFIDENCE_THRESHOLD = 0.6
CORE_MATERIALITIES = frozenset({"major", "critical"})


def materiality_of(value: Mapping[str, Any]) -> str:
    return str(value.get("materiality", "")).strip().lower()


def candidate_is_core(
    value: Mapping[str, Any],
    *,
    allow_low_confidence_text: bool = False,
) -> bool:
    if value.get("defect_present") is not True:
        return False
    if materiality_of(value) not in CORE_MATERIALITIES:
        return False
    if str(value.get("evidence_basis", "")).strip().lower() != "visible_only":
        return False
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        allow_low_confidence_text
        or confidence > CORE_CONFIDENCE_THRESHOLD
    )
