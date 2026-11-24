#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean schema-bound assistant responses in converted aGen OpenAI JSONL.

The unit of removal is one converted agent session (one JSONL record), not an
entire top-level case.  A case is removed as a whole only when it succeeded via
the final-output fallback, because the failed native branch and fallback branch
must not be presented as a native-success trajectory.

Safe normalization:
  * prose/code fences around exactly one valid JSON value -> JSON only
  * duplicated identical JSON values -> one JSON value
  * optional empty strings that violate minLength -> remove the optional key

Drop conditions (main-SFT default):
  * unparseable/truncated JSON, conflicting valid JSON values
  * schema-invalid final response after deterministic repairs
  * empty or unprovenanced primary artifact ids
  * failed sessions, earlier retry sessions, structured repair prompts
  * final-output fallback cases
"""
import argparse
import copy
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


STRUCTURED_MARKERS = re.compile(
    r"return only one json|one json value|matching this schema|return compact json|"
    r"return json|respond with json|exactly one json|final structured response|"
    r"repair only its final structured response|返回\s*json|输出\s*json|"
    r"仅返回\s*json|只返回\s*json|return\s*\{[^\n]{0,200}\}",
    re.I,
)
REPAIR_MARKERS = re.compile(
    r"repair only its final structured response|validation errors\s*:|"
    r"previous response\s*\(truncated\)|修复.*结构化|校验错误\s*:",
    re.I,
)
FALLBACK_MARKERS = re.compile(
    r"final output failure fallback|main workflow failure reason|"
    r"gate-final-output|generate one complete deliverable image as a failure fallback",
    re.I,
)
THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)
DECODER = json.JSONDecoder()
PRIMARY_ARTIFACT_KEYS = ("artifact_id", "gen_artifact_id")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def raw_json_at(text, start):
    try:
        value, used = DECODER.raw_decode(text[start:])
        return value, start + used
    except Exception:
        return None


def json_candidates(text):
    """Return maximal, non-overlapping decodable JSON values in textual order."""
    found = []
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        parsed = raw_json_at(text, idx)
        if parsed:
            value, end = parsed
            found.append((idx, end, value))
    maximal = []
    for item in sorted(found, key=lambda x: (-(x[1] - x[0]), x[0])):
        if not any(a <= item[0] and item[1] <= b for a, b, _ in maximal):
            maximal.append(item)
    return sorted(maximal, key=lambda x: x[0])


def parse_full(text):
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def schema_after_marker(prompt):
    matches = list(re.finditer(r"matching this schema\s*:\s*", prompt, re.I))
    if not matches:
        return None
    tail = prompt[matches[-1].end():]
    pos = tail.find("{")
    if pos < 0:
        return None
    parsed = raw_json_at(tail, pos)
    if not parsed:
        return None
    value, _ = parsed
    return value if isinstance(value, dict) else None


def deterministic_schema_repair(value, schema):
    """Remove only optional empty strings forbidden by minLength."""
    value = copy.deepcopy(value)

    def walk(node, spec):
        if not isinstance(spec, dict):
            return
        if isinstance(node, dict):
            required = set(spec.get("required") or [])
            props = spec.get("properties") or {}
            for key in list(node):
                child = props.get(key)
                if not isinstance(child, dict):
                    continue
                if (key not in required and isinstance(node[key], str)
                        and node[key] == "" and child.get("minLength", 0) > 0):
                    del node[key]
                    continue
                walk(node[key], child)
        elif isinstance(node, list) and isinstance(spec.get("items"), dict):
            for item in node:
                walk(item, spec["items"])

    walk(value, schema)
    return value


def validate_schema(value, schema, path="output"):
    errors = []
    if not isinstance(schema, dict):
        return errors
    typ = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
    }.get(typ, True)
    if not type_ok:
        return [f"{path}:type:{typ}"]
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}:missing:{key}")
        props = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}:additional:{key}")
        for key, child in props.items():
            if key in value:
                errors.extend(validate_schema(value[key], child, f"{path}/{key}"))
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}:minItems:{schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}:maxItems:{schema['maxItems']}")
        child = schema.get("items")
        if child:
            for idx, item in enumerate(value):
                errors.extend(validate_schema(item, child, f"{path}/{idx}"))
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}:minLength:{schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}:maxLength:{schema['maxLength']}")
        if schema.get("enum") and value not in schema["enum"]:
            errors.append(f"{path}:enum")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}:minimum:{schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}:maximum:{schema['maximum']}")
    return errors


def iter_structured_turns(messages):
    """Yield (prompt_index, final_assistant_index, prompt) for schema-bound turns."""
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
            continue
        prompt = msg["content"]
        if not STRUCTURED_MARKERS.search(prompt):
            continue
        end = len(messages)
        for nxt in range(idx + 1, len(messages)):
            if messages[nxt].get("role") == "user":
                end = nxt
                break
        assistants = [j for j in range(idx + 1, end)
                      if messages[j].get("role") == "assistant"
                      and isinstance(messages[j].get("content"), str)
                      and messages[j]["content"].strip()]
        if assistants:
            yield idx, assistants[-1], prompt


def all_prior_text(messages, before_index):
    parts = []
    for msg in messages[:before_index]:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        for call in msg.get("tool_calls") or []:
            parts.append(json.dumps(call, ensure_ascii=False))
    return "\n".join(parts)


def check_artifact_receipt(value, prior_text):
    if not isinstance(value, dict):
        return "not_applicable", []
    errors = []
    seen = False
    for key in PRIMARY_ARTIFACT_KEYS:
        if key not in value:
            continue
        seen = True
        artifact = value.get(key)
        if not isinstance(artifact, str) or not artifact.strip():
            errors.append(f"empty:{key}")
        elif artifact not in prior_text:
            errors.append(f"unprovenanced:{key}")
    return ("verified" if seen and not errors else
            "invalid" if errors else "not_applicable"), errors


def select_structured_value(raw, prompt):
    """Return (value, status, schema_status, errors)."""
    cleaned = THINK_RE.sub("", raw).strip()
    full = parse_full(cleaned)
    candidates = json_candidates(cleaned)
    schema = schema_after_marker(prompt)
    if full is not None:
        candidates = [(0, len(cleaned), full)]
        json_status = "pure"
    elif not candidates:
        return None, "unparseable", "not_checked", ["no_parseable_json"]
    else:
        json_status = "extractable" if len(candidates) == 1 else "multiple"

    valid = []
    invalid_errors = []
    for _, _, candidate in candidates:
        repaired = deterministic_schema_repair(candidate, schema) if schema else candidate
        errors = validate_schema(repaired, schema) if schema else []
        if errors:
            invalid_errors.extend(errors)
        else:
            valid.append(repaired)
    if not valid:
        return None, json_status, "invalid", invalid_errors[:20] or ["schema_invalid"]

    unique = {canonical(v): v for v in valid}
    if len(unique) > 1:
        return None, "multiple_conflicting", "valid", ["multiple_conflicting_json"]
    selected = valid[-1]
    schema_status = "valid" if schema else "not_present"
    return selected, json_status, schema_status, []


def record_text(record):
    parts = []
    for msg in record.get("messages") or []:
        if isinstance(msg.get("content"), str):
            parts.append(msg["content"])
    return "\n".join(parts)


def quality_record(record, drop_repair_prompts=True, drop_retry_sessions=True):
    messages = record.get("messages") or []
    audit = {
        "session_id": (record.get("metadata") or {}).get("session_id"),
        "case": (record.get("metadata") or {}).get("case"),
        "agent_type": (record.get("metadata") or {}).get("agent_type"),
        "category": (record.get("metadata") or {}).get("category"),
        "normalized_turns": 0,
        "structured_turns": 0,
        "json_status": Counter(),
        "schema_status": Counter(),
        "artifact_status": Counter(),
        "errors": [],
    }
    category = (record.get("metadata") or {}).get("category")
    if category == "failed" or record.get("status") == "failed":
        audit["errors"].append("failed_session")
        return None, audit
    if drop_retry_sessions and category == "retry":
        audit["errors"].append("retry_session")
        return None, audit
    if drop_repair_prompts and any(
            msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
            and REPAIR_MARKERS.search(msg["content"])
            for msg in messages):
        audit["errors"].append("structured_repair_prompt")
        return None, audit

    for _, answer_idx, prompt in list(iter_structured_turns(messages)):
        audit["structured_turns"] += 1
        raw = messages[answer_idx]["content"]
        selected, json_status, schema_status, errors = select_structured_value(raw, prompt)
        audit["json_status"][json_status] += 1
        audit["schema_status"][schema_status] += 1
        if errors:
            audit["errors"].extend(errors)
            return None, audit
        artifact_status, artifact_errors = check_artifact_receipt(
            selected, all_prior_text(messages, answer_idx))
        audit["artifact_status"][artifact_status] += 1
        if artifact_errors:
            audit["errors"].extend(artifact_errors)
            return None, audit
        normalized = canonical(selected)
        if raw.strip() != normalized:
            messages[answer_idx]["content"] = normalized
            audit["normalized_turns"] += 1

    meta = record.setdefault("metadata", {})
    meta["quality"] = {
        "structured_turns": audit["structured_turns"],
        "normalized_turns": audit["normalized_turns"],
        "json_status": dict(audit["json_status"]),
        "schema_status": dict(audit["schema_status"]),
        "artifact_status": dict(audit["artifact_status"]),
        "execution_status": "retry" if meta.get("category") == "retry" else "native_success",
    }
    return record, audit


def dump_audit(audit):
    out = dict(audit)
    for key in ("json_status", "schema_status", "artifact_status"):
        out[key] = dict(out[key])
    return out


def load_records(input_dir, limit=0):
    records = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*", "*.jsonl"))):
        rel = os.path.relpath(path, input_dir)
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    records.append((rel, line_no, record))
                except Exception as exc:
                    records.append((rel, line_no, {"__load_error__": str(exc)}))
                if limit and len(records) >= limit:
                    return records
    return records


def write_meta(output_dir, rows):
    datasets = []
    counts = Counter(rel for rel, _ in rows)
    for rel in sorted(counts):
        datasets.append({
            "path": os.path.join(output_dir, rel),
            "length": counts[rel],
            "name": rel[:-6].replace(os.sep, "__") if rel.endswith(".jsonl") else rel,
        })
    path = os.path.join(output_dir, "meta_openai.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(datasets, handle, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="converted OpenAI shard directory")
    parser.add_argument("--output", required=True, help="new, empty output directory")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--keep-fallback", action="store_true")
    parser.add_argument("--keep-repair-prompts", action="store_true")
    parser.add_argument("--keep-retry-sessions", action="store_true",
                        help="keep earlier attempts marked category=retry (recovery-only; not recommended for main SFT)")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="smoke-test first N records")
    args = parser.parse_args()

    if os.path.exists(args.output) and any(Path(args.output).iterdir()) and not args.allow_existing_output:
        raise SystemExit(f"refusing non-empty output directory: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    summary_path = args.summary or os.path.join(args.output, "structured_qc_summary.json")
    audit_path = args.audit or os.path.join(args.output, "structured_qc_audit.jsonl")

    loaded = load_records(args.input, args.limit)
    fallback_cases = set()
    for _, _, record in loaded:
        if "__load_error__" in record:
            continue
        case = (record.get("metadata") or {}).get("case")
        if case and FALLBACK_MARKERS.search(record_text(record)):
            fallback_cases.add(case)

    stats = Counter()
    reason_counts = Counter()
    turn_counts = Counter()
    kept_rows = []
    seen_sessions = set()
    writers = {}
    with open(audit_path, "w", encoding="utf-8") as audit_file:
        for rel, line_no, original in loaded:
            stats["input_records"] += 1
            if "__load_error__" in original:
                audit = {"file": rel, "line": line_no, "action": "drop",
                         "errors": ["invalid_jsonl:" + original["__load_error__"]]}
                reason_counts["invalid_jsonl"] += 1
                stats["dropped_records"] += 1
                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                continue
            record = copy.deepcopy(original)
            meta = record.get("metadata") or {}
            session_id = meta.get("session_id")
            case = meta.get("case")
            if session_id and session_id in seen_sessions:
                audit = {"session_id": session_id, "case": case, "file": rel,
                         "line": line_no, "action": "drop", "errors": ["duplicate_session"]}
                reason_counts["duplicate_session"] += 1
                stats["dropped_records"] += 1
                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                continue
            if session_id:
                seen_sessions.add(session_id)
            if not args.keep_fallback and case in fallback_cases:
                audit = {"session_id": session_id, "case": case, "file": rel,
                         "line": line_no, "action": "drop", "errors": ["fallback_case"]}
                reason_counts["fallback_case"] += 1
                stats["dropped_records"] += 1
                audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                continue

            cleaned, audit = quality_record(
                record,
                drop_repair_prompts=not args.keep_repair_prompts,
                drop_retry_sessions=not args.keep_retry_sessions,
            )
            audit.update({"file": rel, "line": line_no})
            turn_counts.update(audit["json_status"])
            if cleaned is None:
                audit["action"] = "drop"
                stats["dropped_records"] += 1
                for error in audit["errors"] or ["quality_failed"]:
                    reason_counts[error] += 1
            else:
                audit["action"] = "normalize" if audit["normalized_turns"] else "keep"
                stats["normalized_records" if audit["normalized_turns"] else "unchanged_records"] += 1
                stats["output_records"] += 1
                out_path = os.path.join(args.output, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                writer = writers.get(out_path)
                if writer is None:
                    writer = open(out_path, "w", encoding="utf-8")
                    writers[out_path] = writer
                writer.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                kept_rows.append((rel, cleaned))
            audit_file.write(json.dumps(dump_audit(audit), ensure_ascii=False) + "\n")
    for writer in writers.values():
        writer.close()

    meta_path = write_meta(args.output, kept_rows)
    report = {
        "input": args.input,
        "output": args.output,
        "meta_openai": meta_path,
        "policy": {
            "drop_fallback_case": not args.keep_fallback,
            "drop_structured_repair_prompt": not args.keep_repair_prompts,
            "drop_retry_session": not args.keep_retry_sessions,
            "drop_failed_status": True,
            "conflicting_json": "drop_session",
            "optional_empty_minLength_string": "remove_key",
            "artifact_primary_id": "must_be_nonempty_and_present_in_prior_tool_trace",
        },
        "stats": dict(stats),
        "fallback_cases": len(fallback_cases),
        "drop_reasons": dict(reason_counts.most_common()),
        "structured_turn_status": dict(turn_counts),
        "shards": dict(Counter(rel for rel, _ in kept_rows)),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
