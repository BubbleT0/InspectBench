import copy
import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "scripts" / "structured_qc.py"
SPEC = importlib.util.spec_from_file_location("structured_qc", MODULE)
qc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qc)


SCHEMA_PROMPT = '''Return only one JSON value matching this schema:
{"type":"object","required":["artifact_id","mode"],"properties":{
"artifact_id":{"type":"string","minLength":1},
"mode":{"type":"string","enum":["A","B"]},
"source_url":{"type":"string","minLength":1}}}'''


def record(answer, prompt=SCHEMA_PROMPT, prior_tool='{"artifact_id":"img-1"}'):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "content": prior_tool},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"session_id": "s1", "case": "wf1", "category": "success"},
    }


def test_prose_and_fence_are_normalized():
    src = record('done\n```json\n{"artifact_id":"img-1","mode":"A"}\n```')
    out, audit = qc.quality_record(copy.deepcopy(src))
    assert out is not None
    assert out["messages"][-1]["content"] == '{"artifact_id":"img-1","mode":"A"}'
    assert audit["normalized_turns"] == 1


def test_identical_duplicate_json_is_deduped():
    src = record('{"artifact_id":"img-1","mode":"A"}\nagain\n'
                 '{"artifact_id":"img-1","mode":"A"}')
    out, audit = qc.quality_record(copy.deepcopy(src))
    assert out is not None
    assert audit["json_status"]["multiple"] == 1


def test_conflicting_json_drops_session():
    src = record('{"artifact_id":"img-1","mode":"A"}\n'
                 '{"artifact_id":"img-2","mode":"B"}',
                 prior_tool='img-1 img-2')
    out, audit = qc.quality_record(copy.deepcopy(src))
    assert out is None
    assert "multiple_conflicting_json" in audit["errors"]


def test_optional_empty_string_is_removed():
    src = record('{"artifact_id":"img-1","mode":"A","source_url":""}')
    out, _ = qc.quality_record(copy.deepcopy(src))
    assert out is not None
    assert "source_url" not in out["messages"][-1]["content"]


def test_empty_artifact_drops_session():
    src = record('{"artifact_id":"","mode":"A"}', prior_tool="")
    out, audit = qc.quality_record(copy.deepcopy(src))
    assert out is None
    assert any("minLength" in error for error in audit["errors"])


def test_unprovenanced_artifact_drops_session():
    src = record('{"artifact_id":"made-up","mode":"A"}', prior_tool="img-real")
    out, audit = qc.quality_record(copy.deepcopy(src))
    assert out is None
    assert "unprovenanced:artifact_id" in audit["errors"]


def test_repair_prompt_drops_session():
    prompt = SCHEMA_PROMPT + "\nValidation errors: bad\nRepair only its final structured response"
    out, audit = qc.quality_record(record('{"artifact_id":"img-1","mode":"A"}', prompt))
    assert out is None
    assert "structured_repair_prompt" in audit["errors"]


def test_retry_category_drops_from_main_sft():
    src = record('{"artifact_id":"img-1","mode":"A"}')
    src["metadata"]["category"] = "retry"
    out, audit = qc.quality_record(src)
    assert out is None
    assert "retry_session" in audit["errors"]


def test_failed_status_drops_even_when_category_is_retry():
    src = record('{"artifact_id":"img-1","mode":"A"}')
    src["metadata"]["category"] = "retry"
    src["status"] = "failed"
    out, audit = qc.quality_record(src, drop_retry_sessions=False)
    assert out is None
    assert "failed_session" in audit["errors"]
