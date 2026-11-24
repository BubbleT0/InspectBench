import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).parents[1] / "scripts" / "convert_agen.py"
SPEC = importlib.util.spec_from_file_location("convert_agen", MODULE)
convert = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert)


def test_executed_and_delivered_are_success(tmp_path):
    statuses = {
        "wf_executed": "EXECUTED",
        "wf_delivered": "DELIVERED",
        "wf_review": "NEEDS_REVIEW",
        "wf_rejected": "REJECTED",
        "wf_failed": "TASK_FAILED",
    }
    for case, status in statuses.items():
        run_dir = tmp_path / "workflow-runs" / case
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps({"status": status}))

    known, failed = convert.build_case_status(str(tmp_path))
    assert known == set(statuses)
    assert "wf_executed" not in failed
    assert "wf_delivered" not in failed
    assert {"wf_review", "wf_rejected", "wf_failed"} <= failed
