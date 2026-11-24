#!/usr/bin/env python3
"""Verify completed artifacts and write the final per-date pipeline summary."""

import argparse
import json
from pathlib import Path

import process_backflow_date as pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", choices=("0715", "0721"))
    args = parser.parse_args()

    output_root = pipeline.DEFAULT_OUTPUT / f"{args.date}_structured_qc_v1"
    pipeline_dir = output_root / "agent_data_pipeline_main_v1"
    qc_dir = output_root / "openai_structured"
    gate = json.loads(
        (pipeline_dir / "validation_safety_gate.json").read_text(encoding="utf-8")
    )
    preclean_stats = gate["datasets"]
    postclean_stats = pipeline.load_validation_stats(pipeline_dir / "logs_postclean")
    if any(row["invalid_records"] for row in postclean_stats):
        raise RuntimeError("OpenAI post-clean validation has remaining errors")

    middle_meta = json.loads(
        (pipeline_dir / "meta_dir" / "meta_middle_format.json").read_text(encoding="utf-8")
    )
    swift_meta_path, swift_total, mismatches = pipeline.verify_swift(
        pipeline_dir / "data_swift_format"
    )
    if mismatches:
        raise RuntimeError(f"Swift meta length mismatch: {mismatches}")
    swift_validation = json.loads(
        (pipeline_dir / "validate_swift_stdout.log").read_text(encoding="utf-8")
    )
    swift_errors = {
        name: result for name, result in swift_validation.items()
        if not result.get("valid") or result.get("error_count")
    }
    if swift_errors:
        raise RuntimeError(f"Swift validation errors: {swift_errors}")

    preclean_total = sum(row["records"] for row in preclean_stats)
    output_total = sum(int(item["length"]) for item in middle_meta.values())
    summary = {
        "date": args.date,
        "source": str(pipeline.DEFAULT_BACKFLOW / args.date / "extracted"),
        "output_root": str(output_root),
        "flat_harness": json.loads(
            (output_root / "flat_harness" / "flat_manifest.json").read_text(encoding="utf-8")
        ),
        "structured_qc": json.loads(
            (qc_dir / "structured_qc_summary.json").read_text(encoding="utf-8")
        ),
        "main_sft": {
            "deletion_approved_after_safety_gate": gate.get("status") == "blocked",
            "preclean_records": preclean_total,
            "removed_records": preclean_total - output_total,
            "output_records": output_total,
            "preclean_validation": preclean_stats,
            "postclean_validation": postclean_stats,
        },
        "swift": {
            "meta": str(swift_meta_path),
            "records": swift_total,
            "datasets": len(middle_meta),
            "meta_lengths_match": True,
            "validation_errors": 0,
        },
    }
    summary_path = pipeline_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "output_records": output_total,
        "swift_records": swift_total,
        "datasets": len(middle_meta),
        "validation_errors": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
