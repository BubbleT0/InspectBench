#!/usr/bin/env python3
"""Resume a date batch after explicit approval of a failed deletion safety gate."""

import argparse
import json
from pathlib import Path

import process_backflow_date as pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", choices=("0715", "0721"))
    parser.add_argument("--approve-delete", action="store_true", required=True)
    args = parser.parse_args()

    output_root = pipeline.DEFAULT_OUTPUT / f"{args.date}_structured_qc_v1"
    pipeline_dir = output_root / "agent_data_pipeline_main_v1"
    qc_dir = output_root / "openai_structured"
    meta_path = pipeline_dir / "meta_dir" / "meta_middle_format.json"
    gate_path = pipeline_dir / "validation_safety_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "blocked" or not gate.get("over_threshold"):
        raise SystemExit(f"no blocked safety gate to resume: {gate_path}")

    preclean_logs = pipeline_dir / "logs_preclean"
    pipeline.run_logged(
        pipeline.conda_command([
            "agent-data-tools", "util", "delete-error-data",
            "-l", preclean_logs, "-w", "8",
        ]),
        pipeline_dir / "delete_error_stdout.log",
    )
    middle_meta = pipeline.update_meta_lengths(meta_path)

    postclean_logs = pipeline_dir / "logs_postclean"
    pipeline.validate_openai(
        meta_path, postclean_logs, pipeline_dir / "validate_postclean_stdout.log"
    )
    postclean_stats = pipeline.load_validation_stats(postclean_logs)
    remaining_errors = [row for row in postclean_stats if row["invalid_records"]]
    if remaining_errors:
        raise RuntimeError(f"post-clean validation still has errors: {remaining_errors}")

    swift_dir = pipeline_dir / "data_swift_format"
    if swift_dir.exists():
        raise RuntimeError(f"refusing existing Swift output: {swift_dir}")
    pipeline.run_logged(
        pipeline.conda_command([
            "agent-data-tools", "convert-swift-messages", "-i", meta_path,
            "-o", swift_dir, "-w", "8", "-e", "inf",
        ]),
        pipeline_dir / "convert_swift_stdout.log",
    )
    swift_meta, swift_total, swift_mismatches = pipeline.verify_swift(swift_dir)
    if swift_mismatches:
        raise RuntimeError(f"Swift meta length mismatches: {swift_mismatches}")

    swift_logs = pipeline_dir / "logs_swift"
    pipeline.run_logged(
        pipeline.conda_command([
            "agent-data-tools", "validate-swift-messages", "-i", swift_meta,
            "-l", swift_logs, "-w", "8", "-e", "inf",
        ]),
        pipeline_dir / "validate_swift_stdout.log",
    )
    swift_stats = pipeline.load_validation_stats(swift_logs)
    swift_errors = [row for row in swift_stats if row["invalid_records"]]
    if swift_errors:
        raise RuntimeError(f"Swift validation errors remain: {swift_errors}")

    preclean_stats = gate["datasets"]
    preclean_total = sum(row["records"] for row in preclean_stats)
    output_total = sum(int(item["length"]) for item in middle_meta.values())
    summary = {
        "date": args.date,
        "source": str(pipeline.DEFAULT_BACKFLOW / args.date / "extracted"),
        "output_root": str(output_root),
        "structured_qc": json.loads(
            (qc_dir / "structured_qc_summary.json").read_text(encoding="utf-8")
        ),
        "main_sft": {
            "deletion_approved_after_safety_gate": True,
            "preclean_records": preclean_total,
            "removed_records": preclean_total - output_total,
            "output_records": output_total,
            "preclean_validation": preclean_stats,
            "postclean_validation": postclean_stats,
        },
        "swift": {
            "meta": str(swift_meta),
            "records": swift_total,
            "datasets": len(middle_meta),
            "meta_lengths_match": True,
            "validation_errors": 0,
        },
    }
    (pipeline_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
