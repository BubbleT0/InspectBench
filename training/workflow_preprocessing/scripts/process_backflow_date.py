#!/usr/bin/env python3
"""Process one expanded agentv4 backflow date through structured QC and Swift.

The script is intentionally date-scoped and refuses an existing output root.
Run it inside tmux because agent-data-tools may take a long time.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


DEFAULT_SKILL = Path("/path/to/storage/research/.claude/skills/agen-sft-pipeline")
DEFAULT_BACKFLOW = Path("/path/to/storage/multimodal_data/research/backflow")
DEFAULT_OUTPUT = Path("/path/to/storage/multimodal_data/research/agen")
CONDA_SH = Path("/path/to/storage/research/miniconda3/etc/profile.d/conda.sh")
CONDA_ENV = "data_pipeline"
IGNORE_RULES = (
    "assistant_missing_reasoning_content",
    "conversation_missing_nonempty_reasoning_content",
)


def count_lines(path):
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def run_logged(command, log_path, env=None):
    print("RUN", " ".join(map(str, command)), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            list(map(str, command)),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def add_unique_symlinks(source_glob, destination, kind, seen):
    added = 0
    for source in sorted(glob.glob(source_glob)):
        source_path = Path(source)
        if not source_path.is_dir():
            continue
        name = source_path.name
        if name in seen:
            raise RuntimeError(f"duplicate {kind} id: {name}")
        seen.add(name)
        os.symlink(source_path, destination / name, target_is_directory=True)
        added += 1
    return added


def build_flat_harness(source_dir, flat_dir):
    harnesses = sorted(Path(path) for path in glob.glob(str(source_dir / "*" / "wf_*")))
    if not harnesses:
        raise RuntimeError(f"no expanded harnesses under {source_dir}")
    for name in ("sessions", "logs", "workflows"):
        (flat_dir / name).mkdir(parents=True)

    seen_sessions, seen_logs, seen_workflows = set(), set(), set()
    counts = Counter(harnesses=len(harnesses))
    statuses = Counter()
    for harness in harnesses:
        counts["sessions"] += add_unique_symlinks(
            str(harness / "sessions" / "*"), flat_dir / "sessions", "session", seen_sessions
        )
        counts["logs"] += add_unique_symlinks(
            str(harness / "logs" / "*"), flat_dir / "logs", "log session", seen_logs
        )
        for dirname in ("workflow-runs", "workflows"):
            for source in sorted(glob.glob(str(harness / dirname / "wf_*"))):
                source_path = Path(source)
                name = source_path.name
                if name in seen_workflows:
                    raise RuntimeError(f"duplicate workflow id: {name}")
                seen_workflows.add(name)
                os.symlink(source_path, flat_dir / "workflows" / name, target_is_directory=True)
                counts["workflows"] += 1
                result_path = source_path / "result.json"
                try:
                    status = json.loads(result_path.read_text(encoding="utf-8"))["status"]
                except Exception:
                    status = "INVALID_OR_MISSING"
                statuses[str(status)] += 1
    report = {"counts": dict(counts), "workflow_statuses": dict(statuses)}
    (flat_dir / "flat_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def safe_dataset_name(agent_type, category):
    clean_agent = re.sub(r"[^A-Za-z0-9]+", "_", agent_type).strip("_").lower()
    clean_category = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_").lower()
    return f"agen_{clean_agent}_{clean_category}"


def build_middle_data(qc_dir, pipeline_dir, date):
    middle_dir = pipeline_dir / "data_middle_format"
    meta_dir = pipeline_dir / "meta_dir"
    middle_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    meta = {}
    excluded = {}
    thinking_counts = {}
    for source in sorted(qc_dir.glob("*/*.jsonl")):
        agent_type = source.parent.name
        category = source.stem
        length = count_lines(source)
        if category != "success":
            excluded[str(source.relative_to(qc_dir))] = length
            continue
        name = safe_dataset_name(agent_type, category)
        if name in meta:
            raise RuntimeError(f"dataset name collision: {name}")
        destination = middle_dir / f"{name}.jsonl"
        shutil.copyfile(source, destination)
        thinking = Counter()
        with open(destination, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    thinking[str(bool(record.get("enable_thinking"))).lower()] += 1
        thinking_counts[name] = dict(thinking)
        meta[name] = {
            "annotation": str(destination),
            "author": "research",
            "class": "agent",
            "comment": f"{agent_type}/{category}; structured_qc_v1",
            "enable_thinking": bool(thinking["true"]),
            "expect_name": name,
            "from": f"agentv4_backflow_{date}_structured_qc_v1",
            "length": length,
            "repeat_time": 1,
            "team": "research",
            "tools": [],
        }
    if not meta:
        raise RuntimeError("no successful structured-QC shards")
    meta_path = meta_dir / "meta_middle_format.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path, excluded, thinking_counts


def conda_command(arguments):
    quoted = " ".join(subprocess.list2cmdline([str(arg)]) for arg in arguments)
    shell_command = (
        f"source {subprocess.list2cmdline([str(CONDA_SH)])}; "
        f"conda activate {CONDA_ENV}; {quoted}"
    )
    return ["bash", "-lc", shell_command]


def validate_openai(meta_path, log_dir, stdout_log):
    command = [
        "agent-data-tools", "validate-openai", "-i", meta_path,
        "-l", log_dir, "-w", "8", "-e", "inf",
    ]
    for rule in IGNORE_RULES:
        command.extend(["--ignore-rule", rule])
    run_logged(conda_command(command), stdout_log)


def load_validation_stats(log_dir):
    csv_path = log_dir / "validate_stats.csv"
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records = int(row["records"])
            valid = int(row["valid_records"])
            rows.append({
                "dataset": row["dataset"],
                "records": records,
                "valid_records": valid,
                "invalid_records": records - valid,
                "invalid_rate": (records - valid) / records if records else 0.0,
            })
    return rows


def update_meta_lengths(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for item in meta.values():
        item["length"] = count_lines(item["annotation"])
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def verify_swift(swift_dir):
    meta_path = swift_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mismatches = {}
    total = 0
    for name, item in meta.items():
        actual = count_lines(item["annotation"])
        total += actual
        if actual != int(item["length"]):
            mismatches[name] = {"meta": item["length"], "actual": actual}
    return meta_path, total, mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", choices=("0715", "0721"))
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL)
    parser.add_argument("--backflow-root", type=Path, default=DEFAULT_BACKFLOW)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-invalid-rate", type=float, default=0.30)
    args = parser.parse_args()

    source_dir = args.backflow_root / args.date / "extracted"
    output_root = args.output_base / f"{args.date}_structured_qc_v1"
    if output_root.exists():
        raise SystemExit(f"refusing existing output root: {output_root}")
    output_root.mkdir(parents=True)
    flat_dir = output_root / "flat_harness"
    raw_dir = output_root / "openai_raw"
    qc_dir = output_root / "openai_structured"
    pipeline_dir = output_root / "agent_data_pipeline_main_v1"
    flat_report = build_flat_harness(source_dir, flat_dir)

    converter = args.skill_root / "scripts" / "convert_agen.py"
    runner = args.skill_root / "scripts" / "run_pipeline.sh"
    run_logged(
        [sys.executable, converter, "--harness", flat_dir, "--out", raw_dir, "--drop-unknown-case"],
        output_root / "convert.log",
    )
    run_logged(
        ["bash", runner, "structured-qc", raw_dir, qc_dir],
        output_root / "structured_qc.log",
    )
    meta_path, excluded, thinking_counts = build_middle_data(qc_dir, pipeline_dir, args.date)

    preclean_logs = pipeline_dir / "logs_preclean"
    validate_openai(meta_path, preclean_logs, pipeline_dir / "validate_preclean_stdout.log")
    preclean_stats = load_validation_stats(preclean_logs)
    gate_failures = [row for row in preclean_stats if row["invalid_rate"] > args.max_invalid_rate]
    gate_report = {
        "threshold": args.max_invalid_rate,
        "status": "blocked" if gate_failures else "passed",
        "datasets": preclean_stats,
        "over_threshold": gate_failures,
    }
    (pipeline_dir / "validation_safety_gate.json").write_text(
        json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if gate_failures:
        print(json.dumps(gate_report, ensure_ascii=False, indent=2), flush=True)
        print("STOPPED before delete-error-data: invalid rate exceeded safety threshold", flush=True)
        return 42

    run_logged(
        conda_command(["agent-data-tools", "util", "delete-error-data", "-l", preclean_logs, "-w", "8"]),
        pipeline_dir / "delete_error_stdout.log",
    )
    middle_meta = update_meta_lengths(meta_path)
    postclean_logs = pipeline_dir / "logs_postclean"
    validate_openai(meta_path, postclean_logs, pipeline_dir / "validate_postclean_stdout.log")
    postclean_stats = load_validation_stats(postclean_logs)
    remaining_errors = [row for row in postclean_stats if row["invalid_records"]]
    if remaining_errors:
        raise RuntimeError(f"post-clean validation still has errors: {remaining_errors}")

    swift_dir = pipeline_dir / "data_swift_format"
    run_logged(
        conda_command([
            "agent-data-tools", "convert-swift-messages", "-i", meta_path,
            "-o", swift_dir, "-w", "8", "-e", "inf",
        ]),
        pipeline_dir / "convert_swift_stdout.log",
    )
    swift_meta, swift_total, swift_mismatches = verify_swift(swift_dir)
    if swift_mismatches:
        raise RuntimeError(f"Swift meta length mismatches: {swift_mismatches}")
    swift_logs = pipeline_dir / "logs_swift"
    run_logged(
        conda_command([
            "agent-data-tools", "validate-swift-messages", "-i", swift_meta,
            "-l", swift_logs, "-w", "8", "-e", "inf",
        ]),
        pipeline_dir / "validate_swift_stdout.log",
    )

    qc_summary = json.loads((qc_dir / "structured_qc_summary.json").read_text(encoding="utf-8"))
    input_total = sum(row["records"] for row in preclean_stats)
    output_total = sum(int(item["length"]) for item in middle_meta.values())
    summary = {
        "date": args.date,
        "source": str(source_dir),
        "output_root": str(output_root),
        "flat_harness": flat_report,
        "structured_qc": qc_summary,
        "main_sft": {
            "excluded_non_success_shards": excluded,
            "thinking_counts": thinking_counts,
            "preclean_records": input_total,
            "removed_records": input_total - output_total,
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
