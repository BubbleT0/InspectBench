#!/usr/bin/env python3
import argparse
import json
import pathlib


HERE = pathlib.Path(__file__).resolve().parent


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def legacy_id(value):
    if value.startswith("X-AIGD::test::"):
        return value.replace("X-AIGD::test::", "X-AIGD::", 1)
    return value


def main():
    parser = argparse.ArgumentParser(description="Build a gap-filling manifest after an interrupted or incomplete run")
    parser.add_argument("previous_run", type=pathlib.Path)
    parser.add_argument("--candidates", type=pathlib.Path, default=HERE / "candidates.jsonl")
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "continuation_candidates.jsonl")
    args = parser.parse_args()
    decisions_path = args.previous_run / "decisions.jsonl"
    state = json.loads((args.previous_run / "state.json").read_text(encoding="utf-8"))
    retry_statuses = {"request_failed", "skipped_low_yield_pool"}
    retry_rows = []
    seen_ids = set()
    for decision in read_jsonl(decisions_path):
        item_id = decision["item_id"]
        seen_ids.add(item_id)
        if decision.get("status") in retry_statuses:
            retry_rows.append(decision["manifest_row"])

    new_rows = []
    for row in read_jsonl(args.candidates):
        if row["id"] not in seen_ids and legacy_id(row["id"]) not in seen_ids:
            new_rows.append(row)

    deficits = {
        group: {
            leaf: quota - state.get("counts", {}).get(group, {}).get(leaf, 0)
            for leaf, quota in leaves.items()
            if quota - state.get("counts", {}).get(group, {}).get(leaf, 0) > 0
        }
        for group, leaves in json.loads((HERE / "plan.json").read_text(encoding="utf-8"))["leaf_quotas"].items()
    }
    rows = [row for row in retry_rows + new_rows if deficits.get(row["profile"])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "total": len(rows),
        "retry_previous": len(retry_rows),
        "new_candidates": len(new_rows),
        "deficits": deficits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
