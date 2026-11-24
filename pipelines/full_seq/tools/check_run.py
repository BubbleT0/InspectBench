from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.output_roots:
        manifest_path = root / "manifest.json"
        expected = (
            len(json.loads(manifest_path.read_text("utf-8")).get("images", []))
            if manifest_path.exists()
            else 0
        )
        results = read_jsonl(root / "results.jsonl")
        errors = read_jsonl(root / "errors.jsonl")
        successful = {
            row.get("sample_id")
            for row in results
            if row.get("workflow") == "adversarial_consensus"
            and row.get("status") == "ok"
            and isinstance(row.get("consolidated_result"), dict)
        }
        exhausted = {
            row.get("sample_id")
            for row in errors
            if row.get("workflow") == "adversarial_consensus"
            and row.get("sample_id") not in successful
        }
        print(
            json.dumps(
                {
                    "output_root": str(root),
                    "expected": expected,
                    "successful": len(successful),
                    "currently_failed": len(exhausted),
                    "pending": max(0, expected - len(successful) - len(exhausted)),
                    "historical_error_rows": len(errors),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
