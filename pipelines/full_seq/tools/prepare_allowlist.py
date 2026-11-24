from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge annotation JSONL files into a path-only V4.1 allowlist."
    )
    parser.add_argument("--annotations", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    seen_samples: set[str] = set()
    seen_paths: set[Path] = set()
    for annotation in args.annotations:
        with annotation.resolve(strict=True).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                sample_id = str(value.get("sample_id", "")).strip()
                raw_path = value.get("image_absolute_path")
                if not sample_id or not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError(f"{annotation}:{line_number}: missing sample/path")
                image_path = Path(raw_path).resolve(strict=True)
                if sample_id in seen_samples:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                if image_path in seen_paths:
                    raise ValueError(f"duplicate image path: {image_path}")
                seen_samples.add(sample_id)
                seen_paths.add(image_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_absolute_path": str(image_path),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "images": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
