#!/usr/bin/env python3
"""Create local runner manifests from a downloaded InspectBench dataset."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    rows = [json.loads(l) for l in (root/'annotations/annotations.jsonl').read_text().splitlines() if l]
    assert len(rows) == 1039
    inputs = []
    for row in rows:
        image = (root/row['image_path']).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError('Missing or unsafe image for ' + row['sample_id'])
        row['image_absolute_path'] = str(image)
        row['prompt'] = ''
        inputs.append({k: row[k] for k in ['sample_id','image_absolute_path','image_width','image_height']})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [('annotations.jsonl',rows), ('inputs_image_only.jsonl',inputs)]:
        (args.output_dir/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in content))
    print('Prepared 1,039 local annotations and image-only inputs in', args.output_dir)


if __name__ == '__main__':
    main()
