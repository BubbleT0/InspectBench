#!/usr/bin/env python3
"""Validate public sample alignment, annotations, predictions and optional images."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', type=Path)
    ap.add_argument('--hash-images', action='store_true')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    rows = read(repo/'benchmark/annotations.jsonl')
    by_id = {r['sample_id']:r for r in rows}
    assert len(rows) == len(by_id) == 1039
    assert set(by_id) == {f'IMG_{i:04d}' for i in range(1,1040)}
    assert sum(len(r['annotation']['boxes']) for r in rows) == 1983
    assert sum(r['annotation']['no_error'] for r in rows) == 321
    for row in rows:
        w,h = row['image_width'],row['image_height']
        boxes = row['annotation']['boxes']
        assert row['annotation']['no_error'] == (not boxes)
        for box in boxes:
            x,y,bw,bh = box['bbox_xywh']
            assert x >= 0 and y >= 0 and bw > 0 and bh > 0
            assert x+bw <= w+0.01 and y+bh <= h+0.01
    configs = json.loads((repo/'reproducibility/inputs/configurations.json').read_text())
    assert len(configs) == 17
    for config in configs:
        predictions = read(repo/'reproducibility'/config['file'])
        assert len(predictions) == 1039
        assert {r['sample_id'] for r in predictions} == set(by_id)
    scoring_rows = read(repo/'reproducibility/inputs/annotations.jsonl')
    for r in scoring_rows:
        actual = by_id[r['sample_id']]
        assert r['image_width'] == actual['image_width'] and r['image_height'] == actual['image_height']
        assert r['annotation']['no_error'] == actual['annotation']['no_error']
        assert [b['bbox_xywh'] for b in r['annotation']['boxes']] == [b['bbox_xywh'] for b in actual['annotation']['boxes']]
    legacy = [r for p in (repo/'annotations').glob('*.jsonl') for r in read(p)]
    assert len(legacy) == 1039
    for r in legacy:
        assert r['image_path'] == by_id[r['sample_id']]['image_path']
        assert [b['bbox_xywh'] for b in r['boxes']] == [b['bbox_xywh'] for b in by_id[r['sample_id']]['annotation']['boxes']]
    if args.dataset_root:
        from PIL import Image
        root = args.dataset_root.resolve()
        assert read(root/'annotations/annotations.jsonl') == rows
        fingerprints = {r['sample_id']:r for r in read(root/'image_checksums.jsonl')}
        for row in rows:
            p = (root/row['image_path']).resolve()
            assert p.is_relative_to(root) and p.is_file()
            with Image.open(p) as im:
                assert im.size == (row['image_width'],row['image_height'])
            if args.hash_images:
                assert hashlib.sha256(p.read_bytes()).hexdigest() == fingerprints[row['sample_id']]['sha256']
    print('PASS: 1,039 public IDs; 718 flawed / 321 clean; 1,983 regions; 17 aligned prediction sets.')
    if args.dataset_root:print('PASS: dataset annotations and all image dimensions' + (' and SHA-256 hashes.' if args.hash_images else '.'))


if __name__ == '__main__':
    main()
