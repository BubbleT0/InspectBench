#!/usr/bin/env python3
"""Reproduce all 17 point estimates using the original published scorer."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('.local/reproduced_scores'))
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = json.loads((root/'reproducibility/inputs/configurations.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, config in enumerate(configs, 1):
        out = args.output_dir/f'config_{i:02d}.json'
        subprocess.run([sys.executable, str(root/'scorer/bbox_scorer.py'),
                        '--annotations', str(root/'benchmark/annotations.jsonl'),
                        '--predictions', str(root/'reproducibility'/config['file']),
                        '--output', str(out), '--bootstrap', str(args.bootstrap)], check=True, capture_output=True)
        metrics = json.loads(out.read_text())
        score = metrics['headline']['elastic_bbox_score']
        recall = metrics['detection']['recall_0_1']
        assert abs(score-config['expected']['headline']['elastic_bbox_score']) < 1e-8, config['name']
        assert abs(recall-config['expected']['detection']['recall_0_1']) < 1e-8, config['name']
        rows.append({'configuration':config['name'], 'S':score, 'recall_percent':100*recall,
                     'clean_FPR_percent':100*metrics['binary_decision']['clean_false_positive_rate'],
                     'failed_or_missing':metrics['dataset']['failed_or_missing_predictions']})
        print(config['name'], f'S={score:.6f}', 'verified', flush=True)
    with (args.output_dir/'scores.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('All 17 point estimates reproduced with the original published scorer.')


if __name__ == '__main__':
    main()
