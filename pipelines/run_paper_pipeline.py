#!/usr/bin/env python3
"""Run one of the six archived paper pairings with its recorded harness settings."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PAIRINGS = {
    'workflow_full_seq': ('full_seq', 'u1_critic_v41', 'Workflow-SFT'),
    'workflow_conservative': ('inspection', 'u1_critic_v63', 'Workflow-SFT'),
    'workflow_full_sched': ('inspection', 'u1_critic_v63', 'Workflow-SFT'),
    'workflow_selective': ('inspection', 'u1_critic_v63', 'Workflow-SFT'),
    'trajectory_r2_selective': ('inspection', 'u1_critic_v63', 'Trajectory-SFT-R2'),
    'decision_selective': ('inspection', 'u1_critic_v63', 'Decision-SFT'),
}
METADATA = {'configuration', 'scope', 'image_count'}
DERIVED = {
    'pipeline_version', 'bbox_coordinate_space', 'consolidation_enabled',
    'legacy_version_alias', 'v42_model_consolidation_required',
    'v42_online_validation_repair_rounds', 'v423_overlap_coverage_inspection',
    'v423_deterministic_issue_hydration', 'v61_component_repair',
    'v61_streaming_skeptic', 'v62_adaptive_batching', 'v62_component_repair',
    'v62_structured_recovery',
}


def configuration_argv(config):
    argv = []
    for key, value in config.items():
        if key in METADATA | DERIVED:
            continue
        if key == 'workflows':
            for workflow in value:
                argv += ['--workflow', workflow]
        elif isinstance(value, bool):
            if value:
                argv.append('--' + key.replace('_', '-'))
            elif key == 'v42_deterministic_consolidation':
                argv.append('--no-v42-deterministic-consolidation')
        else:
            argv += ['--' + key.replace('_', '-'), str(value)]
    return argv


def verify_configuration(module, args, config, package):
    effective = dict(vars(args))
    effective.update(workflows=list(args.workflows), bbox_coordinate_space='bbox_1000',
                     consolidation_enabled=not args.skip_consolidation)
    if package == 'full_seq':
        effective['pipeline_version'] = 'v4.1' if args.coverage_mode == 'harness_v41' else None
    else:
        effective.update(
            pipeline_version=module.pipeline_version_for_args(args),
            v42_model_consolidation_required=args.v42_quality_first,
            v42_online_validation_repair_rounds=3 if args.v42_quality_first else 1,
            v423_overlap_coverage_inspection=module.flash_optimizations_enabled(args),
            v423_deterministic_issue_hydration=module.flash_optimizations_enabled(args),
            v61_component_repair=module.v61_enabled(args),
            v61_streaming_skeptic=module.v61_enabled(args),
            v62_component_repair=module.v62_enabled(args),
            v62_structured_recovery=module.structured_recovery_enabled(args),
            v62_adaptive_batching=module.structured_recovery_enabled(args),
        )
    mismatches = {k: {'expected': v, 'actual': effective.get(k)} for k, v in config.items()
                  if k not in METADATA and effective.get(k) != v}
    if mismatches:
        raise ValueError('Harness settings differ from the archive: ' + json.dumps(mismatches))
    return {k: effective[k] for k in config if k not in METADATA}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configuration', choices=PAIRINGS, required=True)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--input-jsonl', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', required=True, help='Served model name for the matching paper checkpoint')
    p.add_argument('--api-key-env', default='CRITIC_API_KEY')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--limit-images', type=int, default=1)
    p.add_argument('--dry-run', action='store_true', help='Validate settings without reading images or calling a model')
    opt = p.parse_args(argv)
    package, module_name, checkpoint = PAIRINGS[opt.configuration]
    # Each invocation uses exactly one snapshot; their critic_workflows names overlap.
    sys.path.insert(0, str(ROOT / package / 'src'))
    module = importlib.import_module(module_name + '.run')
    path = ROOT / 'configurations' / (opt.configuration + '.json')
    config = json.loads(path.read_text())
    run_argv = configuration_argv(config) + [
        '--input-root', str(opt.input_root), '--input-jsonl', str(opt.input_jsonl),
        '--output-root', str(opt.output_root), '--base-url', opt.base_url,
        '--model', opt.model, '--api-key-env', opt.api_key_env,
    ]
    run_argv += ['--all', '--confirm-run-all', 'YES'] if opt.all else ['--limit-images', str(opt.limit_images)]
    # Calling main() on the inspection snapshot would inject Selective defaults.
    # Parse the explicit archived flags directly so Conservative/Full-Sched stay intact.
    args = module.parse_args(run_argv)
    effective = verify_configuration(module, args, config, package)
    report = {'configuration': opt.configuration, 'checkpoint_repo': 'BubbleT0/InspectBench-' + checkpoint,
              'package': package, 'configuration_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'effective_settings': effective, 'argv': run_argv,
              'scope': 'Archived settings on available source snapshot; no claim of bitwise regeneration of historical predictions.'}
    if opt.dry_run:
        print(json.dumps(report, indent=2))
        return
    opt.output_root.mkdir(parents=True, exist_ok=True)
    (opt.output_root / 'paper_configuration.json').write_text(json.dumps(report, indent=2) + '\n')
    asyncio.run(module.run_evaluation(args))


if __name__ == '__main__':
    main()
