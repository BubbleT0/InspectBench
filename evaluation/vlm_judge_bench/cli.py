from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .adapters import load_model_configs
from .data import DataValidationError, load_annotations, load_predictions
from .dashboard import build_dashboard
from .metrics import compute_metrics, paired_bootstrap
from .reporting import build_report
from .runner import run_model


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_validate(args: argparse.Namespace) -> int:
    samples = load_annotations(args.annotations, check_images=not args.skip_image_decode)
    summary = {
        "images": len(samples), "positive_images": sum(bool(s.boxes) for s in samples),
        "clean_images": sum(not s.boxes for s in samples), "gt_boxes": sum(len(s.boxes) for s in samples),
        "empty_prompts": sum(not s.prompt.strip() for s in samples),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.rps is not None and args.rps < 0:
        raise ValueError("--rps must be non-negative")
    samples = load_annotations(args.annotations)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        available = {s.sample_id for s in samples}
        missing = sorted(wanted - available)
        if missing:
            raise DataValidationError(f"unknown sample IDs: {missing}")
        samples = [s for s in samples if s.sample_id in wanted]
    configs = load_model_configs(args.models_config)
    missing_models = sorted(set(args.models) - set(configs))
    if missing_models:
        raise ValueError(f"models not found in config: {missing_models}")
    selected_configs = {}
    for name in args.models:
        config = configs[name]
        if args.rps is not None:
            config = replace(config, rps=args.rps)
        selected_configs[name] = config
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "annotations": [str(Path(p).resolve()) for p in args.annotations],
        "models_config": str(Path(args.models_config).resolve()),
        "models": args.models,
        "sample_ids": [s.sample_id for s in samples],
        "workers": args.workers,
        "rps_override": args.rps,
        "max_attempts": args.max_attempts,
    }
    _json_dump(run_dir / "run_manifest.json", manifest)
    results = [
        run_model(
            samples,
            selected_configs[name],
            run_dir,
            resume=not args.no_resume,
            workers=args.workers,
            show_progress=not args.no_progress,
            max_attempts=args.max_attempts,
        )
        for name in args.models
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest = _load_manifest(run_dir)
    samples = load_annotations(manifest["annotations"])
    ids = set(manifest.get("sample_ids") or [])
    if ids:
        samples = [s for s in samples if s.sample_id in ids]
    outputs = {}
    rows_by_model = {}
    for model in manifest["models"]:
        predictions = load_predictions(run_dir / model / "predictions.jsonl")
        metrics, sample_rows = compute_metrics(samples, predictions, bootstrap=args.bootstrap, seed=args.seed)
        rows_by_model[model] = sample_rows
        _json_dump(run_dir / model / "metrics.json", metrics)
        serializable_rows = []
        for row in sample_rows:
            serializable_rows.append({k: v for k, v in row.items() if k not in {"gt_pairs", "matched"}})
        _json_dump(run_dir / model / "per_sample_metrics.json", serializable_rows)
        outputs[model] = metrics["headline"]
    if len(rows_by_model) > 1 and args.bootstrap:
        _json_dump(run_dir / "paired_comparisons.json", paired_bootstrap(rows_by_model, args.bootstrap, args.seed))
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = build_report(args.run_dir)
    print(path)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    path = build_dashboard(
        args.run_dirs,
        args.output,
        thumbnail_max_side=args.thumbnail_max_side,
        paired_iterations=args.paired_bootstrap,
        seed=args.seed,
    )
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm_judge_bench")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate annotation JSONL and referenced images")
    validate.add_argument("--annotations", nargs="+", required=True)
    validate.add_argument("--skip-image-decode", action="store_true")
    validate.set_defaults(func=cmd_validate)
    run = sub.add_parser("run", help="run one or more configured VLM judges")
    run.add_argument("--annotations", nargs="+", required=True)
    run.add_argument("--models-config", required=True)
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--sample-ids", nargs="*")
    run.add_argument("--workers", type=int, default=1, help="number of concurrent sample workers per model")
    run.add_argument("--rps", type=float, help="override global request starts per second; 0 disables rate limiting")
    run.add_argument(
        "--max-attempts", type=int, default=3,
        help="maximum persistent sample-level attempts, including attempts from earlier resumed runs",
    )
    run.add_argument("--no-progress", action="store_true", help="disable the terminal progress bar")
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=cmd_run)
    score = sub.add_parser("score", help="score predictions in an existing run")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--bootstrap", type=int, default=2000)
    score.add_argument("--seed", type=int, default=20260804)
    score.set_defaults(func=cmd_score)
    report = sub.add_parser("report", help="build cross-model Markdown and CSV reports")
    report.add_argument("--run-dir", required=True)
    report.set_defaults(func=cmd_report)
    dashboard = sub.add_parser("dashboard", help="build a Chinese visual metric and bounding-box dashboard")
    dashboard.add_argument("--run-dirs", nargs="+", required=True)
    dashboard.add_argument("--output", required=True)
    dashboard.add_argument("--thumbnail-max-side", type=int, default=900)
    dashboard.add_argument("--paired-bootstrap", type=int, default=0)
    dashboard.add_argument("--seed", type=int, default=20260804)
    dashboard.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataValidationError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from exc
