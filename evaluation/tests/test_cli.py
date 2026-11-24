import json

from PIL import Image

from vlm_judge_bench.cli import main
from vlm_judge_bench.dashboard import _load_run


def test_validate_cli(tmp_path, capsys):
    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(image)
    row = {"sample_id": "s", "image_absolute_path": str(image), "image_width": 8, "image_height": 8, "prompt": "", "annotation": {"no_error": True, "boxes": []}}
    annotations = tmp_path / "a.jsonl"
    annotations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert main(["validate", "--annotations", str(annotations)]) == 0
    assert json.loads(capsys.readouterr().out)["images"] == 1


def test_dashboard_loader_expands_multiple_models(tmp_path):
    run = tmp_path / "run"
    models = ["first", "second"]
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"models": models}), encoding="utf-8"
    )
    for model in models:
        model_dir = run / model
        model_dir.mkdir()
        for filename in ("predictions.jsonl", "metrics.json", "per_sample_metrics.json"):
            (model_dir / filename).write_text("", encoding="utf-8")
    loaded = _load_run(run)
    assert [model for model, _, _ in loaded] == models
