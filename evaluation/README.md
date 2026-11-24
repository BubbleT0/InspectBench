# VLM Judge BBox Benchmark

Validate annotations:

```bash
conda run -n Zcloud python -m vlm_judge_bench validate \
  --annotations annotation/GPT_completed_211_20260803_061931.jsonl
```

API secrets are supplied only through environment variables:

```bash
export QWEN_API_KEY=...
export GEMINI_API_KEY=...
```

Run models (runs are resumable by default), score, and report:

```bash
python -m vlm_judge_bench run --annotations annotation/data.jsonl \
  --models-config configs/models.yaml --models qwen37 gemini31 local_qwen \
  --output-dir runs/example --workers 4 --rps 2 --max-attempts 3
python -m vlm_judge_bench score --run-dir runs/example
python -m vlm_judge_bench report --run-dir runs/example
```

Build a Chinese visual dashboard from one-model run directories after scoring:

```bash
python -m vlm_judge_bench dashboard \
  --run-dirs runs/full_211_qwen37 runs/full_211_gemini31 runs/full_211_local_qwen \
  --output runs/full_211_dashboard/index.html
```

The dashboard contains every aggregate and bucket metric, single-model bootstrap intervals, and paginated overlays for all ground-truth and predicted boxes. It creates one reusable WebP thumbnail per source image next to the HTML.

Pass multiple files after `--annotations` to combine newly completed annotation batches. Sample IDs must remain globally unique. Add OpenAI-compatible models in `configs/models.yaml`; only new wire protocols require a new adapter implementation.

`--workers` controls concurrent images within each model. `--rps` controls the global request start rate shared by those workers; use `--rps 0` only for a local service that has no rate limit. Prediction rows and raw responses are written by the main thread, so interrupted multithreaded runs remain resumable.

The terminal progress bar is enabled by default and includes completed images, percentage, throughput, estimated remaining time, successes, failures, and resumed/skipped rows. Pass `--no-progress` for plain per-sample log lines.

Runs retry every non-successful sample until it succeeds or reaches `--max-attempts` (default: 3). The count is stored in `predictions.jsonl`, so running the same command again preserves successful rows, resumes old failed/skipped rows, and never exceeds the configured total. Each sample-level attempt may additionally use the adapter's transient HTTP retries. Changing operational settings such as `max_tokens`, timeout, or rate limit is compatible with the same output directory; changing the endpoint, wire adapter, model identity, or API-key identity requires a new output directory.

Models return a provider-independent protocol. Corner coordinates avoid ambiguity with the annotation file's internal `xywh` representation:

```json
{
  "no_error": false,
  "boxes": [
    {
      "bbox_1000_xyxy": [100, 200, 300, 400],
      "family": "D1_Artifact/A2_StructuralArtifact",
      "description": "malformed hand"
    }
  ]
}
```

`configs/models.smoke.yaml` applies the same 1024-pixel JPEG preprocessing to every provider and limits each request to one network attempt. The normal configuration preserves original images and uses three transient-error retries.
