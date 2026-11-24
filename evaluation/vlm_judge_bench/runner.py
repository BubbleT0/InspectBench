from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .adapters import ModelConfig, create_adapter
from .data import PredictionRecord, Sample, load_predictions
from .prompting import PredictionParseError, build_prompt, parse_prediction


def config_hash(config: ModelConfig) -> str:
    raw = json.dumps(config.public_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


CONFIG_IDENTITY_FIELDS = ("name", "adapter", "endpoint", "model", "api_key_env")


def config_identity_hash(config: ModelConfig | dict) -> str:
    payload = config.public_dict() if isinstance(config, ModelConfig) else config
    identity = {key: payload.get(key) for key in CONFIG_IDENTITY_FIELDS}
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _write_json(path: Path, value: object) -> None:
    # sample_id may contain dataset namespaces such as ``ArtiBench/item``.
    # Raw responses intentionally mirror that hierarchy, so create it before
    # opening the atomic temporary file.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temp.replace(path)


def _write_predictions(path: Path, records: dict[str, PredictionRecord], samples: list[Sample]) -> None:
    ordered_ids = [sample.sample_id for sample in samples if sample.sample_id in records]
    ordered_ids.extend(sorted(set(records) - set(ordered_ids)))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for sample_id in ordered_ids:
            handle.write(json.dumps(records[sample_id].to_dict(), ensure_ascii=False) + "\n")
    temp.replace(path)


def _record_config_history(path: Path, entries: list[dict]) -> None:
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    known = {item.get("config_hash") for item in history}
    for entry in entries:
        if entry.get("config_hash") not in known:
            history.append(entry)
            known.add(entry.get("config_hash"))
    _write_json(path, history)


class _RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 1 / requests_per_second if requests_per_second > 0 else 0.0
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.interval == 0:
            return
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_start)
            self.next_start = scheduled + self.interval
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


def _evaluate_sample(
    sample: Sample,
    config: ModelConfig,
    adapter,
    limiter: _RateLimiter,
    sample_attempt: int,
) -> tuple[PredictionRecord, dict]:
    limiter.wait()
    started = time.monotonic()
    total_attempts = 0
    raw_attempts: list[dict] = []
    try:
        parsed = None
        parse_error = None
        usage = None
        for format_attempt in range(2):
            if format_attempt:
                limiter.wait()
            response, attempts = adapter.generate(build_prompt(sample.prompt, format_retry=bool(format_attempt)), sample.image_path)
            total_attempts += attempts
            usage = response.usage
            raw_attempts.append({"format_attempt": format_attempt + 1, "response": response.raw})
            try:
                parsed = parse_prediction(response.text)
                break
            except PredictionParseError as exc:
                parse_error = exc
        if parsed is None:
            raise PredictionParseError(str(parse_error))
        no_error, boxes, warnings = parsed
        record = PredictionRecord(
            sample.sample_id, config.name, "ok", boxes, no_error,
            time.monotonic() - started, total_attempts, warnings,
            usage=usage, sample_attempts=sample_attempt,
        )
    except Exception as exc:  # keep the batch alive and make the failure auditable
        record = PredictionRecord(
            sample.sample_id, config.name, "error", [], True,
            time.monotonic() - started, total_attempts, [], f"{type(exc).__name__}: {exc}",
            sample_attempts=sample_attempt,
        )
    raw = {
        "sample_attempt": sample_attempt,
        "provider_attempts": raw_attempts,
        "error": record.error,
        "status": record.status,
    }
    return record, raw


def _existing_attempt_history(path: Path, previous_attempts: int) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload.get("run_attempts"), list):
        return list(payload["run_attempts"])
    if "attempts" in payload or "error" in payload:
        return [{
            "sample_attempt": max(1, previous_attempts),
            "provider_attempts": payload.get("attempts") or [],
            "error": payload.get("error"),
            "status": "error" if payload.get("error") else "ok",
            "legacy_record": True,
        }]
    return []


def _evaluate_until_success(
    sample: Sample,
    config: ModelConfig,
    adapter,
    limiter: _RateLimiter,
    previous: PredictionRecord | None,
    max_attempts: int,
    raw_path: Path,
) -> tuple[PredictionRecord, dict, int]:
    previous_attempts = previous.sample_attempts if previous else 0
    history = _existing_attempt_history(raw_path, previous_attempts)
    record = previous
    retry_count = 0
    for sample_attempt in range(previous_attempts + 1, max_attempts + 1):
        if sample_attempt > 1:
            retry_count += 1
        record, attempt_raw = _evaluate_sample(sample, config, adapter, limiter, sample_attempt)
        history.append(attempt_raw)
        if record.status == "ok":
            break
    if record is None:
        raise RuntimeError(f"no attempt was made for {sample.sample_id}")
    raw = {
        "sample_id": sample.sample_id,
        "model": config.name,
        "run_attempts": history,
        "attempts": history[-1].get("provider_attempts", []) if history else [],
        "error": record.error,
    }
    return record, raw, retry_count


def run_model(
    samples: list[Sample],
    config: ModelConfig,
    output_dir: str | Path,
    resume: bool = True,
    workers: int = 1,
    show_progress: bool = True,
    max_attempts: int = 3,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    model_dir = Path(output_dir) / config.name
    raw_dir = model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = model_dir / "predictions.jsonl"
    config_path = model_dir / "model_config.json"
    config_history_path = model_dir / "model_config_history.json"
    current_hash = config_hash(config)
    current_identity_hash = config_identity_hash(config)
    previous_config = None
    if resume and prediction_path.exists() and config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        previous_identity_hash = previous_config.get("config_identity_hash") or config_identity_hash(previous_config)
        if previous_identity_hash != current_identity_hash:
            raise ValueError(
                f"cannot resume {config.name}: model identity changed; "
                "use a new output directory or --no-resume"
            )
    existing = load_predictions(prediction_path) if resume else {}
    adapter = create_adapter(config)
    public_config = config.public_dict()
    public_config["config_hash"] = current_hash
    public_config["config_identity_hash"] = current_identity_hash
    if previous_config and previous_config.get("config_hash") != current_hash:
        print(
            f"[{config.name}] operational config changed; compatible resume enabled "
            f"({previous_config.get('config_hash')} -> {current_hash})",
            flush=True,
        )
        _record_config_history(config_history_path, [previous_config, public_config])
    _write_json(config_path, public_config)

    skipped = sum(existing.get(sample.sample_id) is not None and existing[sample.sample_id].status == "ok" for sample in samples)
    pre_exhausted = sum(
        existing.get(sample.sample_id) is not None
        and existing[sample.sample_id].status != "ok"
        and existing[sample.sample_id].sample_attempts >= max_attempts
        for sample in samples
    )
    pending = [
        sample for sample in samples
        if sample.sample_id not in existing
        or (existing[sample.sample_id].status != "ok" and existing[sample.sample_id].sample_attempts < max_attempts)
    ]
    records = dict(existing)
    completed = 0
    failures = pre_exhausted
    successes = skipped
    retry_attempts = 0
    limiter = _RateLimiter(config.rps)
    progress = tqdm(
        total=len(samples),
        initial=skipped + pre_exhausted,
        desc=f"{config.name} 推理",
        unit="张",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    progress.set_postfix({"成功": successes, "失败": failures, "重试": retry_attempts, "跳过": skipped}, refresh=True)

    def persist(record: PredictionRecord, raw: dict, retries: int) -> None:
        nonlocal completed, failures, successes, retry_attempts
        if record.status != "ok":
            failures += 1
        else:
            successes += 1
        retry_attempts += retries
        _write_json(raw_dir / f"{record.sample_id}.json", raw)
        records[record.sample_id] = record
        _write_predictions(prediction_path, records, samples)
        completed += 1
        progress.update(1)
        progress.set_postfix({"成功": successes, "失败": failures, "重试": retry_attempts, "跳过": skipped}, refresh=True)
        if not show_progress:
            print(
                f"[{config.name}] {record.sample_id}: {record.status} in "
                f"{record.latency_seconds:.1f}s (sample attempt {record.sample_attempts}/{max_attempts})",
                flush=True,
            )

    try:
        if workers == 1:
            for sample in pending:
                persist(*_evaluate_until_success(
                    sample, config, adapter, limiter, existing.get(sample.sample_id),
                    max_attempts, raw_dir / f"{sample.sample_id}.json",
                ))
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{config.name}-worker") as executor:
                futures = {
                    executor.submit(
                        _evaluate_until_success,
                        sample, config, adapter, limiter, existing.get(sample.sample_id),
                        max_attempts, raw_dir / f"{sample.sample_id}.json",
                    ): sample
                    for sample in pending
                }
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        record, raw, retries = future.result()
                    except Exception as exc:
                        prior = existing.get(sample.sample_id)
                        sample_attempt = min(max_attempts, (prior.sample_attempts if prior else 0) + 1)
                        record = PredictionRecord(
                            sample.sample_id, config.name, "error", [], True, 0.0, 0, [],
                            f"{type(exc).__name__}: {exc}", sample_attempts=sample_attempt,
                        )
                        raw = {"sample_id": sample.sample_id, "model": config.name, "run_attempts": [], "attempts": [], "error": record.error}
                        retries = int(sample_attempt > 1)
                    persist(record, raw, retries)
    finally:
        progress.close()
        close = getattr(adapter, "close", None)
        if close:
            close()
    final_failures = sum(record.status != "ok" for sample_id, record in records.items() if sample_id in {sample.sample_id for sample in samples})
    final_exhausted = sum(
        record.status != "ok" and record.sample_attempts >= max_attempts
        for sample_id, record in records.items() if sample_id in {sample.sample_id for sample in samples}
    )
    return {
        "model": config.name,
        "completed": completed,
        "skipped": skipped,
        "failures": final_failures,
        "exhausted": final_exhausted,
        "retry_attempts": retry_attempts,
        "workers": workers,
        "requests_per_second": config.rps,
        "show_progress": show_progress,
        "max_attempts": max_attempts,
        "prediction_path": str(prediction_path),
    }
