import json
import threading
import time

import pytest
import httpx
from PIL import Image

from vlm_judge_bench.adapters import (
    AdapterResponse,
    AnthropicMessagesAdapter,
    ModelConfig,
    OpenAIChatAdapter,
    _image_payload,
)
from vlm_judge_bench.data import PredictionRecord, Sample, load_predictions
from vlm_judge_bench.runner import run_model


class FakeAdapter:
    def generate(self, prompt, image_path):
        return AdapterResponse('{"no_error":true,"boxes":[]}', {"fake": True}, {"total_tokens": 4}), 1


class ConcurrentFakeAdapter:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, prompt, image_path):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return AdapterResponse('{"no_error":true,"boxes":[]}', {"fake": True}, {"total_tokens": 4}), 1


class FailingAdapter:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def generate(self, prompt, image_path):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"failure {self.calls}")
        return AdapterResponse('{"no_error":true,"boxes":[]}', {"call": self.calls}, None), 1


def test_runner_end_to_end_and_resume(tmp_path, monkeypatch, capsys):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    sample = Sample("s", str(image), 20, 10, "", True, (), "test", 1)
    config = ModelConfig("fake", "openai_chat", "http://unused", "fake")
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: FakeAdapter())
    first = run_model([sample], config, tmp_path / "run")
    second = run_model([sample], config, tmp_path / "run")
    assert first["completed"] == 1 and second["skipped"] == 1
    records = load_predictions(tmp_path / "run" / "fake" / "predictions.jsonl")
    assert records["s"].status == "ok"
    assert "100%" in capsys.readouterr().err


def test_resume_rejects_changed_model_identity(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    sample = Sample("s", str(image), 20, 10, "", True, (), "test", 1)
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: FakeAdapter())
    run_model([sample], ModelConfig("fake", "openai_chat", "http://one", "fake"), tmp_path / "run")
    with pytest.raises(ValueError, match="model identity changed"):
        run_model([sample], ModelConfig("fake", "openai_chat", "http://two", "fake"), tmp_path / "run")


def test_resume_allows_operational_config_change(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    sample = Sample("s", str(image), 20, 10, "", True, (), "test", 1)
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: FakeAdapter())
    run_model([sample], ModelConfig("fake", "openai_chat", "http://one", "fake", max_tokens=100), tmp_path / "run")
    result = run_model(
        [sample], ModelConfig("fake", "openai_chat", "http://one", "fake", max_tokens=32768),
        tmp_path / "run",
    )
    assert result["skipped"] == 1
    assert (tmp_path / "run" / "fake" / "model_config_history.json").exists()


def test_image_preprocessing(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (300, 100), color="red").save(image)
    mime, encoded = _image_payload(str(image), max_side=60)
    assert mime == "image/jpeg"
    assert len(encoded) > 0


def test_image_payload_detects_mislabeled_jpeg(tmp_path):
    image = tmp_path / "mislabeled.png"
    Image.new("RGB", (20, 10), color="red").save(image, format="JPEG")
    mime, encoded = _image_payload(str(image))
    assert mime == "image/jpeg"
    assert len(encoded) > 0


def test_anthropic_messages_adapter(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (1000, 2000), color="red").save(image)
    requests = []

    def handler(request):
        requests.append((request.headers, json.loads(request.content)))
        return httpx.Response(200, json={
            "content": [{
                "type": "tool_use",
                "name": "submit_result",
                "input": {
                    "no_error": False,
                    "boxes": [{
                        "bbox_1000_xyxy": [100, 1200, 400, 1800],
                        "family": "D1_Artifact/A2_StructuralArtifact",
                        "description": "malformed object",
                    }],
                },
            }],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        })

    monkeypatch.setenv("CLAUDE_TEST_KEY", "secret")
    config = ModelConfig(
        "claude", "anthropic_messages", "http://unused", "claude-test",
        api_key_env="CLAUDE_TEST_KEY", max_tokens=32768, image_max_side=2048,
    )
    adapter = AnthropicMessagesAdapter(config)
    adapter.client.close()
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        response = adapter.generate_once("prompt", str(image))
    finally:
        adapter.close()

    headers, payload = requests[0]
    assert headers["x-api-key"] == "secret"
    assert payload["messages"][0]["content"][1]["type"] == "image"
    assert payload["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert payload["max_tokens"] == 32768
    assert payload["tool_choice"] == {"type": "tool", "name": "submit_result"}
    result = json.loads(response.text)
    assert result["boxes"][0]["bbox_1000_xyxy"] == [100, 600, 400, 900]
    assert response.raw["_benchmark_request"]["pixel_bbox_normalized_to_1000"]


def test_openai_adapter_resizes_after_request_too_large(tmp_path):
    image = tmp_path / "large.png"
    Image.new("RGB", (2400, 1200), color="red").save(image)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(413, json={"error": {"message": "too large"}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"no_error":true,"boxes":[]}'}}],
            "usage": {"total_tokens": 10},
        })

    config = ModelConfig(
        "fake", "openai_chat", "http://unused", "fake", max_tokens=32768,
    )
    adapter = OpenAIChatAdapter(config)
    adapter.client.close()
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        response = adapter.generate_once("prompt", str(image))
    finally:
        adapter.close()
    assert len(requests) == 2
    assert requests[0]["max_tokens"] == 32768
    assert response.raw["_benchmark_request"] == {"image_max_side": 2048, "adaptive_resize": True}


def test_multithreaded_runner_writes_every_sample_once(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    samples = [Sample(f"s{i}", str(image), 20, 10, "", True, (), "test", i) for i in range(8)]
    adapter = ConcurrentFakeAdapter()
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: adapter)
    config = ModelConfig("fake", "openai_chat", "http://unused", "fake", rps=0)
    result = run_model(samples, config, tmp_path / "run", workers=4)
    records = load_predictions(tmp_path / "run" / "fake" / "predictions.jsonl")
    assert result["completed"] == 8 and result["failures"] == 0
    assert len(records) == 8
    assert adapter.max_active >= 2


def test_runner_supports_namespaced_sample_id_in_raw_path(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    sample = Sample("ArtiBench/item", str(image), 20, 10, "", True, (), "test", 1)
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: FakeAdapter())
    config = ModelConfig("fake", "openai_chat", "http://unused", "fake", rps=0)
    result = run_model([sample], config, tmp_path / "run")
    raw = tmp_path / "run" / "fake" / "raw" / "ArtiBench" / "item.json"
    assert result["failures"] == 0
    assert raw.exists()


def test_runner_rejects_invalid_worker_count(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        run_model([], ModelConfig("fake", "openai_chat", "http://unused", "fake"), tmp_path, workers=0)


def test_sample_retries_until_success(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    sample = Sample("s", str(image), 20, 10, "", True, (), "test", 1)
    adapter = FailingAdapter(failures=2)
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: adapter)
    result = run_model(
        [sample], ModelConfig("fake", "openai_chat", "http://unused", "fake", rps=0),
        tmp_path / "run", max_attempts=3,
    )
    record = load_predictions(tmp_path / "run" / "fake" / "predictions.jsonl")["s"]
    raw = json.loads((tmp_path / "run" / "fake" / "raw" / "s.json").read_text())
    assert adapter.calls == 3
    assert record.status == "ok" and record.sample_attempts == 3
    assert result["retry_attempts"] == 2 and len(raw["run_attempts"]) == 3


def test_exhausted_failure_is_not_retried_until_limit_increases(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    sample = Sample("s", str(image), 20, 10, "", True, (), "test", 1)
    adapter = FailingAdapter(failures=99)
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: adapter)
    config = ModelConfig("fake", "openai_chat", "http://unused", "fake", rps=0)
    first = run_model([sample], config, tmp_path / "run", max_attempts=3)
    second = run_model([sample], config, tmp_path / "run", max_attempts=3)
    assert adapter.calls == 3
    assert first["exhausted"] == 1 and second["completed"] == 0 and second["exhausted"] == 1
    run_model([sample], config, tmp_path / "run", max_attempts=4)
    assert adapter.calls == 4


def test_old_failed_prediction_is_treated_as_first_attempt(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    sample = Sample("s", str(image), 20, 10, "", True, (), "test", 1)
    model_dir = tmp_path / "run" / "fake"
    model_dir.mkdir(parents=True)
    old = PredictionRecord("s", "fake", "error", [], True, error="old failure").to_dict()
    old.pop("sample_attempts")
    (model_dir / "predictions.jsonl").write_text(json.dumps(old) + "\n")
    config = ModelConfig("fake", "openai_chat", "http://unused", "fake", rps=0)
    from vlm_judge_bench.runner import config_hash, config_identity_hash
    public = config.public_dict()
    public.update(config_hash=config_hash(config), config_identity_hash=config_identity_hash(config))
    (model_dir / "model_config.json").write_text(json.dumps(public))
    adapter = FailingAdapter(failures=0)
    monkeypatch.setattr("vlm_judge_bench.runner.create_adapter", lambda config: adapter)
    run_model([sample], config, tmp_path / "run", max_attempts=3)
    record = load_predictions(model_dir / "predictions.jsonl")["s"]
    assert adapter.calls == 1 and record.status == "ok" and record.sample_attempts == 2
