#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import pathlib
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

from prompts import PROFILES, render_prompt


ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_URL = "https://api.example.invalid/v1beta/models/gemini-3-flash-preview:generateContent"
LOG_LOCK = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                yield row


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def strip_json_fence(text):
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1]
        value = value.rsplit("```", 1)[0]
    return value.strip()


def response_text(payload):
    chunks = []
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                chunks.append(part["text"])
    return "".join(chunks)


class RateLimiter:
    def __init__(self, requests_per_second):
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if delay:
            time.sleep(delay)


class RunState:
    def __init__(self, plan, run_dir, initial=None):
        initial = initial or {}
        self.plan = plan
        self.run_dir = run_dir
        self.counts = {
            group: {
                leaf: int(initial.get("counts", {}).get(group, {}).get(leaf, 0))
                for leaf in leaves
            }
            for group, leaves in plan["leaf_quotas"].items()
        }
        self.requests = int(initial.get("requests", 0))
        self.processed = int(initial.get("processed", 0))
        self.accepted = int(initial.get("accepted", 0))
        self.failed = int(initial.get("failed", 0))
        self.reported_tokens = int(initial.get("reported_tokens", 0))
        self.scheduled_per_dataset = {}
        self.pool_stats = {}
        self.lock = threading.Lock()

    def group_full(self, group):
        with self.lock:
            quotas = self.plan["leaf_quotas"][group]
            return all(self.counts[group][leaf] >= quota for leaf, quota in quotas.items())

    def reserve_request(self):
        with self.lock:
            limit = self.plan["runtime_budget"]["max_http_requests_including_retries"]
            token_limit = self.plan["runtime_budget"]["max_total_reported_tokens"]
            if self.requests >= limit or self.reported_tokens >= token_limit:
                return False
            self.requests += 1
            return True

    def add_usage(self, usage):
        with self.lock:
            self.reported_tokens += int((usage or {}).get("totalTokenCount", 0))

    def set_scheduled_counts(self, counts):
        with self.lock:
            self.scheduled_per_dataset = dict(counts)

    def pool_stopped(self, dataset, group):
        with self.lock:
            stat = self.pool_stats.get(f"{dataset}::{group}", {"processed": 0, "eligible": 0})
            threshold = self.plan["runtime_budget"]["yield_switch_after_candidates"]
            minimum_rate = self.plan["runtime_budget"]["minimum_pool_eligibility_rate"]
            return stat["processed"] >= threshold and stat["eligible"] / stat["processed"] < minimum_rate

    def finalize(self, dataset, group, leaf, candidate_accepted, failed=False):
        with self.lock:
            self.processed += 1
            pool_key = f"{dataset}::{group}"
            stat = self.pool_stats.setdefault(pool_key, {"processed": 0, "eligible": 0, "accepted": 0})
            stat["processed"] += 1
            if failed:
                self.failed += 1
                return False, "request_failed"
            if not candidate_accepted or leaf is None:
                return False, "model_rejected"
            stat["eligible"] += 1
            quota = self.plan["leaf_quotas"][group].get(leaf)
            if quota is None:
                return False, "invalid_leaf"
            if self.counts[group][leaf] >= quota:
                return False, "quota_full"
            self.counts[group][leaf] += 1
            self.accepted += 1
            stat["accepted"] += 1
            return True, "accepted"

    def snapshot(self):
        with self.lock:
            return {
                "updated_at": utc_now(),
                "counts": json.loads(json.dumps(self.counts)),
                "requests": self.requests,
                "processed": self.processed,
                "accepted": self.accepted,
                "failed": self.failed,
                "reported_tokens": self.reported_tokens,
                "scheduled_per_dataset": dict(self.scheduled_per_dataset),
                "pool_stats": json.loads(json.dumps(self.pool_stats)),
            }


def make_schema(allowed_labels):
    label_enum = sorted(allowed_labels)
    return {
        "type": "OBJECT",
        "properties": {
            "eligible": {"type": "BOOLEAN"},
            "primary_leaf": {"type": "STRING", "nullable": True, "enum": label_enum},
            "ranked_labels": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "leaf": {"type": "STRING", "enum": label_enum},
                        "score": {"type": "INTEGER"},
                        "evidence": {"type": "STRING"},
                    },
                    "required": ["leaf", "score", "evidence"],
                },
            },
            "visible_evidence": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
            "reject_reason": {"type": "STRING"},
        },
        "required": [
            "eligible",
            "primary_leaf",
            "ranked_labels",
            "visible_evidence",
            "confidence",
            "reject_reason",
        ],
    }


def validate_decision(decision, profile, plan):
    labels = decision.get("ranked_labels") or []
    allowed = set(PROFILES[profile]["labels"])
    if any(item.get("leaf") not in allowed for item in labels):
        raise ValueError("response contains a leaf outside the active profile")
    scores = [item.get("score") for item in labels]
    if any(not isinstance(score, int) or score < 1 or score > 5 for score in scores):
        raise ValueError("scores must be integers from 1 to 5")
    if scores != sorted(scores, reverse=True):
        raise ValueError("ranked_labels is not sorted")
    primary = decision.get("primary_leaf")
    if labels and primary != labels[0]["leaf"]:
        raise ValueError("primary_leaf does not match ranked_labels[0]")
    acceptance = plan["acceptance"]
    accepted = bool(
        decision.get("eligible")
        and labels
        and scores[0] >= acceptance["minimum_primary_score"]
        and float(decision.get("confidence", 0)) >= acceptance["minimum_confidence"]
        and str(decision.get("visible_evidence", "")).strip()
    )
    return primary, accepted


def classify_one(row, args, plan, state, limiter, run_dir):
    item_id = str(row.get("id") or f"line-{row['_line_number']}")
    dataset = row["dataset"]
    profile = row["profile"]
    image_path = pathlib.Path(row["image_path"]).resolve()
    call_id = f"{row['_line_number']:06d}-{uuid.uuid4().hex[:8]}"
    record = {
        "call_id": call_id,
        "item_id": item_id,
        "dataset": dataset,
        "profile": profile,
        "image_path": str(image_path),
        "started_at": utc_now(),
        "manifest_row": {key: value for key, value in row.items() if key != "_line_number"},
        "attempts": [],
    }
    if profile not in PROFILES or profile == "D2-B" and not row.get("target_text"):
        record["status"] = "invalid_manifest"
        record["error"] = "unknown profile or D2-B without target_text"
        write_record(record, run_dir, state)
        return
    if state.group_full(profile):
        record["status"] = "skipped_group_quota_full"
        write_record(record, run_dir, state)
        return
    if state.pool_stopped(dataset, profile):
        record["status"] = "skipped_low_yield_pool"
        write_record(record, run_dir, state)
        return
    if not image_path.is_file():
        record["status"] = "missing_image"
        record["error"] = str(image_path)
        write_record(record, run_dir, state)
        return

    image_bytes = image_path.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    metadata = json.dumps(row.get("metadata", {}), ensure_ascii=False, indent=2)[:6000]
    prompt = render_prompt(profile, dataset, metadata)
    record["image_sha256"] = image_hash
    record["image_bytes"] = len(image_bytes)
    record["prompt"] = prompt
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode("ascii")}},
            ],
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": plan["runtime_budget"]["max_output_tokens"],
            "responseMimeType": "application/json",
            "responseSchema": make_schema(PROFILES[profile]["labels"]),
        },
    }

    decision = None
    error = None
    for attempt_number in range(plan["runtime_budget"]["max_retries"] + 1):
        if not state.reserve_request():
            error = "global request budget exhausted"
            break
        limiter.wait()
        request = urllib.request.Request(
            args.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": args.api_key},
            method="POST",
        )
        attempt = {"attempt": attempt_number + 1, "started_at": utc_now()}
        try:
            with urllib.request.urlopen(request, timeout=plan["runtime_budget"]["timeout_seconds"]) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
                attempt["http_status"] = response.status
                payload = json.loads(raw_text)
                attempt["raw_response"] = payload
                state.add_usage(payload.get("usageMetadata"))
                output = response_text(payload)
                finish_reasons = [candidate.get("finishReason") for candidate in payload.get("candidates", [])]
                if "MAX_TOKENS" in finish_reasons:
                    raise ValueError("model output truncated by MAX_TOKENS")
                parsed = json.loads(strip_json_fence(output))
                primary, model_accepted = validate_decision(parsed, profile, plan)
                decision = parsed
                attempt["completed_at"] = utc_now()
                record["attempts"].append(attempt)
                accepted, status = state.finalize(dataset, profile, primary, model_accepted)
                record.update({"decision": decision, "accepted": accepted, "status": status})
                break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempt["error"] = error
            attempt["completed_at"] = utc_now()
            record["attempts"].append(attempt)
            if attempt_number < plan["runtime_budget"]["max_retries"]:
                time.sleep((2 ** attempt_number) + random.random())

    if decision is None:
        state.finalize(dataset, profile, None, False, failed=True)
        record["accepted"] = False
        record["status"] = "request_failed"
        record["error"] = error
    record["completed_at"] = utc_now()
    write_record(record, run_dir, state)


def write_record(record, run_dir, state):
    record.setdefault("completed_at", utc_now())
    call_path = run_dir / "calls" / f"{record['call_id']}.json"
    atomic_json(call_path, record)
    with LOG_LOCK:
        with (run_dir / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        atomic_json(run_dir / "state.json", state.snapshot())


def main():
    parser = argparse.ArgumentParser(description="Quota-aware VLM benchmark pre-labeler")
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--run-dir", type=pathlib.Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", DEFAULT_URL))
    parser.add_argument("--api-key", default=os.environ.get("VLM_API_KEY"))
    parser.add_argument("--initial-state", type=pathlib.Path, help="Carry quota and global budget counters from a previous run")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("set VLM_API_KEY or pass --api-key; environment variable is recommended")

    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_dir = (args.run_dir or ROOT / "runs" / run_id).resolve()
    (run_dir / "calls").mkdir(parents=True, exist_ok=True)
    initial_state = None
    if args.initial_state:
        initial_state = json.loads(args.initial_state.read_text(encoding="utf-8"))
    atomic_json(run_dir / "run_config.json", {
        "run_id": run_id,
        "started_at": utc_now(),
        "manifest": str(args.manifest.resolve()),
        "base_url": args.base_url,
        "plan_version": plan["plan_version"],
        "initial_state": str(args.initial_state.resolve()) if args.initial_state else None,
    })
    state = RunState(plan, run_dir, initial_state)
    budget = plan["runtime_budget"]
    workers = min(max(args.workers or budget["workers"], 1), 32)
    max_candidates = min(args.max_candidates or budget["max_candidates"], budget["max_candidates"])
    rows = []
    scheduled_counts = {}
    default_dataset_cap = budget["max_candidates_per_dataset"]
    dataset_caps = budget.get("dataset_candidate_caps", {})
    for row in load_jsonl(args.manifest):
        dataset = row.get("dataset", "unknown")
        cap = min(dataset_caps.get(dataset, default_dataset_cap), default_dataset_cap)
        if scheduled_counts.get(dataset, 0) >= cap:
            continue
        rows.append(row)
        scheduled_counts[dataset] = scheduled_counts.get(dataset, 0) + 1
        if len(rows) >= max_candidates:
            break
    state.set_scheduled_counts(scheduled_counts)
    limiter = RateLimiter(budget["requests_per_second"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(classify_one, row, args, plan, state, limiter, run_dir) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    final = state.snapshot()
    final["completed_at"] = utc_now()
    atomic_json(run_dir / "state.json", final)
    print(json.dumps({"run_dir": str(run_dir), **final}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
