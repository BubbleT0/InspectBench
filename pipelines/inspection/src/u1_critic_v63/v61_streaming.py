from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Awaitable, Callable

from critic_workflows import (
    CriticContext,
    CriticResult,
    Finding,
    ImageAsset,
    TraceEvent,
    Verdict,
    VerificationStatus,
)
from critic_workflows.utils import safe_verdict

from .v42_fast import (
    V42AdaptiveContentCoverageAuditor,
    V42EditGroundedConsensusCritic,
    _chunks,
)


class V61AdaptiveContentCoverageAuditor(
    V42AdaptiveContentCoverageAuditor
):
    """V6.1 coverage auditor with completion-driven finding publication."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._stable_finding_sink: (
            Callable[[Finding], Awaitable[None]] | None
        ) = None

    def set_stable_finding_sink(
        self,
        sink: Callable[[Finding], Awaitable[None]] | None,
    ) -> None:
        self._stable_finding_sink = sink

    async def _inspect_regions_core(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        indexed = list(enumerate(regions))
        text_regions = [
            item
            for item in indexed
            if item[1].get("content_kind") == "text"
        ]
        artifact_regions = [
            item
            for item in indexed
            if item[1].get("content_kind") != "text"
        ]
        primary_batches = [
            *_chunks(text_regions, self.text_batch_size),
            *_chunks(artifact_regions, self.artifact_batch_size),
        ]
        crosscheck_batches = (
            _chunks(text_regions, self.text_crosscheck_batch_size)
            if self.text_crosscheck
            else []
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def call_batch(batch, inspection_pass):
            async with semaphore:
                values = await self._inspect_batch(
                    image,
                    batch,
                    inspection_pass=inspection_pass,
                )
            missing = [
                item
                for item, value in zip(batch, values)
                if value is None
            ]
            if not missing or len(batch) <= 1:
                return values
            if len(missing) == len(batch):
                midpoint = max(1, len(missing) // 2)
                retry_batches = [missing[:midpoint], missing[midpoint:]]
            else:
                retry_batches = [missing]
            retry_batches = [item for item in retry_batches if item]
            retry_outputs = await asyncio.gather(
                *(
                    call_batch(retry_batch, inspection_pass)
                    for retry_batch in retry_batches
                )
            )
            recovered = {
                indexed_item[0]: value
                for retry_batch, retry_values in zip(
                    retry_batches,
                    retry_outputs,
                )
                for indexed_item, value in zip(retry_batch, retry_values)
            }
            return [
                recovered.get(indexed_item[0]) if value is None else value
                for indexed_item, value in zip(batch, values)
            ]

        async def tagged_call(kind, batch, inspection_pass):
            return (
                kind,
                batch,
                await call_batch(batch, inspection_pass),
            )

        batch_tasks = [
            asyncio.create_task(tagged_call("primary", batch, "primary"))
            for batch in primary_batches
        ] + [
            asyncio.create_task(
                tagged_call(
                    "crosscheck",
                    batch,
                    "text_crosscheck",
                )
            )
            for batch in crosscheck_batches
        ]
        primary_by_index: dict[
            int, tuple[Finding | None, int, bool] | None
        ] = {}
        crosscheck_by_index: dict[
            int, tuple[Finding | None, int, bool] | None
        ] = {}
        output: dict[int, tuple[Finding | None, int, bool]] = {}
        finalize_tasks: dict[int, asyncio.Task] = {}

        async def finalize(
            index: int,
            content: Mapping[str, Any],
        ) -> tuple[int, tuple[Finding | None, int, bool]]:
            value = primary_by_index.get(index)
            if self.text_crosscheck and content.get("content_kind") == "text":
                value = self._merge_crosschecked_inspection(
                    value,
                    crosscheck_by_index.get(index),
                )
            if value is None:
                async with semaphore:
                    value = await super(
                        V61AdaptiveContentCoverageAuditor,
                        self,
                    )._inspect_region(image, content)
            finding = value[0]
            if finding is not None and self._stable_finding_sink is not None:
                await self._stable_finding_sink(finding)
            return index, value

        def ready(index: int, content: Mapping[str, Any]) -> bool:
            if index not in primary_by_index:
                return False
            return not (
                self.text_crosscheck
                and content.get("content_kind") == "text"
                and index not in crosscheck_by_index
            )

        try:
            for completed in asyncio.as_completed(batch_tasks):
                kind, batch, values = await completed
                destination = (
                    primary_by_index
                    if kind == "primary"
                    else crosscheck_by_index
                )
                for (index, content), value in zip(batch, values):
                    destination[index] = value
                    if index not in finalize_tasks and ready(index, content):
                        finalize_tasks[index] = asyncio.create_task(
                            finalize(index, content)
                        )
            for index, content in indexed:
                if index not in finalize_tasks:
                    finalize_tasks[index] = asyncio.create_task(
                        finalize(index, content)
                    )
            finalized = await asyncio.gather(
                *(finalize_tasks[index] for index, _ in indexed)
            )
        except BaseException:
            for task in [*batch_tasks, *finalize_tasks.values()]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *batch_tasks,
                *finalize_tasks.values(),
                return_exceptions=True,
            )
            raise
        output.update(finalized)
        return [output[index] for index in range(len(regions))]


class V61EditGroundedConsensusCritic(V42EditGroundedConsensusCritic):
    """V6.1 consensus critic with candidate-streamed Skeptic scheduling."""

    _QUEUE_END = object()

    @staticmethod
    def _candidate_fingerprint(role_index: int, finding: Finding) -> str:
        payload = {
            "role_index": role_index,
            "region_id": finding.region_id,
            "bbox_norm": finding.bbox.as_list(),
            "artifact_type": finding.artifact_type,
            "observation": finding.observation,
            "abnormality": finding.abnormality,
            "source": finding.source,
        }
        digest = hashlib.sha1(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        return f"c{role_index:02d}_{digest}"

    @staticmethod
    def _candidate_alias_key(finding: Finding) -> tuple[Any, ...]:
        return (
            tuple(finding.bbox.as_list()),
            finding.artifact_type,
            finding.observation.strip(),
            finding.abnormality.strip(),
            str(finding.metadata.get("edit_target", "")).strip(),
        )

    async def _stream_challenge_batch(
        self,
        image: ImageAsset,
        context: CriticContext,
        batch: Sequence[tuple[str, Finding]],
        semaphore: asyncio.Semaphore,
    ) -> list[tuple[str, VerificationStatus, dict[str, Any]]]:
        output: list[tuple[str, VerificationStatus, dict[str, Any]]] = []
        pending: list[tuple[str, Finding]] = []
        for candidate_id, finding in batch:
            if (
                self.trust_preverified
                and finding.source == "role:text_coverage"
                and finding.metadata.get("v42_batched_inspection") is True
            ):
                output.append(
                    self._apply_challenge_review(
                        candidate_id,
                        finding,
                        {
                            "status": "supported",
                            "confidence": finding.confidence,
                            "counterargument": "多尺度检查已在原图与上下文中确认。",
                            "missing_evidence": "",
                            "defect_present": True,
                            "materiality": finding.metadata.get(
                                "materiality", "major"
                            ),
                            "evidence_basis": "visible_only",
                            "grounding_status": "valid",
                            "revised_bbox_norm": None,
                            "grounding_rationale": (
                                "V6.1 多尺度检查已确认该区域适合局部重绘。"
                            ),
                            "v42_preverified_fast_path": True,
                        },
                    )
                )
            else:
                pending.append((candidate_id, finding))
        if not pending:
            return output
        async with semaphore:
            parsed = await self._request_challenge_batch(
                image,
                context,
                pending,
            )
        missing: list[tuple[str, Finding]] = []
        for candidate_id, finding in pending:
            values = parsed.get(candidate_id, [])
            if len(values) != 1:
                missing.append((candidate_id, finding))
            else:
                output.append(
                    self._apply_challenge_review(
                        candidate_id,
                        finding,
                        values[0],
                    )
                )
        for candidate_id, finding in missing:
            async with semaphore:
                output.append(
                    await self._challenge(
                        image,
                        context,
                        candidate_id,
                        finding,
                    )
                )
        return output

    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ) -> CriticResult:
        context = context or CriticContext()
        critique_started = time.monotonic()
        trace: list[TraceEvent] = []
        proposal_semaphore = asyncio.Semaphore(self.config.max_concurrency)
        challenge_semaphore = asyncio.Semaphore(self.challenge_concurrency)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        candidates: dict[str, Finding] = {}
        candidate_ids_by_role: dict[int, list[str]] = {
            index: [] for index in range(len(self.config.roles))
        }
        candidate_ids_by_identity: dict[tuple[int, int], str] = {}
        challenge_started_at: float | None = None
        challenge_completed_at: float | None = None
        challenge_batch_count = 0
        challenge_failure: asyncio.Future = (
            asyncio.get_running_loop().create_future()
        )
        async def register_candidate(
            role_index: int,
            role_name: str,
            finding: Finding,
        ) -> str | None:
            identity_key = (role_index, id(finding))
            if identity_key in candidate_ids_by_identity:
                return candidate_ids_by_identity[identity_key]
            if (
                len(candidate_ids_by_role[role_index])
                >= self.config.max_findings_per_role
            ):
                return None
            candidate_id = self._candidate_fingerprint(role_index, finding)
            if candidate_id in candidates and candidates[candidate_id] is not finding:
                suffix = 2
                base = candidate_id
                while f"{base}_{suffix}" in candidates:
                    suffix += 1
                candidate_id = f"{base}_{suffix}"
            finding.metadata["candidate_id"] = candidate_id
            finding.metadata["role"] = role_name
            finding.metadata["v61_candidate_state"] = "skeptic_pending"
            finding.metadata["v61_stable_offset_seconds"] = round(
                time.monotonic() - critique_started,
                4,
            )
            candidates[candidate_id] = finding
            candidate_ids_by_role[role_index].append(candidate_id)
            candidate_ids_by_identity[identity_key] = candidate_id
            await queue.put((candidate_id, finding))
            return candidate_id

        async def consume_challenges():
            nonlocal challenge_started_at
            nonlocal challenge_completed_at
            nonlocal challenge_batch_count
            pending: list[tuple[str, Finding]] = []
            tasks: list[asyncio.Task] = []
            def start_batch(values):
                nonlocal challenge_started_at
                nonlocal challenge_batch_count
                task = asyncio.create_task(
                    self._stream_challenge_batch(
                        image,
                        context,
                        tuple(values),
                        challenge_semaphore,
                    )
                )

                def publish_failure(completed: asyncio.Task) -> None:
                    if completed.cancelled():
                        return
                    exception = completed.exception()
                    if exception is not None and not challenge_failure.done():
                        challenge_failure.set_result(exception)

                task.add_done_callback(publish_failure)
                tasks.append(task)
                challenge_batch_count += 1
                if challenge_started_at is None:
                    challenge_started_at = time.monotonic()

            try:
                while True:
                    item = await queue.get()
                    if item is self._QUEUE_END:
                        if pending:
                            start_batch(pending)
                            pending = []
                        break
                    pending.append(item)
                    if len(pending) >= self.challenge_batch_size:
                        start_batch(pending)
                        pending = []
                grouped = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            challenge_completed_at = time.monotonic()
            return [value for group in grouped for value in group]

        challenge_task = asyncio.create_task(consume_challenges())
        role_results: dict[int, tuple[Any, list[Finding], dict[str, Any]]] = {}

        async def propose(role_index: int, role):
            async with proposal_semaphore:
                result = await self._propose(image, context, role)
            for finding in result[1]:
                await register_candidate(role_index, role.name, finding)
            return role_index, result

        text_role_index = next(
            (
                index
                for index, role in enumerate(self.config.roles)
                if role.name == "text_coverage"
            ),
            None,
        )
        sink_setter = getattr(
            self.text_auditor,
            "set_stable_finding_sink",
            None,
        )
        if text_role_index is not None and callable(sink_setter):
            text_role_name = self.config.roles[text_role_index].name

            async def publish_text_finding(finding: Finding) -> None:
                await register_candidate(
                    text_role_index,
                    text_role_name,
                    finding,
                )

            sink_setter(publish_text_finding)

        proposal_tasks = [
            asyncio.create_task(propose(index, role))
            for index, role in enumerate(self.config.roles)
        ]
        try:
            pending_proposals = set(proposal_tasks)
            while pending_proposals:
                done, still_pending = await asyncio.wait(
                    [*pending_proposals, challenge_failure],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending_proposals = {
                    task
                    for task in still_pending
                    if task is not challenge_failure
                }
                if challenge_failure in done:
                    raise challenge_failure.result()
                for completed in done:
                    role_index, role_result = completed.result()
                    role_results[role_index] = role_result
            discovery_completed_at = time.monotonic()
            await queue.put(self._QUEUE_END)
            challenged = await challenge_task
        except BaseException:
            for task in proposal_tasks:
                if not task.done():
                    task.cancel()
            if not challenge_task.done():
                challenge_task.cancel()
            await asyncio.gather(
                *proposal_tasks,
                challenge_task,
                return_exceptions=True,
            )
            raise
        finally:
            if callable(sink_setter):
                sink_setter(None)

        role_summaries: list[dict[str, Any]] = []
        for role_index, role in enumerate(self.config.roles):
            _, _, summary = role_results[role_index]
            role_summaries.append(
                {
                    "role": role.name,
                    "verdict": summary.get("verdict", "uncertain"),
                    "confidence": summary.get("confidence", 0.5),
                    "candidate_ids": candidate_ids_by_role[role_index],
                }
            )
        trace.append(
            TraceEvent(
                stage="consensus.proposals",
                message=(
                    f"{len(self.config.roles)} roles proposed "
                    f"{len(candidates)} claims."
                ),
                data={"roles": role_summaries},
            )
        )

        challenge_records: dict[str, dict[str, Any]] = {}
        for candidate_id, status, record in challenged:
            finding = candidates[candidate_id]
            finding.verification = status
            finding.metadata["skeptic"] = record
            finding.metadata["v61_candidate_state"] = "skeptic_complete"
            finding.metadata["v61_skeptic_completed_offset_seconds"] = round(
                time.monotonic() - critique_started,
                4,
            )
            challenge_records[candidate_id] = record
        trace.append(
            TraceEvent(
                stage="consensus.challenges",
                message=f"Skeptic reviewed {len(challenge_records)} claims.",
                data=challenge_records,
            )
        )

        alias_groups: dict[tuple[Any, ...], list[str]] = {}
        for candidate_id, finding in candidates.items():
            alias_groups.setdefault(
                self._candidate_alias_key(finding),
                [],
            ).append(candidate_id)
        candidate_aliases: dict[str, str] = {}
        judge_candidates = dict(candidates)
        judge_challenges = {
            candidate_id: copy.deepcopy(record)
            for candidate_id, record in challenge_records.items()
        }
        for group in alias_groups.values():
            if len(group) <= 1:
                continue
            canonical_id = min(group)
            alias_ids = sorted(
                candidate_id
                for candidate_id in group
                if candidate_id != canonical_id
            )
            canonical_finding = candidates[canonical_id]
            canonical_finding.metadata["v61_candidate_aliases"] = alias_ids
            canonical_finding.metadata["v61_alias_sources"] = [
                candidates[candidate_id].source
                for candidate_id in alias_ids
            ]
            canonical_review = judge_challenges[canonical_id]
            canonical_review["alias_reviews"] = [
                {
                    "candidate_id": candidate_id,
                    "review": copy.deepcopy(
                        challenge_records[candidate_id]
                    ),
                }
                for candidate_id in alias_ids
            ]
            for candidate_id in alias_ids:
                candidate_aliases[candidate_id] = canonical_id
                judge_candidates.pop(candidate_id, None)
                judge_challenges.pop(candidate_id, None)
        if candidate_aliases:
            for summary in role_summaries:
                folded_ids = [
                    candidate_aliases.get(candidate_id, candidate_id)
                    for candidate_id in summary["candidate_ids"]
                ]
                summary["candidate_ids"] = list(dict.fromkeys(folded_ids))

        judge_payload = await self._judge(
            image=image,
            context=context,
            role_summaries=role_summaries,
            candidates=judge_candidates,
            challenges=judge_challenges,
        )
        accepted_ids = self._valid_ids(
            judge_payload.get("accepted_candidate_ids", []),
            judge_candidates,
        )
        uncertain_ids = self._valid_ids(
            judge_payload.get("uncertain_candidate_ids", []),
            judge_candidates,
        )
        accepted = self._deduplicate(
            [judge_candidates[item] for item in accepted_ids]
        )
        accepted_set = set(accepted_ids)
        uncertain = self._deduplicate(
            [
                judge_candidates[item]
                for item in uncertain_ids
                if item not in accepted_set
            ]
        )
        for finding in accepted:
            finding.verification = VerificationStatus.SUPPORTED
        for finding in uncertain:
            finding.verification = VerificationStatus.UNCERTAIN

        raw_verdict = safe_verdict(judge_payload.get("verdict"))
        if accepted:
            verdict = Verdict.ARTIFACT_PRESENT
        elif uncertain:
            verdict = Verdict.UNCERTAIN
        elif raw_verdict is Verdict.NO_ARTIFACT:
            verdict = Verdict.NO_ARTIFACT
        else:
            verdict = Verdict.UNCERTAIN
        confidence = float(judge_payload.get("confidence", 0.5))
        trace.append(
            TraceEvent(
                stage="consensus.judge",
                message=f"Judge returned {verdict.value}.",
                data=judge_payload,
            )
        )
        result = CriticResult(
            verdict=verdict,
            confidence=confidence,
            findings=accepted,
            uncertain_findings=uncertain,
            trace=trace,
            metadata={
                "workflow": "adversarial_consensus",
                "roles": [role.name for role in self.config.roles],
                "judge_rationale": str(judge_payload.get("rationale", "")),
            },
        )

        coverage_summary = getattr(self.text_auditor, "last_summary", {})
        if self.text_auditor is not None:
            coverage_status = str(
                coverage_summary.get("coverage_status", "failed")
            )
            result.metadata.update(
                {
                    "coverage_status": coverage_status,
                    "coverage_ledger": copy.deepcopy(
                        coverage_summary.get("coverage_ledger", [])
                    ),
                    "semantic_zones": copy.deepcopy(
                        coverage_summary.get("semantic_zones", [])
                    ),
                    "findings_non_exhaustive": coverage_status != "ok",
                    "harness_inventory_health": copy.deepcopy(
                        coverage_summary.get("harness_inventory_health", {})
                    ),
                    "coverage_channels": copy.deepcopy(
                        coverage_summary.get("coverage_channels", {})
                    ),
                    "text_coverage": copy.deepcopy(
                        coverage_summary.get("text_coverage", {})
                    ),
                    "artifact_coverage": copy.deepcopy(
                        coverage_summary.get("artifact_coverage", {})
                    ),
                }
            )
            if coverage_status != "ok":
                result.metadata["quality_verdict"] = result.verdict.value
                result.metadata["harness_abstention_reason"] = str(
                    coverage_summary.get(
                        "harness_abstention_reason",
                        "内容覆盖账本未全部核销。",
                    )
                )
                result.verdict = Verdict.UNCERTAIN
                result.confidence = min(result.confidence, 0.5)

        self._last_challenge_batch_count = challenge_batch_count
        self._last_challenge_wave_count = (
            (
                challenge_batch_count + self.challenge_concurrency - 1
            )
            // self.challenge_concurrency
            if challenge_batch_count
            else 0
        )
        overlapped = bool(
            challenge_started_at is not None
            and challenge_started_at < discovery_completed_at
        )
        result.metadata.update(
            {
                "pipeline_version": "v6.1",
                "legacy_version_alias": "v4.1-flash.1",
                "v42_challenge_batch_size": self.challenge_batch_size,
                "v42_challenge_concurrency": self.challenge_concurrency,
                "v42_challenge_batch_count": challenge_batch_count,
                "v42_challenge_wave_count": self._last_challenge_wave_count,
                "v42_trust_preverified": self.trust_preverified,
                "v42_deterministic_judge": self.deterministic_judge,
                "v42_coverage_only": self.coverage_only,
                "v42_single_artifact_proposer": self.single_artifact_proposer,
                "v42_review_original_max_side": self.review_original_max_side,
                "v61_streaming_skeptic": True,
                "v61_candidate_count": len(candidates),
                "v61_skeptic_completed_count": len(challenge_records),
                "v61_candidate_aliases": candidate_aliases,
                "v61_skeptic_overlapped_discovery": overlapped,
                "v61_skeptic_overlap_seconds": round(
                    max(
                        0.0,
                        discovery_completed_at - challenge_started_at,
                    )
                    if challenge_started_at is not None
                    else 0.0,
                    4,
                ),
                "v61_challenge_span_seconds": round(
                    max(
                        0.0,
                        (challenge_completed_at or discovery_completed_at)
                        - challenge_started_at,
                    )
                    if challenge_started_at is not None
                    else 0.0,
                    4,
                ),
            }
        )
        result.trace.append(
            TraceEvent(
                stage="consensus.v61_streaming",
                message=(
                    "V6.1 streamed stable candidates into Skeptic before "
                    "the global Judge gate."
                ),
                data={
                    "candidate_count": len(candidates),
                    "skeptic_completed_count": len(challenge_records),
                    "challenge_batch_count": challenge_batch_count,
                    "challenge_wave_count": self._last_challenge_wave_count,
                    "overlapped_discovery": overlapped,
                    "overlap_seconds": result.metadata[
                        "v61_skeptic_overlap_seconds"
                    ],
                    "challenge_span_seconds": result.metadata[
                        "v61_challenge_span_seconds"
                    ],
                },
            )
        )
        return result
