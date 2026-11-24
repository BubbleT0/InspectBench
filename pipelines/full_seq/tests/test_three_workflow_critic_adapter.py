from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from critic_workflows import ImageAsset, ModelRequest
from u1_critic_v41.model_adapter import (
    OpenAICompatibleModelClient,
    TraceWriter,
    build_anthropic_payload,
    build_payload,
    parse_anthropic_response,
    parse_assistant_message,
)


SCHEMA = {
    "type": "object",
    "required": ["present", "confidence"],
    "properties": {
        "present": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
}


class FakeTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.payloads: list[dict] = []
        self.urls: list[str] = []
        self.headers: list[dict] = []

    async def post(self, url, payload, headers, timeout):
        self.payloads.append(payload)
        self.urls.append(url)
        self.headers.append(dict(headers))
        return self.response


class ModelAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = ModelRequest(
            stage="specialist.detect",
            system_prompt="system",
            user_prompt="inspect",
            images=(ImageAsset(id="img_anon", data=b"pixels", mime_type="image/jpeg"),),
            response_schema=SCHEMA,
            metadata={"artifact_type": "structural_artifact"},
        )

    def test_payload_forces_submit_result_without_source_identity(self) -> None:
        payload = build_payload("model", self.request, 1600)

        self.assertEqual(payload["tools"][0]["function"]["name"], "submit_result")
        self.assertEqual(
            payload["tool_choice"]["function"]["name"],
            "submit_result",
        )
        self.assertEqual(payload["tools"][0]["function"]["parameters"], SCHEMA)
        serialized = json.dumps(payload)
        self.assertNotIn("LEAK_LABEL", serialized)
        self.assertNotIn("secret.jpg", serialized)

    def test_payload_requires_simplified_chinese_natural_language_fields(self) -> None:
        payload = build_payload("model", self.request, 1600)
        messages = payload["messages"]

        self.assertIn("简体中文", messages[0]["content"])
        self.assertIn("简体中文", messages[1]["content"][0]["text"])

    def test_tool_arguments_take_precedence_over_content(self) -> None:
        parsed, raw = parse_assistant_message(
            {
                "content": '{"present": false, "confidence": 0.2}',
                "tool_calls": [
                    {
                        "function": {
                            "name": "submit_result",
                            "arguments": '{"present": true, "confidence": 0.8}',
                        }
                    }
                ],
            }
        )

        self.assertTrue(parsed["present"])
        self.assertEqual(raw["transport"], "result-tool")

    def test_content_json_is_fallback(self) -> None:
        parsed, raw = parse_assistant_message(
            {
                "content": (
                    "```json\n"
                    '{"present": false, "confidence": 0.7}'
                    "\n```"
                )
            }
        )

        self.assertFalse(parsed["present"])
        self.assertEqual(raw["transport"], "prompt")

    def test_anthropic_payload_and_tool_response(self) -> None:
        payload = build_anthropic_payload("claude", self.request, 1600)
        self.assertEqual(payload["tools"][0]["name"], "submit_result")
        self.assertEqual(payload["tool_choice"]["name"], "submit_result")
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image")
        parsed, raw, text = parse_anthropic_response(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "submit_result",
                        "input": {"present": True, "confidence": 0.8},
                    }
                ]
            }
        )
        self.assertTrue(parsed["present"])
        self.assertEqual(raw["transport"], "result-tool")
        self.assertEqual(text, "")

    def test_payload_uses_file_signature_when_suffix_mime_is_wrong(self) -> None:
        request = ModelRequest(
            stage="smoke.image",
            system_prompt="system",
            user_prompt="inspect",
            images=(
                ImageAsset(
                    id="mislabeled",
                    data=b"\xff\xd8\xfffake-jpeg",
                    mime_type="image/png",
                ),
            ),
            response_schema=SCHEMA,
        )

        anthropic = build_anthropic_payload("claude", request, 64)
        openai = build_payload("model", request, 64)

        self.assertEqual(
            anthropic["messages"][0]["content"][1]["source"]["media_type"],
            "image/jpeg",
        )
        self.assertTrue(
            openai["messages"][1]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )

    def test_anthropic_client_uses_full_messages_endpoint_once(self) -> None:
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_result",
                    "input": {"present": False, "confidence": 0.9},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
        transport = FakeTransport(response)
        with tempfile.TemporaryDirectory() as directory:
            client = OpenAICompatibleModelClient(
                base_url="https://model/v1/messages",
                model="claude",
                sample_id="img_anon",
                workflow="adversarial_consensus",
                trace_writer=TraceWriter(Path(directory) / "trace.jsonl"),
                limiter=asyncio.Semaphore(1),
                transport=transport,
                api_key="secret",
                api_format="anthropic_messages",
                max_retries=0,
            )
            result = asyncio.run(client.complete(self.request))
        self.assertFalse(result.parsed["present"])
        self.assertEqual(transport.urls, ["https://model/v1/messages"])
        self.assertEqual(transport.headers[0]["x-api-key"], "secret")

    def test_client_traces_without_pixels_or_headers(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "submit_result",
                                    "arguments": (
                                        '{"present": false, "confidence": 0.9}'
                                    ),
                                }
                            }
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 12},
        }
        transport = FakeTransport(response)
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "traces.jsonl"
            client = OpenAICompatibleModelClient(
                base_url="http://model",
                model="model",
                sample_id="img_anon",
                workflow="specialist_cascade",
                trace_writer=TraceWriter(trace_path),
                limiter=asyncio.Semaphore(2),
                transport=transport,
                api_key="secret-key",
                max_retries=0,
            )
            result = asyncio.run(client.complete(self.request))
            trace = trace_path.read_text("utf-8")

        self.assertFalse(result.parsed["present"])
        self.assertNotIn("base64", trace.lower())
        self.assertNotIn("authorization", trace.lower())
        self.assertNotIn("secret-key", trace)
        self.assertNotIn("pixels", trace)
        self.assertIn('"transport": "result-tool"', trace)

    def test_every_workflow_stage_uses_the_same_result_tool(self) -> None:
        stages = (
            "specialist.detect",
            "specialist.verify",
            "evidence.plan",
            "evidence.force_finish",
            "evidence.verify",
            "consensus.propose",
            "consensus.challenge",
            "consensus.judge",
        )
        for stage in stages:
            with self.subTest(stage=stage):
                request = ModelRequest(
                    stage=stage,
                    system_prompt="system",
                    user_prompt="prompt",
                    response_schema=SCHEMA,
                )
                payload = build_payload("model", request, 800)
                self.assertEqual(
                    payload["tool_choice"]["function"]["name"],
                    "submit_result",
                )

    def test_bbox_schema_enforces_normalized_coordinates_in_actions_and_findings(self) -> None:
        request = ModelRequest(
            stage="evidence.plan",
            system_prompt="system",
            user_prompt="prompt",
            response_schema={
                "type": "object",
                "properties": {
                    "bbox_norm": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "findings": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
            },
        )

        schema = build_payload("model", request, 800)["tools"][0]["function"][
            "parameters"
        ]

        self.assertEqual(schema["properties"]["bbox_norm"]["items"]["minimum"], 0)
        self.assertEqual(
            schema["properties"]["bbox_norm"]["items"]["type"],
            "integer",
        )
        self.assertEqual(
            schema["properties"]["bbox_norm"]["items"]["maximum"],
            1000,
        )
        finding_bbox = schema["properties"]["findings"]["items"]["properties"][
            "bbox_norm"
        ]
        self.assertEqual(finding_bbox["items"]["minimum"], 0)
        self.assertEqual(finding_bbox["items"]["type"], "integer")
        self.assertEqual(finding_bbox["items"]["maximum"], 1000)


if __name__ == "__main__":
    unittest.main()
