#!/usr/bin/env python3
"""Send one tiny image request through the V4.1 model adapter."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
from pathlib import Path

from critic_workflows import ImageAsset, ModelRequest
from u1_critic_v41.model_adapter import OpenAICompatibleModelClient, TraceWriter


async def run(args: argparse.Namespace) -> None:
    client = OpenAICompatibleModelClient(
        base_url=args.base_url,
        model=args.model,
        sample_id="smoke",
        workflow="adapter_smoke",
        trace_writer=TraceWriter(args.trace),
        limiter=asyncio.Semaphore(1),
        api_key=os.environ.get(args.api_key_env),
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_retries=args.retries,
        api_format=args.api_format,
    )
    response = await client.complete(
        ModelRequest(
            stage="smoke.image",
            system_prompt="你是图片连通性测试器。",
            user_prompt="查看图片并通过工具返回测试成功。",
            images=(ImageAsset(id="smoke", path=args.image),),
            response_schema={
                "type": "object",
                "required": ["status"],
                "properties": {"status": {"type": "string"}},
            },
        )
    )
    print(json.dumps(response.parsed, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-format",
        choices=("openai_chat", "anthropic_messages"),
        default="openai_chat",
    )
    parser.add_argument("--api-key-env", default="CRITIC_API_KEY")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=2)
    try:
        asyncio.run(run(parser.parse_args()))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
