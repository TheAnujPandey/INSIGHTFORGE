"""Thin Anthropic client with prompt caching + a Bedrock fallback path.

Why two paths:
- Local / interview demo → direct Anthropic API.
- AWS deployment → Bedrock (same Claude model family) so we don't ship API keys
  into ECS tasks.

Both expose the same `complete()` signature so the rest of the codebase doesn't
care which is wired up.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    tool_result: Dict[str, Any] | None = None


class _DummyLLM:
    """Used when no API key is configured - keeps the pipeline runnable in tests/CI."""

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        log.warning("ANTHROPIC_API_KEY not set - returning stub LLM response.")
        return LLMResponse(
            text=(
                "Customer Risk Analysis\n"
                "Customer shows multiple churn drivers. Recommended Action:\n"
                "- Assign dedicated CSM\n- Offer 15% loyalty discount for 6 months\n- Priority support access\n"
                "Expected Retention Probability: 72%"
            ),
            input_tokens=0,
            output_tokens=0,
        )

    def complete_with_tools(
        self, *, system: str, user: str, tools: List[dict], max_tokens: int | None = None,
    ) -> LLMResponse:
        log.warning("ANTHROPIC_API_KEY not set - returning stub tool response.")
        return LLMResponse(
            text="",
            input_tokens=0,
            output_tokens=0,
            tool_result=None,
        )


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        from anthropic import Anthropic

        self.model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return LLMResponse(
            text=text,
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
        )

    def complete_with_tools(
        self, *, system: str, user: str, tools: List[dict], max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call the LLM with tool definitions; extract structured tool_use result."""
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        text_parts = []
        tool_result = None
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_result = block.input
        return LLMResponse(
            text="".join(text_parts),
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            tool_result=tool_result,
        )


class BedrockLLM:
    """AWS Bedrock Claude - same interface as AnthropicLLM."""

    def __init__(self, model_id: str | None = None, max_tokens: int | None = None):
        import boto3

        self.model_id = model_id or settings.bedrock_model_id
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        import json

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        text = "".join(b.get("text", "") for b in payload.get("content", []))
        usage = payload.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


class _AsyncDummyLLM:
    """Async stub for DummyLLM."""

    async def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        log.warning("ANTHROPIC_API_KEY not set - returning async stub response.")
        return LLMResponse(
            text=(
                "Customer Risk Analysis\n"
                "Customer shows multiple churn drivers. Recommended Action:\n"
                "- Assign dedicated CSM\n- Offer 15% loyalty discount for 6 months\n- Priority support access\n"
                "Expected Retention Probability: 72%"
            ),
            input_tokens=0,
            output_tokens=0,
        )

    async def complete_with_tools(
        self, *, system: str, user: str, tools: List[dict], max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(text="", input_tokens=0, output_tokens=0, tool_result=None)


class AsyncAnthropicLLM:
    """Async Anthropic client for non-blocking API calls in async endpoints."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        from anthropic import AsyncAnthropic

        self.model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return LLMResponse(
            text=text,
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
        )

    async def complete_with_tools(
        self, *, system: str, user: str, tools: List[dict], max_tokens: int | None = None,
    ) -> LLMResponse:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        text_parts = []
        tool_result = None
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_result = block.input
        return LLMResponse(
            text="".join(text_parts),
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            tool_result=tool_result,
        )


def get_llm(provider: Optional[str] = None):
    """Factory. provider ∈ {'anthropic', 'bedrock', 'dummy', None}."""
    provider = (provider or "anthropic").lower()
    if provider == "dummy" or not settings.anthropic_api_key and provider == "anthropic":
        return _DummyLLM()
    if provider == "bedrock":
        return BedrockLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_async_llm(provider: Optional[str] = None):
    """Async factory. provider ∈ {'anthropic', 'dummy', None}."""
    provider = (provider or "anthropic").lower()
    if provider == "dummy" or not settings.anthropic_api_key and provider == "anthropic":
        return _AsyncDummyLLM()
    if provider == "anthropic":
        return AsyncAnthropicLLM()
    raise ValueError(f"Async LLM not supported for provider: {provider}")
