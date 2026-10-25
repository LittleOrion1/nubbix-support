from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import requests

from .base import BaseProvider, ProviderError, ProviderUsage


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    default_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "max_tokens": 800,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        start = time.perf_counter()
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=60,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        if response.status_code >= 400:
            raise ProviderError(f"Anthropic API error: {response.status_code} - {response.text[:400]}")

        data = response.json()
        blocks = data.get("content", [])
        text = ""
        for block in blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))

        try:
            response_json = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Anthropic returned non-JSON content: {text[:200]}") from exc

        return {
            "content": response_json,
            "usage": ProviderUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=self.estimate_cost(input_tokens, output_tokens),
            ),
            "provider": self.name,
            "model": self.model,
        }

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Approximate pricing for Claude 3.5 Haiku; used for instrumentation only.
        cost_per_1k_in = 0.0008
        cost_per_1k_out = 0.0004
        return ((input_tokens / 1000.0) * cost_per_1k_in) + ((output_tokens / 1000.0) * cost_per_1k_out)
