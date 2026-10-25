from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import requests

from .base import BaseProvider, ProviderError, ProviderUsage


class OpenAIProvider(BaseProvider):
    name = "openai"
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        start = time.perf_counter()
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        if response.status_code >= 400:
            raise ProviderError(f"OpenAI API error: {response.status_code} - {response.text[:400]}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))

        try:
            response_json = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"OpenAI returned non-JSON content: {content[:200]}") from exc

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
        # Conservative approximate pricing for gpt-4o-mini in 2026; valid for rough instrumentation.
        cost_per_1k_in = 0.00015
        cost_per_1k_out = 0.0006
        return ((input_tokens / 1000.0) * cost_per_1k_in) + ((output_tokens / 1000.0) * cost_per_1k_out)
