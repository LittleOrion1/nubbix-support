from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


class ProviderError(RuntimeError):
    """Raised when a provider fails to answer or returns invalid output."""


@dataclass
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0


class BaseProvider(ABC):
    name = "base"
    default_model = "mock-model"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.default_model

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return the raw payload with provider usage information."""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


def create_provider(provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None):
    provider_name = (provider_name or "mock").lower()

    if provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=api_key, model=model)
    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=model)
    if provider_name == "mock":
        from .mock_provider import MockProvider

        return MockProvider(model=model)

    raise ValueError(f"Unsupported provider: {provider_name}")
