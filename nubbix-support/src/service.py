from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .guardrails import mask_sensitive_text, moderate_input, moderate_output
from .metrics import log_metric
from .prompts import render_prompt
from .providers.base import create_provider
from .schemas import SupportResponse


class TicketSupportService:
    def __init__(
        self,
        provider_name: str = "mock",
        model: Optional[str] = None,
        prompt_version: str = "v2_few_shot",
        api_key: Optional[str] = None,
    ) -> None:
        self.provider_name = (provider_name or os.getenv("DEFAULT_PROVIDER", "mock")).lower()
        self.model = model or os.getenv("DEFAULT_MODEL")
        self.prompt_version = prompt_version
        self.api_key = api_key

    def _build_response_contract(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "category": {"type": "string", "enum": ["billing", "technical", "account", "other"]},
                "actions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
            },
            "required": ["answer", "confidence", "category", "actions"],
            "additionalProperties": False,
        }

    def _safe_fallback(self, query: str) -> Dict[str, Any]:
        lowered = query.lower()
        if any(token in lowered for token in ["factura", "pago", "cobro", "cargo", "reembolso", "suscripción"]):
            category = "billing"
        elif any(token in lowered for token in ["error", "fallo", "bug", "login", "contraseña", "pantalla", "api", "crash"]):
            category = "technical"
        elif any(token in lowered for token in ["cuenta", "usuario", "perfil", "rol", "acceso", "configuración"]):
            category = "account"
        else:
            category = "other"

        return {
            "answer": "He revisado la consulta y la he categorizado para que el agente reciba el contexto correcto. Por favor, confirma el caso con el cliente y valida los detalles antes de responder.",
            "confidence": 0.72,
            "category": category,
            "actions": [
                "Confirmar detalles del cliente antes de responder.",
                "Revisar historial de pagos o incidencias relevantes.",
                "Escalar al equipo especialista si el caso requiere intervención técnica.",
            ],
        }

    def handle(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("The query text cannot be empty.")

        query = query.strip()
        safe, reasons = moderate_input(query)
        if not safe:
            return {
                "answer": "No puedo atender esta solicitud porque incluye instrucciones de ingeniería social o intentos de manipular el sistema. Por favor, reenvía la consulta de soporte de forma normal.",
                "confidence": 0.0,
                "category": "other",
                "actions": [
                    "Marcar ticket como sospechoso de prompt injection.",
                    "Solicitar al cliente que reescriba la consulta sin instrucciones ocultas.",
                    "Escalar a seguridad si la incidencia persiste.",
                ],
            }

        prompt_bundle = render_prompt(self.prompt_version, query)
        provider = create_provider(self.provider_name, api_key=self.api_key, model=self.model)
        provider.prompt_version = self.prompt_version
        last_error = None

        for attempt in range(3):
            try:
                raw = provider.generate(
                    system_prompt=prompt_bundle["system"],
                    user_prompt=prompt_bundle["user"],
                    response_schema=self._build_response_contract(),
                )
                payload = raw.get("content", {})
                if not isinstance(payload, dict):
                    raise ValueError("Provider responded with a non-object payload.")

                parsed = SupportResponse.model_validate(payload)
                safe_answer, _ = moderate_output(parsed.answer)
                if not safe_answer:
                    parsed.answer = "La respuesta ha sido bloqueada por motivos de seguridad. Reenvía la consulta en un formato estándar para continuar con soporte humano."
                parsed.answer = mask_sensitive_text(parsed.answer)

                usage = raw.get("usage", None)
                metrics = {
                    "provider": provider.name,
                    "model": provider.model,
                    "prompt_version": self.prompt_version,
                    "category": parsed.category,
                    "query_length": len(query),
                    "tokens_in": getattr(usage, "input_tokens", 0),
                    "tokens_out": getattr(usage, "output_tokens", 0),
                    "latency_ms": getattr(usage, "latency_ms", 0),
                    "cost_usd": getattr(usage, "cost_usd", 0.0),
                }
                log_metric(metrics)
                return parsed.model_dump()
            except Exception as exc:  # pragma: no cover - guarded retry path
                last_error = exc
                if attempt == 2:
                    break

        fallback = self._safe_fallback(query)
        log_metric(
            {
                "provider": provider.name,
                "model": provider.model,
                "prompt_version": self.prompt_version,
                "category": fallback["category"],
                "query_length": len(query),
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "cost_usd": 0.0,
                "fallback": True,
                "error": str(last_error),
            }
        )
        return fallback
