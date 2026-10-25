from __future__ import annotations

import json
import re
from typing import Any, Dict

from .base import BaseProvider, ProviderUsage


class MockProvider(BaseProvider):
    name = "mock"
    default_model = "mock-classifier"

    def _classify(self, query: str) -> str:
        lowered = query.lower()
        prompt_version = getattr(self, "prompt_version", "v2_few_shot")

        if prompt_version == "v1_zero_shot":
            if any(token in lowered for token in ["factura", "cobro", "cargo", "pago", "reembolso", "deuda", "suscripción", "cobran"]):
                return "billing"
            if any(token in lowered for token in ["error", "fallo", "bug", "pantalla", "login", "contraseña", "api", "servidor", "crash", "bloqueado"]):
                return "technical"
            if any(token in lowered for token in ["cuenta", "usuario", "perfil", "configuración", "rol", "acceso", "contraseña", "plan"]):
                return "account"
            return "other"

        # v2_few_shot: clearer precedence and more nuanced account/billing patterns.
        billing_terms = ["factura", "cobro", "cargo", "pago", "reembolso", "deuda", "suscripción", "facturación", "recibo", "tarifa", "corte"]
        technical_terms = ["error", "fallo", "no funciona", "bug", "pantalla", "login", "contraseña", "api", "servidor", "crash", "lento", "sincroniz"]
        account_terms = ["cuenta", "usuario", "perfil", "rol", "acceso", "configuración", "administrador", "cambio de email", "cambiar nombre", "contacto principal"]

        if any(token in lowered for token in billing_terms) and not any(token in lowered for token in ["cambiar cuenta", "usuario", "perfil"]):
            return "billing"
        if any(token in lowered for token in technical_terms):
            return "technical"
        if any(token in lowered for token in account_terms):
            return "account"
        if "cancelar" in lowered and "factura" in lowered:
            return "billing"
        if "nombre" in lowered and "cuenta" in lowered:
            return "account"
        return "other"

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
        query = user_prompt.split("Consulta del cliente:")[-1].strip()
        category = self._classify(query)
        actions = {
            "billing": [
                "Validar la última factura y los cargos asociados.",
                "Confirmar si fueron cobrados por error o por plan.",
                "Revisar si corresponde reembolso o ajuste manual.",
            ],
            "technical": [
                "Solicitar evidencia del error y navegador o versión.",
                "Comprobar si hubo cambios en permisos o integraciones.",
                "Escalar a soporte técnico si persiste el bloqueo.",
            ],
            "account": [
                "Verificar datos del cliente y permisos de la cuenta.",
                "Confirmar si requiere cambio de plan, perfil o acceso.",
                "Registrar la solicitud en la historia del cliente.",
            ],
            "other": [
                "Clasificar la consulta y derivarla al equipo correcto.",
                "Solicitar más contexto si la solicitud es ambigua.",
                "Entregar respuesta operativa del primer nivel.",
            ],
        }[category]

        answer_map = {
            "billing": "Revisé que la consulta tiene que ver con facturación. Te recomiendo validar la factura, confirmar si el cobro es correcto y, si corresponde, emitir un ajuste o respuesta con la explicación del cargo.",
            "technical": "La solicitud parece relacionada con un problema técnico. Solicita el detalle del error, el paso que lo provocó y la plataforma afectada para que el agente pueda resolverlo o escalarlo correctamente.",
            "account": "Parece un tema de gestión de cuenta. Verifica datos del cliente, permisos, plan y cambios recientes antes de responder o derivar a administración.",
            "other": "La consulta no encaja claramente en las categorías principales, así que recomienda pedir más contexto para atenderla con precisión.",
        }

        payload = {
            "answer": answer_map[category],
            "confidence": 0.94 if "v2_few_shot" in (getattr(self, "prompt_version", "v2_few_shot") or "") else 0.87,
            "category": category,
            "actions": actions,
        }

        return {
            "content": payload,
            "usage": ProviderUsage(
                input_tokens=max(80, len(query) // 2),
                output_tokens=max(40, len(payload["answer"]) // 3),
                latency_ms=24,
                cost_usd=0.00008,
            ),
            "provider": self.name,
            "model": self.model,
        }
