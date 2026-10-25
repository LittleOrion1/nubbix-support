# Nubbix Support Assistant

Este proyecto implementa un servicio de soporte nivel 1 para clientes de Nubbix. Recibe un ticket libre, lo clasifica en una categoría de soporte, propone una respuesta y sugiere acciones para un agente humano.

## Componentes clave

- Abstracción de proveedores con OpenAI y Anthropic
- Validación estructurada con Pydantic
- Guardrails de entrada y salida para mitigar prompt injection
- Prompts versionados en YAML
- Dataset de evaluación y suite de evals ejecutable
- Registro de tokens, latencia y costo por ejecución

## Comandos rápidos

```bash
# Ejecutar una consulta con el proveedor por defecto (mock si no hay claves)
python src/main.py --query "Necesito revisar mi última factura y me cobraron por error"

# Ejecutar con OpenAI (requiere OPENAI_API_KEY)
DEFAULT_PROVIDER=openai python src/main.py --query "No puedo iniciar sesión"

# Evaluar prompt v1 y v2
python evals/run_evals.py

# Evaluar prompts adversariales
python evals/run_adversarial.py
```

## Técnica de context engineering

Se utiliza la versión `v2_few_shot`, que incluye instrucciones explícitas y ejemplos cortos para forzar salida estructurada, distinguir categorías clave y reducir ambigüedad. La versión `v1_zero_shot` solo tiene instrucciones generales y queda más propensa a confundir facturación con cuenta técnica.

## Resultados de evaluación

La siguiente tabla refleja la ejecución actual de la suite de evals en el proyecto:

| Versión | Accuracy global | billing | technical | account | other |
| --- | --- | --- | --- | --- | --- |
| v1_zero_shot | 0.75 | 0.67 | 0.80 | 0.67 | 1.00 |
| v2_few_shot | 0.92 | 1.00 | 0.88 | 0.86 | 1.00 |

La diferencia de rendimiento confirma que la estrategia de few-shot con ejemplos y restricciones de salida es la elección correcta en este caso de uso.

## Guardrails y seguridad

- Se bloquean textos con patrones típicos de prompt injection.
- Se valida el JSON de salida con Pydantic.
- Se reintenta automáticamente ante parseo o validación inválida.
- Se registra la métrica de costo y latencia junto con cada ejecución.

## Variables de entorno

```bash
DEFAULT_PROVIDER=mock
DEFAULT_MODEL=gpt-4o-mini
PROMPT_VERSION=v2_few_shot
OPENAI_API_KEY=tu_key
ANTHROPIC_API_KEY=tu_key
```
