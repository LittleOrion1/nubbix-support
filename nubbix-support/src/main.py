from __future__ import annotations

import argparse
import json
import os
import sys

from service import TicketSupportService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nubbix support assistant CLI")
    parser.add_argument("--query", help="Consulta del cliente en texto libre")
    parser.add_argument("--provider", choices=["openai", "anthropic", "mock"], default=os.getenv("DEFAULT_PROVIDER", "mock"))
    parser.add_argument("--model", default=os.getenv("DEFAULT_MODEL"))
    parser.add_argument("--prompt-version", default=os.getenv("PROMPT_VERSION", "v2_few_shot"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.query:
        if sys.stdin.isatty():
            parser.error("Se requiere --query o entrada por stdin.")
        args.query = sys.stdin.read().strip()

    service = TicketSupportService(
        provider_name=args.provider,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    result = service.handle(args.query)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
