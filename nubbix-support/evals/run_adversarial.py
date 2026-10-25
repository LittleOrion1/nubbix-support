from __future__ import annotations

import json
from pathlib import Path

from src.guardrails import moderate_input, moderate_output
from src.service import TicketSupportService

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "adversarial.jsonl"


def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    rows = load_rows(DATASET_PATH)
    service = TicketSupportService(provider_name="mock", prompt_version="v2_few_shot")
    blocked = 0
    results = []

    for row in rows:
        response = service.handle(row["query"])
        answer = response["answer"]
        safe_input, input_reasons = moderate_input(row["query"])
        safe_output, output_reasons = moderate_output(answer)
        blocked_flag = bool((not safe_input) or (not safe_output))
        blocked += int(blocked_flag)
        results.append({
            "query": row["query"],
            "blocked": blocked_flag,
            "input_reasons": input_reasons,
            "output_reasons": output_reasons,
            "category": response["category"],
        })

    total = len(rows)
    print(json.dumps({
        "total_adversarial_prompts": total,
        "blocked": blocked,
        "block_rate": round(blocked / total, 4) if total else 0.0,
        "details": results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
