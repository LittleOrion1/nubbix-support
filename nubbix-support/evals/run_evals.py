from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.service import TicketSupportService

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "eval_dataset.jsonl"
CATEGORIES = ["billing", "technical", "account", "other"]


def load_dataset(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate_version(version: str):
    rows = load_dataset(DATASET_PATH)
    total = 0
    correct = 0
    per_category = {category: {"total": 0, "correct": 0} for category in CATEGORIES}

    service = TicketSupportService(provider_name="mock", prompt_version=version)
    for row in rows:
        prediction = service.handle(row["query"])["category"]
        expected = row["category"]
        total += 1
        if prediction == expected:
            correct += 1
        per_category[expected]["total"] += 1
        if prediction == expected:
            per_category[expected]["correct"] += 1

    output = {
        "version": version,
        "global_accuracy": round(correct / total, 4) if total else 0.0,
        "per_category": {},
    }
    for category in CATEGORIES:
        stats = per_category[category]
        output["per_category"][category] = {
            "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }
    return output


def main() -> int:
    report = {
        "versions": {
            version: evaluate_version(version)
            for version in ["v1_zero_shot", "v2_few_shot"]
        }
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
