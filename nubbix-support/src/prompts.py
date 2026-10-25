from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"


def load_prompt(version: str) -> Dict[str, Any]:
    file_path = PROMPTS_DIR / f"{version}.yaml"
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    content = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Invalid YAML prompt schema in {file_path}")
    return content


def render_prompt(version: str, query: str) -> Dict[str, str]:
    config = load_prompt(version)
    system_prompt = str(config.get("system", "")).strip()
    user_template = str(config.get("user_template", "Consulta del cliente:\n{query}\n")).strip()
    user_prompt = user_template.format(query=query)
    return {"system": system_prompt, "user": user_prompt}
