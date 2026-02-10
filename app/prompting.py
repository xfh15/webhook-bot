from __future__ import annotations

from pathlib import Path

from .config import Settings


def load_system_prompt(settings: Settings) -> str:
    if settings.system_prompt_path:
        path = Path(settings.system_prompt_path)
        if path.exists() and path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content
    return settings.system_prompt
