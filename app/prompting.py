from __future__ import annotations

from pathlib import Path

from .config import Settings


def load_system_prompt(settings: Settings) -> str:
    prompt = settings.system_prompt

    if settings.system_prompt_path:
        path = Path(settings.system_prompt_path)
        if path.exists() and path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                prompt = content

    fallback_language = (settings.default_response_language or "ja").strip().lower()
    if fallback_language == "ja":
        language_instruction = (
            "回答はユーザーの直近の質問で使われた言語に合わせてください。"
            "言語を判別できない場合のみ日本語で回答してください。"
        )
    else:
        language_instruction = (
            "回答はユーザーの直近の質問で使われた言語に合わせてください。"
            f"言語を判別できない場合のみ {fallback_language} で回答してください。"
        )

    parts = [
        prompt.strip(),
        language_instruction,
    ]

    if settings.knowledge_path:
        knowledge_path = Path(settings.knowledge_path)
        if knowledge_path.exists() and knowledge_path.is_file():
            knowledge = knowledge_path.read_text(encoding="utf-8").strip()
            if knowledge:
                parts.append("以下は社内ナレッジです。回答時に最優先で参照してください。")
                parts.append(knowledge)

    return "\n\n".join(part for part in parts if part)
