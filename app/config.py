import os
from dataclasses import dataclass


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    chatwoot_base_url: str
    chatwoot_api_token: str | None
    chatwoot_inbox_identifier: str | None
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    system_prompt: str
    history_messages: int
    request_timeout_seconds: float
    rag_enabled: bool
    rag_store_path: str
    rag_top_k: int
    rag_chunk_size: int
    rag_chunk_overlap: int
    openai_embed_model: str
    system_prompt_path: str | None
    tools_enabled: bool
    tools_config_path: str
    tool_choice: str
    max_tool_rounds: int


def load_settings() -> Settings:
    return Settings(
        chatwoot_base_url=_get_env("CHATWOOT_BASE_URL", "http://localhost:3000", required=True),
        chatwoot_api_token=_get_env("CHATWOOT_API_TOKEN", None),
        chatwoot_inbox_identifier=_get_env("CHATWOOT_INBOX_IDENTIFIER", None),
        openai_api_key=_get_env("OPENAI_API_KEY", required=True),
        openai_base_url=_get_env("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_model=_get_env("OPENAI_MODEL", "openai/gpt-4o-mini"),
        system_prompt=_get_env(
            "SYSTEM_PROMPT",
            "You are a helpful assistant in a customer support chat. Keep replies concise and friendly.",
        ),
        history_messages=int(_get_env("HISTORY_MESSAGES", "10")),
        request_timeout_seconds=float(_get_env("REQUEST_TIMEOUT_SECONDS", "30")),
        rag_enabled=_get_env("RAG_ENABLED", "0") == "1",
        rag_store_path=_get_env("RAG_STORE_PATH", "rag_store.jsonl"),
        rag_top_k=int(_get_env("RAG_TOP_K", "4")),
        rag_chunk_size=int(_get_env("RAG_CHUNK_SIZE", "800")),
        rag_chunk_overlap=int(_get_env("RAG_CHUNK_OVERLAP", "120")),
        openai_embed_model=_get_env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        system_prompt_path=_get_env("SYSTEM_PROMPT_PATH", None),
        tools_enabled=_get_env("TOOLS_ENABLED", "0") == "1",
        tools_config_path=_get_env("TOOLS_CONFIG_PATH", "tools.json"),
        tool_choice=_get_env("TOOL_CHOICE", "auto"),
        max_tool_rounds=int(_get_env("MAX_TOOL_ROUNDS", "2")),
    )
