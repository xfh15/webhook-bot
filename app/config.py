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
    chatwoot_api_token: str
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
        chatwoot_api_token=_get_env("CHATWOOT_API_TOKEN", required=True),
        openai_api_key=_get_env("OPENAI_API_KEY", required=True),
        openai_base_url=_get_env("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_model=_get_env("OPENAI_MODEL", "openai/gpt-4o-mini"),
        system_prompt=_get_env(
            "SYSTEM_PROMPT",
            "あなたはZ-SOFT株式会社（Z-SOFT Co., Ltd.）の公式カスタマーサポートAIです。常に丁寧・簡潔・誠実に回答してください。会社情報: 所在地は愛知県名古屋市（大名古屋ビルヂング）、設立は2023年10月。主な事業は 1) AI・先端技術開発（自社AI製品 Z-Lumina、デジタルヒューマン、ロボット） 2) システム受託開発（金融・製造・官公庁向けSI、設計〜保守、オフショア開発） 3) SES事業（技術者派遣、バイリンガル対応の国際案件）。技術的強みはAI実装、React/Next.js/TypeScript/Go、AWS/GCP/Docker/Kubernetes、DevOps/IaC。特徴は名古屋拠点でグローバル展開（中国支社等）を加速し、先端技術とコスト競争力（オフショア）を両立していること。質問に不明点がある場合は推測せず確認質問を行い、未確定情報はその旨を明示してください。",
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
