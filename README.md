# Chatwoot Webhook + OpenAI-Compatible Bot

A minimal FastAPI service that receives Chatwoot webhooks, calls any OpenAI-compatible API, and replies back to the conversation.

## Setup

1. Create and edit `.env` based on `.env.example`.
2. Install dependencies.
3. Run the server.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

## Configuration

Key `.env` settings:

- `CHATWOOT_BASE_URL`: Your Chatwoot base URL, e.g. `http://localhost:3000`
- `CHATWOOT_API_TOKEN`: Agent bot API token (required for account API).
- `OPENAI_API_KEY`: API key for your OpenAI-compatible provider
- `OPENAI_BASE_URL`: Base URL for the provider
- `OPENAI_MODEL`: Chat completion model
- `OPENAI_EMBED_MODEL`: Embedding model (used by RAG)

## Webhook

Configure your Chatwoot agent bot webhook URL as:

```
http://<your-host>:3000/webhook/chatwoot
```

## LLM (OpenAI-Compatible)

Set your API base URL and key in `.env`:

```
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

You can point `OPENAI_BASE_URL` to any OpenAI-compatible provider.

## Prompt (Configurable)

You can either set `SYSTEM_PROMPT` in `.env` or point to a file:

```
SYSTEM_PROMPT_PATH=./system_prompt.txt
```

## Tools / Function Calling (Configurable)

Enable tools and optionally provide a `tools.json` config.

```
TOOLS_ENABLED=1
TOOLS_CONFIG_PATH=tools.json
TOOL_CHOICE=auto
MAX_TOOL_ROUNDS=2
```

Example file: `tools.json.example` (copy to `tools.json`).
Tool specs map to local handlers in `app/tools.py`.

## RAG (Optional)

1. Enable RAG in `.env`:

```
RAG_ENABLED=1
RAG_STORE_PATH=rag_store.jsonl
```

2. Ingest files into the vector store (OpenAI-compatible embeddings):

```bash
python -m app.ingest ./docs
```

3. The webhook will add retrieved context as a system message.

## Notes

- The webhook handler ignores non-incoming or private messages to prevent loops.
- The bot fetches the last N messages (default 10) to build context.
- Embeddings use `OPENAI_EMBED_MODEL`.
