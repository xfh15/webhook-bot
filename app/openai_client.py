from __future__ import annotations

import httpx
import json

from .config import Settings


def _headers(settings: Settings) -> dict:
    return {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }


def _tool_choice(settings: Settings) -> str | dict | None:
    if settings.tool_choice in {"auto", "none"}:
        return settings.tool_choice
    return {"type": "function", "function": {"name": settings.tool_choice}}


async def _chat_completion(
    settings: Settings,
    payload: dict,
) -> dict:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(base_url=settings.openai_base_url, timeout=timeout) as client:
        response = await client.post("/chat/completions", headers=_headers(settings), json=payload)
        response.raise_for_status()
        return response.json()


async def generate_reply(
    settings: Settings,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_handlers: dict | None = None,
) -> str:
    tool_handlers = tool_handlers or {}

    for _ in range(max(1, settings.max_tool_rounds)):
        payload = {
            "model": settings.openai_model,
            "messages": messages,
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _tool_choice(settings)

        data = await _chat_completion(settings, payload)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        content = (message.get("content") or "").strip()

        if not tool_calls:
            if not content:
                raise RuntimeError("OpenRouter returned empty content")
            return content

        messages.append(message)

        for call in tool_calls:
            name = call.get("function", {}).get("name")
            arguments = call.get("function", {}).get("arguments")
            try:
                if isinstance(arguments, str):
                    parsed_args = json.loads(arguments)
                elif isinstance(arguments, dict):
                    parsed_args = arguments
                else:
                    parsed_args = {}
            except Exception:
                parsed_args = {}

            handler = tool_handlers.get(name)
            if handler:
                result = await handler(parsed_args)
            else:
                result = f"Tool '{name}' is not available"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": result,
                }
            )

    raise RuntimeError("Tool call loop exceeded")


async def embed_texts(settings: Settings, texts: list[str]) -> list[list[float]]:
    payload = {
        "model": settings.openai_embed_model,
        "input": texts,
    }
    timeout = httpx.Timeout(settings.request_timeout_seconds)

    async with httpx.AsyncClient(base_url=settings.openai_base_url, timeout=timeout) as client:
        response = await client.post("/embeddings", headers=_headers(settings), json=payload)
        response.raise_for_status()
        data = response.json()

    items = data.get("data") or []
    items.sort(key=lambda x: x.get("index", 0))
    embeddings = [item.get("embedding") for item in items]
    if not embeddings or any(e is None for e in embeddings):
        raise RuntimeError("OpenRouter returned empty embeddings")
    return embeddings
