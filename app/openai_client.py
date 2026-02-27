from __future__ import annotations

import httpx
import json
from typing import Any

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


def _parse_sse_chat_completion(body_text: str) -> dict:
    choice_states: dict[int, dict[str, Any]] = {}
    first_chunk: dict[str, Any] | None = None

    for raw_line in body_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break

        chunk = json.loads(payload)
        if first_chunk is None:
            first_chunk = chunk

        for choice in chunk.get("choices") or []:
            idx = int(choice.get("index", 0))
            state = choice_states.setdefault(
                idx,
                {
                    "message": {"role": "assistant", "content": ""},
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )

            delta = choice.get("delta") or {}
            if isinstance(delta.get("role"), str):
                state["message"]["role"] = delta["role"]
            if isinstance(delta.get("content"), str):
                state["message"]["content"] += delta["content"]

            for tc in delta.get("tool_calls") or []:
                tc_idx = int(tc.get("index", 0))
                tc_state = state["tool_calls"].setdefault(
                    tc_idx,
                    {"type": "function", "id": "", "function": {"name": "", "arguments": ""}},
                )
                if isinstance(tc.get("id"), str):
                    tc_state["id"] = tc["id"]
                if isinstance(tc.get("type"), str):
                    tc_state["type"] = tc["type"]

                fn = tc.get("function") or {}
                if isinstance(fn.get("name"), str):
                    tc_state["function"]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    tc_state["function"]["arguments"] += fn["arguments"]

            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice.get("finish_reason")

    if first_chunk is None:
        raise ValueError("empty SSE body")

    choices: list[dict[str, Any]] = []
    for idx in sorted(choice_states.keys()):
        state = choice_states[idx]
        message = state["message"]
        tool_calls = [state["tool_calls"][k] for k in sorted(state["tool_calls"].keys())]
        if tool_calls:
            message["tool_calls"] = tool_calls
        choices.append(
            {
                "index": idx,
                "message": message,
                "finish_reason": state["finish_reason"],
            }
        )

    return {
        "id": first_chunk.get("id"),
        "object": "chat.completion",
        "created": first_chunk.get("created"),
        "model": first_chunk.get("model"),
        "choices": choices,
    }


async def _chat_completion(
    settings: Settings,
    payload: dict,
) -> dict:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(base_url=settings.openai_base_url, timeout=timeout) as client:
        response = await client.post("/chat/completions", headers=_headers(settings), json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            details = response.text.strip()
            raise RuntimeError(
                f"Chat completion request failed ({response.status_code} {response.reason_phrase}): {details}"
            ) from exc
        content_type = response.headers.get("content-type", "")
        body_text = response.text or ""
        try:
            return response.json()
        except ValueError as exc:
            is_sse = "text/event-stream" in content_type.lower() or body_text.lstrip().startswith("data:")
            if is_sse:
                try:
                    return _parse_sse_chat_completion(body_text)
                except Exception:
                    pass
            preview = body_text.strip().replace("\n", " ")[:300]
            raise RuntimeError(
                "Chat completion returned non-JSON response "
                f"(status={response.status_code}, content_type={content_type!r}, body_preview={preview!r})"
            ) from exc


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
            "stream": False,
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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            details = response.text.strip()
            raise RuntimeError(
                f"Embedding request failed ({response.status_code} {response.reason_phrase}): {details}"
            ) from exc
        data = response.json()

    items = data.get("data") or []
    items.sort(key=lambda x: x.get("index", 0))
    embeddings = [item.get("embedding") for item in items]
    if not embeddings or any(e is None for e in embeddings):
        raise RuntimeError("OpenRouter returned empty embeddings")
    return embeddings
