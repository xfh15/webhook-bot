from __future__ import annotations

import httpx

from .config import Settings


def _build_headers(settings: Settings) -> dict:
    headers = {"Content-Type": "application/json"}
    if settings.chatwoot_api_token:
        headers["api_access_token"] = settings.chatwoot_api_token
    return headers


async def list_messages_public(
    settings: Settings,
    inbox_identifier: str,
    contact_identifier: str,
    conversation_id: int,
    limit: int,
) -> list[dict]:
    if limit <= 0:
        return []

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    url = (
        f"/public/api/v1/inboxes/{inbox_identifier}/contacts/"
        f"{contact_identifier}/conversations/{conversation_id}/messages"
    )
    params = {"limit": limit}

    async with httpx.AsyncClient(base_url=settings.chatwoot_base_url, timeout=timeout) as client:
        response = await client.get(url, headers=_build_headers(settings), params=params)
        response.raise_for_status()
        data = response.json()

    return data.get("payload", [])


async def create_message_public(
    settings: Settings,
    inbox_identifier: str,
    contact_identifier: str,
    conversation_id: int,
    content: str,
) -> None:
    url = (
        f"/public/api/v1/inboxes/{inbox_identifier}/contacts/"
        f"{contact_identifier}/conversations/{conversation_id}/messages"
    )
    payload = {
        "content": content,
        "message_type": "outgoing",
        "private": False,
        "content_type": "text",
    }
    timeout = httpx.Timeout(settings.request_timeout_seconds)

    async with httpx.AsyncClient(base_url=settings.chatwoot_base_url, timeout=timeout) as client:
        response = await client.post(url, headers=_build_headers(settings), json=payload)
        response.raise_for_status()
