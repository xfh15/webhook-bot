from __future__ import annotations

import httpx

from .config import Settings


def _build_headers(settings: Settings) -> dict:
    return {
        "api_access_token": settings.chatwoot_api_token,
        "Content-Type": "application/json",
    }


async def list_messages(
    settings: Settings,
    account_id: int,
    conversation_id: int,
    limit: int,
) -> list[dict]:
    if limit <= 0:
        return []

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    params = {"limit": limit}

    async with httpx.AsyncClient(base_url=settings.chatwoot_base_url, timeout=timeout) as client:
        response = await client.get(url, headers=_build_headers(settings), params=params)
        response.raise_for_status()
        data = response.json()

    return data.get("payload", [])


async def create_message(
    settings: Settings,
    account_id: int,
    conversation_id: int,
    content: str,
) -> None:
    url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
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


async def assign_conversation(
    settings: Settings,
    account_id: int,
    conversation_id: int,
    team_id: int | None = None,
) -> None:
    url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/assignments"
    payload = {"team_id": team_id} if team_id is not None else {"assignee_id": None}
    timeout = httpx.Timeout(settings.request_timeout_seconds)

    async with httpx.AsyncClient(base_url=settings.chatwoot_base_url, timeout=timeout) as client:
        response = await client.post(url, headers=_build_headers(settings), json=payload)
        response.raise_for_status()


async def open_conversation_from_bot(
    settings: Settings,
    account_id: int,
    conversation_id: int,
) -> None:
    url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/toggle_status"
    timeout = httpx.Timeout(settings.request_timeout_seconds)

    async with httpx.AsyncClient(base_url=settings.chatwoot_base_url, timeout=timeout) as client:
        response = await client.post(
            url,
            headers=_build_headers(settings),
            json={"status": "open"},
        )
        response.raise_for_status()


async def handoff_conversation(
    settings: Settings,
    account_id: int,
    conversation_id: int,
    team_id: int | None,
    message: str,
) -> None:
    await create_message(settings, account_id, conversation_id, message)
    await assign_conversation(settings, account_id, conversation_id)
    if team_id is not None:
        await assign_conversation(settings, account_id, conversation_id, team_id)
    await open_conversation_from_bot(settings, account_id, conversation_id)
