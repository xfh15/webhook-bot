from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from .chatwoot import create_message, handoff_conversation, list_messages
from .config import load_settings
from .openai_client import generate_reply
from .prompting import load_system_prompt
from .rag import retrieve_context
from .tools import load_tools

load_dotenv()
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO))
logger = logging.getLogger("chatwoot-bot")

app = FastAPI(title="Chatwoot Bot Webhook")


class HandoffRequested(Exception):
    """Raised after the Chatwoot handoff API calls have completed."""


def _is_incoming(payload: dict) -> bool:
    msg_type = payload.get("message_type")
    if isinstance(msg_type, str):
        return msg_type == "incoming"
    if isinstance(msg_type, int):
        return msg_type == 0

    message = payload.get("message") or {}
    msg_type = message.get("message_type")
    if isinstance(msg_type, str):
        return msg_type == "incoming"
    if isinstance(msg_type, int):
        return msg_type == 0

    sender = payload.get("sender") or message.get("sender") or {}
    return sender.get("type") == "contact"


def _is_private(payload: dict) -> bool:
    if payload.get("private") is True:
        return True
    message = payload.get("message") or {}
    return message.get("private") is True


def _extract_content(payload: dict) -> str:
    if isinstance(payload.get("content"), str):
        return payload.get("content", "").strip()
    message = payload.get("message") or {}
    return (message.get("content") or "").strip()


def _extract_account_id(payload: dict) -> int | None:
    account = payload.get("account") or {}
    if isinstance(account.get("id"), int):
        return account.get("id")
    if isinstance(payload.get("account_id"), int):
        return payload.get("account_id")
    return None


def _extract_conversation_id(payload: dict) -> int | None:
    conversation = payload.get("conversation") or {}
    if isinstance(conversation.get("id"), int):
        return conversation.get("id")
    if isinstance(payload.get("conversation_id"), int):
        return payload.get("conversation_id")
    return None


def _is_sender_bot(payload: dict) -> bool:
    sender = payload.get("sender") or (payload.get("message") or {}).get("sender") or {}
    sender_type = sender.get("type")
    return sender_type in {"agent", "agent_bot"}


def _is_bot_handled_conversation(payload: dict) -> bool:
    status = (payload.get("conversation") or {}).get("status")
    if status is None:
        return True
    return status in {"pending", 2}


def _map_history_to_messages(history: list[dict], current_content: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    for item in sorted(history, key=lambda x: x.get("id") or 0):
        if item.get("private") is True:
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue

        sender_type = item.get("sender_type")
        message_type = item.get("message_type")

        if sender_type == "contact" or message_type in (0, "incoming"):
            role = "user"
        else:
            role = "assistant"

        messages.append({"role": role, "content": content})

    if current_content:
        if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != current_content:
            messages.append({"role": "user", "content": current_content})

    return messages


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/chatwoot")
async def chatwoot_webhook(request: Request) -> dict[str, Any]:
    payload = await request.json()
    logger.info("Webhook event received: %s", payload.get("event"))
    logger.debug(
        "payload keys=%s inbox=%s contact=%s conversation=%s message=%s sender=%s",
        list(payload.keys()),
        payload.get("inbox"),
        payload.get("contact"),
        payload.get("conversation"),
        payload.get("message"),
        payload.get("sender"),
    )

    if payload.get("event") != "message_created":
        return {"ignored": True, "reason": "unsupported_event"}

    if _is_private(payload):
        return {"ignored": True, "reason": "private_message"}

    if not _is_incoming(payload):
        return {"ignored": True, "reason": "not_incoming"}

    if _is_sender_bot(payload):
        return {"ignored": True, "reason": "sender_is_bot"}

    if not _is_bot_handled_conversation(payload):
        return {"ignored": True, "reason": "conversation_handed_to_human"}

    content = _extract_content(payload)
    if not content:
        return {"ignored": True, "reason": "empty_content"}

    account_id = _extract_account_id(payload)
    conversation_id = _extract_conversation_id(payload)
    missing = []
    if account_id is None:
        missing.append("account_id")
    if conversation_id is None:
        missing.append("conversation_id")
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing identifiers: {', '.join(missing)}")

    settings = load_settings()

    async def request_handoff(arguments: dict[str, Any]) -> str:
        await handoff_conversation(
            settings,
            account_id,
            conversation_id,
            settings.handoff_team_id,
            settings.handoff_message,
        )
        raise HandoffRequested(arguments.get("reason", ""))

    history = await list_messages(settings, account_id, conversation_id, settings.history_messages)
    system_prompt = load_system_prompt(settings)
    if settings.handoff_enabled:
        system_prompt += (
            "\n\n转人工规则：如果用户明确要求人工客服、人工介入，或你无法可靠处理请求，"
            "必须调用 handoff_to_human。调用后不要继续生成普通客服答复。"
        )
    llm_messages = [{"role": "system", "content": system_prompt}]
    if settings.rag_enabled:
        rag = await retrieve_context(settings, content)
        if rag.context:
            llm_messages.append({"role": "system", "content": rag.context})
    llm_messages.extend(_map_history_to_messages(history, content))

    tools = None
    tool_handlers = None
    if settings.tools_enabled or settings.handoff_enabled:
        tools, tool_handlers = load_tools(
            settings,
            handoff_handler=request_handoff if settings.handoff_enabled else None,
            handoff_only=not settings.tools_enabled,
        )

    try:
        reply = await generate_reply(settings, llm_messages, tools=tools, tool_handlers=tool_handlers)
    except HandoffRequested:
        return {"ok": True, "handoff": True}

    await create_message(settings, account_id, conversation_id, reply)

    return {"ok": True}
