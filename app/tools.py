from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import Settings

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


def _schema_time() -> ToolSpec:
    async def _handle(_: dict[str, Any]) -> str:
        return datetime.now(timezone.utc).isoformat()

    return ToolSpec(
        name="get_current_time",
        description="Get the current UTC time in ISO 8601 format.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_handle,
    )


def _builtin_specs() -> dict[str, ToolSpec]:
    time_tool = _schema_time()
    return {time_tool.name: time_tool}


def _load_custom_specs(path: Path) -> list[ToolSpec]:
    if not path.exists() or not path.is_file():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    specs: list[ToolSpec] = []
    builtin = _builtin_specs()

    for item in data.get("tools", []):
        name = item.get("name")
        handler_key = item.get("handler")
        if not name or not handler_key:
            continue
        if handler_key not in builtin:
            continue
        spec = builtin[handler_key]
        specs.append(
            ToolSpec(
                name=name,
                description=item.get("description", spec.description),
                parameters=item.get("parameters", spec.parameters),
                handler=spec.handler,
            )
        )

    return specs


def load_tools(settings: Settings) -> tuple[list[dict[str, Any]], dict[str, ToolHandler]]:
    builtin = _builtin_specs()
    specs = list(builtin.values())

    custom = _load_custom_specs(Path(settings.tools_config_path))
    if custom:
        specs = custom

    tools = []
    handlers: dict[str, ToolHandler] = {}
    for spec in specs:
        tools.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        })
        handlers[spec.name] = spec.handler

    return tools, handlers
