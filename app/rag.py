from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .openai_client import embed_texts
from .rag_store import RagDocument, RagStore


@dataclass(frozen=True)
class RagResult:
    context: str
    sources: list[dict]


def _format_context(docs: list[RagDocument]) -> str:
    if not docs:
        return ""

    lines = ["Use the following context to answer the user question."]
    for idx, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("source") or f"doc-{idx}"
        lines.append(f"[{idx}] {title}\n{doc.text}")
    return "\n\n".join(lines)


def _sources(docs: list[RagDocument]) -> list[dict]:
    output = []
    for doc in docs:
        output.append({
            "id": doc.id,
            "title": doc.metadata.get("title"),
            "source": doc.metadata.get("source"),
        })
    return output


async def retrieve_context(settings: Settings, question: str) -> RagResult:
    store = RagStore(settings.rag_store_path)
    query_embedding = await embed_texts(settings, [question])
    docs = store.query(query_embedding[0], settings.rag_top_k)
    return RagResult(context=_format_context(docs), sources=_sources(docs))
