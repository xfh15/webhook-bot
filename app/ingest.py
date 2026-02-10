from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from .config import load_settings
from .openai_client import embed_texts
from .rag_store import RagDocument, RagStore


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _hash_id(path: str, chunk_index: int) -> str:
    raw = f"{path}:{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _load_files(root: Path) -> list[tuple[str, str]]:
    items = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        items.append((str(path), text))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into RAG store")
    parser.add_argument("root", help="Folder to ingest")
    args = parser.parse_args()

    settings = load_settings()
    store = RagStore(settings.rag_store_path)

    items = _load_files(Path(args.root))
    texts = []
    meta = []
    ids = []

    for path, text in items:
        for i, chunk in enumerate(_chunk_text(text, settings.rag_chunk_size, settings.rag_chunk_overlap)):
            if not chunk.strip():
                continue
            ids.append(_hash_id(path, i))
            texts.append(chunk)
            meta.append({"source": path, "title": os.path.basename(path)})

    if not texts:
        print("No text found to ingest")
        return

    embeddings = __import__("asyncio").run(embed_texts(settings, texts))
    docs = [
        RagDocument(id=ids[i], text=texts[i], metadata=meta[i], embedding=embeddings[i])
        for i in range(len(texts))
    ]
    store.add_many(docs)
    print(f"Ingested {len(docs)} chunks into {settings.rag_store_path}")


if __name__ == "__main__":
    main()
