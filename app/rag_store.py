from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RagDocument:
    id: str
    text: str
    metadata: dict
    embedding: list[float]


class RagStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._docs: list[RagDocument] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._docs = []
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._docs.append(
                        RagDocument(
                            id=data["id"],
                            text=data["text"],
                            metadata=data.get("metadata", {}),
                            embedding=data["embedding"],
                        )
                    )
        self._loaded = True

    def add_many(self, docs: Iterable[RagDocument]) -> None:
        self.load()
        with open(self.path, "a", encoding="utf-8") as f:
            for doc in docs:
                self._docs.append(doc)
                f.write(
                    json.dumps(
                        {
                            "id": doc.id,
                            "text": doc.text,
                            "metadata": doc.metadata,
                            "embedding": doc.embedding,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def query(self, query_embedding: list[float], top_k: int) -> list[RagDocument]:
        self.load()
        if not self._docs:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        doc_vecs = np.array([doc.embedding for doc in self._docs], dtype=np.float32)

        denom = (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec))
        denom = np.where(denom == 0, 1e-8, denom)
        sims = doc_vecs @ query_vec / denom

        top_k = max(1, top_k)
        idxs = np.argsort(-sims)[:top_k]
        return [self._docs[i] for i in idxs]
