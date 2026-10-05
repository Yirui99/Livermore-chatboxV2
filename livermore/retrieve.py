from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .index import Index


@dataclass(frozen=True)
class Hit:
    id: int        # row in the index == position in the notes list
    text: str
    score: float   # cosine similarity (inner product of L2-normalised vectors)


def search(index: "Index", query: str, k: int = 3) -> list[Hit]:
    q = index.embed(query)
    D, I = index.faiss_index.search(q, k)
    return [Hit(int(i), index.docs[i], float(s)) for i, s in zip(I[0], D[0]) if i != -1]
