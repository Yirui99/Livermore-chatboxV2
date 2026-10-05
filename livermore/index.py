"""Build / load the vector index over a notes corpus.

Notes format (unchanged from build_dataset.py): JSONL, one {"prompt", "response"}
object per line, each becoming one document "Question: …\nAnswer: …". No chunking.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import time
from dataclasses import dataclass, field

from .config import Settings
from .retrieve import Hit, search

DEFAULT_EMBED_MODEL = Settings.embed_model

# File names inside an index directory. The legacy names are what build_dataset.py wrote.
_FILES = ("index.faiss", "docs.pkl")
_LEGACY_FILES = ("trading_index.faiss", "trading_docs.pkl")


def _faiss():
    # torch must be loaded before faiss: both bundle libomp, and with faiss first a
    # CPU SentenceTransformer.encode segfaults on macOS (exit 139).
    import torch  # noqa: F401
    import faiss
    return faiss


def load_notes(path: str) -> list[str]:
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.jsonl")))
    docs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                q = item["prompt"].strip()
                a = item["response"].strip()
                docs.append(f"Question: {q}\nAnswer: {a}")
    return docs


@dataclass
class Index:
    faiss_index: object
    docs: list[str]
    embed_model: str = DEFAULT_EMBED_MODEL
    path: str | None = None
    device: str = "auto"
    _embedder: object = field(default=None, repr=False)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            from ._device import resolve_torch_device
            self._embedder = SentenceTransformer(self.embed_model, device=resolve_torch_device(self.device))
        return self._embedder

    @property
    def dim(self) -> int:
        return self.faiss_index.d

    def __len__(self) -> int:
        return len(self.docs)

    def embed(self, query: str):
        faiss = _faiss()
        emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        faiss.normalize_L2(emb)
        return emb

    def search(self, query: str, k: int = 3) -> list[Hit]:
        return search(self, query, k)

    def save(self, path: str) -> None:
        faiss = _faiss()
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.faiss_index, os.path.join(path, _FILES[0]))
        with open(os.path.join(path, _FILES[1]), "wb") as f:
            pickle.dump(self.docs, f)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"embed_model": self.embed_model, "n_docs": len(self.docs), "dim": self.dim,
                       "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=2)
        self.path = path

    def info(self) -> dict:
        return {"path": self.path, "n_docs": len(self), "dim": self.dim, "embed_model": self.embed_model}


def build(notes_dir: str, out_dir: str | None = None, embed_model: str = DEFAULT_EMBED_MODEL,
          batch_size: int = 64, device: str | None = None) -> Index:
    """Embed every note and build an exact inner-product index (IndexFlatIP on L2-normalised vectors)."""
    faiss = _faiss()
    from sentence_transformers import SentenceTransformer

    docs = load_notes(notes_dir)
    if not docs:
        raise ValueError(f"no notes found in {notes_dir} (expected *.jsonl with prompt/response)")

    model = SentenceTransformer(embed_model, device=device)
    emb = model.encode(docs, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=False)
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)

    idx = Index(index, docs, embed_model=embed_model)
    if out_dir:
        idx.save(out_dir)
    return idx


def load(path: str, device: str = "auto") -> Index:
    faiss = _faiss()

    meta_path = os.path.join(path, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    for names in (_FILES, _LEGACY_FILES):
        ip, dp = (os.path.join(path, n) for n in names)
        if os.path.exists(ip) and os.path.exists(dp):
            break
    else:
        raise FileNotFoundError(f"no index in {path} (looked for {_FILES} or {_LEGACY_FILES})")

    with open(dp, "rb") as f:
        docs = pickle.load(f)
    return Index(faiss.read_index(ip), docs, embed_model=meta.get("embed_model", DEFAULT_EMBED_MODEL),
                 path=path, device=device)
