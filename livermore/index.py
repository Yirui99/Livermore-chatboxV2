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

from . import models
from .config import Settings
from .errors import IndexNotFound, NotesNotFound
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
    embed_revision: str | None = None
    _embedder: object = field(default=None, repr=False)

    @property
    def embedder(self):
        if self._embedder is None:
            from ._device import resolve_torch_device
            self._embedder = load_embedder(self.embed_model, self.embed_revision, resolve_torch_device(self.device))
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
            json.dump({"embed_model": self.embed_model, "embed_revision": models.revision_for(self.embed_model, self.embed_revision),
                       "n_docs": len(self.docs), "dim": self.dim,
                       "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=2)
        self.path = path

    def info(self) -> dict:
        return {"path": self.path, "n_docs": len(self), "dim": self.dim, "embed_model": self.embed_model}


def load_embedder(embed_model: str, revision: str | None = None, device: str | None = None):
    from sentence_transformers import SentenceTransformer
    from .errors import ModelFileError
    path = models.ensure(embed_model, revision)
    try:
        return SentenceTransformer(path, device=device)
    except (OSError, ValueError, RuntimeError) as e:
        raise ModelFileError(f"could not load embedding model {embed_model} from {path}: {type(e).__name__}",
                             f"run `livermore doctor --verify`, or `livermore models fetch {embed_model} --force`")


def build(notes_dir: str, out_dir: str | None = None, embed_model: str = DEFAULT_EMBED_MODEL,
          batch_size: int = 64, device: str | None = None, embed_revision: str | None = None) -> Index:
    """Embed every note and build an exact inner-product index (IndexFlatIP on L2-normalised vectors)."""
    faiss = _faiss()

    if not os.path.exists(notes_dir):
        raise NotesNotFound(f"notes directory {notes_dir} does not exist",
                            "pass --notes <dir> or set notes_dir in ~/.livermore/config.yaml")
    try:
        docs = load_notes(notes_dir)
    except (json.JSONDecodeError, KeyError) as e:
        raise NotesNotFound(f"could not read notes in {notes_dir}: {type(e).__name__}: {e}",
                            'each line must be a JSON object with "prompt" and "response"')
    if not docs:
        raise NotesNotFound(f"no notes in {notes_dir}",
                            'add *.jsonl files there, one {"prompt": ..., "response": ...} per line')

    model = load_embedder(embed_model, embed_revision, device)
    emb = model.encode(docs, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=False)
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)

    idx = Index(index, docs, embed_model=embed_model, embed_revision=embed_revision)
    if out_dir:
        idx.save(out_dir)
    return idx


def load(path: str, device: str = "auto") -> Index:
    faiss = _faiss()

    if not os.path.isdir(path):
        raise IndexNotFound(f"no index at {path}",
                            "build one with `livermore build --notes <dir>` (or set index_dir in ~/.livermore/config.yaml)")
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
        raise IndexNotFound(f"{path} has no index files ({' + '.join(_FILES)})",
                            "build one with `livermore build --notes <dir>`")
    try:
        with open(dp, "rb") as f:
            docs = pickle.load(f)
        fi = faiss.read_index(ip)
    except Exception as e:
        raise IndexNotFound(f"the index in {path} is unreadable ({type(e).__name__})",
                            "rebuild it with `livermore build`")
    if fi.ntotal != len(docs):
        raise IndexNotFound(f"the index in {path} is inconsistent: {fi.ntotal} vectors but {len(docs)} notes",
                            "rebuild it with `livermore build`")
    return Index(fi, docs, embed_model=meta.get("embed_model", DEFAULT_EMBED_MODEL), path=path, device=device,
                 embed_revision=meta.get("embed_revision"))
