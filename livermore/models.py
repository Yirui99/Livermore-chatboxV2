"""Model files under ~/.livermore/models/<owner>--<name>/, pinned to a revision and checksummed.

On first use a model is fetched from, in order:
  1. the local Hugging Face cache, if it holds the pinned revision (no network; each blob's
     file name *is* its hash, so the copy is verified offline), or
  2. the Hugging Face Hub (progress bars; each file checked against the Hub's sha256 / git sha1).
A manifest.json records every file's size and hash. Loading does a quick size check;
`livermore doctor --verify` re-hashes everything.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import sys
import time

from .config import models_dir
from .errors import ModelFileError, ModelUnavailable

# Revisions the shipped index, prompts and benchmark numbers were produced with.
PINNED = {
    "meta-llama/Llama-3.2-1B-Instruct": "9213176726f574b556790deb65791e0c5aa438b6",
    "mlx-community/Llama-3.2-1B-Instruct-4bit": "08231374eeacb049a0eade7922910865b8fce912",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
ALLOW = ["*.json", "*.safetensors", "*.txt"]
IGNORE = ["original/*", "onnx/*", "openvino/*"]
MANIFEST = "manifest.json"


def local_dir(repo_id: str) -> str:
    return os.path.join(models_dir(), repo_id.replace("/", "--"))


def revision_for(repo_id: str, revision: str | None) -> str:
    return revision or PINNED.get(repo_id, "main")


def _wanted(path: str) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in ALLOW) and not any(fnmatch.fnmatch(path, p) for p in IGNORE)


def _log(msg: str):
    print(f"livermore: {msg}", file=sys.stderr, flush=True)


# ---------------- hashing ----------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha1_file(path: str) -> str:
    h = hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_matches(path: str, expected: dict) -> bool:
    if expected.get("sha256"):
        return sha256_file(path) == expected["sha256"]
    return git_sha1_file(path) == expected["git_sha1"]


# ---------------- status ----------------

def read_manifest(repo_id: str) -> dict | None:
    p = os.path.join(local_dir(repo_id), MANIFEST)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def check(repo_id: str, revision: str | None = None, verify: bool = False) -> list[str]:
    """Problems with the local copy ([] = fine). Quick: existence + size; verify=True: hashes too."""
    d = local_dir(repo_id)
    if not os.path.isdir(d):
        return [f"not downloaded ({d} does not exist)"]
    m = read_manifest(repo_id)
    if m is None:
        return [f"{MANIFEST} missing or unreadable in {d}"]
    problems = []
    want_rev = revision_for(repo_id, revision)
    if m.get("revision") != want_rev:
        problems.append(f"revision {m.get('revision', '?')[:10]} on disk, {want_rev[:10]} configured")
    for rel, meta in m["files"].items():
        p = os.path.join(d, rel)
        if not os.path.exists(p):
            problems.append(f"{rel} is missing")
        elif os.path.getsize(p) != meta["size"]:
            problems.append(f"{rel} has {os.path.getsize(p):,} bytes, expected {meta['size']:,} (truncated or corrupted)")
        elif verify and not _hash_matches(p, meta):
            problems.append(f"{rel} fails its checksum (corrupted)")
    return problems


def ensure(repo_id: str, revision: str | None = None) -> str:
    """Local directory of a verified copy, fetching it on first use."""
    rev = revision_for(repo_id, revision)
    problems = check(repo_id, rev)
    if not problems:
        return local_dir(repo_id)
    if os.path.isdir(local_dir(repo_id)) and read_manifest(repo_id) is not None:
        raise ModelFileError(f"model files for {repo_id} are damaged: {problems[0]}",
                             f"re-download with `livermore models fetch {repo_id} --force`")
    return fetch(repo_id, rev)


def fetch(repo_id: str, revision: str | None = None, force: bool = False) -> str:
    rev = revision_for(repo_id, revision)
    dest = local_dir(repo_id)
    if force and os.path.isdir(dest):
        shutil.rmtree(dest)
    elif not force and not check(repo_id, rev):
        return dest
    os.makedirs(models_dir(), exist_ok=True)
    tmp = dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    files = _from_hf_cache(repo_id, rev, tmp)
    if files is None:
        files = _from_hub(repo_id, rev, tmp)
    with open(os.path.join(tmp, MANIFEST), "w") as f:
        json.dump({"repo_id": repo_id, "revision": rev, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "files": files}, f, indent=2)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest


def _from_hf_cache(repo_id: str, rev: str, tmp: str) -> dict | None:
    """Copy the pinned snapshot out of the local HF cache, verifying each blob against its name."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    snap = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"), "snapshots", rev)
    if not os.path.isdir(snap):
        return None
    entries = []
    for root, _, names in os.walk(snap):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), snap)
            if _wanted(rel):
                entries.append(rel)
    if not entries:
        return None
    _log(f"{repo_id}: copying revision {rev[:10]} from the local Hugging Face cache into {local_dir(repo_id)}")
    files = {}
    for rel in sorted(entries):
        src = os.path.join(snap, rel)
        blob = os.path.basename(os.path.realpath(src))
        expected = {"sha256": blob} if len(blob) == 64 else {"git_sha1": blob} if len(blob) == 40 else None
        if expected is None:
            return None  # not a standard cache layout; fall back to the Hub
        out = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(src, out)
        if not _hash_matches(out, expected):
            shutil.rmtree(tmp, ignore_errors=True)
            _log(f"{repo_id}: cached {rel} fails its checksum; downloading instead")
            return None
        files[rel] = {"size": os.path.getsize(out), **expected}
    return files


def _from_hub(repo_id: str, rev: str, tmp: str) -> dict:
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError as e:
        raise ModelUnavailable(f"cannot download {repo_id}: huggingface_hub is not installed ({e})")
    hint_net = "the first run needs internet access to download it; later runs work offline"
    try:
        info = HfApi().model_info(repo_id, revision=rev, files_metadata=True)
    except GatedRepoError:
        raise ModelUnavailable(f"{repo_id} is a gated model and this machine has no access to it",
                               f"accept the license at https://huggingface.co/{repo_id}, then run `hf auth login`")
    except (RepositoryNotFoundError, RevisionNotFoundError):
        raise ModelUnavailable(f"{repo_id} @ {rev[:10]} does not exist on the Hugging Face Hub",
                               "check hf_model / mlx_model / embed_model and the *_revision settings")
    except Exception as e:
        raise ModelUnavailable(f"could not reach the Hugging Face Hub to download {repo_id} ({type(e).__name__})",
                               hint_net)
    expected = {}
    for s in info.siblings:
        if not _wanted(s.rfilename):
            continue
        lfs = getattr(s, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        expected[s.rfilename] = {"size": s.size, **({"sha256": sha} if sha else {"git_sha1": s.blob_id})}
    total = sum(m["size"] or 0 for m in expected.values())
    _log(f"{repo_id}: downloading revision {rev[:10]} ({total / 1e6:,.0f} MB, {len(expected)} files) "
         f"into {local_dir(repo_id)}")
    try:
        snapshot_download(repo_id, revision=rev, local_dir=tmp, allow_patterns=list(expected))
    except Exception as e:
        raise ModelUnavailable(f"download of {repo_id} failed ({type(e).__name__}: {e})", hint_net)
    shutil.rmtree(os.path.join(tmp, ".cache"), ignore_errors=True)
    _log(f"{repo_id}: verifying checksums")
    files = {}
    for rel, meta in expected.items():
        p = os.path.join(tmp, rel)
        if not os.path.exists(p) or not _hash_matches(p, meta):
            raise ModelFileError(f"downloaded {repo_id}/{rel} fails its checksum",
                                 f"try again: `livermore models fetch {repo_id} --force`")
        files[rel] = {"size": os.path.getsize(p), **{k: v for k, v in meta.items() if k != "size"}}
    return files
