"""`livermore doctor`: check the install, config, models, index and trace directory.

Each line is ✓ ok, ! warning (works, but you should know) or ✗ problem (something will fail).
Exit code is 1 if there is any ✗. `--verify` also re-hashes every model file.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys

from . import __version__, models
from .config import Settings, default_config_path, models_dir
from .errors import LivermoreError

OK, WARN, FAIL = "✓", "!", "✗"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, what, detail, fix=None):
        self.rows.append((status, what, detail, fix))
        print(f"  {status} {what:<12} {detail}")
        if fix and status != OK:
            print(f"    {'':<12} fix: {fix}")

    @property
    def failures(self):
        return sum(1 for r in self.rows if r[0] == FAIL)


def _version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed"), None
    except Exception as e:  # ImportError, or a broken native lib
        return None, f"{type(e).__name__}: {e}"


def _model_size(repo_id):
    m = models.read_manifest(repo_id)
    return sum(f["size"] for f in m["files"].values()) if m else None


def run(config_path: str | None = None, verify: bool = False) -> int:
    print(f"livermore doctor (v{__version__})")
    r = Report()

    # --- runtime
    v = sys.version_info
    r.add(OK if v >= (3, 10) else FAIL, "python", f"{platform.python_version()} ({sys.executable})",
          None if v >= (3, 10) else "livermore needs Python 3.10 or newer")
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    r.add(OK if apple else WARN, "platform", f"{platform.system()} {platform.machine()}"
          + (" (Apple Silicon)" if apple else ""), None if apple else "the mlx backend needs Apple Silicon; use backend: torch")

    # --- config
    try:
        s = Settings.load(config_path)
        path = s.config_path
        r.add(OK, "config", f"{path}" if os.path.exists(path) else f"{path} not present, using defaults",
              None)
    except LivermoreError as e:
        r.add(FAIL, "config", e.message, e.hint)
        s = Settings()
    print(f"    {'':<12} backend={s.backend} device={s.device} top_k={s.top_k} temperature={s.temperature} "
          f"max_tokens={s.max_tokens}")

    # --- packages (torch before faiss: see index._faiss)
    for mod, needed in (("torch", True), ("transformers", True), ("sentence_transformers", True), ("faiss", True),
                        ("huggingface_hub", True), ("mlx", s.backend == "mlx"), ("mlx_lm", s.backend == "mlx")):
        ver, err = _version(mod)
        if ver:
            r.add(OK, mod, ver)
        else:
            r.add(FAIL if needed else WARN, mod, f"not importable ({err})",
                  "pip install -e ." if needed else "only needed for the mlx backend")

    # --- accelerators
    try:
        import mlx.core as mx
        metal = mx.metal.is_available()
        r.add(OK if metal else (FAIL if s.backend == "mlx" else WARN), "mlx metal",
              "available" if metal else "not available (MLX would run on CPU)")
    except Exception:
        pass
    try:
        import torch
        mps = torch.backends.mps.is_available()
        r.add(OK if mps or s.device != "mps" else WARN, "torch mps", "available" if mps else "not available",
              "device: mps will fall back to cpu" if s.device == "mps" and not mps else None)
    except Exception:
        pass

    # --- models
    print(f"    {'':<12} model cache: {models_dir()}")
    needed = {s.embed_model: s.embed_revision}
    if s.backend == "torch":
        needed[s.hf_model] = s.hf_revision
    if s.backend == "mlx":
        needed[s.mlx_model] = s.mlx_revision
    optional = {s.hf_model: s.hf_revision, s.mlx_model: s.mlx_revision}
    for repo, rev in list(needed.items()) + [(k, v) for k, v in optional.items() if k not in needed]:
        is_needed = repo in needed
        problems = models.check(repo, rev, verify=verify)
        label = "model" if is_needed else "model (opt)"
        pinned = models.revision_for(repo, rev)[:10]
        if not problems:
            size = _model_size(repo)
            r.add(OK, label, f"{repo} @ {pinned}, {size / 1e6:,.0f} MB" + (", checksums verified" if verify else ""))
        elif problems[0].startswith("not downloaded"):
            r.add(WARN if is_needed else OK, label,
                  f"{repo} not downloaded yet" + (" (fetched automatically on first use)" if is_needed else " (not used by current backend)"),
                  f"livermore models fetch {repo}" if is_needed else None)
        else:
            r.add(FAIL if is_needed else WARN, label, f"{repo}: " + "; ".join(problems[:3])
                  + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""),
                  f"livermore models fetch {repo} --force")
    if s.backend == "scratch" or os.path.exists(s.scratch_ckpt):
        ok = os.path.exists(s.scratch_ckpt) and os.path.exists(s.scratch_tokenizer)
        r.add(OK if ok else (FAIL if s.backend == "scratch" else WARN), "scratch",
              f"{s.scratch_ckpt}" + ("" if ok else " (missing checkpoint or tokenizer)"),
              None if ok else "set scratch_ckpt / scratch_tokenizer in config.yaml")

    # --- notes + index
    from .index import load, load_notes
    if os.path.exists(s.notes_dir):
        try:
            n = len(load_notes(s.notes_dir))
            r.add(OK if n else WARN, "notes", f"{s.notes_dir}: {n} notes", None if n else "add *.jsonl notes there")
        except Exception as e:
            r.add(FAIL, "notes", f"{s.notes_dir}: unreadable ({type(e).__name__}: {e})",
                  'each line must be {"prompt": ..., "response": ...}')
    else:
        r.add(WARN, "notes", f"{s.notes_dir} does not exist", "only needed to (re)build the index")
    try:
        idx = load(s.index_dir)
        detail = f"{s.index_dir}: {len(idx)} notes, dim {idx.dim}, embed {idx.embed_model}"
        status, fix = OK, None
        if idx.embed_model != s.embed_model:
            status, fix = FAIL, "the index was built with a different embed_model; run `livermore build`"
        else:
            cfg_path = os.path.join(models.local_dir(s.embed_model), "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    dim = json.load(f).get("hidden_size")
                if dim and dim != idx.dim:
                    status, fix = FAIL, f"index dim {idx.dim} != embedding dim {dim}; run `livermore build`"
        r.add(status, "index", detail, fix)
    except LivermoreError as e:
        r.add(FAIL, "index", e.message, e.hint)

    # --- traces
    from .trace import read_index, trace_root
    root = trace_root()
    try:
        os.makedirs(root, exist_ok=True)
        probe = os.path.join(root, ".doctor_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        rows = [x for x in read_index() if x.get("name") == "ask"]
        last = max((x["time"] for x in rows), default=None)
        r.add(OK, "traces", f"{root}: {len(rows)} queries recorded" + (f", last {last[:16]}Z" if last else ""))
    except OSError as e:
        r.add(FAIL, "traces", f"{root} is not writable ({e.strerror})", "check permissions on ~/.livermore")

    print()
    if r.failures:
        print(f"{r.failures} problem(s) found.")
        return 1
    print("No problems found.")
    return 0
