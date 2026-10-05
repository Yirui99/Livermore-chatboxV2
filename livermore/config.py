"""Settings: built-in defaults < ~/.livermore/config.yaml < command-line flags.

    livermore config init     write a commented config.yaml with the defaults
    livermore config show     effective values and where each came from

LIVERMORE_HOME moves the whole ~/.livermore directory (config, models, traces);
LIVERMORE_CONFIG points at a different config file.
"""
from __future__ import annotations

import dataclasses
import difflib
import os
import typing
from dataclasses import dataclass, field

from .errors import ConfigError
from .generate import BACKENDS

# Repo root when installed with `pip install -e .`. The trading-notes corpus and its index ship there.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICES = ("auto", "cpu", "mps", "cuda")
PATH_FIELDS = ("notes_dir", "index_dir", "scratch_ckpt", "scratch_tokenizer")


def home() -> str:
    return os.path.expanduser(os.environ.get("LIVERMORE_HOME", "~/.livermore"))


def default_config_path() -> str:
    return os.path.expanduser(os.environ.get("LIVERMORE_CONFIG", os.path.join(home(), "config.yaml")))


def models_dir() -> str:
    return os.path.join(home(), "models")


def _repo_or_home(repo_rel: str, home_rel: str) -> str:
    p = os.path.join(PROJECT_ROOT, repo_rel)
    return p if os.path.exists(p) else os.path.join(home(), home_rel)


@dataclass
class Settings:
    # corpus. There is no chunking: each JSONL record ({"prompt", "response"}) is one note.
    notes_dir: str = field(default_factory=lambda: _repo_or_home("data", "notes"))
    index_dir: str = field(default_factory=lambda: _repo_or_home("kb_data", "index"))
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_revision: str | None = None  # None = the revision pinned in livermore.models.PINNED
    top_k: int = 3

    # generation
    backend: str = "mlx"
    device: str = "auto"  # torch/scratch backends; mps falls back to cpu with a warning if unavailable
    hf_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    hf_revision: str | None = None
    mlx_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    mlx_revision: str | None = None
    scratch_ckpt: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "transformer", "checkpoints", "sft_best.pt"))
    scratch_tokenizer: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "transformer", "tokenizer", "corpus", "processed", "tokenizer.json"))
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    timeout_s: float = 120.0  # generation wall-clock limit; exceeded -> generation_timeout span

    # server
    host: str = "127.0.0.1"
    port: int = 8000

    # where each value came from: default | <config path> | cli
    source: dict = field(default_factory=dict, repr=False, compare=False)
    config_path: str | None = field(default=None, repr=False, compare=False)

    # ---------------- loading ----------------

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls) if f.name not in ("source", "config_path")]

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        import yaml

        s = cls()
        s.source = {n: "default" for n in cls.names()}
        s.config_path = path = os.path.expanduser(path or default_config_path())
        if not os.path.exists(path):
            return s
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            where = f" (line {mark.line + 1})" if mark else ""
            raise ConfigError(f"{path} is not valid YAML{where}", "fix the file or delete it to use defaults")
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must be a mapping of setting: value", "see `livermore config init`")
        s._apply(data, origin=path, base_dir=os.path.dirname(path))
        return s

    def override(self, **values) -> "Settings":
        """Command-line flags. None means 'not given'."""
        self._apply({k: v for k, v in values.items() if v is not None}, origin="cli", base_dir=os.getcwd())
        return self

    def _apply(self, data: dict, origin: str, base_dir: str):
        hints = typing.get_type_hints(type(self))
        valid = self.names()
        for k, v in data.items():
            if k not in valid:
                close = difflib.get_close_matches(k, valid, n=1)
                raise ConfigError(f"unknown setting '{k}' in {origin}",
                                  f"did you mean '{close[0]}'?" if close else f"valid settings: {', '.join(valid)}")
            setattr(self, k, self._coerce(k, v, hints[k], origin, base_dir))
            self.source[k] = origin
        self._validate(origin)

    @staticmethod
    def _coerce(k, v, typ, origin, base_dir):
        optional = type(None) in typing.get_args(typ)
        base = next((t for t in typing.get_args(typ) if t is not type(None)), typ)
        if v is None:
            if optional:
                return None
            raise ConfigError(f"'{k}' in {origin} cannot be empty")
        try:
            if base is bool:
                if isinstance(v, str):
                    v = v.lower() in ("1", "true", "yes", "on")
                return bool(v)
            if base is int and isinstance(v, float) and not v.is_integer():
                raise ValueError
            v = base(v)
        except (TypeError, ValueError):
            raise ConfigError(f"'{k}' in {origin} should be {base.__name__}, got {v!r}")
        if k in PATH_FIELDS:
            v = os.path.expanduser(v)
            if not os.path.isabs(v):
                v = os.path.normpath(os.path.join(base_dir, v))
        return v

    def _validate(self, origin):
        if self.backend not in BACKENDS:
            raise ConfigError(f"backend '{self.backend}' ({origin}) is not one of {', '.join(BACKENDS)}")
        if self.device not in DEVICES:
            raise ConfigError(f"device '{self.device}' ({origin}) is not one of {', '.join(DEVICES)}")
        for k, lo in (("top_k", 1), ("max_tokens", 1), ("port", 1)):
            if getattr(self, k) < lo:
                raise ConfigError(f"'{k}' ({origin}) must be >= {lo}")
        if not 0 <= self.temperature <= 2:
            raise ConfigError(f"'temperature' ({origin}) must be between 0 and 2")
        if not 0 < self.top_p <= 1:
            raise ConfigError(f"'top_p' ({origin}) must be in (0, 1]")
        if self.timeout_s <= 0:
            raise ConfigError(f"'timeout_s' ({origin}) must be > 0")

    # ---------------- use ----------------

    def backend_kwargs(self, name: str | None = None) -> dict:
        name = name or self.backend
        if name == "torch":
            return {"model": self.hf_model, "revision": self.hf_revision, "device": self.device}
        if name == "mlx":
            return {"model": self.mlx_model, "revision": self.mlx_revision}
        if name == "scratch":
            return {"ckpt": self.scratch_ckpt, "tokenizer": self.scratch_tokenizer, "device": self.device}
        raise ConfigError(f"unknown backend {name!r}")

    def describe(self) -> list[tuple[str, object, str]]:
        return [(n, getattr(self, n), self.source.get(n, "default")) for n in self.names()]


CONFIG_TEMPLATE = """\
# Livermore settings. Precedence: built-in default < this file < command-line flag.
# Delete a line to go back to the default. `livermore config show` prints the effective values.

# --- corpus ---
# JSONL notes, one {{"prompt": ..., "response": ...}} per line. Each record is one note (no chunking).
notes_dir: {notes_dir}
index_dir: {index_dir}            # built by `livermore build`
embed_model: {embed_model}
top_k: {top_k}                    # notes retrieved per question

# --- generation ---
backend: {backend}                # mlx | torch | scratch
device: {device}                  # auto | cpu | mps | cuda  (torch/scratch; mps falls back to cpu)
hf_model: {hf_model}
mlx_model: {mlx_model}
max_tokens: {max_tokens}
temperature: {temperature}        # used by the app, the server, the CLI and `livermore eval` alike
top_p: {top_p}
timeout_s: {timeout_s}

# --- server ---
host: {host}
port: {port}
"""


def render_template(s: Settings | None = None) -> str:
    s = s or Settings()
    return CONFIG_TEMPLATE.format(**{n: getattr(s, n) for n in Settings.names()})
