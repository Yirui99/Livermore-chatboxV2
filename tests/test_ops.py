"""Stage 5: config precedence and errors, model cache + checksums, device fallback, human errors."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVERMORE = [sys.executable, "-m", "livermore.cli"]


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVERMORE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LIVERMORE_CONFIG", raising=False)
    return tmp_path / "home"


def write_config(home, text):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text)


def run_cli(*args, env=None):
    return subprocess.run(LIVERMORE + list(args), capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, **(env or {})})


# ---------------- config ----------------

def test_defaults_when_no_config(home):
    from livermore.config import Settings
    s = Settings.load()
    assert s.temperature == 0.2 and s.backend == "mlx" and s.top_k == 3
    assert set(s.source.values()) == {"default"}


def test_precedence_default_config_cli(home):
    from livermore.config import Settings
    write_config(home, "temperature: 0.5\ntop_k: 5\nindex_dir: myindex\n")
    s = Settings.load()
    assert s.temperature == 0.5 and s.top_k == 5
    assert s.index_dir == str(home / "myindex")  # relative paths resolve against the config file
    s.override(temperature=0.0, top_k=None)
    assert s.temperature == 0.0 and s.source["temperature"] == "cli"
    assert s.top_k == 5 and s.source["top_k"] == str(home / "config.yaml")


@pytest.mark.parametrize("text, msg", [
    ("temprature: 0.3\n", "did you mean 'temperature'"),
    ("top_k: three\n", "'top_k'"),
    ("backend: gpt\n", "backend 'gpt'"),
    ("temperature: 5\n", "between 0 and 2"),
    ("top_k: [1\n", "not valid YAML"),
    ("- a\n- b\n", "must be a mapping"),
])
def test_config_errors_are_one_line(home, text, msg):
    from livermore.config import Settings
    from livermore.errors import ConfigError
    write_config(home, text)
    with pytest.raises(ConfigError) as e:
        Settings.load()
    assert msg in str(e.value)


def test_cli_config_error_has_no_traceback(home):
    write_config(home, "temprature: 0.3\n")
    r = run_cli("config", "show")
    assert r.returncode == 1
    assert "livermore: error: unknown setting 'temprature'" in r.stderr
    assert "Traceback" not in r.stderr


def test_config_init_roundtrips(home):
    r = run_cli("config", "init")
    assert r.returncode == 0, r.stderr
    from livermore.config import Settings
    s = Settings.load()
    assert s.temperature == 0.2 and s.backend == "mlx"
    assert run_cli("config", "init").returncode == 1  # refuses to overwrite


# ---------------- model cache ----------------

def _fake_hf_cache(tmp_path, repo, rev, files):
    """Build an HF-cache-shaped snapshot whose blob names are the files' hashes."""
    base = tmp_path / "hfcache" / ("models--" + repo.replace("/", "--"))
    (base / "blobs").mkdir(parents=True)
    snap = base / "snapshots" / rev
    for rel, data in files.items():
        if rel.endswith(".safetensors"):
            name = hashlib.sha256(data).hexdigest()
        else:
            name = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        (base / "blobs" / name).write_bytes(data)
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).symlink_to(base / "blobs" / name)
    return tmp_path / "hfcache"


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    import huggingface_hub.constants as C
    repo, rev = "acme/tiny", "a" * 40
    files = {"config.json": b'{"hidden_size": 4}', "model.safetensors": b"\x00" * 1000,
             "sub/vocab.txt": b"a\nb\n", "README.md": b"not wanted"}
    monkeypatch.setattr(C, "HF_HUB_CACHE", str(_fake_hf_cache(tmp_path, repo, rev, files)))
    return repo, rev


def test_fetch_from_cache_writes_manifest_and_filters(fake_model):
    from livermore import models
    repo, rev = fake_model
    d = models.ensure(repo, rev)
    m = models.read_manifest(repo)
    assert m["revision"] == rev and set(m["files"]) == {"config.json", "model.safetensors", "sub/vocab.txt"}
    assert "sha256" in m["files"]["model.safetensors"] and "git_sha1" in m["files"]["config.json"]
    assert models.check(repo, rev, verify=True) == []
    assert os.path.isdir(d)


def test_missing_truncated_corrupted_files(fake_model):
    from livermore import models
    from livermore.errors import ModelFileError
    repo, rev = fake_model
    d = models.ensure(repo, rev)
    w = os.path.join(d, "model.safetensors")

    with open(w, "r+b") as f:  # same size, different bytes: only --verify can see it
        f.write(b"\x01")
    assert models.check(repo, rev) == []
    assert models.check(repo, rev, verify=True) == ["model.safetensors fails its checksum (corrupted)"]

    with open(w, "wb") as f:
        f.write(b"\x00" * 10)
    assert "truncated or corrupted" in models.check(repo, rev)[0]
    with pytest.raises(ModelFileError) as e:
        models.ensure(repo, rev)
    assert "models fetch acme/tiny --force" in e.value.hint

    os.remove(w)
    assert models.check(repo, rev) == ["model.safetensors is missing"]
    models.fetch(repo, rev, force=True)
    assert models.check(repo, rev, verify=True) == []


def test_revision_mismatch_is_reported(fake_model):
    from livermore import models
    repo, rev = fake_model
    models.ensure(repo, rev)
    assert "configured" in models.check(repo, "b" * 40)[0]


def test_unknown_model_offline_is_human_error(monkeypatch, tmp_path):
    import huggingface_hub.constants as C
    from livermore import models
    from livermore.errors import ModelUnavailable
    monkeypatch.setattr(C, "HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelUnavailable) as e:
        models.ensure("acme/not-here", "c" * 40)
    assert "acme/not-here" in e.value.message


# ---------------- device fallback ----------------

def test_mps_unavailable_falls_back_to_cpu_with_warning(monkeypatch, capsys):
    import torch
    from livermore._device import resolve_torch_device
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_torch_device("mps") == "cpu"
    assert "MPS is not available" in capsys.readouterr().err
    assert resolve_torch_device("cpu") == "cpu"


def test_fallback_recorded_in_trace(monkeypatch, home):
    import torch
    from livermore import load
    from livermore.ask import ask
    from livermore.trace import read_trace

    class Fake:
        name, model, dtype = "fake", "m", "x"

        def __init__(self):
            from livermore._device import resolve_torch_device
            self.device_requested = "mps"
            self.device = resolve_torch_device("mps")
            self.last_usage = {}

        def apply_chat_template(self, m):
            return m[1]["content"]

        def generate(self, p, n, **_):
            yield "ok"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    a = ask("q", load(os.path.join(ROOT, "kb_data")), Fake(), entry="test")
    gen = next(s for s in read_trace(a.trace_id) if s["name"] == "generate")["attributes"]
    assert gen["device"] == "cpu" and gen["device_requested"] == "mps"


# ---------------- index / notes errors ----------------

def test_index_not_found_and_notes_empty(tmp_path):
    from livermore.errors import IndexNotFound, NotesNotFound
    from livermore.index import build, load
    with pytest.raises(IndexNotFound) as e:
        load(str(tmp_path / "nope"))
    assert "livermore build" in e.value.hint
    (tmp_path / "empty").mkdir()
    with pytest.raises(IndexNotFound):
        load(str(tmp_path / "empty"))
    with pytest.raises(NotesNotFound) as e:
        build(str(tmp_path / "empty"))
    assert "no notes in" in e.value.message
    with pytest.raises(NotesNotFound):
        build(str(tmp_path / "missing"))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "x.jsonl").write_text('{"question": "no prompt key"}\n')
    with pytest.raises(NotesNotFound) as e:
        build(str(tmp_path / "bad"))
    assert '"prompt" and "response"' in e.value.hint


def test_cli_ask_with_missing_index_prints_one_line(home):
    r = run_cli("ask", "hi", "--index", "/nonexistent/idx")
    assert r.returncode == 1
    assert r.stderr.strip().splitlines()[0] == "livermore: error: no index at /nonexistent/idx"
    assert "Traceback" not in r.stderr


def test_mlx_cpu_fallback_really_runs_on_cpu(monkeypatch):
    """Regression: the fallback used to only relabel the device while mlx_lm kept its GPU stream."""
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from livermore import models
    if models.check("mlx-community/Llama-3.2-1B-Instruct-4bit") and \
            not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit")):
        pytest.skip("MLX model not available offline")
    gen_mod = sys.modules["mlx_lm.generate"]  # the module; `mlx_lm.generate` attribute is the function
    saved = gen_mod.generation_stream
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    try:
        from livermore.backends.mlx import MLXBackend
        b = MLXBackend()
        assert b.device == "cpu"
        assert "cpu" in str(gen_mod.generation_stream).lower()
    finally:
        gen_mod.generation_stream = saved
        mx.set_default_device(mx.gpu)
