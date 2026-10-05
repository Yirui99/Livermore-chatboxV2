"""Tracing: span tree, the three failure kinds, feedback, stats. Uses a fake backend
(no model load) and the real index; LIVERMORE_HOME is redirected to a temp dir."""
import os
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeBackend:
    name, model, device, dtype = "fake", "fake-model", "cpu", "none"

    def __init__(self, chunks=("Cut ", "losses ", "early."), fail=False, delay=0.0):
        self.chunks, self.fail, self.delay = chunks, fail, delay
        self.last_usage = {}

    def apply_chat_template(self, messages):
        return messages[0]["content"] + "\n" + messages[1]["content"]

    def generate(self, prompt, max_tokens, **_):
        for c in self.chunks:
            time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("model exploded")
            yield c
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": len(self.chunks)}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVERMORE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def index():
    from livermore import load
    return load(os.path.join(ROOT, "kb_data"))


def spans_of(tid):
    from livermore.trace import read_trace
    return {s["name"]: s for s in read_trace(tid)}


def index_rows():
    from livermore.trace import read_index
    return read_index()


def test_ok_trace_tree(index, home):
    from livermore import ask
    a = ask("When should I cut my losses?", index, FakeBackend(), entry="test")
    assert a.text == "Cut losses early."
    files = list((home / "traces").glob("*/trace_*.jsonl"))
    assert len(files) == 1 and files[0].name == f"trace_{a.trace_id}.jsonl"
    sp = spans_of(a.trace_id)
    assert set(sp) == {"ask", "embed_query", "retrieve", "build_prompt", "generate"}
    root = sp["ask"]
    assert root["parent_span_id"] is None and root["status"] == "ok"
    assert all(sp[n]["parent_span_id"] == root["span_id"] for n in ("embed_query", "retrieve", "build_prompt", "generate"))
    assert root["attributes"]["backend"] == "fake" and root["attributes"]["device"] == "cpu"
    g = sp["generate"]["attributes"]
    assert g["completion_tokens"] == 3 and g["prompt_tokens"] == 100 and g["ttft_ms"] is not None
    assert sp["retrieve"]["attributes"]["n_hits"] == 3
    assert sp["embed_query"]["attributes"]["device"] == "cpu"
    (row,) = index_rows()
    assert row["status"] == "ok" and row["query"] == "When should I cut my losses?" and row["error_kinds"] == []
    assert row["completion_tokens"] == 3 and row["backend"] == "fake" and row["entry"] == "test"


def test_backend_error(index):
    from livermore import ask
    with pytest.raises(RuntimeError):
        ask("q", index, FakeBackend(fail=True), entry="test")
    (row,) = index_rows()
    assert row["status"] == "error" and row["error_kinds"] == ["backend_error"]
    sp = spans_of(row["trace_id"])
    assert sp["generate"]["status"] == "error" and "model exploded" in sp["generate"]["error"]
    assert sp["ask"]["status"] == "error"


def test_generation_timeout_keeps_partial(index):
    from livermore.ask import GenerationTimeout, ask
    with pytest.raises(GenerationTimeout) as ei:
        ask("q", index, FakeBackend(chunks=("a", "b", "c", "d"), delay=0.05), timeout_s=0.08, entry="test")
    assert ei.value.partial and ei.value.partial != "abcd"
    (row,) = index_rows()
    assert row["status"] == "error" and row["error_kinds"] == ["generation_timeout"]
    sp = spans_of(row["trace_id"])
    assert sp["generate"]["status"] == "error" and sp["generate"]["attributes"]["partial_chars"] >= 1


def test_empty_retrieval_is_an_error_span_but_still_answers(index):
    import faiss
    from livermore import ask
    from livermore.index import Index
    empty = Index(faiss.IndexFlatIP(index.dim), [], embed_model=index.embed_model, device="cpu")
    empty._embedder = index.embedder
    a = ask("anything", empty, FakeBackend(), entry="test")
    assert a.hits == [] and a.text
    sp = spans_of(a.trace_id)
    assert sp["retrieve"]["status"] == "error" and sp["retrieve"]["attributes"]["error_kind"] == "empty_retrieval"
    assert sp["ask"]["status"] == "ok"
    (row,) = index_rows()
    assert row["error_kinds"] == ["empty_retrieval"]


def test_feedback_written_to_trace_and_index(index):
    from livermore import ask
    from livermore.trace import record_feedback
    a = ask("q", index, FakeBackend(), entry="test")
    assert record_feedback(a.trace_id, -1, "wrong note", entry="test")
    assert record_feedback(a.trace_id[:6], 1)  # unique prefix; last click wins
    assert not record_feedback("deadbeefdeadbeef", 1)
    assert not record_feedback("../../etc", 1)
    from livermore.trace import read_trace
    fb = [s for s in read_trace(a.trace_id) if s["name"] == "feedback"]
    assert [s["attributes"]["rating"] for s in fb] == [-1, 1]
    from livermore.stats import load
    asks, _ = load("all", include_eval=True)
    assert asks[0]["rating"] == 1


def test_stats_excludes_eval_and_computes_percentiles(index, capsys):
    from livermore import ask, stats
    for _ in range(3):
        ask("q", index, FakeBackend(), entry="cli")
    ask("q", index, FakeBackend(), entry="eval")
    with pytest.raises(RuntimeError):
        ask("q", index, FakeBackend(fail=True), entry="streamlit")
    asks, _ = stats.load("7d")
    s = stats.summarize(asks)
    assert s["queries"] == 4 and s["failed"] == 1 and s["failure_rate"] == 0.25
    assert s["latency_ms"]["n"] == 3 and s["error_kinds"] == {"backend_error": 1}
    assert stats.pct([10, 20, 30, 40], 0.5) == 20 and stats.pct([10, 20, 30, 40], 0.95) == 40
    stats.report("7d")
    out = capsys.readouterr().out
    assert "queries      4" in out and "By backend" in out and "By day" in out


def test_trace_false_writes_nothing(index, home):
    from livermore import ask
    ask("q", index, FakeBackend(), trace=False)
    assert not (home / "traces").exists()
