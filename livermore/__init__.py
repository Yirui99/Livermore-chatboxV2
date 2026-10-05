"""Livermore: a local notes Q&A engine for Apple Silicon."""

__version__ = "0.1.0"

__all__ = ["__version__", "Index", "Hit", "Answer", "build", "load", "ask", "get_backend"]


_EXPORTS = {
    "Index": "index", "build": "index", "load": "index",
    "Hit": "retrieve",
    "Answer": "ask", "ask": "ask",
    "get_backend": "generate",
}


def __getattr__(name):
    # Lazy so that `import livermore` / `livermore --version` don't pull in torch/faiss.
    if name not in _EXPORTS:
        raise AttributeError(f"module 'livermore' has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
    # Importing livermore.ask binds the *module* as livermore.ask; rebind every export
    # from that module so livermore.ask is the function.
    for k, v in _EXPORTS.items():
        if v == _EXPORTS[name]:
            globals()[k] = getattr(mod, k)
    return globals()[name]
