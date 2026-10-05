"""Livermore: a local notes Q&A engine for Apple Silicon."""

__version__ = "0.1.0"

# These modules are light (torch/faiss/mlx are imported inside functions), so import
# eagerly. That also guarantees `livermore.ask` is the function: the submodule
# livermore/ask.py is bound here first and then shadowed, and a later
# `from livermore.ask import X` does not rebind it.
from .ask import Answer, GenerationTimeout, ask  # noqa: E402
from .generate import get_backend  # noqa: E402
from .index import Index, build, load  # noqa: E402
from .retrieve import Hit  # noqa: E402

__all__ = ["__version__", "Index", "Hit", "Answer", "GenerationTimeout", "build", "load", "ask", "get_backend"]
