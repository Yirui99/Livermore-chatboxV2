"""From-scratch Transformer (encoder-decoder, 13.26M params). Teaching code + optional backend."""
import dataclasses
import inspect
import sys


def load_checkpoint(path, **kwargs):
    """torch.load that also accepts checkpoints saved before this code became a package.

    seq2seq_best.pt / seq2seq_final.pt pickle their config dataclasses as
    `config.ModelConfig` etc. (the old top-level module name). Map that name to
    livermore.scratch.config for the duration of the load.
    """
    import torch
    from torch.serialization import add_safe_globals

    from . import config

    classes = [c for _, c in inspect.getmembers(config) if inspect.isclass(c) and dataclasses.is_dataclass(c)]
    add_safe_globals([(c, f"config.{c.__name__}") for c in classes])
    prev = sys.modules.get("config")
    sys.modules["config"] = config
    try:
        return torch.load(path, **kwargs)
    finally:
        if prev is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prev
