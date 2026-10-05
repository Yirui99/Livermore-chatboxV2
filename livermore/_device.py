import sys


def warn(msg: str):
    print(f"livermore: warning: {msg}", file=sys.stderr, flush=True)


def resolve_torch_device(pref: str = "auto") -> str:
    """auto -> cuda if available, else cpu (same policy as the pre-refactor rag_qa.RAGQA).

    MPS is deliberately not chosen under "auto": running torch on MPS from Streamlit's
    script threads deadlocks. Pass device="mps" to opt in; if MPS (or CUDA) is not
    available we fall back to CPU with a warning, and the trace records both.
    """
    import torch
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps" and not torch.backends.mps.is_available():
        warn("device 'mps' requested but MPS is not available on this machine; using CPU")
        return "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        warn("device 'cuda' requested but CUDA is not available; using CPU")
        return "cpu"
    return pref
