def resolve_torch_device(pref: str = "auto") -> str:
    """Same policy as the pre-refactor rag_qa.RAGQA.

    MPS is deliberately not chosen under "auto": running torch on MPS from
    Streamlit's script threads deadlocks. Pass device="mps" explicitly to opt in.
    """
    if pref != "auto":
        return pref
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
