"""Streamlit chat UI. UI only: all retrieval/generation goes through livermore.*

    streamlit run apps/streamlit_app.py
"""
import datetime
import os
import sys

import streamlit as st

st.set_page_config(page_title="Livermore", layout="centered", initial_sidebar_state="expanded")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import CHAT_CSS, GLOBAL_CSS  # noqa: E402

import livermore  # noqa: E402
from livermore.config import Settings  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown(CHAT_CSS, unsafe_allow_html=True)

SETTINGS = Settings()

# label -> backend name
MODELS = {
    "RAG Llama (default)": "torch",
    "RAG Llama · MLX 4-bit": "mlx",
    "My Custom Transformer (no RAG)": "scratch",
}


@st.cache_resource
def get_index(path: str):
    return livermore.load(path)


@st.cache_resource
def get_backend(name: str, ckpt: str | None = None):
    kw = SETTINGS.backend_kwargs(name)
    if name == "scratch" and ckpt:
        kw["ckpt"] = ckpt
    return livermore.get_backend(name, **kw)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(livermore.config.PROJECT_ROOT, path)


# ---------------- state ----------------
ss = st.session_state
if "chat_sessions" not in ss:
    ss.chat_sessions = ["Current Chat"]
    ss.current_chat_id = "Current Chat"
if "chat_messages" not in ss:
    ss.chat_messages = {"Current Chat": []}
if "chatbot_settings" not in ss:
    ss.chatbot_settings = {"model_label": next(iter(MODELS)), "max_new_tokens": 256,
                           "temperature": 0.8, "top_p": 0.9, "top_k": 3,
                           "custom_ckpt_path": os.path.relpath(SETTINGS.scratch_ckpt, livermore.config.PROJECT_ROOT)}
cfg = ss.chatbot_settings

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("Chat List")
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ss.chat_sessions.append(name)
        ss.chat_messages[name] = []
        ss.current_chat_id = name
        st.rerun()
    st.markdown("---")
    st.subheader("Chat History")
    for i, chat_id in enumerate(ss.chat_sessions):
        if st.button(chat_id, key=f"chat_btn_{i}_{hash(chat_id)}", use_container_width=True,
                     type="primary" if chat_id == ss.current_chat_id else "secondary"):
            ss.current_chat_id = chat_id
            st.rerun()
    st.markdown("---")

    with st.expander("⚙️ Settings", expanded=False):
        st.subheader("Model")
        labels = list(MODELS)
        cfg["model_label"] = st.selectbox("Select model", labels, index=labels.index(cfg["model_label"]),
                                          key="sidebar_model_label")
        backend_name = MODELS[cfg["model_label"]]
        if backend_name == "scratch":
            cfg["custom_ckpt_path"] = st.text_input(
                "Checkpoint path", value=cfg["custom_ckpt_path"], key="sidebar_custom_ckpt",
                help="seq2seq_best.pt / sft_best.pt / seq2seq_final.pt / sft.pt under transformer/checkpoints/",
            ).strip()
        else:
            st.code(SETTINGS.hf_model if backend_name == "torch" else SETTINGS.mlx_model, language="bash")

        st.subheader("Generation Parameters")
        cfg["max_new_tokens"] = st.slider("Max new tokens", 32, 512, cfg["max_new_tokens"], step=16,
                                          key="sidebar_max_tokens")
        cfg["temperature"] = st.slider("Temperature", 0.0, 2.0, cfg["temperature"], step=0.05,
                                       key="sidebar_temperature")
        cfg["top_p"] = st.slider("Top-p", 0.1, 1.0, cfg["top_p"], step=0.05, key="sidebar_top_p")
        if backend_name == "scratch":
            cfg["top_k"] = st.slider("Top-k (sampling)", 0, 50, min(cfg["top_k"], 50), step=1, key="sidebar_top_k")
        else:
            cfg["top_k"] = st.slider("Top-k (retrieval, RAG only)", 1, 10, max(1, min(cfg["top_k"], 10)), step=1,
                                     key="sidebar_top_k")

# ---------------- main ----------------
st.title("Livermore-style Trading Chatbox")
st.markdown("A Jesse Livermore Style Chatbox for trading questions. You can switch between different models.")
st.caption(f"Current Chat: {ss.current_chat_id}")

try:
    index = get_index(SETTINGS.index_dir)
    backend = get_backend(backend_name, _resolve(cfg["custom_ckpt_path"]) if backend_name == "scratch" else None)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    msg = str(e)
    if "gated" in msg.lower() or "403" in msg:
        st.error(f"❌ **Model Access Required**\n\n{msg}\n\n**How to fix:**\n"
                 f"1. Visit https://huggingface.co/{SETTINGS.hf_model} and click 'Agree and access repository'\n"
                 "2. Log in: `huggingface-cli login`\n3. Restart the application")
    else:
        st.error(f"Failed to load model/index: {type(e).__name__}: {msg}")
    st.stop()

messages = ss.chat_messages.setdefault(ss.current_chat_id, [])
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Type your question here...")
if not query:
    st.stop()

messages.append({"role": "user", "content": query})
with st.chat_message("user"):
    st.markdown(query)

with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        try:
            if backend_name == "scratch":
                # The scratch model gets only the question; its top_k is a sampling parameter.
                ans = livermore.ask(query, index, backend, k=SETTINGS.top_k, max_tokens=cfg["max_new_tokens"],
                                    temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"])
            else:
                ans = livermore.ask(query, index, backend, k=cfg["top_k"], max_tokens=cfg["max_new_tokens"],
                                    temperature=cfg["temperature"], top_p=cfg["top_p"])
        except Exception as e:
            st.error(f"Error during generation: {type(e).__name__}: {e}")
            st.stop()
    st.markdown(ans.text)
    if backend_name != "scratch":
        with st.expander("🔍 Retrieved notes"):
            for i, h in enumerate(ans.hits):
                st.markdown(f"**Note {i + 1}** (score = `{h.score:.4f}`)")
                st.write(h.text[:600] + ("..." if len(h.text) > 600 else ""))
                st.markdown("---")

messages.append({"role": "assistant", "content": ans.text})
