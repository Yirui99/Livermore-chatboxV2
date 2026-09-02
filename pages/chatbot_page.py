# pages/chatbot_page.py
import streamlit as st
import os
import sys
import datetime
import time

# Set environment variables to suppress tensorflow warnings and avoid protobuf issues
# Must be set BEFORE any imports that might trigger tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Try to suppress protobuf version warnings
try:
    import warnings

    warnings.filterwarnings('ignore', category=UserWarning)
except:
    pass

from rag_qa import RAGQA, load_yaml_config

# config
import os
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_rag.yaml")

# Define the models the user can choose from.
# KEY = model name, VALUE = model id
AVAILABLE_MODELS = {
    "RAG LLama (default)": {
        "backend": "rag_llama",
        "hf_model": "meta-llama/Llama-3.2-1B-Instruct",
        "ckpt_path": None,
    },
    "My Custom Transformer (no RAG)": {
        "backend": "custom_only",
        "hf_model": None,
        "ckpt_path": "",
    },
}


# Cache model loading so switching questions is fast
@st.cache_resource
def load_ragqa(config_path: str, model_name: str):
    # Delayed import to avoid importing torch and other dependencies when loading the main page
    # Set environment variable to suppress tensorflow warnings if not needed
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress tensorflow warnings

    config = load_yaml_config(config_path)
    config["model_name"] = model_name
    rag_qa = RAGQA(config)
    return rag_qa


# Cache seq2seq model loading for custom transformer
@st.cache_resource
def load_custom_transformer_model(ckpt_path: str, tokenizer_path: str = None, device: str = None):
    """
    Load seq2seq model from checkpoint for custom transformer.
    
    Args:
        ckpt_path: Path to checkpoint file (e.g., checkpoints/seq2seq_best.pt or checkpoints/sft_best.pt)
        tokenizer_path: Path to tokenizer JSON file (default: transformer/tokenizer/corpus/processed/tokenizer.json)
        device: Device to use (default: cuda if available, else cpu)
    
    Returns:
        Tuple of (tokenizer, config, token_emb, pos_emb, model, model_kind, pad_id)
    """
    import sys
    import torch
    
    # Add transformer directory to path
    transformer_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformer")
    if transformer_dir not in sys.path:
        sys.path.insert(0, transformer_dir)
    
    from generate import build_modules
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if tokenizer_path is None:
        # Default tokenizer path
        tokenizer_path = os.path.join(transformer_dir, "tokenizer", "corpus", "processed", "tokenizer.json")
    
    # Handle relative paths - convert to absolute if needed
    if not os.path.isabs(ckpt_path):
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        abs_ckpt_path = os.path.join(project_root, ckpt_path)
        if os.path.exists(abs_ckpt_path):
            ckpt_path = abs_ckpt_path
        elif not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path} (also tried: {abs_ckpt_path})")
    elif not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    device_obj = torch.device(device)
    return build_modules(ckpt_path, tokenizer_path, device_obj)


def generate_custom_transformer_response(src: str, ckpt_path: str, tokenizer_path: str = None,
                                         max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0,
                                         min_new_tokens: int = 12, no_repeat_ngram_size: int = 3,
                                         repetition_penalty: float = 1.1, device: str = None) -> str:
    """
    Generate response using custom transformer (seq2seq) model.
    
    Args:
        src: Source text (user query)
        ckpt_path: Path to checkpoint file
        tokenizer_path: Path to tokenizer JSON file
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling (0 = disabled)
        min_new_tokens: Minimum number of tokens to generate
        no_repeat_ngram_size: N-gram size for repetition penalty
        repetition_penalty: Repetition penalty factor
        device: Device to use
    
    Returns:
        Generated response text
    """
    import sys
    import torch
    
    # Add transformer directory to path
    transformer_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformer")
    if transformer_dir not in sys.path:
        sys.path.insert(0, transformer_dir)
    
    from generate import generate_seq2seq
    
    # Load model (cached)
    tok, cfg, tok_emb, pos_emb, model, model_kind, pad_id = load_custom_transformer_model(
        ckpt_path, tokenizer_path, device
    )
    
    if model_kind != "seq2seq":
        raise ValueError(f"Model is not seq2seq type, got: {model_kind}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    
    # Generate response
    response = generate_seq2seq(
        src, tok, cfg, tok_emb, pos_emb, model, pad_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        min_new_tokens=min_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        device=device_obj
    )
    
    return response

def show_chatbot_page():
    """Display chatbot page"""

    # Add Navy Blue background styling and Gold text for titles
    st.markdown("""
        <style>
            .stApp {
                background: #1a2332 !important;
            }
            
            /* All titles - Gold color, clear and visible */
            h1, h2, h3, h4, h5, h6 {
                color: #d4af37 !important;
                font-weight: bold !important;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
            }
            
            /* Ensure all title elements are gold */
            .main h1,
            .main h2,
            .main h3,
            [data-testid="stAppViewContainer"] h1,
            [data-testid="stAppViewContainer"] h2,
            [data-testid="stAppViewContainer"] h3,
            div[data-testid="stAppViewContainer"] h1,
            div[data-testid="stAppViewContainer"] h2,
            div[data-testid="stAppViewContainer"] h3 {
                color: #d4af37 !important;
                font-weight: bold !important;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
            }
            
            /* Sidebar titles and labels - Gold color */
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] h4,
            section[data-testid="stSidebar"] h5,
            section[data-testid="stSidebar"] h6 {
                color: #d4af37 !important;
                font-weight: bold !important;
            }
            
            /* Sidebar general labels - Gold (but settings labels override to white) */
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                color: #d4af37 !important;
            }
            
            /* Settings expander content - All labels and text white to match Config path style */
            section[data-testid="stSidebar"] .streamlit-expanderContent label,
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown,
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown p,
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown strong,
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown p strong,
            section[data-testid="stSidebar"] .streamlit-expanderContent p,
            section[data-testid="stSidebar"] .streamlit-expanderContent p strong {
                color: #ffffff !important;
                font-weight: normal !important;
            }
            
            /* Settings expander header - White and bold */
            section[data-testid="stSidebar"] .streamlit-expanderHeader,
            section[data-testid="stSidebar"] .streamlit-expanderHeader *,
            section[data-testid="stSidebar"] .streamlit-expanderHeader p,
            section[data-testid="stSidebar"] .streamlit-expanderHeader span,
            section[data-testid="stSidebar"] .streamlit-expanderHeader div {
                color: #ffffff !important;
                font-weight: bold !important;
            }
            
            /* Settings labels - White color to match Config path style */
            section[data-testid="stSidebar"] .stTextInput label,
            section[data-testid="stSidebar"] .stSelectbox label,
            section[data-testid="stSidebar"] .stSelectbox > label,
            section[data-testid="stSidebar"] .stSelectbox label p,
            section[data-testid="stSidebar"] .stSelectbox label span,
            section[data-testid="stSidebar"] .stSelectbox label div,
            section[data-testid="stSidebar"] .stSlider label,
            section[data-testid="stSidebar"] .streamlit-expanderContent label {
                color: #ffffff !important;
                font-weight: normal !important;
                background-color: transparent !important;
            }
            
            /* Model name label - White to match Config path style (more specific selector) */
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown strong,
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown p strong,
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown p,
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown {
                color: #ffffff !important;
                font-weight: normal !important;
            }
            
            /* Override any gold color for model name specifically */
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown p strong[style*="color"],
            section[data-testid="stSidebar"] .streamlit-expanderContent .stMarkdown strong[style*="color"] {
                color: #ffffff !important;
            }
            
            /* Uniform styling for all input fields in settings - match Config path style */
            /* Text input fields */
            section[data-testid="stSidebar"] .stTextInput > div > div > input {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ffffff !important;
                border-radius: 0.25rem !important;
            }
            
            /* Selectbox dropdown - match text input style */
            section[data-testid="stSidebar"] .stSelectbox > div > div {
                background-color: #ffffff !important;
                border: 1px solid #ffffff !important;
                border-radius: 0.25rem !important;
            }
            
            /* Selectbox text - black and bold */
            section[data-testid="stSidebar"] .stSelectbox > div > div > div,
            section[data-testid="stSidebar"] .stSelectbox > div > div > div > div,
            section[data-testid="stSidebar"] .stSelectbox input,
            section[data-testid="stSidebar"] .stSelectbox span,
            section[data-testid="stSidebar"] .stSelectbox p {
                background-color: #ffffff !important;
                color: #000000 !important;
                font-weight: bold !important;
            }
            
            /* Selectbox dropdown options - black text */
            section[data-testid="stSidebar"] [data-baseweb="select"] > div,
            section[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
            section[data-testid="stSidebar"] [data-baseweb="popover"] li,
            section[data-testid="stSidebar"] [data-baseweb="popover"] li > div {
                color: #000000 !important;
                font-weight: bold !important;
            }
            
            /* Code block (model name) - match text input style - black text */
            section[data-testid="stSidebar"] .stCodeBlock,
            section[data-testid="stSidebar"] .stCodeBlock *,
            section[data-testid="stSidebar"] code,
            section[data-testid="stSidebar"] code *,
            section[data-testid="stSidebar"] .stCodeBlock code,
            section[data-testid="stSidebar"] .stCodeBlock span,
            section[data-testid="stSidebar"] .stCodeBlock div,
            section[data-testid="stSidebar"] .stCodeBlock p {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ffffff !important;
                border-radius: 0.25rem !important;
                padding: 0.5rem !important;
                font-weight: bold !important;
            }
            
            section[data-testid="stSidebar"] pre,
            section[data-testid="stSidebar"] pre *,
            section[data-testid="stSidebar"] pre code,
            section[data-testid="stSidebar"] pre span {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ffffff !important;
                border-radius: 0.25rem !important;
                font-weight: bold !important;
            }
            
            /* Main content description text - Gold color */
            .main .stMarkdown,
            .main .stMarkdown p,
            .main .stMarkdown div {
                color: #d4af37 !important;
            }
            
            /* Caption text - Gold color */
            .main .stCaption,
            .main .stCaption p,
            [data-testid="stCaption"],
            [data-testid="stCaption"] p {
                color: #d4af37 !important;
            }
            
            /* All markdown text in main area - Gold */
            [data-testid="stAppViewContainer"] .stMarkdown,
            [data-testid="stAppViewContainer"] .stMarkdown p,
            [data-testid="stAppViewContainer"] .stCaption,
            [data-testid="stAppViewContainer"] .stCaption p {
                color: #d4af37 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    # Initialize chat list
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = ["Current Chat"]
        st.session_state.current_chat_id = "Current Chat"

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = {"Current Chat": []}

    # Initialize settings
    if "chatbot_settings" not in st.session_state:
        st.session_state.chatbot_settings = {
            "config_path": DEFAULT_CONFIG_PATH,
            "model_label": list(AVAILABLE_MODELS.keys())[0],
            "custom_ckpt_path": "",  # Checkpoint path for custom transformer
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 3
        }

    # Sidebar: Chat list and settings
    with st.sidebar:
        st.title("Chat List")

        # New chat button
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            import datetime
            import time
            # Use timestamp to ensure uniqueness
            timestamp = time.time()
            new_chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            st.session_state.chat_sessions.append(new_chat_name)
            st.session_state.chat_messages[new_chat_name] = []
            st.session_state.current_chat_id = new_chat_name
            st.rerun()

        st.markdown("---")

        # Chat list - Use index to ensure key uniqueness
        st.subheader("Chat History")
        if len(st.session_state.chat_sessions) > 0:
            for idx, chat_id in enumerate(st.session_state.chat_sessions):
                # Use index to ensure key uniqueness, even if chat names are the same
                button_key = f"chat_btn_{idx}_{hash(chat_id)}"
                if st.button(
                        chat_id,
                        key=button_key,
                        use_container_width=True,
                        type="primary" if chat_id == st.session_state.current_chat_id else "secondary"
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
        else:
            st.info("No chat history")

        st.markdown("---")

        # Collapsible settings panel - always present
        with st.expander("⚙️ Settings", expanded=False):
            # Step 1: Model selection (first)
            model_label = st.selectbox(
                "Select model",
                options=list(AVAILABLE_MODELS.keys()),
                index=list(AVAILABLE_MODELS.keys()).index(
                    st.session_state.chatbot_settings.get("model_label", list(AVAILABLE_MODELS.keys())[0])
                ) if st.session_state.chatbot_settings.get("model_label") in AVAILABLE_MODELS else 0,
                key="sidebar_model_label"
            )
            st.session_state.chatbot_settings["model_label"] = model_label

            model_cfg = AVAILABLE_MODELS[model_label]
            backend_type = model_cfg["backend"]

            st.markdown("---")

            # Step 2: Model-specific configuration (based on selected model)
            if backend_type == "rag_llama":
                st.subheader("RAG LLama Configuration")
                config_path = st.text_input(
                    "Config path",
                    value=st.session_state.chatbot_settings.get("config_path", DEFAULT_CONFIG_PATH),
                    help="Path to your YAML config (same one used for rag_qa.py)",
                    key="sidebar_config_path"
                )
                st.session_state.chatbot_settings["config_path"] = config_path
                
                st.write("**Model name:**")
                st.code(model_cfg["hf_model"], language="bash")
                
            elif backend_type == "custom_only":
                st.subheader("Custom Transformer Configuration")
                # Checkpoint path input for custom transformer model
                ckpt_path = st.text_input(
                    "Checkpoint path",
                    value=st.session_state.chatbot_settings.get("custom_ckpt_path", ""),
                    help="""Path to checkpoint file. Available checkpoints:
                    - transformer/checkpoints/seq2seq_best.pt
                    - transformer/checkpoints/sft_best.pt
                    - transformer/checkpoints/seq2seq_final.pt
                    - transformer/checkpoints/sft.pt
                    
                    Use relative path from project root or absolute path.""",
                    placeholder="transformer/checkpoints/seq2seq_best.pt",
                    key="sidebar_custom_ckpt"
                )
                # Strip whitespace and save to session state
                ckpt_path = ckpt_path.strip() if ckpt_path else ""
                st.session_state.chatbot_settings["custom_ckpt_path"] = ckpt_path
                if ckpt_path:
                    st.write("**Model name:**")
                    st.code(ckpt_path, language="bash")
                    # Check if file exists
                    if not os.path.isabs(ckpt_path):
                        project_root = os.path.dirname(os.path.dirname(__file__))
                        abs_ckpt_path = os.path.join(project_root, ckpt_path)
                        if os.path.exists(abs_ckpt_path):
                            st.success(f"✓ Checkpoint found: {abs_ckpt_path}")
                        elif not os.path.exists(ckpt_path):
                            st.warning(f"⚠️ File not found. Tried:\n- {ckpt_path}\n- {abs_ckpt_path}")
                    elif os.path.exists(ckpt_path):
                        st.success(f"✓ Checkpoint found: {ckpt_path}")
                    else:
                        st.warning(f"⚠️ File not found: {ckpt_path}")
                else:
                    st.warning("⚠️ Please provide a checkpoint path (e.g., transformer/checkpoints/seq2seq_best.pt)")
            
            st.markdown("---")

            # Step 3: Generation parameters (common for all models)
            st.subheader("Generation Parameters")
            max_new_tokens = st.slider(
                "Max new tokens",
                32, 512,
                st.session_state.chatbot_settings.get("max_new_tokens", 256),
                step=16,
                key="sidebar_max_tokens"
            )
            st.session_state.chatbot_settings["max_new_tokens"] = max_new_tokens

            temperature = st.slider(
                "Temperature",
                0.0, 2.0,
                st.session_state.chatbot_settings.get("temperature", 0.8),
                step=0.05,
                key="sidebar_temperature"
            )
            st.session_state.chatbot_settings["temperature"] = temperature

            top_p = st.slider(
                "Top-p",
                0.1, 1.0,
                st.session_state.chatbot_settings.get("top_p", 0.9),
                step=0.05,
                key="sidebar_top_p"
            )
            st.session_state.chatbot_settings["top_p"] = top_p

            # Top-k slider - different labels for different backends
            if backend_type == "rag_llama":
                top_k = st.slider(
                    "Top-k (retrieval, RAG only)",
                    1, 10,
                    st.session_state.chatbot_settings.get("top_k", 3),
                    step=1,
                    key="sidebar_top_k"
                )
            elif backend_type == "custom_only":
                top_k = st.slider(
                    "Top-k (sampling)",
                    0, 50,
                    st.session_state.chatbot_settings.get("top_k", 0),
                    step=1,
                    key="sidebar_top_k"
                )
            else:
                top_k = st.session_state.chatbot_settings.get("top_k", 3)
            st.session_state.chatbot_settings["top_k"] = top_k

        st.markdown("---")

        # Back to main page button - always present
        if st.button("← Back to Main", use_container_width=True, key="back_to_main_btn"):
            st.session_state.page = "main"
            st.rerun()

    # Main content area
    st.title("Livermore-style Trading Chatbox")
    st.markdown(
        "A Jesse Livermore Style Chatbox for trading questions. "
        "You can switch between different models."
    )

    # Display current chat ID
    st.caption(f"Current Chat: {st.session_state.current_chat_id}")

    # Get current chat messages
    current_messages = st.session_state.chat_messages.get(st.session_state.current_chat_id, [])

    # Get settings from session_state
    config_path = st.session_state.chatbot_settings.get("config_path", DEFAULT_CONFIG_PATH)
    model_label = st.session_state.chatbot_settings.get("model_label", list(AVAILABLE_MODELS.keys())[0])
    model_cfg = AVAILABLE_MODELS[model_label]
    backend_type = model_cfg["backend"]
    max_new_tokens = st.session_state.chatbot_settings.get("max_new_tokens", 256)
    temperature = st.session_state.chatbot_settings.get("temperature", 0.8)
    top_p = st.session_state.chatbot_settings.get("top_p", 0.9)
    top_k = st.session_state.chatbot_settings.get("top_k", 3)

    # load backend
    rag_qa = None
    custom_model = None
    custom_ckpt_path = None

    if backend_type == "rag_llama":
        if not config_path:
            st.error("Please provide a valid config path.")
            return

        try:
            rag_qa = load_ragqa(config_path, model_cfg["hf_model"])
        except ImportError as e:
            st.error(
                f"Unable to import required dependencies. Please ensure all dependencies are installed:\n\n```bash\npip install torch transformers sentence-transformers faiss-cpu pyyaml\n```\n\nError details: {e}")
            return
        except FileNotFoundError as e:
            st.error(
                f"File not found. Please check:\n- Config file: {config_path}\n- Index file: kb_data/trading_index.faiss\n- Docs file: kb_data/trading_docs.pkl\n\nError: {e}")
            return
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = str(e)
            if "gated" in error_msg.lower() or "403" in error_msg or "access" in error_msg.lower():
                st.error(
                    f"❌ **Model Access Required**\n\n{error_msg}\n\n**How to fix:**\n1. Visit https://huggingface.co/{model_cfg['hf_model']} and click 'Agree and access repository'\n2. After approval, ensure you're logged in: `huggingface-cli login`\n3. Restart the application")
            else:
                st.error(
                    f"Failed to load RAG model/index. Check paths and config.\n\nError: {str(e)}\n\nDetails:\n```\n{error_details}\n```")
            return
    elif backend_type == "custom_only":
        # Get checkpoint path from session state (already updated by sidebar)
        custom_ckpt_path = st.session_state.chatbot_settings.get("custom_ckpt_path", "")
        
        # Strip whitespace from path
        if custom_ckpt_path:
            custom_ckpt_path = custom_ckpt_path.strip()
            st.session_state.chatbot_settings["custom_ckpt_path"] = custom_ckpt_path
        
        # Only show error if user tries to use the model but path is empty
        # Don't block the page from loading
        if not custom_ckpt_path:
            st.info("""
            💡 **Please provide a checkpoint path in Settings**
            
            To use the Custom Transformer model, you need to provide the path to a checkpoint file in the Settings panel (⚙️ Settings in the sidebar).
            
            **Available checkpoints:**
            - `transformer/checkpoints/seq2seq_best.pt`
            - `transformer/checkpoints/sft_best.pt`
            - `transformer/checkpoints/seq2seq_final.pt`
            - `transformer/checkpoints/sft.pt`
            
            **Note:** The checkpoint file is the trained model weights, typically saved after training.
            """)
            # Don't return here - allow user to see the chat interface and input path
        else:
            # Check if path exists (handle both relative and absolute paths)
            path_valid = False
            final_ckpt_path = None
            
            if not os.path.isabs(custom_ckpt_path):
                project_root = os.path.dirname(os.path.dirname(__file__))
                abs_ckpt_path = os.path.join(project_root, custom_ckpt_path)
                if os.path.exists(abs_ckpt_path):
                    final_ckpt_path = abs_ckpt_path
                    path_valid = True
                elif os.path.exists(custom_ckpt_path):
                    final_ckpt_path = custom_ckpt_path
                    path_valid = True
            elif os.path.exists(custom_ckpt_path):
                final_ckpt_path = custom_ckpt_path
                path_valid = True
            
            if not path_valid:
                st.error(f"""
                ❌ **Checkpoint file not found**
                
                The checkpoint path you provided could not be found:
                - **Input path:** `{custom_ckpt_path}`
                
                **Please check:**
                1. The path is correct (e.g., `transformer/checkpoints/seq2seq_best.pt`)
                2. The file exists in the project directory
                3. There are no typos in the path
                
                **Available checkpoints:**
                - `transformer/checkpoints/seq2seq_best.pt`
                - `transformer/checkpoints/sft_best.pt`
                - `transformer/checkpoints/seq2seq_final.pt`
                - `transformer/checkpoints/sft.pt`
                """)
                # Don't return - allow user to fix the path
            else:
                # Update the path to the validated absolute path
                custom_ckpt_path = final_ckpt_path
                st.session_state.chatbot_settings["custom_ckpt_path"] = custom_ckpt_path
    else:
        # Custom transformer model loading would go here
        st.warning("Custom transformer model is not yet implemented.")
        return

    # Display current chat history
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # user input
    user_query = st.chat_input("Type your question here...")

    if not user_query:
        return

    # add user message to history
    current_messages.append({"role": "user", "content": user_query})
    st.session_state.chat_messages[st.session_state.current_chat_id] = current_messages

    with st.chat_message("user"):
        st.markdown(user_query)

    # generate answers
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if backend_type == "rag_llama":
                    # Use your existing RAG pipeline
                    answer, docs, scores = rag_qa.answer(
                        user_query,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )

                    st.markdown(answer)

                    # Show retrieved notes in an expander
                    with st.expander("🔍 Retrieved notes"):
                        for i, (d, s) in enumerate(zip(docs, scores)):
                            st.markdown(f"**Note {i + 1}** (score = `{s:.4f}`)")
                            st.write(d[:600] + ("..." if len(d) > 600 else ""))
                            st.markdown("---")

                elif backend_type == "custom_only":
                    # Validate checkpoint path before using
                    if not custom_ckpt_path:
                        st.error("❌ Please provide a checkpoint path in Settings before using the model.")
                        return
                    
                    # Re-validate path (in case it changed)
                    path_valid = False
                    final_ckpt_path = None
                    
                    if not os.path.isabs(custom_ckpt_path):
                        project_root = os.path.dirname(os.path.dirname(__file__))
                        abs_ckpt_path = os.path.join(project_root, custom_ckpt_path)
                        if os.path.exists(abs_ckpt_path):
                            final_ckpt_path = abs_ckpt_path
                            path_valid = True
                        elif os.path.exists(custom_ckpt_path):
                            final_ckpt_path = custom_ckpt_path
                            path_valid = True
                    elif os.path.exists(custom_ckpt_path):
                        final_ckpt_path = custom_ckpt_path
                        path_valid = True
                    
                    if not path_valid:
                        st.error(f"❌ Checkpoint file not found: `{custom_ckpt_path}`. Please check the path in Settings.")
                        return
                    
                    # Use custom transformer (seq2seq) model
                    answer = generate_custom_transformer_response(
                        src=user_query,
                        ckpt_path=final_ckpt_path,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        min_new_tokens=12,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.1
                    )
                    st.markdown(answer)

                else:
                    # Custom transformer: no RAG, just pass user query as prompt
                    answer = custom_model.generate(
                        user_query,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    st.markdown(answer)

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Error during generation: {e}\n\nDetails:\n```\n{error_details}\n```")
                return

    # add bot answer to history
    current_messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_messages[st.session_state.current_chat_id] = current_messages
    st.rerun()


if __name__ == "__main__":
    show_chatbot_page()
