# pages/main_page.py
import streamlit as st


def show_main_page():
    """Display main page"""
    st.markdown("""
        <style>
            /* Hide sidebar on main page */
            section[data-testid="stSidebar"] {
                display: none !important;
            }

            /* Remove left padding/margin */
            .main .block-container {
                background-color: rgba(26, 35, 50, 0.8);
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                max-width: 100% !important;
            }

            /* Navy Blue Background */
            .stApp {
                background: #1a2332 !important;
                color: #ffffff;
            }

            h1, h2, h3 {
                color: #d4af37 !important;
            }

            p, div, span {
                color: #f5f5f5;
            }

            .stButton > button[type="primary"] {
                background-color: #d4af37;
                color: #1a2332;
                border: none;
                font-weight: bold;
            }

            .stButton > button[type="primary"]:hover {
                background-color: #f4d03f;
                box-shadow: 0 4px 8px rgba(212, 175, 55, 0.3);
            }

            .stButton > button[type="secondary"] {
                background-color: #1e3a5f;
                color: #d4af37;
                border: 2px solid #d4af37;
            }

            .stButton > button[type="secondary"]:hover {
                background-color: #2a4f7a;
                border-color: #f4d03f;
            }

            hr {
                border-color: #d4af37;
                opacity: 0.3;
            }
        </style>
        """, unsafe_allow_html=True)
    st.title("Livermore Trading System")
    st.markdown("---")
    st.markdown("### Welcome to Livermore Trading System")
    st.markdown("Please select the feature you want to use:")
    st.markdown("")
    col1_text, col2_text = st.columns(2, gap="large")

    with col1_text:
        st.subheader("AI Chatbot Assistant")
        st.write(
            "Chat with an AI assistant based on Livermore trading strategies to get trading advice and strategy analysis.")

    with col2_text:
        st.subheader("Stock Selection & Analysis")
        st.write("Select stocks and view technical analysis and trading signals based on Livermore strategies.")

    st.write("")
    st.write("")

    col1_btn, col2_btn = st.columns(2, gap="large")

    with col1_btn:
        if st.button("Enter Chat Interface", use_container_width=True, type="primary"):
            st.session_state.page = "chatbot"
            st.rerun()

    with col2_btn:
        if st.button("Enter Stock Selection", use_container_width=True, type="primary"):
            st.session_state.page = "stock_selection"
            st.rerun()

