# streamlit_app.py
import streamlit as st

# set_page_config must be called first, before any other Streamlit commands
# This MUST be the very first Streamlit command in the entire script
st.set_page_config(
    page_title="Livermore Trading System",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Hide the header and page navigation in sidebar
# Global styling: Navy Blue background and Gold buttons
st.markdown("""
    <style>
        [data-testid="stHeader"] {
            display: none;
        }
        /* Hide Streamlit's automatic page navigation in sidebar */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        /* Hide navigation links in sidebar - target only nav elements */
        nav[data-testid="stSidebarNav"],
        div[data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] nav {
            display: none !important;
        }
        .main .block-container {
            padding-top: 2rem;
        }

        /* Global Navy Blue Background */
        .stApp {
            background: #1a2332 !important;
        }

        /* All buttons - Gold color */
        .stButton > button {
            background-color: #d4af37 !important;
            color: #1a2332 !important;
            border: none !important;
            font-weight: bold !important;
        }

        .stButton > button:hover {
            background-color: #f4d03f !important;
            box-shadow: 0 4px 8px rgba(212, 175, 55, 0.3) !important;
        }

        /* Primary buttons - Gold */
        .stButton > button[type="primary"] {
            background-color: #d4af37 !important;
            color: #1a2332 !important;
            border: none !important;
            font-weight: bold !important;
        }

        .stButton > button[type="primary"]:hover {
            background-color: #f4d03f !important;
            box-shadow: 0 4px 8px rgba(212, 175, 55, 0.3) !important;
        }

        /* Secondary buttons - Gold with border */
        .stButton > button[type="secondary"] {
            background-color: #d4af37 !important;
            color: #1a2332 !important;
            border: 2px solid #d4af37 !important;
            font-weight: bold !important;
        }

        .stButton > button[type="secondary"]:hover {
            background-color: #f4d03f !important;
            border-color: #f4d03f !important;
            box-shadow: 0 4px 8px rgba(212, 175, 55, 0.3) !important;
        }

        /* Sidebar styling - Navy Blue background */
        section[data-testid="stSidebar"] {
            background-color: #1a2332 !important;
        }

        /* Sidebar text colors */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6 {
            color: #d4af37 !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #f5f5f5 !important;
        }

        /* Sidebar separators */
        section[data-testid="stSidebar"] hr {
            border-color: #d4af37 !important;
            opacity: 0.3;
        }

        /* Sidebar expander */
        section[data-testid="stSidebar"] .streamlit-expanderHeader {
            color: #d4af37 !important;
        }

        section[data-testid="stSidebar"] .streamlit-expanderContent {
            color: #f5f5f5 !important;
        }

        /* Sidebar input fields */
        section[data-testid="stSidebar"] .stTextInput > div > div > input,
        section[data-testid="stSidebar"] .stSelectbox > div > div > select {
            background-color: rgba(26, 35, 50, 0.8) !important;
            color: #f5f5f5 !important;
            border-color: #d4af37 !important;
        }

        /* Sidebar sliders */
        section[data-testid="stSidebar"] .stSlider {
            color: #d4af37 !important;
        }

        section[data-testid="stSidebar"] .stSlider > div > div > div {
            background-color: #d4af37 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    # Import page functions inside main() to ensure set_page_config is called first
    # This prevents any Streamlit decorators in imported modules from executing before set_page_config
    from pages.main_page import show_main_page
    from pages.chatbot_page import show_chatbot_page
    from pages.stock_selection_page import show_stock_selection_page

    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "main"

    # Display different pages based on page state
    if st.session_state.page == "main":
        show_main_page()
    elif st.session_state.page == "chatbot":
        show_chatbot_page()
    elif st.session_state.page == "stock_selection":
        show_stock_selection_page()


if __name__ == "__main__":
    main()
