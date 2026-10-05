"""CSS carried over unchanged from the pre-refactor streamlit_app.py and pages/chatbot_page.py."""

GLOBAL_CSS = """
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
    """

CHAT_CSS = """
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
        """
