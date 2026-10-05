"""Livermore breakout-strategy backtest (example; not part of the livermore package).

    pip install -e ".[app,examples]"
    streamlit run examples/backtest/app.py
"""
import os
import sys

import streamlit as st

st.set_page_config(page_title="Livermore Backtest", layout="centered", initial_sidebar_state="expanded")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps"))
from _style import GLOBAL_CSS  # noqa: E402
from page import show_stock_selection_page  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
show_stock_selection_page()
