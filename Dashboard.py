
from pathlib import Path  
import streamlit as st


import streamlit as st

st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📡",
    layout="wide",
)

PAGES_DIR = Path("Dashboard/Pages")

pages = [
    st.Page(PAGES_DIR / "Dashboard.py", title="Dashboard", default=True),
    st.Page(PAGES_DIR / "EDA.py", title="EDA"),
    st.Page(PAGES_DIR / "Classification.py", title="Classification"),
    st.Page(PAGES_DIR / "Reggresion.py", title="Regression"),
    st.Page(PAGES_DIR / "predict.py", title="Predict"),
]

st.navigation(pages).run()
