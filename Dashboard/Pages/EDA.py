from pathlib import Path
import pandas as pd
import streamlit as st
from config import CLEAN_DATA_PATH, DASHBOARD_Pics_PATH
 
 
@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_DATA_PATH)
 
 
def show_image(name: str):
    st.image(str(Path(DASHBOARD_Pics_PATH) / name), width="stretch")
 
 
df = load_data()
churn_rate = df["Churn Value"].mean()
 
st.title(" Exploratory Data Analysis")

 
st.header("1. Customer profile and services")
show_image("graph_1.png")
st.markdown("Gender is balanced. Most customers are non-senior with no dependents. "
           "Add-on services (Security, Backup, Tech Support) are mostly unused. "
           "Month-to-month is the dominant contract type.")
 
st.divider()
 
st.header("2. Churn balance")
c1, c2 = st.columns([1, 1.2])
with c1:
    show_image("graph_4.png")
with c2:
    st.markdown(f"**{churn_rate:.1%} churn rate** - imbalanced, so we score models "
                "on Recall/F1/ROC-AUC, not accuracy.")
 
st.divider()
 
st.header("3. Numeric features")
show_image("graph_2.png")
st.markdown("Tenure is U-shaped (many new + many loyal customers). Monthly Charges "
           "has a low phone-only cluster and a high internet cluster. Total Charges "
           "is right-skewed (grows with tenure).")
show_image("graph_3.png")
st.markdown("No extreme outliers. Median tenure ~29 months, monthly charge ~$70, CLTV ~4,500.")
 
st.divider()
 
st.header("4. What drives churn?")
 
c1, c2 = st.columns([1, 1.2])
with c1:
    show_image("graph_5.png")
with c2:
    st.markdown("Month-to-month churns at **~43%** vs **~3%** on two-year contracts — "
                "the strongest single churn signal.\n\n**Action:** push longer contracts.")
 
c1, c2 = st.columns([1, 1.2])
with c1:
    show_image("graph_6.png")
with c2:
    st.markdown("Churners on average pay more than non-churners")
 
c1, c2 = st.columns([1.2, 1])
with c1:
    show_image("graph_7.png")
with c2:
    st.markdown("Electronic check churns at **~45%**, ~3x automatic payment (~15-17%).\n\n"
                "**Action:** push customers toward autopay.")
 
c1, c2 = st.columns([1.2, 1])
with c1:
    show_image("graph_8.png")
with c2:
    st.markdown("Top reasons: support attitude and competitor offers (speed/data/price), "
                "not price alone.\n\n**Action:** improve support, benchmark competitors.")
