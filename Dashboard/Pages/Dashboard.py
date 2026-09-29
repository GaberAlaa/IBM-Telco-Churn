from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from config import CLEAN_DATA_PATH

@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_DATA_PATH)
    df["Churn"] = df["Churn Value"].map({1: "Churned", 0: "Retained"})
    return df

df = load_data()

contract = st.sidebar.multiselect("Contract", df["Contract"].unique(), default=df["Contract"].unique())
payment = st.sidebar.multiselect("Payment Method", df["Payment Method"].unique(), default=df["Payment Method"].unique())


f = df[
    df["Contract"].isin(contract)
    & df["Payment Method"].isin(payment)
]

st.title("Telco Customer Churn Dashboard")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", len(f))
k2.metric("Churn Rate", f"{f['Churn Value'].mean():.1%}")
k3.metric("Avg Monthly Charges", f"${f['Monthly Charges'].mean():.2f}")
k4.metric("Avg CLTV", f"${f['CLTV'].mean():.0f}")
k5.metric("Total CLTV", f"${f['CLTV'].sum()}")

c1, c2, c3 = st.columns(3)
with c1:
    counts = f["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Customers"]
    st.plotly_chart(px.pie(counts, names="Churn", values="Customers", title="Churn split"))

with c2:
    g = f.groupby("Contract", observed=True)["Churn Value"].mean().reset_index()
    st.plotly_chart(px.bar(g, x="Contract", y="Churn Value", title="Churn rate by contract"))

with c3:
    g = f.groupby("Internet Service", observed=True)["Churn Value"].mean().reset_index()
    st.plotly_chart(px.bar(g, x="Internet Service", y="Churn Value", title="Churn rate by internet service"))


c1, c2 = st.columns(2)
with c1:
    g = f.groupby("Payment Method", observed=True)["Churn Value"].mean().reset_index()
    st.plotly_chart(px.bar(g, x="Churn Value", y="Payment Method", orientation="h", title="Churn rate by payment method"))
    
with c2:
    services = ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
    rows = []
    for s in services:
        sub = f[f[s].isin(["Yes", "No"])]
        for val in ["Yes", "No"]:
            part = sub[sub[s] == val]
            if len(part):
                rows.append({"Service": s, "Has service": val, "Churn Rate": part["Churn Value"].mean()})
    st.plotly_chart(px.bar(pd.DataFrame(rows), x="Service", y="Churn Rate", color="Has service", barmode="group", title="Churn rate by add-on"))

with st.expander("Filtered data"):
    st.dataframe(f)