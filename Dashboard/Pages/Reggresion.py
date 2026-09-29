import pandas as pd
import streamlit as st
from config import REGRESSION_RESULTS_PATH 

@st.cache_data
def load():
    res = pd.read_csv(REGRESSION_RESULTS_PATH)
    return res 


res = load()

st.title("📈 Regression: predicting CLTV")
st.markdown(
    "Goal: predict a customer's **Customer Lifetime Value (CLTV)**."
    "I compared **6 regression models**."
)


st.header("1. Model results")
styled = (
    res.style.format({"RMSE": "{:,.1f}", "MAE": "{:,.1f}", "R2": "{:.3f}"})
    .highlight_max(subset=["R2"], color="#c8e6c9")
    .highlight_min(subset=["RMSE", "MAE"], color="#c8e6c9")
)
st.dataframe(styled,hide_index=True)



st.header("2. How we choose: what the metrics mean")
st.markdown(
    f"""
- **RMSE**: typical prediction error in CLTV units, and it **penalises large errors** more.
- **MAE**: the average absolute error, easy to read as "off by about X on average".
- **R2**: the share of CLTV variation the model explains (1 = perfect, 0 = no better than always guessing the mean).

We rank primarily on **R2 and RMSE**, and use MAE as a sanity check. All three metrics agree on the best two models; the linear models are nearly tied.
"""
)

st.header("3. Chosen model")
st.success(
    "### Gradient Boosting Regressor"

)

second = res.iloc[1]
st.markdown(
f"""
**Why Gradient Boosting Regressor ?** \n
1. It is **best on every metric**:  R2, RMSE and MAE.\n
"""
)


