import pandas as pd
import streamlit as st
from config import CLASSIFICATION_RESULTS_PATH 


@st.cache_data
def load():
    res = pd.read_csv(CLASSIFICATION_RESULTS_PATH)
    return res


res = load()




st.title("Classification: predicting churn")
st.markdown("""
    Goal: predict whether a customer will **churn (1)** or **stay (0)**. \n
    We trained and compared **7 classifiers** on the same data and evaluated them.
    """
)

# ------------------------------------------------------------------ Results
st.header("1. Model results")

styled = (
    res.style.format({"Accuracy": "{:1f}", "Precision": "{:1f}", "Recall": "{:3f}","F1":"{:1f}","ROC_AUC":"{:1f}"})
    .highlight_max(subset=["Accuracy","Precision","Recall","F1","ROC_AUC"], color="#c8e6c9")
)
st.dataframe(styled,hide_index=True)
# ------------------------------------------------------------------ Why
st.header("2. How we choose: which metric matters?")
st.markdown(
    f"""
Only **~25** of customers churn, so the data is **imbalanced**. This changes how we read the metrics:

- **Accuracy is misleading.** A model that predicts "stays" for everyone would already score about **~75** and catch zero churners.
- **Recall** = the number of the customers who really churn, how many do we catch? A missed churner is a lost customer and lost revenue, so this is the metric the business cares about most.
- **Precision** = the number of the customers we flag, how many really churn? Low precision means retention offers are wasted on people who would have stayed.
- **F1** balances Precision and Recall in a single number.
"""
)

st.header("3. Chosen model")
st.success(f"### Random Forest")

st.markdown(
f"""
**Why Random forest ?**
It has the **highest F1**  the best balance between catching churners and avoiding false alarms.\n
Its **recall is ~80**: it catches about **80 of every 100 churners**.
"""
)