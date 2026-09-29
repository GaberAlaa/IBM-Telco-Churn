import pandas as pd
import streamlit as st
from config import CLEAN_DATA_PATH , DASHBOARD_Pics_PATH 



@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_DATA_PATH)


def show(name: str, caption: str | None = None):
    st.image(str(DASHBOARD_Pics_PATH / name), caption=caption, width="stretch")


df = load_data()
churn_rate = df["Churn Value"].mean()

st.title("🔍 Exploratory Data Analysis")
st.markdown(
    f"The dataset has **{len(df):,} customers** and **{df.shape[1]} columns** "
    "(customer profile, services, billing, churn and CLTV). It has no missing values. "
    "Each section below shows a chart and what we learned from it."
)

# ------------------------------------------------------------------ 1
st.header("1. Customer profile and services")
show("graph_1.png")
st.markdown(
    """
**Insights**
- **Gender is balanced** (roughly 50/50), so it is unlikely to drive churn.
- **Most customers are not senior citizens** and most have **no dependents**. Seniors are a small but distinct group.
- **Phone service is almost universal.** Internet is split across **Fiber optic (largest)**, DSL and no internet.
- **Add-on services are under-used.** Online Security, Tech Support, Device Protection and Online Backup all have more "No" than "Yes".
- **Contracts are dominated by Month-to-month.** Two-year and one-year contracts are much smaller.
- **Electronic check is the most common payment method**, and most customers use paperless billing.
- The last chart already shows that churn is a **minority class**.
"""
)

st.divider()

# ------------------------------------------------------------------ 2
st.header("2. Target: churn balance")
c1, c2 = st.columns([1, 1.2])
with c1:
    show("graph_4.png")
with c2:
    st.markdown(
        f"""
**Insights**
- About **{churn_rate:.1%}** of customers churned, roughly **1 in 4**.
- The classes are **imbalanced** (about 73.5% stay vs 26.5% leave). A model that always predicts "stay" would be ~73% accurate and still useless.
- Because of this, we judge classification models with **Recall, F1 and ROC-AUC** and not accuracy alone (see the Classification page).
"""
    )

st.divider()

# ------------------------------------------------------------------ 3
st.header("3. Numeric features: distributions")
show("graph_2.png")
st.markdown(
    """
**Insights**
- **Tenure Months** is U-shaped: many brand-new customers (0-5 months) and a second peak of very loyal customers (~70 months).
- **Monthly Charges** has two groups: a large low-price cluster (~$20, phone-only customers) and a broad hump around $70-$100 (internet customers).
- **Total Charges** is heavily **right-skewed**, because it grows with tenure. Most customers have not paid much in total yet.
- **Churn Value** is binary and imbalanced (see section 2).
- **CLTV** ranges from about 2,000 to 6,500 with a fairly flat spread and a bulge around 4,000-6,000.
"""
)

st.subheader("Outlier check")
show("graph_3.png")
st.markdown(
    """
**Insights**
- The boxplots show **no extreme outliers**; the whiskers cover the full range in every numeric column, so no rows need to be removed.
- The **median tenure is about 29 months**, the **median monthly charge about $70**, and the **median CLTV about 4,500**.
- Total Charges has a median (~1,400) far below its maximum, confirming the right skew.
"""
)

st.divider()

# ------------------------------------------------------------------ 4
st.header("4. What drives churn?")

st.subheader("Contract type")
c1, c2 = st.columns([1, 1.2])
with c1:
    show("graph_5.png")
with c2:
    st.markdown(
        """
**Insights**
- **Month-to-month customers churn at ~43%**, versus **~11%** on one-year and **~3%** on two-year contracts.
- Contract length is the **strongest single churn signal** in the data. Customers with no commitment can leave at any time.
- **Business action:** offer incentives to move month-to-month customers onto longer contracts.
"""
    )

st.subheader("Monthly charges")
c1, c2 = st.columns([1, 1.2])
with c1:
    show("graph_6.png")
with c2:
    st.markdown(
        """
**Insights**
- **Churners pay more.** Their median monthly charge is about **$80**, against about **$64** for customers who stay.
- Customers who leave rarely sit in the cheap (~$20) range; they are mostly on higher-priced internet plans.
- **Business action:** review pricing and value for high-bill customers, especially fiber.
"""
    )

st.subheader("Payment method")
c1, c2 = st.columns([1.2, 1])
with c1:
    show("graph_7.png")
with c2:
    st.markdown(
        """
**Insights**
- **Electronic check users churn at ~45%**, about **3x the rate** of customers on automatic payments (~15-17%) and well above mailed check (~19%).
- Automatic payment (card or bank transfer) is linked with **higher retention**, probably because it removes the friction of paying each month.
- **Business action:** encourage customers to switch to automatic payment.
"""
    )
    st.caption("Note: the title on this chart says 'Contract Type', but the x-axis (and data) is Payment Method.")

st.subheader("Reasons customers give for leaving")
c1, c2 = st.columns([1.2, 1])
with c1:
    show("graph_8.png")
with c2:
    st.markdown(
        """
**Insights**
- The top reasons are about **support and competitors**, not price. The main ones are *attitude of support person*, and competitors offering *higher download speeds*, *more data* or *a better offer*.
- **Price too high** is much further down the list.
- Many churn reasons are therefore **service quality and competitive offers**, which the company can act on.
- **Business action:** improve support training and compare speed, data and device offers with competitors.
"""
    )

