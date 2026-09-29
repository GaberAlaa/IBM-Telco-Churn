import pandas as pd
import streamlit as st
import joblib
from config import CLASSIFICATION_MODEL_PATH

@st.cache_resource
def load_model():
    return joblib.load(CLASSIFICATION_MODEL_PATH)

model = load_model()
st.title("Customer Churn Prediction")
st.markdown("Enter customer details below to calculate real-time **churn risk**.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account & Charges")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    paperless = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=10000.0, value=840.0, step=10.0)
    cltv = st.number_input("CLTV", min_value=0, max_value=10000, value=4000, step=100)

with col2:
    st.subheader("Services")
    phone_service = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_bak = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_supp = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)

st.divider()


if st.button("Calculate Churn Risk", type="primary", use_container_width=True):

    input_data = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Senior Citizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": tenure,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet,
                "Online Security": online_sec,
                "Online Backup": online_bak,
                "Device Protection": device_prot,
                "Tech Support": tech_supp,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_mov,
                "Contract": contract,
                "Paperless Billing": paperless,
                "Payment Method": payment,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
                "CLTV": cltv,
            }
        ]
    )

    # Predict probability
    probabilities = model.predict_proba(input_data)[0]
    p_stay = probabilities[0] * 100
    p_churn = probabilities[1] * 100
    prediction = 1 if p_churn >= 50 else 0

    # Display results
    st.subheader("Prediction Result")
    r1, r2, r3 = st.columns(3)

    with r1:
        if prediction == 1:
            st.error("⚠️ **High Churn Risk**")
        else:
            st.success("✅ **Low Churn Risk**")

    with r2:
        st.metric(label="Churn Probability", value=f"{p_churn:.1f}%")

    with r3:
        st.metric(label="Retention Probability", value=f"{p_stay:.1f}%")

    st.progress(int(p_churn), text=f"Churn Risk Level: {p_churn:.1f}%")