"""Streamlit UI for Customer Churn Prediction."""
import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.markdown("Predict whether a customer will churn based on their profile.")

st.divider()

# Customer Info Section
st.subheader("Customer Info")
col1, col2 = st.columns(2)
with col1:
    customer_id = st.number_input("Customer ID", min_value=1, value=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)

# Usage Info Section
st.subheader("Usage Info")
col3, col4 = st.columns(2)
with col3:
    usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=50, value=10)
    support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2)
with col4:
    payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=60, value=5)
    last_interaction = st.number_input("Last Interaction (days)", min_value=0, max_value=60, value=10)

# Subscription Info Section
st.subheader("Subscription Info")
col5, col6 = st.columns(2)
with col5:
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
with col6:
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0)

st.divider()

# Predict Button
if st.button("Predict Churn", type="primary", use_container_width=True):
    # Prepare request payload
    payload = {
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage_Frequency": usage_frequency,
        "Support_Calls": support_calls,
        "Payment_Delay": payment_delay,
        "Subscription_Type": subscription_type,
        "Contract_Length": contract_length,
        "Total_Spend": total_spend,
        "Last_Interaction": last_interaction
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            churn = result["churn"]
            probability = result["churn_probability"]

            st.divider()

            if churn:
                st.error(f"**Prediction: Will Churn**")
            else:
                st.success(f"**Prediction: Will Not Churn**")

            st.metric(label="Churn Probability", value=f"{probability:.1%}")

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the server is running at localhost:8000")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.caption("Ensure the FastAPI server is running: `python src/serve.py`")
