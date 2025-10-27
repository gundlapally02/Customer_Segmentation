import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🎯 Customer Segmentation using K-Means")
st.write("Enter key customer details to predict which cluster they belong to.")

# --- Load Model and Scaler ---
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("kmeans_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model/scaler: {e}")
    st.stop()

# --- Define full feature order used in training ---
feature_order = [
    'Income', 'Age', 'TotalSpend', 'Family_Size', 'Recency',
    'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases',
    'NumWebVisitsMonth', 'Spend_per_Visit', 'Deal_Ratio',
    'CampaignAcceptedCount', 'Education_enc', 'Marital_Status_enc',
    'Customer_Tenure_Days'
]

# --- Manual Input Section (Simplified) ---
st.header("📝 Predict Cluster for Key Inputs")
with st.form("manual_input_form"):
    Income = st.number_input("Income", min_value=0, value=50000)
    Age = st.number_input("Age", min_value=18, value=30)
    TotalSpend = st.number_input("Total Spend", min_value=0, value=1000)
    Family_Size = st.number_input("Family Size", min_value=1, value=3)
    Recency = st.number_input("Recency (days since last purchase)", min_value=0, value=10)
    NumWebPurchases = st.number_input("Num Web Purchases", min_value=0, value=5)
    NumStorePurchases = st.number_input("Num Store Purchases", min_value=0, value=3)
    NumCatalogPurchases = st.number_input("Num Catalog Purchases", min_value=0, value=2)
    Education_enc = st.selectbox("Education (encoded)", [0, 1, 2, 3, 4])
    Marital_Status_enc = st.selectbox("Marital Status (encoded)", [0, 1, 2, 3, 4])

    submitted = st.form_submit_button("Predict Cluster")

    if submitted:
        try:
            # Automatically calculate derived/default features
            Spend_per_Visit = TotalSpend / (NumWebPurchases + NumStorePurchases + NumCatalogPurchases + 1)
            Deal_Ratio = 0  # default since we don't ask NumDealsPurchases
            CampaignAcceptedCount = 0  # default
            NumWebVisitsMonth = 0  # default
            Customer_Tenure_Days = 365  # default

            # Prepare input array in correct order
            input_array = np.array([
                Income, Age, TotalSpend, Family_Size, Recency,
                NumWebPurchases, NumStorePurchases, NumCatalogPurchases,
                NumWebVisitsMonth, Spend_per_Visit, Deal_Ratio,
                CampaignAcceptedCount, Education_enc, Marital_Status_enc,
                Customer_Tenure_Days
            ]).reshape(1, -1)

            # Scale and predict
            X_scaled = scaler.transform(input_array)
            cluster_pred = model.predict(X_scaled)[0]

            st.success(f"🎯 The input belongs to **Cluster {cluster_pred}**")

        except Exception as e:
            st.error(f"Error predicting cluster: {e}")
