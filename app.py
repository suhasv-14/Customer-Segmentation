import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation System")

# Load model & scaler
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaled_features = scaler.transform(features)

    df['Cluster'] = model.predict(scaled_features)

    st.subheader("Segmented Customers")
    st.dataframe(df)

    st.subheader("Visualization")
    st.scatter_chart(
        df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Cluster'
    )
