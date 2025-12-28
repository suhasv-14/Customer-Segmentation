import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/mall_customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save model AND scaler
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved successfully")
