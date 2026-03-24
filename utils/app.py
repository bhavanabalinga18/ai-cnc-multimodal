import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from utils.quantum import quantum_transform

# Load model & scaler
model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("model/scaler.save")

st.title("🚀 CNC Predictive Maintenance System")

# ---------- AUGMENTATION ----------
def augment_data(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise

# ---------- PROCESS ----------
def process_input(X):
    X = augment_data(X)        # 🔥 augmentation added
    X = quantum_transform(X)
    X = scaler.transform(X)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X

# ---------- CSV UPLOAD ----------
st.header("📂 Upload CSV")
file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Predict CSV"):

        # Remove target column if present
        if "tool_wear" in df.columns:
            X = df.drop("tool_wear", axis=1).values
        else:
            X = df.values

        X = process_input(X)

        preds = model.predict(X)

        df["Prediction"] = preds

        st.success("Prediction Complete ✅")
        st.dataframe(df)

# ---------- MANUAL INPUT ----------
st.header("⚙️ Manual Input")

vibration = st.number_input("Vibration", value=0.0)
temperature = st.number_input("Temperature", value=30.0)
force = st.number_input("Force", value=50.0)
spindle = st.number_input("Spindle Speed", value=2000.0)
feed = st.number_input("Feed Rate", value=200.0)

if st.button("Predict Single"):

    X = np.array([[vibration, temperature, force, spindle, feed]])
    X = process_input(X)

    pred = model.predict(X)

    st.success(f"Tool Wear Prediction: {pred[0][0]:.4f}")
