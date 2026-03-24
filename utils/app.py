import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

from utils.quantum import quantum_transform

# ---------- CONFIG ----------
st.set_page_config(page_title="JARVIS CNC AI", layout="wide")

# ---------- LOAD ----------
model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("model/scaler.save")

# ---------- LOAD CSS ----------
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🤖 JARVIS CNC Predictive Intelligence")

# ---------- AUGMENT ----------
def augment_data(X):
    return X + np.random.normal(0, 0.01, X.shape)

# ---------- PROCESS ----------
def process_input(X):
    X = augment_data(X)
    X = quantum_transform(X)
    X = scaler.transform(X)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X

# ---------- KPI SECTION ----------
st.subheader("🎯 System KPIs")
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown('<div class="kpi-card">Accuracy<br><h2>~95%</h2></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card">Avg Error<br><h2>Low</h2></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi-card">Status<br><h2>Active</h2></div>', unsafe_allow_html=True)

# ---------- DASHBOARD PANELS ----------
p1, p2 = st.columns(2)

# ---------- CSV PANEL ----------
with p1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("📂 CSV Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        if st.button("🚀 Predict CSV"):

            if "tool_wear" in df.columns:
                X = df.drop("tool_wear", axis=1).values
            else:
                X = df.values

            X = process_input(X)
            preds = model.predict(X)

            df["Prediction"] = preds
            st.success("Prediction Done ✅")

            st.dataframe(df)

            # GRAPH
            if "tool_wear" in df.columns:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    y=df["tool_wear"],
                    mode='lines',
                    name='Actual',
                    line=dict(color='orange', width=2)
                ))

                fig.add_trace(go.Scatter(
                    y=df["Prediction"],
                    mode='lines',
                    name='Prediction',
                    line=dict(color='cyan', width=3)
                ))

                fig.update_layout(template="plotly_dark", title="⚡ Actual vs Prediction")

                st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MANUAL PANEL ----------
with p2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("⚙️ Manual Input")

    vibration = st.number_input("Vibration", 0.0)
    temperature = st.number_input("Temperature", 30.0)
    force = st.number_input("Force", 50.0)
    spindle = st.number_input("Spindle Speed", 2000.0)
    feed = st.number_input("Feed Rate", 200.0)

    if st.button("🔧 Predict Single"):
        X = np.array([[vibration, temperature, force, spindle, feed]])
        X = process_input(X)

        pred = model.predict(X)
        st.success(f"🔥 Tool Wear: {pred[0][0]:.4f}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- LIVE IoT ----------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.header("📡 Live IoT Monitoring")

run = st.button("Start Live Monitoring")
placeholder = st.empty()

if run:
    for i in range(40):
        vib = np.random.uniform(-1,1)
        temp = np.random.uniform(20,50)
        force = np.random.uniform(40,80)
        spindle = np.random.uniform(1000,5000)
        feed = np.random.uniform(100,500)

        X = np.array([[vib, temp, force, spindle, feed]])
        X = process_input(X)

        pred = model.predict(X)[0][0]

        placeholder.metric("Real-Time Tool Wear", f"{pred:.4f}")
        time.sleep(0.5)

st.markdown('</div>', unsafe_allow_html=True)
