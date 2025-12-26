import streamlit as st
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="Industrial AI Health System")

st.title("🏭 Industrial Asset Health Prediction")

# Load model safely
MODEL_PATH = Path("model.pkl")

if not MODEL_PATH.exists():
    st.error("model.pkl not found in repo")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Inputs
vibration = st.number_input("Vibration", 0.0, 10.0, 2.5)
temperature = st.number_input("Temperature", 20.0, 100.0, 45.0)
current = st.number_input("Current", 0.0, 30.0, 10.0)
frequency = st.number_input("Frequency", 30.0, 70.0, 50.0)
quality = st.number_input("Quality Index", 0.0, 1.0, 0.9)

if st.button("Predict"):
    vib_temp_ratio = vibration / (temperature + 1)
    curr_quality = current * quality

    X = np.array([[vibration, temperature, current, frequency, quality,
                   vib_temp_ratio, curr_quality]])

    pred = model.predict(X)[0]

    if pred == 0:
        st.success("🟢 HEALTHY")
    elif pred == 1:
        st.warning("🟠 WARNING")
    else:
        st.error("🔴 CRITICAL")
