import streamlit as st
import numpy as np

st.set_page_config(page_title="Industrial ML Health System")

st.title("🏭 Industrial Asset Health Prediction")
st.write("ML model (KNN) deployed on Streamlit Cloud")

# -----------------------------
# Training data (synthetic)
# -----------------------------
np.random.seed(42)
n = 500

vibration = np.random.uniform(0.5, 8.0, n)
temperature = np.random.uniform(30, 90, n)
current = np.random.uniform(5, 25, n)
frequency = np.random.uniform(35, 60, n)
quality = np.random.uniform(0.4, 1.0, n)

X_train = np.column_stack(
    (vibration, temperature, current, frequency, quality)
)

y_train = np.where(
    (vibration < 3) & (temperature < 50) & (quality > 0.85), 0,
    np.where((vibration < 6) & (temperature < 70), 1, 2)
)

# -----------------------------
# KNN from scratch (ML)
# -----------------------------
def knn_predict(X, y, x, k=5):
    distances = np.linalg.norm(X - x, axis=1)
    nearest = distances.argsort()[:k]
    labels = y[nearest]
    return np.bincount(labels).argmax()

# -----------------------------
# Inputs
# -----------------------------
v = st.number_input("Vibration", 0.0, 10.0, 2.5)
t = st.number_input("Temperature (°C)", 20.0, 100.0, 45.0)
c = st.number_input("Current / Load", 0.0, 30.0, 10.0)
f = st.number_input("Frequency", 30.0, 70.0, 50.0)
q = st.number_input("Quality Index", 0.0, 1.0, 0.9)

if st.button("Predict Health"):
    x = np.array([v, t, c, f, q])
    pred = knn_predict(X_train, y_train, x)

    if pred == 0:
        st.success("🟢 HEALTHY")
    elif pred == 1:
        st.warning("🟠 WARNING")
    else:
        st.error("🔴 CRITICAL")
