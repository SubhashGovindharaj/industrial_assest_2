import streamlit as st
import numpy as np

st.set_page_config(page_title="Industrial ML Health System")

st.title("🏭 Industrial Asset Health Prediction (ML Deployed)")

st.write("KNN-based Machine Learning model deployed on Streamlit Cloud")

# ---------------------------------------
# 1. Create synthetic training dataset
# ---------------------------------------
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

# Labels (supervised learning)
y_train = np.where(
    (vibration < 3) & (temperature < 50) & (quality > 0.85), 0,
    np.where((vibration < 6) & (temperature < 70), 1, 2)
)

# ---------------------------------------
# 2. KNN implementation (ML logic)
# ---------------------------------------
def knn_predict(X_train, y_train, x_test, k=5):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    nearest_idx = distances.argsort()[:k]
    nearest_labels = y_train[nearest_idx]
    return np.bincount(nearest_labels).argmax()

# ---------------------------------------
# 3. Streamlit Inputs
# ---------------------------------------
v = st.number_input("Vibration", 0.0, 10.0, 2.5)
t = st.number_input("Temperature (°C)", 20.0, 100.0, 45.0)
c = st.number_input("Current / Load", 0.0, 30.0, 10.0)
f = st.number_input("Frequency", 30.0, 70.0, 50.0)
q = st.number_input("Quality Index", 0.0, 1.0, 0.9)

if st.button("Predict Health"):
    x_test = np.array([v, t, c, f, q])
    pred = knn_predict(X_train, y_train, x_test)

    if pred == 0:
        st.success("🟢 HEALTHY CONDITION")
    elif pred == 1:
        st.warning("🟠 WARNING CONDITION")
    else:
        st.error("🔴 CRITICAL CONDITION")
