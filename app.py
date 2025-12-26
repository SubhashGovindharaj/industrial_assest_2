import streamlit as st
import numpy as np
import pandas as pd
from faker import Faker
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Industrial AI Health System")

st.title("Industrial Asset Health Prediction")

# -----------------------------------
# 1. Generate synthetic dataset
# -----------------------------------
fake = Faker()
records = []

for _ in range(800):
    vibration = round(random.uniform(0.5, 8.0), 2)
    temperature = round(random.uniform(30, 90), 1)
    current = round(random.uniform(5, 25), 2)
    frequency = round(random.uniform(35, 60), 2)
    quality = round(random.uniform(0.4, 1.0), 2)

    if vibration < 3 and temperature < 50 and quality > 0.85:
        health = 0      # Healthy
    elif vibration < 6 and temperature < 70:
        health = 1      # Warning
    else:
        health = 2      # Critical

    records.append([vibration, temperature, current,
                    frequency, quality, health])

df = pd.DataFrame(
    records,
    columns=[
        "vibration",
        "temperature",
        "current",
        "frequency",
        "quality",
        "health"
    ]
)

# -----------------------------------
# 2. Feature Engineering
# -----------------------------------
df["vib_temp_ratio"] = df["vibration"] / (df["temperature"] + 1)
df["current_quality"] = df["current"] * df["quality"]

X = df.drop("health", axis=1)
y = df["health"]

# -----------------------------------
# 3. Train model (runtime)
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# 4. Streamlit Inputs
# -----------------------------------
vibration = st.number_input("Vibration", 0.0, 10.0, 2.5)
temperature = st.number_input("Temperature (°C)", 20.0, 100.0, 45.0)
current = st.number_input("Current / Load", 0.0, 30.0, 10.0)
frequency = st.number_input("Frequency", 30.0, 70.0, 50.0)
quality = st.number_input("Quality Index", 0.0, 1.0, 0.9)

if st.button("Predict Health"):
    vib_temp_ratio = vibration / (temperature + 1)
    current_quality = current * quality

    X_input = np.array([[vibration, temperature, current,
                          frequency, quality,
                          vib_temp_ratio, current_quality]])

    pred = model.predict(X_input)[0]

    if pred == 0:
        st.success("HEALTHY CONDITION")
    elif pred == 1:
        st.warning("WARNING CONDITION")
    else:
        st.error("CRITICAL CONDITION")
