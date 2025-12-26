import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Industrial Health System",
    page_icon="🏭",
    layout="wide"
)

# Title
st.title("🏭 Industrial Asset Health Prediction System")

# Generate synthetic data
@st.cache_data
def generate_training_data():
    random.seed(42)
    np.random.seed(42)
    
    data = []
    for _ in range(1000):
        vibration = round(random.uniform(0.5, 8.0), 2)
        temperature = round(random.uniform(30, 90), 1)
        current = round(random.uniform(5, 25), 2)
        frequency = round(random.uniform(35, 60), 2)
        quality = round(random.uniform(0.4, 1.0), 2)
        
        # Health rules
        if vibration < 3 and temperature < 50 and quality > 0.85:
            health = 0  # Healthy
        elif vibration < 6 and temperature < 70:
            health = 1  # Warning
        else:
            health = 2  # Critical
        
        data.append([vibration, temperature, current, frequency, quality, health])
    
    df = pd.DataFrame(data, columns=['vibration', 'temperature', 'current', 'frequency', 'quality_index', 'health'])
    
    # Feature engineering
    df['vib_temp_ratio'] = df['vibration'] / (df['temperature'] + 1)
    df['current_quality'] = df['current'] * df['quality_index']
    
    return df

# Train model
@st.cache_resource
def train_model():
    df = generate_training_data()
    
    X = df.drop('health', axis=1)
    y = df['health']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Load model
with st.spinner("Loading model..."):
    model, accuracy = train_model()

st.sidebar.success(f"✅ Model Loaded (Accuracy: {accuracy:.1%})")

# Sidebar inputs
st.sidebar.header("⚙️ Sensor Parameters")

preset = st.sidebar.selectbox(
    "Quick Preset",
    ["Custom", "Healthy", "Warning", "Critical"]
)

if preset == "Healthy":
    v, t, c, f, q = 2.0, 45, 12.0, 50.0, 0.95
elif preset == "Warning":
    v, t, c, f, q = 5.0, 65, 18.0, 48.0, 0.70
elif preset == "Critical":
    v, t, c, f, q = 7.5, 85, 23.0, 38.0, 0.45
else:
    v, t, c, f, q = 3.0, 60, 15.0, 50.0, 0.85

vibration = st.sidebar.slider("🔊 Vibration (mm/s)", 0.5, 8.0, v, 0.1)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", 30, 90, t, 1)
current = st.sidebar.slider("⚡ Current (A)", 5.0, 25.0, c, 0.5)
frequency = st.sidebar.slider("📊 Frequency (Hz)", 35.0, 60.0, f, 0.5)
quality = st.sidebar.slider("⭐ Quality Index", 0.4, 1.0, q, 0.01)

# Predict button
if st.sidebar.button("🔍 PREDICT", use_container_width=True):
    # Feature engineering
    vib_temp_ratio = vibration / (temperature + 1)
    current_quality = current * quality
    
    # Create input
    input_df = pd.DataFrame([[
        vibration, temperature, current, frequency, quality,
        vib_temp_ratio, current_quality
    ]], columns=['vibration', 'temperature', 'current', 'frequency', 'quality_index', 'vib_temp_ratio', 'current_quality'])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    # Display results
    st.subheader("📊 Prediction Results")
    
    status_map = {
        0: ("✅ HEALTHY", "green", "Normal operation"),
        1: ("⚠️ WARNING", "orange", "Requires attention"),
        2: ("🚨 CRITICAL", "red", "Immediate action needed")
    }
    
    status, color, message = status_map[prediction]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### :{color}[{status}]")
        st.info(message)
        
        # Confidence chart
        chart_data = pd.DataFrame({
            'Status': ['Healthy', 'Warning', 'Critical'],
            'Confidence': probabilities * 100
        })
        st.bar_chart(chart_data.set_index('Status'))
    
    with col2:
        st.metric("Healthy", f"{probabilities[0]:.1%}")
        st.metric("Warning", f"{probabilities[1]:.1%}")
        st.metric("Critical", f"{probabilities[2]:.1%}")
    
    # Input summary
    st.subheader("📋 Input Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vibration", f"{vibration} mm/s")
        st.metric("Temperature", f"{temperature}°C")
    with col2:
        st.metric("Current", f"{current} A")
        st.metric("Frequency", f"{frequency} Hz")
    with col3:
        st.metric("Quality", f"{quality:.2f}")

# Info section
with st.expander("ℹ️ About"):
    st.markdown("""
    ### Industrial Asset Health Monitoring
    
    This system uses machine learning to predict equipment health status based on sensor readings.
    
    **Input Parameters:**
    - Vibration: Mechanical oscillations (mm/s)
    - Temperature: Operating temperature (°C)
    - Current: Electrical current (A)
    - Frequency: Operating frequency (Hz)
    - Quality Index: Performance metric (0-1)
    
    **Output:**
    - Healthy: Normal operation
    - Warning: Needs attention
    - Critical: Immediate action required
    
    **Model:** Random Forest with 100 trees
    """)

st.markdown("---")
st.markdown("🏭 **Industrial Health System** | Built with Streamlit & Scikit-learn")