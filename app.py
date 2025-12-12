import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
model = joblib.load('models/random_forest_model.pkl')

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

st.title("🌾 Crop Recommendation System")
st.markdown("Enter the soil and environmental parameters to get a crop recommendation.")

# Input fields
c1, c2 = st.columns(2)
with c1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)

with c2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Predict Crop"):
    # Prepare input
    # The training data features order: N, P, K, temperature, humidity, ph, rainfall
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Predict
    prediction = model.predict(input_data)
    recommended_crop = prediction[0]
    
    st.success(f"🌱 Recommended Crop: **{recommended_crop.upper()}**")
    
    st.info("Note: This recommendation is based on a Random Forest model trained on historical crop data.")

st.markdown("---")
st.caption("Developed for Crop Recommendation Assessment")
