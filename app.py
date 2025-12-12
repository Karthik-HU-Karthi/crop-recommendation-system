import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
model = joblib.load('models/random_forest_model.pkl')

# Page Configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ecc71;
    }
    .result-box {
        padding: 20px;
        background-color: #dff0d8;
        border-left: 6px solid #3c763d;
        border-radius: 5px;
        margin-top: 20px;
    }
    .crop-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Crop Dictionary with Descriptions (Images will be fetched dynamically to ensure they always load)
crop_info = {
    'rice': 'Rice needs heavy rainfall, high humidity, and clayey soil.',
    'maize': 'Maize grows well in warm weather with moderate rain and loamy soil.',
    'jute': 'Jute is a rain-fed crop that requires alluvial soil and standing water.',
    'cotton': 'Cotton requires plenty of sunshine, moderate rainfall, and black soil.',
    'coconut': 'Coconut thrives in tropical coastal regions with high humidity.',
    'papaya': 'Papaya needs a warm climate and fertile, well-drained soil.',
    'orange': 'Oranges require sunshine and moderate water to sweeten.',
    'apple': 'Apples grow best in cooler climates with well-drained loamy soil.',
    'muskmelon': 'Muskmelon needs dry, warm weather and sandy loam soil.',
    'watermelon': 'Watermelon thrives in hot, dry climates and sandy soil.',
    'grapes': 'Grapes need dry, hot summers and plenty of sunshine.',
    'mango': 'Mango is a tropical fruit that loves heat and well-drained soil.',
    'banana': 'Bananas need high humidity, moisture, and rich soil.',
    'pomegranate': 'Pomegranate is drought-tolerant and grows in semiarid zones.',
    'lentil': 'Lentils grow well in cool weather and don\'t need much water.',
    'blackgram': 'Blackgram is a warm-weather pulse crop suited for loamy soil.',
    'mungbean': 'Mungbean is perfect for summer cultivation and tolerates heat.',
    'mothbeans': 'Mothbeans are extremely drought-resistant and hardy.',
    'pigeonpeas': 'Pigeonpeas are deep-rooted and handle dry spells well.',
    'kidneybeans': 'Kidney beans need moderate rainfall and distinct dry season.',
    'chickpea': 'Chickpeas need cool climate and low rainfall (winter crop).',
    'coffee': 'Coffee prefers cool to warm tropical highland climates.'
}

# ... (Sidebar code remains same) ...

# Main Content
st.markdown("<h1 class='main-header'>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("Enter your farm's Soil & Weather details below:")

with st.form("prediction_form"):
    st.subheader("🧪 Soil Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen (N)", 0, 140, 50, help="Ratio of Nitrogen content in soil")
    with c2:
        P = st.number_input("Phosphorus (P)", 0, 145, 50, help="Ratio of Phosphorus content in soil")
    with c3:
        K = st.number_input("Potassium (K)", 0, 205, 50, help="Ratio of Potassium content in soil")

    st.subheader("🌤️ Weather Parameters")
    c4, c5 = st.columns(2)
    with c4:
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    with c5:
        ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)
    
    submit_btn = st.form_submit_button("🚜 Predict Best Crop")

# Logic
if submit_btn:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    result_crop = prediction[0].lower()
    
    st.markdown(f"""
    <div class='result-box'>
        <h3>🌟 Best Crop to Grow: {result_crop.upper()}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Image and Desc
    col_img, col_desc = st.columns([1, 2])
    
    # Use loremflickr for reliable dynamic images matching the crop name
    image_url = f"https://loremflickr.com/400/300/{result_crop},farming"
    desc = crop_info.get(result_crop, 'An excellent choice for your farm.')
    
    with col_img:
        st.image(image_url, caption=f"{result_crop.capitalize()} Field", use_column_width=True)
    with col_desc:
        st.subheader("📋 Crop Report")
        st.write(f"**Description:** {desc}")
        st.markdown("---")
        st.write("**Why this matches your soil:**")
        st.write(f"- **Nutrient Profile:** Your soil has N: {N}, P: {P}, K: {K}, which aligns with the requirements for {result_crop}.")
        st.write(f"- **Climate Suitability:** {result_crop.capitalize()} thrives in {temperature}°C temperatures with {humidity}% humidity and requires around {rainfall}mm of rain.")

