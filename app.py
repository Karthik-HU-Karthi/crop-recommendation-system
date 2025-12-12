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
        color: #2c3e50;
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

# Crop Dictionary with Static Images (More reliable than dynamic search)
crop_info = {
    'rice': {'image': 'https://upload.wikimedia.org/wikipedia/commons/1/19/Rice_Plants_%28IRRI%29.jpg', 'desc': 'Rice needs heavy rainfall, high humidity, and clayey soil.'},
    'maize': {'image': 'https://upload.wikimedia.org/wikipedia/commons/3/30/Maize_plant_in_details.jpg', 'desc': 'Maize grows well in warm weather with moderate rain and loamy soil.'},
    'jute': {'image': 'https://upload.wikimedia.org/wikipedia/commons/0/03/Jute_field.jpg', 'desc': 'Jute is a rain-fed crop that requires alluvial soil and standing water.'},
    'cotton': {'image': 'https://upload.wikimedia.org/wikipedia/commons/c/c2/Cotton_plant.jpg', 'desc': 'Cotton requires plenty of sunshine, moderate rainfall, and black soil.'},
    'coconut': {'image': 'https://upload.wikimedia.org/wikipedia/commons/8/82/Coconut_palms.jpg', 'desc': 'Coconut thrives in tropical coastal regions with high humidity.'},
    'papaya': {'image': 'https://upload.wikimedia.org/wikipedia/commons/6/6b/Papaya_tree_with_fruits.jpg', 'desc': 'Papaya needs a warm climate and fertile, well-drained soil.'},
    'orange': {'image': 'https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange_Fruit_Pieces.jpg', 'desc': 'Oranges require sunshine and moderate water to sweeten.'},
    'apple': {'image': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg', 'desc': 'Apples grow best in cooler climates with well-drained loamy soil.'},
    'muskmelon': {'image': 'https://upload.wikimedia.org/wikipedia/commons/e/e0/Cantaloupes.jpg', 'desc': 'Muskmelon needs dry, warm weather and sandy loam soil.'},
    'watermelon': {'image': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Watermelon.jpg', 'desc': 'Watermelon thrives in hot, dry climates and sandy soil.'},
    'grapes': {'image': 'https://upload.wikimedia.org/wikipedia/commons/b/bb/Table_grapes_on_white.jpg', 'desc': 'Grapes need dry, hot summers and plenty of sunshine.'},
    'mango': {'image': 'https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg', 'desc': 'Mango is a tropical fruit that loves heat and well-drained soil.'},
    'banana': {'image': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/Bananas.jpg', 'desc': 'Bananas need high humidity, moisture, and rich soil.'},
    'pomegranate': {'image': 'https://upload.wikimedia.org/wikipedia/commons/8/8d/Pomegranate_02.jpg', 'desc': 'Pomegranate is drought-tolerant and grows in semiarid zones.'},
    'lentil': {'image': 'https://upload.wikimedia.org/wikipedia/commons/b/b3/Lentils_in_spoon_and_bowl.jpg', 'desc': 'Lentils grow well in cool weather and don\'t need much water.'},
    'blackgram': {'image': 'https://upload.wikimedia.org/wikipedia/commons/a/a2/Vigna_mungo_seeds.jpg', 'desc': 'Blackgram is a warm-weather pulse crop suited for loamy soil.'},
    'mungbean': {'image': 'https://upload.wikimedia.org/wikipedia/commons/7/7b/Mung_beans.jpg', 'desc': 'Mungbean is perfect for summer cultivation and tolerates heat.'},
    'mothbeans': {'image': 'https://upload.wikimedia.org/wikipedia/commons/d/d3/Vigna_aconitifolia_seeds.jpg', 'desc': 'Mothbeans are extremely drought-resistant and hardy.'},
    'pigeonpeas': {'image': 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Pigeon_peas.jpg', 'desc': 'Pigeonpeas are deep-rooted and handle dry spells well.'},
    'kidneybeans': {'image': 'https://upload.wikimedia.org/wikipedia/commons/7/79/Kidney_beans.jpg', 'desc': 'Kidney beans need moderate rainfall and distinct dry season.'},
    'chickpea': {'image': 'https://upload.wikimedia.org/wikipedia/commons/e/e2/Chickpeas.jpg', 'desc': 'Chickpeas need cool climate and low rainfall (winter crop).',},
    'coffee': {'image': 'https://upload.wikimedia.org/wikipedia/commons/4/45/Coffee_berries_1.jpg', 'desc': 'Coffee prefers cool to warm tropical highland climates.'}
}

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6064/6064973.png", width=100)
    st.header("About App")
    st.markdown("""
    This **Smart Crop Recommendation System** analyzes:
    - Soil Nutrients (N, P, K)
    - Weather Conditions (Temp, Humidity, Rainfall)
    - Soil pH
    
    ...to suggest the **optimal crop** for maximum yield!
    """)
    st.markdown("---")
    st.info("Built with Machine Learning (Random Forest)")

# Main Content
st.markdown("<h1 class='main-header'>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("Enter your farm's Soil & Weather details below:")

# Columns for Input (No Form)
st.subheader("🧪 Soil Parameters")
c1, c2, c3 = st.columns(3)
with c1:
    N = st.number_input("Nitrogen (N)", 0, 140, 40, help="Ratio of Nitrogen content in soil")
with c2:
    P = st.number_input("Phosphorus (P)", 0, 145, 50, help="Ratio of Phosphorus content in soil")
with c3:
    K = st.number_input("Potassium (K)", 0, 205, 50, help="Ratio of Potassium content in soil")

st.subheader("🌤️ Weather Parameters")
c4, c5 = st.columns(2)
with c4:
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 71.0)
with c5:
    ph = st.number_input("pH Level", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 103.0)

# Predict Button
if st.button("🚜 Predict Best Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    # Store result in session state
    st.session_state['prediction'] = prediction[0].lower()

# Display Result if it exists in state
if 'prediction' in st.session_state:
    result_crop = st.session_state['prediction']
    
    st.markdown(f"""
    <div class='result-box'>
        <h3>🌟 Best Crop to Grow: {result_crop.upper()}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get Info
    info = crop_info.get(result_crop, {'image': 'https://via.placeholder.com/300?text=No+Image', 'desc': 'An excellent choice for your farm.'})
    
    # Display Description Only (Image removed as requested)
    st.subheader("📋 Crop Report")
    st.write(f"**Description:** {info['desc']}")
    st.markdown("---")
    st.write("**Why this matches your soil:**")
    st.write(f"- **Nutrient Profile:** Your soil has N: {N}, P: {P}, K: {K}, which aligns with the requirements for {result_crop}.")
    st.write(f"- **Conditions:** {result_crop.capitalize()} is suitable for {temperature}°C temperatures, {humidity}% humidity, and {rainfall}mm rainfall.")


