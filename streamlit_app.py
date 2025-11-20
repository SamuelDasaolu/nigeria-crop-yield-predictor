import streamlit as st
import pandas as pd
import joblib

# 1. Load the Trained Model
# The pipeline includes the OneHotEncoder, so we can feed it raw text like "Maize"
try:
    model = joblib.load('models/nigeria_yield_model.pkl')
except FileNotFoundError:
    st.error("❌ Model not found. Please run 'train_model.py' first.")
    st.stop()

# 2. App Title & Configuration
st.set_page_config(page_title="AgroPredict Nigeria", page_icon="🇳🇬", layout="centered")

st.title("🇳🇬 Nigeria Crop Yield Predictor")
st.markdown("Ai-Powered Precision Agriculture | *Predict harvest efficiency based on climate data*")
st.write("---")

# 3. User Input Form
with st.form("yield_form"):
    st.subheader("🌱 Farm & Crop Details")

    # Crop Selection (These match the training data EXACTLY)
    crop_list = [
        'Maize (corn)', 'Cassava, fresh', 'Yams', 'Rice, paddy',
        'Sorghum', 'Millet', 'Tomatoes', 'Potatoes', 'Sweet potatoes',
        'Beans, dry', 'Groundnuts, with shell', 'Cocoa, beans'
    ]
    selected_crop = st.selectbox("Select Crop", crop_list)

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Prediction Year", min_value=1990, max_value=2030, value=2024)
        area = st.number_input("Farm Area (Hectares)", min_value=1.0, value=1000.0)

    with col2:
        # Default values are based on Nigeria's averages
        rain = st.number_input("Expected Rainfall (mm/year)", min_value=0.0, value=1200.0)
        temp = st.number_input("Avg Temperature (°C)", min_value=0.0, value=28.0)

    # Submit Button
    submitted = st.form_submit_button("🚀 Predict Yield")

# 4. Prediction Logic
if submitted:
    # Create a DataFrame similar to the training data
    # Columns: Item, Year, area_harvested_ha, avg_temp_c, avg_rain_mm
    input_data = pd.DataFrame({
        'Item': [selected_crop],
        'Year': [year],
        'area_harvested_ha': [area],
        'avg_temp_c': [temp],
        'avg_rain_mm': [rain]
    })

    # Make Prediction
    try:
        prediction = model.predict(input_data)[0]

        # Display Result
        st.success(f"🌾 Predicted Yield for {selected_crop}:")

        # Formatting the big numbers
        st.metric(label="Yield Efficiency", value=f"{prediction:,.2f} kg/ha")

        # Calculate Total Production (Yield * Area)
        total_production = prediction * area
        st.info(f"🚜 Estimated Total Harvest: **{total_production / 1000:,.1f} Tonnes**")

    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Please check if the Crop Name matches the training data exactly.")