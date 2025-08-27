import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD ARTIFACTS ---

# Load the machine learning model, scaler, and encoders
# These files were created by your 'training-optimised.py' script
try:
    model = joblib.load('crop_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    metadata = joblib.load('model_metadata.pkl')
    # Load the original dataset to get the unique values for the dropdowns
    df = pd.read_csv('APY.csv')
    df.columns = df.columns.str.strip().str.lower() # Clean column names
except FileNotFoundError:
    st.error("Model files not found! Please run the training script first.")
    st.stop()

# Get the list of feature columns from the metadata
feature_columns = metadata.get('feature_columns', [])

# --- STREAMLIT APP INTERFACE ---

st.set_page_config(layout="wide")

# App title
st.title("🌾 Advanced Crop Recommendation System")
st.markdown("This app recommends the best crop to plant based on your farm's conditions.")

# Create columns for user input
col1, col2 = st.columns(2)

with col1:
    # Get unique sorted values for dropdowns from the dataframe
    state_options = sorted(df['state'].unique())
    season_options = sorted(df['season'].unique())
    
    selected_state = st.selectbox('Select your State', state_options)
    
    # Filter districts based on selected state
    district_options = sorted(df[df['state'] == selected_state]['district'].unique())
    selected_district = st.selectbox('Select your District', district_options)
    
    selected_season = st.selectbox('Select the Season', season_options)

with col2:
    crop_year = st.number_input('Enter Crop Year', min_value=1997, max_value=2025, value=2024)
    area = st.number_input('Enter Area of Land (in Hectares)', min_value=0.01, value=1.0, step=0.1)

# Prediction button
if st.button('🌱 Recommend Crop', use_container_width=True):
    # --- DATA PREPARATION FOR PREDICTION ---
    
    # Create a dictionary with user inputs
    user_input = {
        'state': selected_state,
        'district': selected_district,
        'season': selected_season,
        'crop_year': crop_year,
        'area': area
    }
    
    # Create a DataFrame from the user input
    input_df = pd.DataFrame([user_input])
    
    # --- ENCODING ---
    # We need to encode the inputs just like we did in training
    # We also add the engineered features your model expects
    
    # Encode categorical features using the loaded encoders
    input_df['state_encoded'] = encoders['state'].transform(input_df['state'])
    input_df['district_encoded'] = encoders['district'].transform(input_df['district'])
    input_df['season_encoded'] = encoders['season'].transform(input_df['season'])
    
    # Create engineered features (simplified for single prediction)
    input_df['area_category_encoded'] = encoders['area_category'].transform(pd.qcut(input_df['area'], q=4, labels=['small', 'medium', 'large', 'very_large']))
    input_df['season_type_encoded'] = encoders['season_type'].transform(input_df['season'].str.strip().str.lower())
    
    # Ensure all required feature columns are present and in the correct order
    final_input_df = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in input_df.columns:
            final_input_df[col] = input_df[col]

    # --- SCALING ---
    # Scale the numerical features using the loaded scaler
    final_input_scaled = scaler.transform(final_input_df)
    
    # --- PREDICTION ---
    # Make a prediction using the loaded model
    prediction_encoded = model.predict(final_input_scaled)
    
    # --- DISPLAY RESULT ---
    # Decode the prediction back to the original crop name
    predicted_crop = encoders['crop'].inverse_transform(prediction_encoded)
    
    st.success(f"**Based on the provided data, the recommended crop is:**")
    st.header(f"**{predicted_crop[0]}**")

# Add a footer
st.markdown("---")
st.write("Project by you for your internship.")