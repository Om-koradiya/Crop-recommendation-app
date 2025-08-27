import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# --- LOAD SAVED ARTIFACTS ---
# This section loads all the files created by your training script.
# It uses a try-except block to handle errors gracefully if files are missing.
try:
    model = joblib.load('crop_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    metadata = joblib.load('model_metadata.pkl')
    df = pd.read_csv('APY.csv')
    df.columns = df.columns.str.strip().str.lower() # Clean column names
except FileNotFoundError:
    st.error("ERROR: Critical model files are missing! Please run the training script to generate them.")
    st.stop()

# Extract feature columns from metadata
feature_columns = metadata.get('feature_columns', [])

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Information", "Data Exploration", "Crop Recommender"])
st.sidebar.markdown("---")
st.sidebar.info("This project recommends the best crop to grow based on environmental and agricultural data.")

# =====================================================================================
# --- PAGE 1: PROJECT INFORMATION ---
# =====================================================================================
if page == "Project Information":
    st.title("🌾 Advanced Crop Recommendation System")
    st.markdown("---")
    
    st.header("Project Goal")
    st.write("The primary objective of this project is to build a machine learning model that provides farmers with a reliable crop recommendation. By analyzing historical agricultural data from India, the system predicts the most suitable crop to cultivate based on specific environmental and geographical factors, aiming to increase yield and support agricultural planning.")
    
    st.header("How to Use This App")
    st.write("This application is designed with a simple, multi-page interface:")
    st.markdown("""
    - **Data Exploration**: This page provides a visual overview of the dataset used for training, including key statistics and distributions. It showcases the Exploratory Data Analysis (EDA) performed.
    - **Crop Recommender**: This is the main tool. Users can input specific details like their state, district, season, and land area to receive an instant crop recommendation from our trained XGBoost model.
    """)

    st.header("Dataset Information")
    st.write("This project is built upon a comprehensive agricultural dataset sourced from the official **Indian Government portal (data.gov.in)**. It contains district-wise, season-wise crop production statistics, providing a rich foundation for our predictive model.")

# =====================================================================================
# --- PAGE 2: DATA EXPLORATION & PLOTS ---
# =====================================================================================
elif page == "Data Exploration":
    st.title("📊 Data Exploration and Insights")
    st.markdown("---")
    
    st.header("Dataset Overview")
    st.write("Here is a small sample of the dataset used to train our model:")
    st.dataframe(df.head())
    
    st.header("Exploratory Data Analysis (EDA) Plots")
    st.write("These plots were generated during the model training phase to understand the data's characteristics and relationships.")
    
    plot_dir = 'plots'
    if os.path.exists(plot_dir):
        plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        for plot_file in plot_files:
            st.image(os.path.join(plot_dir, plot_file), caption=plot_file.replace('_', ' ').replace('.png', '').title())
    else:
        st.warning("The 'plots' directory was not found. Please run the training script to generate plots.")

# =====================================================================================
# --- PAGE 3: CROP RECOMMENDER (PREDICTION TOOL) ---
# =====================================================================================
elif page == "Crop Recommender":
    st.title("🌱 Crop Recommender Tool")
    st.markdown("---")
    st.header("Enter Your Farm's Details")

    # Create columns for user input
    col1, col2 = st.columns(2)

    with col1:
        state_options = sorted(df['state'].unique())
        selected_state = st.selectbox('Select your State', state_options)
        
        district_options = sorted(df[df['state'] == selected_state]['district'].unique())
        selected_district = st.selectbox('Select your District', district_options)

    with col2:
        season_options = sorted(df['season'].unique())
        selected_season = st.selectbox('Select the Season', season_options)
        
        crop_year = st.number_input('Enter Crop Year', min_value=1997, max_value=2025, value=2024)
        area = st.number_input('Enter Area of Land (in Hectares)', min_value=0.01, value=1.0, step=0.1)

    # Prediction button
    if st.button('Recommend Crop', use_container_width=True, type="primary"):
        # Create a dictionary with user inputs
        user_input = {
            'state': selected_state, 'district': selected_district,
            'season': selected_season, 'crop_year': crop_year, 'area': area
        }
        input_df = pd.DataFrame([user_input])

        # --- DATA PREPARATION PIPELINE ---
        
        # 1. Encode categorical text features
        input_df['state_encoded'] = encoders['state'].transform(input_df['state'])
        input_df['district_encoded'] = encoders['district'].transform(input_df['district'])
        input_df['season_encoded'] = encoders['season'].transform(input_df['season'])
        
        # 2. Engineer the new features your model expects
        # We define simple logic for single predictions
        input_df['season_type_encoded'] = encoders['season_type'].transform(input_df['season'].str.strip().str.lower())
        
        area_bins = [0, 100, 1000, 10000, np.inf]
        area_labels = ['small', 'medium', 'large', 'very_large']
        input_df['area_category'] = pd.cut(input_df['area'], bins=area_bins, labels=area_labels, right=False)
        input_df['area_category_encoded'] = encoders['area_category'].transform(input_df['area_category'])
        
        # 3. Ensure all feature columns are present and in the correct order
        final_input_df = input_df[feature_columns]

        # 4. Scale the features using the loaded scaler
        final_input_scaled = scaler.transform(final_input_df)
        
        # --- PREDICTION ---
        prediction_encoded = model.predict(final_input_scaled)
        
        # Decode the prediction back to the original crop name
        predicted_crop = encoders['crop'].inverse_transform(prediction_encoded)
        
        # --- DISPLAY RESULT ---
        st.success(f"**Based on the provided data, the recommended crop is:**")
        st.header(f"**{predicted_crop[0].title()}**")