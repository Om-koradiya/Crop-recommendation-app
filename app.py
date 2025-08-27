import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Page setup
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Load model, scaler, encoders, metadata, and data
@st.cache_resource
def load_artifacts():
    model = joblib.load('crop_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    metadata = joblib.load('model_metadata.pkl')
    df = pd.read_csv('APY.csv')
    df.columns = df.columns.str.strip().str.lower()
    return model, scaler, encoders, metadata, df

try:
    model, scaler, encoders, metadata, df = load_artifacts()
except FileNotFoundError:
    st.error("ERROR: Critical model files are missing! Please run the training script to generate them.")
    st.stop()

feature_columns = metadata.get('feature_columns', [])

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Information", "Data Exploration", "Crop Recommender"])
st.sidebar.markdown("---")
st.sidebar.info("This project recommends the best crop to grow based on environmental and agricultural data.")

# Project info page
if page == "Project Information":
    st.title("🌾 Crop Recommendation System")
    st.markdown("---")
    
    st.header("Project Goal")
    st.write("The primary objective of this project was to build a machine learning model that provides farmers with a reliable crop recommendation. By analyzing historical agricultural data from India, the system predicts the most suitable crop to cultivate based on specific environmental and geographical factors, aiming to increase yield and support agricultural planning.")
    
    st.header("How to Use This App")
    st.write("Use the navigation bar to jump between pages:")
    st.markdown("""
    - **Data Exploration**: This page provides a visual overview of the dataset used for training, including key statistics and distributions. It showcases the Exploratory Data Analysis (EDA) performed.
    - **Crop Recommender**: This is the main tool. Users can input specific details like their state, district, season, and land area to receive instant crop recommendations from our trained XGBoost model.
    """)

    st.header("Dataset Information")
    st.write("This project is built upon a  agricultural dataset of India sourced from kaggle. It contains district-wise, season-wise crop production statistics, providing a rich foundation for our predictive model.")
    
    st.markdown("---")
    st.header("About This Project")
    st.write("Hi! I'm **Om Koradiya**, and I built this project.")
    st.write("If you'd like to know more about it or see the code, check out the project's [GitHub repository](https://github.com/Om-koradiya/Crop-recommendation-app).")

# Data exploration page
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

# Crop recommender page
elif page == "Crop Recommender":
    st.title("🌱 Crop Recommender Tool")
    st.markdown("---")
    st.header("Enter Your Farm's Details")

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

    if st.button('Recommend Crop', use_container_width=True, type="primary"):
        user_input = {
            'state': selected_state, 'district': selected_district,
            'season': selected_season, 'crop_year': crop_year, 'area': area
        }
        input_df = pd.DataFrame([user_input])

        # Prepare input for model
        input_df['state_encoded'] = encoders['state'].transform(input_df['state'])
        input_df['district_encoded'] = encoders['district'].transform(input_df['district'])
        input_df['season_encoded'] = encoders['season'].transform(input_df['season'])
        input_df['season_type_encoded'] = encoders['season_type'].transform(input_df['season'].str.strip().str.lower())
        area_bins = [0, 100, 1000, 10000, np.inf]
        area_labels = ['small', 'medium', 'large', 'very_large']
        input_df['area_category'] = pd.cut(input_df['area'], bins=area_bins, labels=area_labels, right=False)
        input_df['area_category_encoded'] = encoders['area_category'].transform(input_df['area_category'])
        final_input_df = input_df[feature_columns]
        final_input_scaled = scaler.transform(final_input_df)
        
        # Get top 3 crop predictions
        probabilities = model.predict_proba(final_input_scaled)
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_3_probabilities = probabilities[0][top_3_indices]
        top_3_crops = encoders['crop'].inverse_transform(top_3_indices)
        
        st.success("**Based on the provided data, here are your top 3 crop recommendations:**")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.subheader("🥇 1st Choice")
            st.header(f"{top_3_crops[0].title()}")
            st.metric(label="Confidence Score", value=f"{top_3_probabilities[0]*100:.2f}%")

        with res_col2:
            st.subheader("🥈 2nd Choice")
            st.header(f"{top_3_crops[1].title()}")
            st.metric(label="Confidence Score", value=f"{top_3_probabilities[1]*100:.2f}%")

        with res_col3:
            st.subheader("🥉 3rd Choice")
            st.header(f"{top_3_crops[2].title()}")
            st.metric(label="Confidence Score", value=f"{top_3_probabilities[2]*100:.2f}%")
