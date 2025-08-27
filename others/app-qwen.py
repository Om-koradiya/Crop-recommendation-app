# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page config
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Load model
        with open('crop_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load encoders
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, encoders, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_agricultural_data.csv')
        return df.head(10)  # For preview
    except Exception as e:
        st.warning("Could not load dataset preview.")
        return pd.DataFrame()

# Load model artifacts
model, encoders, scaler, metadata = load_artifacts()
data_preview = load_data()

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Info", "Plots"])

# Title & Description
if page == "Home":
    st.title("🌾 Crop Recommendation System")
    st.markdown("""
    This AI-powered tool recommends the best crops to grow based on your farm's location, season, and land area.
    
    Use the sidebar to navigate to:
    - **Prediction**: Get top 3 recommended crops
    - **Info**: Learn about the data and model
    - **Plots**: View insights from training
    """)

elif page == "Prediction":
    st.title("🔍 Predict Best Crops")

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("Select your State", options=encoders['state'].classes_)
        district = st.selectbox("Select your District", options=encoders['district'].classes_)
        season = st.selectbox("Select the Season", options=encoders['season'].classes_)

    with col2:
        crop_year = st.number_input("Enter Crop Year", min_value=2000, max_value=2050, value=2024)
        area = st.number_input("Enter Area of Land (in Hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)

    # Prediction button
    if st.button("🌱 Recommend Crop"):
        if model is None:
            st.error("Model not loaded. Please check files.")
        else:
            try:
                # Encode inputs
                state_enc = encoders['state'].transform([state])[0]
                district_enc = encoders['district'].transform([district])[0]
                season_enc = encoders['season'].transform([season])[0]

                # Prepare input features
                input_features = np.array([[state_enc, district_enc, season_enc, 0, 0, crop_year, area]])  # Fill dummy values for missing encoded features
                # Note: If you have more features like season_type, area_category, etc., we need to encode them too.

                # Scale input
                scaled_input = scaler.transform(input_features)

                # Predict
                probabilities = model.predict_proba(scaled_input)[0]
                top_indices = np.argsort(probabilities)[-3:][::-1]  # Top 3 indices
                top_crops = [metadata['crop_classes'][i] for i in top_indices]
                top_probs = [probabilities[i] for i in top_indices]

                # Display results
                st.subheader("✅ Top 3 Recommended Crops")
                for i, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                    st.write(f"**{i+1}. {crop}** — Confidence: {prob:.2%}")

            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "Info":
    st.title("📊 About the System")

    st.header("📌 Dataset Overview")
    st.write("The model was trained on agricultural data from across India.")
    st.dataframe(data_preview)

    st.header("🧠 Model Details")
    st.write("Best Model: **XGBoost Classifier**")
    st.write(f"Test Accuracy: {metadata.get('test_accuracy', 'N/A')}")
    st.write(f"Test F1-Score: {metadata.get('test_f1', 'N/A')}")
    st.write(f"Number of Crops: {metadata.get('num_classes', 55)}")
    st.write(f"Features Used: {metadata.get('feature_names', ['state_encoded', 'district_encoded', 'season_encoded', 'area', 'crop_year'])}")

    st.header("📈 Feature Importance")
    if 'feature_importance' in metadata:
        fig, ax = plt.subplots(figsize=(8, 5))
        importance_df = pd.DataFrame({
            'Feature': metadata['feature_importance'].index,
            'Importance': metadata['feature_importance'].values
        }).sort_values('Importance', ascending=False)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title("Feature Importance (XGBoost)")
        st.pyplot(fig)
    else:
        st.write("Feature importance not available.")

elif page == "Plots":
    st.title("📈 Exploratory Data Analysis")

    # Load plots
    plot_files = [
        'plots/eda_overview.png',
        'plots/correlation_heatmap.png',
        'plots/state_analysis.png',
        'plots/feature_engineering_insights.png',
        'plots/model_performance.png'
    ]

    for file in plot_files:
        if file in st.session_state.get('plot_cache', {}):
            img = st.session_state.plot_cache[file]
        else:
            try:
                img = Image.open(file)
                st.session_state.plot_cache[file] = img
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
                continue

        st.image(img, caption=file.split('/')[-1], use_column_width=True)

# Footer
st.markdown("---")
st.caption("🌱 Powered by AI • Built for Agricultural Planning • Project Demo")