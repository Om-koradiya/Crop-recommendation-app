# This is your advanced script, with only the feature list corrected for classification.
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

def create_plots(df, df_clean, results, best_model, feature_columns, model_name):
    """Create and save EDA and model performance plots"""
    print("\nCREATING EXPLORATORY DATA ANALYSIS PLOTS")
    print("=" * 60)
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # EDA Overview
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    sns.histplot(df['crop_year'], bins=20, kde=True, ax=axes[0, 0]).set_title('Distribution of Crop Year')
    sns.countplot(y='season', data=df, order=df['season'].value_counts().index, ax=axes[0, 1]).set_title('Distribution of Season')
    sns.histplot(df['area'], bins=50, kde=True, ax=axes[1, 0]).set_title('Distribution of Area').set_xscale('log')
    sns.histplot(df['production'], bins=50, kde=True, ax=axes[1, 1]).set_title('Distribution of Production').set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eda_overview.png'))
    plt.close()
    print(f"Saved: {os.path.join(plot_dir, 'eda_overview.png')}")

    # Correlation Heatmap
    numeric_df = df_clean.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.png'))
    plt.close()
    print(f"Saved: {os.path.join(plot_dir, 'correlation_heatmap.png')}")

    # State Analysis
    plt.figure(figsize=(15, 12))
    sns.countplot(y='state', data=df, order=df['state'].value_counts().index[:20])
    plt.title('Top 20 States by Number of Crop Records')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'state_analysis.png'))
    plt.close()
    print(f"Saved: {os.path.join(plot_dir, 'state_analysis.png')}")
    
    # Feature Engineering Insights
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.boxplot(x='area_category', y='productivity_ratio', data=df_clean, ax=axes[0]).set_title('Productivity Ratio by Area Category')
    sns.countplot(y='yield_category', data=df_clean, order=df_clean['yield_category'].value_counts().index, ax=axes[1]).set_title('Distribution of Yield Category')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_engineering_insights.png'))
    plt.close()
    print(f"Saved: {os.path.join(plot_dir, 'feature_engineering_insights.png')}")

    # Model Performance
    model_names = list(results.keys())
    accuracies = [res['Accuracy'] for res in results.values()]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Model Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(plot_dir, 'model_performance.png'))
    plt.close()
    print(f"Saved: {os.path.join(plot_dir, 'model_performance.png')}")


def load_and_explore_data(file_path):
    """Load data and perform initial exploration"""
    print("Loading and exploring data...")
    df = pd.read_csv(file_path, on_bad_lines='skip')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    df.columns = df.columns.str.strip().str.lower()
    print(f"Columns: {list(df.columns)}")
    print("\nData Overview:")
    print(df.head().to_string())
    print("\nData Types:")
    print(df.dtypes)
    
    # Handle missing values by dropping rows where 'crop' is NaN
    df.dropna(subset=['crop'], inplace=True)
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print(f"\nUnique crops in dataset: {df['crop'].nunique()}")
    print("Top 10 crops by frequency:")
    print(df['crop'].value_counts().head(10).to_string())
    
    return df

def preprocess_and_engineer_features(df):
    """Clean, preprocess, and engineer features"""
    print("\nDATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)
    
    df_clean = df.copy()

    # Handle potential missing values in production and area
    df_clean['production'].fillna(0, inplace=True)
    df_clean['area'].fillna(df_clean['area'].median(), inplace=True)
    
    print(f"Dataset is already clean! Shape: {df_clean.shape}")

    print("Creating enhanced features...")
    df_clean['productivity_ratio'] = df_clean['production'] / (df_clean['area'] + 1e-6)
    print("    Created productivity_ratio feature (production/area)")
    
    # Create categorical features based on quantiles
    df_clean['area_category'] = pd.qcut(df_clean['area'], q=4, labels=['small', 'medium', 'large', 'very_large'])
    print("    Created area_category feature")
    
    # Create yield and handle it
    df_clean['yield'] = df_clean['production'] / (df_clean['area'] + 1e-6)
    df_clean['yield_category'] = pd.qcut(df_clean['yield'].rank(method='first'), q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    print("    Created yield_category feature")
    
    df_clean['season_type'] = df_clean['season'].str.strip().str.lower()
    print("    Created season_type feature")
    
    print("Filtering crops with sufficient training data...")
    crop_counts = df_clean['crop'].value_counts()
    crops_to_keep = crop_counts[crop_counts >= 30].index
    df_clean = df_clean[df_clean['crop'].isin(crops_to_keep)]
    print(f"    Keeping {len(crops_to_keep)} crops with at least 30 samples each")
    print(f"    Final dataset shape: {df_clean.shape}")
    
    print("\nTop crops in final dataset:")
    print(df_clean['crop'].value_counts().head(10).to_string())
    
    return df_clean

def encode_features(df):
    """Encode categorical features"""
    print("\nENCODING CATEGORICAL FEATURES")
    print("=" * 60)
    
    df_encoded = df.copy()
    encoders = {}
    
    categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'crop' in categorical_columns:
        categorical_columns.remove('crop')

    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        print(f"    Encoded {col}: {len(le.classes_)} unique values")
        
    # Encode target variable 'crop'
    le_crop = LabelEncoder()
    df_encoded['crop_encoded'] = le_crop.fit_transform(df_encoded['crop'].astype(str))
    encoders['crop'] = le_crop
    print(f"    Encoded crop (target): {len(le_crop.classes_)} classes")
    
    print("\nCrop classes in your model:")
    for i, crop_name in enumerate(le_crop.classes_[:10]):
        print(f"    {i}: {crop_name}")
    if len(le_crop.classes_) > 10:
        print(f"    ... and {len(le_crop.classes_) - 10} more")
        
    return df_encoded, encoders

def prepare_features_target(df_encoded):
    """Prepare feature matrix and target vector"""
    print("\nPREPARING FEATURES AND TARGET")
    print("=" * 60)
    
    # --- THIS IS THE ONLY PART I HAVE CHANGED ---
    # The feature list is now corrected to exclude leaky variables
    feature_columns = [
        'state_encoded', 
        'district_encoded', 
        'season_encoded', 
        'season_type_encoded', 
        'area_category_encoded',
        'crop_year', 
        'area'
    ]
    
    X = df_encoded[feature_columns]
    y = df_encoded['crop_encoded'] # Target is the encoded crop column
    
    print(f"    Features shape: {X.shape}")
    print(f"    Target shape: {y.shape}")
    print(f"    Feature columns: {feature_columns}")
    print(f"    Number of crop classes: {len(np.unique(y))}")
    
    return X, y, feature_columns

def train_and_evaluate_models(X, y):
    """Split data, scale, train and evaluate classification models"""
    print("\nSPLITTING DATA")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"    Training set: {X_train.shape}")
    print(f"    Test set: {X_test.shape}")

    print("\nSCALING FEATURES")
    print("=" * 60)
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"    Scaled {len(numerical_cols)} numerical features: {numerical_cols}")

    print("\nTRAINING CLASSIFICATION MODELS")
    print("=" * 60)
    models = {
        'XGBoost Classifier': xgb.XGBClassifier(objective='multi:softmax', random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss'),
        'Random Forest Classifier': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Extra Trees Classifier': ExtraTreesClassifier(random_state=42, n_jobs=-1)
    }
    
    results = {}
    best_model_name = ''
    best_accuracy = 0.0

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {'Accuracy': accuracy, 'F1-Score': f1, 'Training Time': training_time, 'model_object': model}
        
        print(f"    Training time: {training_time:.2f} seconds")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    print(f"\nBest Model: {best_model_name}")
    print(f"    Best Accuracy: {results[best_model_name]['Accuracy']:.4f}")
    print(f"    Best F1-Score: {results[best_model_name]['F1-Score']:.4f}")
    
    best_model = results[best_model_name]['model_object']
    
    return best_model, scaler, results, best_model_name, X.columns.tolist()

def perform_cross_validation(model, X, y, model_name):
    """Perform cross-validation"""
    print(f"\nCROSS-VALIDATION FOR {model_name.upper()}")
    print("=" * 60)
    cv_accuracy = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
    cv_f1 = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    print(f"    CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"    CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    return {'cv_accuracy_mean': cv_accuracy.mean(), 'cv_f1_mean': cv_f1.mean()}

def print_feature_importance(model, feature_columns, model_name):
    """Print feature importance"""
    print(f"\nFEATURE IMPORTANCE - {model_name.upper()}")
    print("=" * 60)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(importance_df.to_string(index=False))

def save_artifacts(model, encoders, scaler, metadata):
    """Save all model artifacts"""
    print("\nSAVING MODEL ARTIFACTS")
    print("=" * 60)
    joblib.dump(model, 'crop_model.pkl')
    print("    Model saved as 'crop_model.pkl'")
    joblib.dump(encoders, 'encoders.pkl')
    print("    Encoders saved as 'encoders.pkl'")
    joblib.dump(scaler, 'scaler.pkl')
    print("    Scaler saved as 'scaler.pkl'")
    joblib.dump(metadata, 'model_metadata.pkl')
    print("    Metadata saved as 'model_metadata.pkl'")

def main():
    """Main training pipeline"""
    print("ADVANCED CROP RECOMMENDATION SYSTEM")
    print("=" * 60)
    total_start_time = time.time()

    try:
        # Step 1
        df = load_and_explore_data('APY.csv')
        # Step 2
        df_clean = preprocess_and_engineer_features(df)
        # Step 3
        df_encoded, encoders = encode_features(df_clean)
        # Step 4
        X, y, feature_columns = prepare_features_target(df_encoded)
        # Step 5
        best_model, scaler, results, best_model_name, feature_columns_final = train_and_evaluate_models(X, y)
        # Step 6
        create_plots(df, df_clean, results, best_model, feature_columns, best_model_name)
        # Step 7
        cv_results = perform_cross_validation(best_model, X, y, best_model_name)
        results[best_model_name].update(cv_results)
        # Step 8
        print_feature_importance(best_model, feature_columns, best_model_name)
        # Step 9
        metadata = {
            'feature_columns': feature_columns_final,
            'model_name': best_model_name,
            'model_results': {name: {k: v for k, v in res.items() if k != 'model_object'} for name, res in results.items()},
            'encoded_columns': list(encoders.keys())
        }
        save_artifacts(best_model, encoders, scaler, metadata)

        # Final Summary
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        total_time = time.time() - total_start_time
        print(f"⏱️ Total training time: {total_time:.2f} seconds")
        print(f"🏆 Best model: {best_model_name.upper()}")
        best_results = results[best_model_name]
        print(f"📈 Test accuracy: {best_results['Accuracy']:.4f}")
        print(f"📊 Test F1-score: {best_results['F1-Score']:.4f}")
        print(f"✅ CV accuracy: {best_results['cv_accuracy_mean']:.4f}")
        print(f"📋 Number of crop classes: {len(encoders['crop'].classes_)}")
        print(f"📜 Features used: {len(feature_columns_final)}")
        print("\nFiles created:")
        print("    crop_model.pkl")
        print("    encoders.pkl")
        print("    scaler.pkl")
        print("    model_metadata.pkl")
        print("\nPlots created in 'plots' folder.")
        print("\n🚀 Ready to build your Streamlit app!")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check your data file and try again.")

if __name__ == "__main__":
    main()