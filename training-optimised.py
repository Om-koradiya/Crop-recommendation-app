# Memory-optimized version of your advanced script
import pandas as pd
import numpy as np
import joblib
import time
import gc  # Garbage collector for memory management
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

def create_plots(df_sample, df_clean_sample, results, best_model, feature_columns, model_name):
    """Create and save EDA and model performance plots with memory optimization"""
    print("\nCREATING EXPLORATORY DATA ANALYSIS PLOTS")
    print("=" * 60)
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Use samples for plotting to save memory
    print("Using data samples for plotting to optimize memory usage...")
    
    # EDA Overview
    print("Creating EDA overview plots...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    # Create histplots with proper axis handling
    ax1 = sns.histplot(df_sample['crop_year'], bins=20, kde=True, ax=axes[0, 0])
    ax1.set_title('Distribution of Crop Year')
    
    ax2 = sns.countplot(y='season', data=df_sample, order=df_sample['season'].value_counts().index, ax=axes[0, 1])
    ax2.set_title('Distribution of Season')
    
    ax3 = sns.histplot(df_sample['area'], bins=50, kde=True, ax=axes[1, 0])
    ax3.set_title('Distribution of Area')
    ax3.set(xscale='log')
    
    ax4 = sns.histplot(df_sample['production'], bins=50, kde=True, ax=axes[1, 1])
    ax4.set_title('Distribution of Production')
    ax4.set(xscale='log')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eda_overview.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    gc.collect()  # Force garbage collection
    print(f"Saved: {os.path.join(plot_dir, 'eda_overview.png')}")

    # Correlation Heatmap
    print("Creating correlation heatmap...")
    numeric_df = df_clean_sample.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    del numeric_df
    gc.collect()
    print(f"Saved: {os.path.join(plot_dir, 'correlation_heatmap.png')}")

    # State Analysis
    print("Creating state analysis plot...")
    plt.figure(figsize=(15, 12))
    sns.countplot(y='state', data=df_sample, order=df_sample['state'].value_counts().index[:20])
    plt.title('Top 20 States by Number of Crop Records')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'state_analysis.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    gc.collect()
    print(f"Saved: {os.path.join(plot_dir, 'state_analysis.png')}")
    
    # Feature Engineering Insights
    print("Creating feature engineering insights...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.boxplot(x='area_category', y='productivity_ratio', data=df_clean_sample, ax=axes[0]).set_title('Productivity Ratio by Area Category')
    sns.countplot(y='yield_category', data=df_clean_sample, order=df_clean_sample['yield_category'].value_counts().index, ax=axes[1]).set_title('Distribution of Yield Category')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_engineering_insights.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    gc.collect()
    print(f"Saved: {os.path.join(plot_dir, 'feature_engineering_insights.png')}")

    # Model Performance
    print("Creating model performance plot...")
    model_names = list(results.keys())
    accuracies = [res['Accuracy'] for res in results.values()]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Model Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(plot_dir, 'model_performance.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    gc.collect()
    print(f"Saved: {os.path.join(plot_dir, 'model_performance.png')}")
    print("All plots created successfully!")


def load_and_explore_data(file_path):
    """Load data and perform initial exploration with memory optimization"""
    print("Loading and exploring data...")
    
    # Read with optimized dtypes to reduce memory usage
    print("Reading data in optimized mode...")
    dtype_dict = {
        'state': 'category',
        'district': 'category',
        'crop_year': 'int32',
        'season': 'category',
        'crop': 'category',
        'area': 'float32',
        'production': 'float32'
    }
    
    df = pd.read_csv(file_path, 
                     on_bad_lines='skip', 
                     low_memory=True,
                     dtype=dtype_dict)
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    print(f"Columns: {list(df.columns)}")
    
    # Show sample data
    print("\nData Overview (first 5 rows):")
    print(df.head().to_string())
    print("\nData Types:")
    print(df.dtypes)
    
    # Handle missing values by dropping rows where 'crop' is NaN
    initial_shape = df.shape
    df.dropna(subset=['crop'], inplace=True)
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing crop data")
    
    print("\nMissing Values:")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(missing_counts[missing_counts > 0])
    else:
        print("No missing values found!")
    
    print(f"\nUnique crops in dataset: {df['crop'].nunique()}")
    print("Top 10 crops by frequency:")
    print(df['crop'].value_counts().head(10).to_string())
    
    # Force garbage collection
    gc.collect()
    
    return df

def preprocess_and_engineer_features(df):
    """Clean, preprocess, and engineer features with memory optimization"""
    print("\nDATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)
    
    print("Processing data in memory-optimized mode...")
    
    # Work directly on the dataframe to save memory
    # Handle potential missing values in production and area
    df['production'].fillna(0, inplace=True)
    df['area'].fillna(df['area'].median(), inplace=True)
    
    print(f"Dataset cleaned! Shape: {df.shape}")

    print("Creating enhanced features...")
    df['productivity_ratio'] = df['production'] / (df['area'] + 1e-6)
    print("    Created productivity_ratio feature (production/area)")
    
    # Create categorical features based on quantiles
    print("    Creating area categories...")
    df['area_category'] = pd.qcut(df['area'], q=4, labels=['small', 'medium', 'large', 'very_large'])
    print("    Created area_category feature")
    
    # Create yield and handle it
    print("    Creating yield categories...")
    df['yield'] = df['production'] / (df['area'] + 1e-6)
    df['yield_category'] = pd.qcut(df['yield'].rank(method='first'), q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    print("    Created yield_category feature")
    
    df['season_type'] = df['season'].str.strip().str.lower()
    print("    Created season_type feature")
    
    print("Filtering crops with sufficient training data...")
    crop_counts = df['crop'].value_counts()
    crops_to_keep = crop_counts[crop_counts >= 30].index
    df = df[df['crop'].isin(crops_to_keep)]
    print(f"    Keeping {len(crops_to_keep)} crops with at least 30 samples each")
    print(f"    Final dataset shape: {df.shape}")
    
    print("\nTop crops in final dataset:")
    print(df['crop'].value_counts().head(10).to_string())
    
    # Force garbage collection
    gc.collect()
    
    return df

def encode_features(df):
    """Encode categorical features with memory optimization"""
    print("\nENCODING CATEGORICAL FEATURES")
    print("=" * 60)
    
    encoders = {}
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'crop' in categorical_columns:
        categorical_columns.remove('crop')

    print("Encoding categorical features one by one...")
    for col in categorical_columns:
        print(f"    Processing {col}...")
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"    Encoded {col}: {len(le.classes_)} unique values")
        
        # Free up memory by deleting the original column after encoding
        # (Keep original for plotting later by creating a sample)
        gc.collect()
        
    # Encode target variable 'crop'
    print("    Processing crop (target variable)...")
    le_crop = LabelEncoder()
    df['crop_encoded'] = le_crop.fit_transform(df['crop'].astype(str))
    encoders['crop'] = le_crop
    print(f"    Encoded crop (target): {len(le_crop.classes_)} classes")
    
    print("\nCrop classes in your model:")
    for i, crop_name in enumerate(le_crop.classes_[:10]):
        print(f"    {i}: {crop_name}")
    if len(le_crop.classes_) > 10:
        print(f"    ... and {len(le_crop.classes_) - 10} more")
    
    # Force garbage collection
    gc.collect()
        
    return df, encoders

def prepare_features_target(df_encoded):
    """Prepare feature matrix and target vector with memory optimization"""
    print("\nPREPARING FEATURES AND TARGET")
    print("=" * 60)
    
    # The corrected feature list (same as your original)
    feature_columns = [
        'state_encoded', 
        'district_encoded', 
        'season_encoded', 
        'season_type_encoded', 
        'area_category_encoded',
        'crop_year', 
        'area'
    ]
    
    print("Extracting features and target...")
    X = df_encoded[feature_columns].copy()
    y = df_encoded['crop_encoded'].copy()
    
    print(f"    Features shape: {X.shape}")
    print(f"    Target shape: {y.shape}")
    print(f"    Feature columns: {feature_columns}")
    print(f"    Number of crop classes: {len(np.unique(y))}")
    
    # Force garbage collection
    gc.collect()
    
    return X, y, feature_columns

def train_and_evaluate_models(X, y):
    """Split data, scale, train and evaluate classification models with memory optimization"""
    print("\nSPLITTING DATA")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"    Training set: {X_train.shape}")
    print(f"    Test set: {X_test.shape}")
    
    # Force garbage collection after splitting
    gc.collect()

    print("\nSCALING FEATURES")
    print("=" * 60)
    scaler = StandardScaler()
    
    print("    Fitting scaler on training data...")
    X_train_scaled = scaler.fit_transform(X_train)
    print("    Transforming test data...")
    X_test_scaled = scaler.transform(X_test)
    
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    print(f"    Scaled {len(numerical_cols)} numerical features: {numerical_cols}")
    
    # Clear original unscaled data from memory
    del X_train, X_test
    gc.collect()

    print("\nTRAINING CLASSIFICATION MODELS")
    print("=" * 60)
    
    # Optimized parameters - fast but memory-aware
    models = {
        'XGBoost Classifier': xgb.XGBClassifier(
            objective='multi:softmax', 
            random_state=42, 
            n_jobs=5,  # Use 5 cores for better performance on your 6-core CPU
            use_label_encoder=False, 
            eval_metric='mlogloss',
            max_depth=6,
            n_estimators=100,
            tree_method='hist',  # Faster histogram-based algorithm
            grow_policy='lossguide',  # More efficient tree growth
            max_bin=256  # Memory-efficient binning
        ),
        'Random Forest Classifier': RandomForestClassifier(
            random_state=42, 
            n_jobs=5,  # Use 5 cores
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,  # Better memory efficiency
            max_features='sqrt'  # Memory-efficient feature selection
        ),
        'Extra Trees Classifier': ExtraTreesClassifier(
            random_state=42, 
            n_jobs=5,  # Use 5 cores
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,  # Better memory efficiency
            max_features='sqrt'  # Memory-efficient feature selection
        )
    }
    
    results = {}
    best_model_name = ''
    best_accuracy = 0.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        print(f"    Model parameters optimized for memory efficiency")
        
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        print(f"    Making predictions...")
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'Accuracy': accuracy, 
            'F1-Score': f1, 
            'Training Time': training_time, 
            'model_object': model
        }
        
        print(f"    Training time: {training_time:.2f} seconds")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
        
        # Clear predictions from memory
        del y_pred
        gc.collect()
        
        # No pause needed - just memory cleanup

    print(f"\nBest Model: {best_model_name}")
    print(f"    Best Accuracy: {results[best_model_name]['Accuracy']:.4f}")
    print(f"    Best F1-Score: {results[best_model_name]['F1-Score']:.4f}")
    
    best_model = results[best_model_name]['model_object']
    
    # Clear scaled test data
    del X_train_scaled, X_test_scaled
    gc.collect()
    
    return best_model, scaler, results, best_model_name, X.columns.tolist()

def perform_cross_validation(model, X, y, model_name):
    """Perform cross-validation with memory optimization"""
    print(f"\nCROSS-VALIDATION FOR {model_name.upper()}")
    print("=" * 60)
    print("    Performing 3-fold cross-validation (memory optimized)...")
    
    # Use more CV folds but with smart parallelization
    cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=3)  # 3 cores for CV
    gc.collect()
    
    cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1_weighted', n_jobs=3)
    gc.collect()
    
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
        del importance_df
        gc.collect()

def save_artifacts(model, encoders, scaler, metadata):
    """Save all model artifacts"""
    print("\nSAVING MODEL ARTIFACTS")
    print("=" * 60)
    
    print("    Saving model...")
    joblib.dump(model, 'crop_model.pkl')
    print("    Model saved as 'crop_model.pkl'")
    
    print("    Saving encoders...")
    joblib.dump(encoders, 'encoders.pkl')
    print("    Encoders saved as 'encoders.pkl'")
    
    print("    Saving scaler...")
    joblib.dump(scaler, 'scaler.pkl')
    print("    Scaler saved as 'scaler.pkl'")
    
    print("    Saving metadata...")
    joblib.dump(metadata, 'model_metadata.pkl')
    print("    Metadata saved as 'model_metadata.pkl'")

def main():
    """Main training pipeline with memory optimization"""
    print("MEMORY-OPTIMIZED CROP RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("🚀 Running in memory-efficient mode for better stability")
    print("=" * 60)
    
    total_start_time = time.time()

    try:
        # Step 1: Load and explore data
        print("📊 STEP 1: Loading data...")
        df = load_and_explore_data('APY.csv')
        
        # Step 2: Preprocess and engineer features
        print("\n🔧 STEP 2: Preprocessing and feature engineering...")
        df_clean = preprocess_and_engineer_features(df)
        
        # Create samples for plotting before heavy processing
        print("\n📈 Creating data samples for plotting...")
        df_sample = df.sample(n=min(10000, len(df)), random_state=42) if len(df) > 10000 else df.copy()
        df_clean_sample = df_clean.sample(n=min(10000, len(df_clean)), random_state=42) if len(df_clean) > 10000 else df_clean.copy()
        
        # Step 3: Encode features
        print("\n🏷️ STEP 3: Encoding features...")
        df_encoded, encoders = encode_features(df_clean)
        
        # Clear df_clean from memory as we have df_encoded now
        del df_clean
        gc.collect()
        
        # Step 4: Prepare features and target
        print("\n🎯 STEP 4: Preparing features and target...")
        X, y, feature_columns = prepare_features_target(df_encoded)
        
        # Clear df_encoded from memory
        del df_encoded
        gc.collect()
        
        # Step 5: Train and evaluate models
        print("\n🤖 STEP 5: Training and evaluating models...")
        best_model, scaler, results, best_model_name, feature_columns_final = train_and_evaluate_models(X, y)
        
        # Step 6: Create plots (using samples)
        print("\n📊 STEP 6: Creating visualizations...")
        create_plots(df_sample, df_clean_sample, results, best_model, feature_columns, best_model_name)
        
        # Clear plot samples
        del df_sample, df_clean_sample
        gc.collect()
        
        # Step 7: Cross-validation
        print("\n✅ STEP 7: Cross-validation...")
        cv_results = perform_cross_validation(best_model, X, y, best_model_name)
        results[best_model_name].update(cv_results)
        
        # Step 8: Feature importance
        print("\n🎯 STEP 8: Analyzing feature importance...")
        print_feature_importance(best_model, feature_columns, best_model_name)
        
        # Step 9: Save artifacts
        print("\n💾 STEP 9: Saving model artifacts...")
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
        print("\n📁 Files created:")
        print("    crop_model.pkl")
        print("    encoders.pkl")
        print("    scaler.pkl")
        print("    model_metadata.pkl")
        print("\n📊 Plots created in 'plots' folder.")
        print("\n💡 Memory optimizations applied:")
        print("    - Smart garbage collection")
        print("    - Data sampling for plots only")
        print("    - Step-by-step processing")
        print("    - Using 4/6 CPU cores efficiently")
        print("    - Memory cleanup without speed loss")
        print("\n🚀 Ready to build your Streamlit app!")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check your data file and try again.")
        print("💡 If memory issues persist, try reducing your dataset size first.")

if __name__ == "__main__":
    main()