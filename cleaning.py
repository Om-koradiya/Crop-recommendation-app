import pandas as pd
import numpy as np
import os

def clean_agricultural_data():
    """
    Complete data cleaning for agricultural CSV file in the same folder
    """
    # Check if the CSV file exists
    csv_file = "APY.csv"
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' not found in the current directory!")
        return None
    
    print(f"Reading data from: {csv_file}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Display initial info
    print("\n=== INITIAL DATA INFO ===")
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    print("\nFirst few rows:")
    print(df.head())
    
    # Standardize column names (remove special characters, make lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    print(f"\nStandardized column names: {df.columns.tolist()}")
    
    # Check if we need to split combined columns (like 'Crop_Year Season' from your preview)
    for col in df.columns:
        if any(x in col for x in ['year', 'season']):
            # Check if this column might contain combined data
            sample_value = df[col].iloc[0] if len(df) > 0 else ''
            if isinstance(sample_value, str) and len(str(sample_value).split()) >= 2:
                print(f"Column '{col}' might contain combined data: '{sample_value}'")
    
    # Handle missing values - based on your preview data structure
    # Assuming columns: state, district, crop, crop_year, season, area, production, yield
    
    # Fix the production/yield relationship
    if 'production' in df.columns and 'yield' in df.columns and 'area' in df.columns:
        # If production is missing but area and yield exist, calculate production
        missing_production = df['production'].isnull() & df['yield'].notnull() & df['area'].notnull()
        df.loc[missing_production, 'production'] = df.loc[missing_production, 'area'] * df.loc[missing_production, 'yield']
        
        # If yield is missing but production and area exist, calculate yield
        missing_yield = df['yield'].isnull() & df['production'].notnull() & df['area'].notnull()
        df.loc[missing_yield, 'yield'] = df.loc[missing_yield, 'production'] / df.loc[missing_yield, 'area']
    
    # Fill remaining missing numerical values with 0
    numerical_cols = ['area', 'production', 'yield']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Clean text columns (remove extra spaces)
    text_cols = ['state', 'district', 'crop', 'season']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Standardize season names based on your preview
    if 'season' in df.columns:
        season_mapping = {
            'whole yee': 'whole year',
            'kharif': 'kharif', 
            'rabi': 'rabi',
            'autumn': 'autumn',
            'summer': 'summer',
            'whole year': 'whole year'
        }
        df['season'] = df['season'].str.lower().replace(season_mapping)
    
    # Ensure year is integer if there's a year column
    year_cols = [col for col in df.columns if 'year' in col]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Remove duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(f"\nRemoved {initial_count - final_count} duplicate rows")
    
    # Data Validation
    print("\n=== DATA VALIDATION ===")
    
    # Check for negative values
    numerical_cols = ['area', 'production', 'yield']
    for col in numerical_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            print(f"Negative values in {col}: {negative_count}")
    
    # Check for unrealistic values
    if 'yield' in df.columns:
        # Assuming yield should typically be between 0.01 and 20 for most crops
        unrealistic_yield = ((df['yield'] > 20) | (df['yield'] < 0.01)).sum()
        print(f"Potential unrealistic yield values: {unrealistic_yield}")
    
    # Basic Exploration
    print("\n=== DATA EXPLORATION ===")
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Count by categorical fields
    categorical_cols = ['state', 'district', 'crop', 'season']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\nUnique {col}s: {df[col].nunique()}")
            print(f"Top 5 {col}s by count:")
            print(df[col].value_counts().head())
    
    # Save the cleaned data
    output_file = 'cleaned_agricultural_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")
    
    # Display final info
    print("\n=== FINAL DATA INFO ===")
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

# Run the cleaning process
if __name__ == "__main__":
    cleaned_df = clean_agricultural_data()
    
    if cleaned_df is not None:
        print("\nData cleaning completed successfully!")
        print("\nFirst few rows of cleaned data:")
        print(cleaned_df.head())
    else:
        print("Data cleaning failed!")