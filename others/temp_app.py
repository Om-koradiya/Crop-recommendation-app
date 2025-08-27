# Load state-district mapping
@st.cache_data
def load_state_district_mapping():
    try:
        # Read the CSV file
        df = pd.read_csv('APY.csv')
        
        # Fix the column names
        df.columns = [col.strip() for col in df.columns]
        
        # Now we can safely use the cleaned column names
        df['State'] = df['State'].str.strip()
        df['District'] = df['District'].str.strip()
        
        # Create state-district mapping
        state_districts = df.groupby('State')['District'].unique().to_dict()
        
        # Clean and sort districts for each state
        mapping = {}
        for state, districts in state_districts.items():
            clean_districts = sorted(list(set([d.strip() for d in districts if isinstance(d, str)])))
            if clean_districts:  # Only add if there are valid districts
                mapping[state.strip()] = clean_districts
        return mapping
    except Exception as e:
        st.error(f"Error loading state-district mapping: {str(e)}")
        return {}
