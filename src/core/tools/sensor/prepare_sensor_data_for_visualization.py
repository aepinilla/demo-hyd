import streamlit as st
import pandas as pd
from datetime import datetime

def prepare_sensor_data_for_visualization() -> str:
    """
    Prepare the latest fetched sensor data for visualization with the plotting tools.
    This function makes the sensor data available to the visualization tools without
    requiring the user to save and load a CSV file.
    
    Returns:
        str: Information about the prepared dataset
    """
    
    # Check if we have sensor data in session state
    if 'latest_data' in st.session_state and not st.session_state.latest_data.empty:
        # Get the data and make a copy to avoid modifying the original
        df = st.session_state.latest_data.copy()
        
        # Verify that we have a timestamp column that can be used for time series visualization
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            st.warning("Converting timestamp column to datetime format...")
            try:
                # Try to convert the timestamp column to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Check if conversion was successful (no NaT values)
                if df['timestamp'].isna().all():
                    st.warning("Could not convert timestamp column to datetime format. Some visualizations may not work properly.")
                
                # Drop rows with NaT values if any
                if df['timestamp'].isna().any():
                    original_len = len(df)
                    df = df.dropna(subset=['timestamp'])
                    st.warning(f"Dropped {original_len - len(df)} rows with invalid timestamps.")
                    
            except Exception as e:
                st.warning(f"Error converting timestamp column: {str(e)}")
        
        # Rename latitude and longitude columns to lat and lon for compatibility with plot_sensor_map
        if 'latitude' in df.columns and 'lat' not in df.columns:
            df = df.rename(columns={'latitude': 'lat'})
        if 'longitude' in df.columns and 'lon' not in df.columns:
            df = df.rename(columns={'longitude': 'lon'})
            
        # Store the data in both session state variables for consistency
        # This ensures both sensor-specific and standard visualization tools use the same data
        st.session_state.latest_data = df
        st.session_state.dataset = df
        
        # Update the data fetch time
        st.session_state.data_fetch_time = datetime.now()
        
        # Get information about the dataset
        return f"""
## Sensor Data Prepared for Visualization

**Source:** sensor.community API
**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Columns:** {', '.join(df.columns.tolist())}

**Column Types:**
```
{df.dtypes.to_string()}
```

**Sample Data:**
```
{df.head(5).to_string()}
```

You can now use visualization tools like plot_scatter, plot_histogram, etc. directly on this data.
        """
    else:
        return "Error: No sensor data available. Please fetch sensor data first using fetch_latest_sensor_data."


