"""
Synchronize sensor data between different session state variables.

This tool ensures that data fetched from the sensor API is properly available
to functions that expect data in different session state variables.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

def sync_sensor_data() -> str:
    """
    Synchronize sensor data from latest_data to dataset in session state.
    
    This ensures that data fetched from the sensor API is properly available
    to functions that expect data in st.session_state.dataset.
    
    Returns:
        str: Status message about the synchronization
    """
    if 'latest_data' not in st.session_state or st.session_state.latest_data is None or st.session_state.latest_data.empty:
        return "No sensor data available to synchronize. Please fetch data first using fetch_latest_sensor_data."
    
    # Print debug info
    print("==== SYNC_SENSOR_DATA FUNCTION START ====")
    print(f"latest_data exists: {'latest_data' in st.session_state}")
    if 'latest_data' in st.session_state:
        print(f"latest_data shape: {st.session_state.latest_data.shape if not st.session_state.latest_data.empty else 'empty'}")
        print(f"latest_data columns: {st.session_state.latest_data.columns.tolist() if not st.session_state.latest_data.empty else []}")
    
    # Copy the data to ensure both references have the same data
    st.session_state.dataset = st.session_state.latest_data.copy()
    
    # Log information about the synchronized data
    rows = st.session_state.dataset.shape[0]
    cols = st.session_state.dataset.shape[1]
    
    return f"""
## Sensor Data Synchronized

**Source:** sensor.community API data
**Shape:** {rows} rows × {cols} columns
**Columns:** {', '.join(st.session_state.dataset.columns.tolist())}

The sensor data has been successfully synchronized and is now available for:
- Outlier removal
- Statistical analysis
- Visualization

You can now use functions like remove_outliers_iqr on this data.
"""
