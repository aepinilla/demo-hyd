"""
Data bridge utilities to ensure proper data flow between different components.

This module provides functions to ensure data is properly transferred between
different components of the application, especially between API functions and
data processing functions.
"""

import streamlit as st
import pandas as pd
import logging

def sync_sensor_data_to_dataset():
    """
    Synchronize sensor data from latest_data to dataset in session state.
    
    This ensures that data fetched from the sensor API is properly available
    to functions that expect data in st.session_state.dataset.
    
    Returns:
        str: Status message about the synchronization
    """
    if 'latest_data' not in st.session_state or st.session_state.latest_data is None or st.session_state.latest_data.empty:
        return "No sensor data available to synchronize. Please fetch data first."
    
    # Copy the data to ensure both references have the same data
    st.session_state.dataset = st.session_state.latest_data.copy()
    
    # Log information about the synchronized data
    rows = st.session_state.dataset.shape[0]
    cols = st.session_state.dataset.shape[1]
    
    return f"Successfully synchronized sensor data to dataset. {rows} rows × {cols} columns available for processing."
