"""
Utility for automatically loading the most recent processed data on application startup.

This module checks for processed data files and loads the most recent one
into the Streamlit session state when the application starts.
"""

import os
import glob
import pandas as pd
import streamlit as st
import logging
from typing import Optional

def auto_load_processed_data() -> Optional[str]:
    """
    Automatically load the most recent processed dataset if available.
    
    This function is called when the application starts. It checks for
    processed data files in the data/processed directory and loads the
    most recent one into the Streamlit session state.
    
    Returns:
        Optional[str]: A message about the loaded data or None if no data was loaded.
    """
    try:
        # Define the directory where processed datasets are stored
        data_dir = os.path.join(os.getcwd(), 'data', 'processed')
        
        # Check if the directory exists
        if not os.path.exists(data_dir):
            return None
        
        # Get all processed data files
        files = glob.glob(os.path.join(data_dir, "cleaned_data_*.csv"))
        if not files:
            return None
        
        # Get the most recent file
        filepath = max(files, key=os.path.getctime)
        
        # Load the data
        df = pd.read_csv(filepath)
        
        # Store in session state
        st.session_state.dataset = df
        st.session_state.processed_data_path = filepath
        
        # Generate message
        message = f"Automatically loaded most recent processed data: {os.path.basename(filepath)}"
        logging.info(message)
        
        return message
    
    except Exception as e:
        logging.error(f"Error auto-loading processed data: {str(e)}")
        return None
