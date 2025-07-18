"""
Utility function to load processed data from disk.

This module provides functionality to load cleaned datasets that have been 
processed by the outlier removal tool and saved to disk.
"""

import os
import pandas as pd
import streamlit as st
import glob
from typing import Optional, List


def load_processed_data(filename: Optional[str] = None, latest: bool = True) -> str:
    """
    Load a processed dataset from disk into Streamlit session state.
    
    Args:
        filename: Optional specific filename to load. If not provided and latest=True, 
                 loads the most recent processed file.
        latest: If True and filename is None, loads the most recent processed file.
    
    Returns:
        str: A report about the loaded data or an error message.
    """
    try:
        # Define the directory where processed datasets are stored
        data_dir = os.path.join(os.getcwd(), 'data', 'processed')
        
        # Check if the directory exists
        if not os.path.exists(data_dir):
            return "No processed data directory found. Process a dataset first using the remove_outliers tool."
        
        # If filename is provided, use it
        if filename:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
        # Otherwise, get the latest file if requested
        elif latest:
            files = glob.glob(os.path.join(data_dir, "cleaned_data_*.csv"))
            if not files:
                return "No processed data files found. Process a dataset first using the remove_outliers tool."
            filepath = max(files, key=os.path.getctime)  # Get the most recently created file
        else:
            # List all available files
            files = glob.glob(os.path.join(data_dir, "cleaned_data_*.csv"))
            if not files:
                return "No processed data files found. Process a dataset first using the remove_outliers tool."
            
            file_list = "\n".join([os.path.basename(f) for f in files])
            return f"Available processed data files:\n\n{file_list}\n\nSpecify a filename to load."
        
        # Load the data
        df = pd.read_csv(filepath)
        
        # Store in session state
        st.session_state.dataset = df
        st.session_state.processed_data_path = filepath
        
        # Generate report
        report = "## Processed Data Loaded\n\n"
        report += f"**File:** {os.path.basename(filepath)}\n"
        report += f"**Path:** {filepath}\n"
        report += f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        report += f"**Columns:** {', '.join(df.columns.tolist())}\n\n"
        
        # Show a preview of the data
        report += "**Data Preview:**\n\n"
        report += df.head(5).to_markdown()
        
        return report
    
    except Exception as e:
        import traceback
        st.error(f"Error loading processed data: {str(e)}")
        st.error(traceback.format_exc())
        return f"Error loading processed data: {str(e)}"
