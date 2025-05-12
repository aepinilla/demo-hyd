"""
Utility functions for handling dataframes in Streamlit.
"""

import pandas as pd
import streamlit as st

def prepare_dataframe_for_streamlit(df):
    """
    Prepare a dataframe for display in Streamlit by converting object columns to strings.
    This helps avoid PyArrow serialization errors.
    
    Args:
        df: The pandas DataFrame to prepare
        
    Returns:
        A DataFrame with object columns converted to strings
    """
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert object columns to strings to avoid PyArrow serialization issues
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            
    return df

# Add an alias for the function to maintain compatibility    
def ensure_string_columns(df):
    """
    Alias for prepare_dataframe_for_streamlit.
    Converts object columns to strings to avoid PyArrow serialization errors.
    
    Args:
        df: The pandas DataFrame to prepare
        
    Returns:
        A DataFrame with object columns converted to strings
    """
    return prepare_dataframe_for_streamlit(df)
