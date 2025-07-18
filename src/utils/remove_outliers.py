"""
Outlier detection and removal utilities.

This module provides functions for detecting and removing outliers from datasets
using various statistical methods.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
from datetime import datetime
import re
from typing import List, Optional, Union, Tuple


def clean_column_name(column_name: str) -> str:
    """
    Clean column name by trimming whitespace and normalizing it.
    
    Args:
        column_name (str): The column name to clean
        
    Returns:
        str: Cleaned column name
    """
    if not isinstance(column_name, str):
        return str(column_name)
    
    # Trim whitespace
    cleaned = column_name.strip()
    
    return cleaned

def find_matching_column(column_name: str, available_columns: List[str]) -> Optional[str]:
    """
    Find a matching column name in the available columns list.
    Handles cases with extra spaces or slight variations.
    
    Args:
        column_name (str): The column name to find (or JSON object containing column name)
        available_columns (List[str]): List of available column names
        
    Returns:
        str: Matched column name or None if no match found
    """
    print(f"DEBUG - Input column_name: {repr(column_name)}")
    
    # Handle JSON input
    import json
    if isinstance(column_name, str) and column_name.strip().startswith('{'):
        try:
            # Try to parse as JSON
            params = json.loads(column_name)
            # Extract column name from various possible formats
            if isinstance(params, dict):
                if "column" in params:
                    column_name = params["column"]
                elif "columns" in params and isinstance(params["columns"], list) and params["columns"]:
                    column_name = params["columns"][0]
                print(f"DEBUG - Extracted column_name from JSON: {repr(column_name)}")
        except Exception as e:
            # If JSON parsing fails, continue with original string
            print(f"DEBUG - JSON parsing failed: {str(e)}")
            pass
    
    # Clean the input column name
    clean_name = clean_column_name(column_name)
    
    # Direct match
    if clean_name in available_columns:
        return clean_name
    
    # Try with space variations
    for col in available_columns:
        # Compare without spaces
        if clean_name.replace(' ', '') == col.replace(' ', ''):
            return col
    
    # Try with case insensitive match
    for col in available_columns:
        if clean_name.lower() == col.lower():
            return col
    
    # No match found
    return None


def remove_outliers_iqr(
    columns: Union[List[str], str] = "all",
    iqr_multiplier: float = 1.5,
    drop_outliers: bool = True,
    fill_method: Optional[str] = None
) -> str:
    """
    Remove outliers from the dataset using the Interquartile Range (IQR) method.
    
    Args:
        columns (Union[List[str], str]): List of columns or 'all' for all numeric columns
        iqr_multiplier (float): Multiplier for IQR to determine outlier threshold
        drop_outliers (bool): Whether to drop outliers (True) or replace them (False)
        fill_method (Optional[str]): Method to fill outliers if drop_outliers is False.
                                    Options: 'mean', 'median', 'mode', 'nearest', None
    
    Returns:
        str: Information about the outlier removal process
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        import debugpy; debugpy.breakpoint()
        # Get the dataset
        df = st.session_state.dataset.copy()
        original_shape = df.shape
        
        # Get all numeric columns - use a broader approach to detect numeric columns
        numeric_cols = []
        
        # First try pandas' built-in detection
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return "Error: The dataset has no numeric columns to remove outliers from. Please ensure your dataset contains numeric data."
        
        # Print available columns for debugging
        st.info(f"Available columns in dataset: {', '.join(df.columns.tolist())}")
        st.info(f"Detected numeric columns: {', '.join(numeric_cols)}")
        
        # Determine which columns to process
        valid_columns = []
        
        # Default to all numeric columns
        if not numeric_cols:
            return "Error: No numeric columns found in the dataset. Outlier removal requires numeric data."
            
        # Handle the 'all' case or empty input - use all numeric columns
        if not columns or (isinstance(columns, str) and columns.lower() == "all"):
            valid_columns = numeric_cols
            st.info(f"Processing all numeric columns: {', '.join(valid_columns)}")
            
        # Handle LangChain JSON input format
        elif isinstance(columns, str) and '{"columns":' in columns:
            import re
            
            # Extract the columns value using regex
            match = re.search(r'"columns"\s*:\s*"([^"]+)"', columns)
            
            if match:
                columns_value = match.group(1)
                
                # Process the extracted value
                if columns_value.lower() == "all":
                    valid_columns = numeric_cols
                else:
                    # Split by comma if it's a comma-separated string
                    user_columns = [col.strip() for col in columns_value.split(',')]
                    valid_columns = [col for col in user_columns if col in numeric_cols]
            else:
                # Fallback to all numeric columns
                valid_columns = numeric_cols
                
        # Handle direct column specification
        else:
            # Parse the user's column input
            user_columns = []
            if isinstance(columns, str):
                if ',' in columns:
                    user_columns = [col.strip() for col in columns.split(',')]
                else:
                    user_columns = [columns.strip()]
            elif isinstance(columns, list):
                user_columns = columns
            else:
                user_columns = [str(columns)]
                
            # Filter to only include numeric columns
            valid_columns = [col for col in user_columns if col in numeric_cols]
            
        # Show which columns will be processed
        if valid_columns:
            st.info(f"Processing columns: {', '.join(valid_columns)}")
        else:
            return "Error: No valid numeric columns specified for outlier removal. Please check column names."
        
        # Track outliers per column
        outliers_info = {}
        
        # Process each column
        for col in valid_columns:
            # Calculate Q1, Q3, and IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define outlier boundaries
            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)
            
            # Identify outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outliers_count = len(outliers)
            
            # Store outlier information
            outliers_info[col] = {
                'count': outliers_count,
                'percentage': (outliers_count / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': df[col].min(),
                'max_value': df[col].max()
            }
            
            # Handle outliers
            if drop_outliers:
                # Drop rows with outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif fill_method:
                # Replace outliers with specified method
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
                if fill_method == 'mean':
                    fill_value = df[col].mean()
                elif fill_method == 'median':
                    fill_value = df[col].median()
                elif fill_method == 'mode':
                    fill_value = df[col].mode()[0]
                elif fill_method == 'nearest':
                    # Replace lower outliers with lower bound, upper outliers with upper bound
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    continue
                else:
                    return f"Error: Invalid fill method '{fill_method}'. Use 'mean', 'median', 'mode', or 'nearest'."
                
                df.loc[mask, col] = fill_value
        
        # Create directory for saving processed datasets if it doesn't exist
        data_dir = os.path.join(os.getcwd(), 'data', 'processed')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cleaned_data_{timestamp}.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Save the processed data to disk
        df.to_csv(filepath, index=False)
        
        # Store the filepath in session state for future reference
        st.session_state.processed_data_path = filepath
        
        # Handle potential pyarrow serialization issues by converting object columns to string
        # This prevents errors when Streamlit tries to convert DataFrame to Arrow table
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            df[col] = df[col].astype(str)
            
        # Update the dataset in session state
        st.session_state.dataset = df
        
        # Generate report
        report = "## Outlier Removal Report\n\n"
        report += f"**Method:** IQR (Interquartile Range) with multiplier {iqr_multiplier}\n\n"
        report += f"**Action:** {'Dropped outliers' if drop_outliers else f'Replaced outliers using {fill_method}'}\n\n"
        report += f"**Original dataset shape:** {original_shape[0]} rows × {original_shape[1]} columns\n"
        report += f"**New dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns\n"
        report += f"**Rows removed:** {original_shape[0] - df.shape[0]} ({((original_shape[0] - df.shape[0]) / original_shape[0]) * 100:.2f}%)\n\n"
        report += f"**Data saved to:** {filepath}\n\n"
        
        report += "**Outliers per column:**\n\n"
        for col, info in outliers_info.items():
            report += f"- **{col}**: {info['count']} outliers ({info['percentage']:.2f}%)\n"
            report += f"  - Range: [{info['min_value']:.2f}, {info['max_value']:.2f}]\n"
            report += f"  - Outlier boundaries: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]\n"
        
        return report
    
    except Exception as e:
        import traceback
        st.error(f"Error removing outliers: {str(e)}")
        st.error(traceback.format_exc())
        return f"Error removing outliers: {str(e)}"
