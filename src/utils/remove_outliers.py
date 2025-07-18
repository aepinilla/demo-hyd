"""
Outlier detection and removal utilities.

This module provides functions for detecting and removing outliers from datasets
using various statistical methods.
"""

import pandas as pd
import numpy as np
import streamlit as st
import json
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
    Remove outliers from the dataset using the IQR method.
    
    Args:
        columns: Columns to check for outliers. Can be 'all' for all numeric columns,
                a single column name, or a list of column names.
        iqr_multiplier: Multiplier for IQR to determine outlier boundaries.
        drop_outliers: Whether to drop outliers or replace them.
        fill_method: Method to fill outliers if drop_outliers is False.
                    Options: 'mean', 'median', 'mode', 'ffill', 'bfill'.
    
    Returns:
        str: A report on the outlier removal process.
    """
    
    # Check if dataset exists in session state
    if 'dataset' not in st.session_state or st.session_state.dataset is None or st.session_state.dataset.empty:
        return "Error: No dataset found in session state. Please load or create a dataset first."
    
    # Make a copy of the dataset to avoid modifying the original
    df = st.session_state.dataset.copy()
    original_shape = df.shape
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Define columns to exclude from outlier detection
    excluded_columns = ["location_id", "sensor_id", "latitude", "longitude", "lat", "lon", "lng", "gps_lat", "gps_lon", "gps_longitude", "gps_latitude"]
    numeric_cols = [col for col in numeric_cols if col not in excluded_columns]
    
    if not numeric_cols:
        return "Error: No numeric columns found in the dataset. Outlier removal requires numeric data."
    
    # Process columns parameter to determine target columns
    target_columns = []
    
    # Handle JSON format from LangChain
    if isinstance(columns, str) and '{' in columns:
        import re
        column_match = re.search(r'"columns"\s*:\s*"([^"]+)"', columns)
        if column_match:
            columns = column_match.group(1)
    
    # Process the columns parameter
    if columns == "all" or (isinstance(columns, str) and columns.strip().lower() == "all"):
        target_columns = numeric_cols
    elif isinstance(columns, str) and ',' in columns:
        # Handle comma-separated list
        column_list = [col.strip() for col in columns.split(',')]
        for col in column_list:
            if col in df.columns and col in numeric_cols:
                target_columns.append(col)
    elif isinstance(columns, str):
        # Handle single column name
        if columns in df.columns and columns in numeric_cols:
            target_columns.append(columns)
    elif isinstance(columns, list):
        # Handle list of column names
        for col in columns:
            if col in df.columns and col in numeric_cols:
                target_columns.append(col)
    else:
        return f"Error: Invalid columns parameter: {columns}. Use 'all' or specify column names."
    
    # Check if we have valid columns to process
    if not target_columns:
        return "Error: No valid numeric columns found for outlier removal. Please check your column names."
    
    # Track outliers per column
    outliers_info = {}
    
    # Create a mask to track rows with outliers
    outlier_mask = pd.Series(False, index=df.index)
    
    # Process each column
    for col in target_columns:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        # Identify outliers in this column
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = col_outliers.sum()
        
        # Store outlier info for reporting
        outliers_info[col] = {
            "count": outlier_count,
            "percentage": (outlier_count / len(df)) * 100,
            "range": [round(df[col].min(), 2), round(df[col].max(), 2)],
            "boundaries": [round(lower_bound, 2), round(upper_bound, 2)]
        }
        
        # Update the overall outlier mask
        outlier_mask = outlier_mask | col_outliers
    
    # Handle outliers
    if drop_outliers:
        # Drop rows with outliers in any of the target columns
        df = df[~outlier_mask]
    else:
        # Replace outliers with specified method for each column
        for col in target_columns:
            # Calculate IQR again for this column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Create mask for this column's outliers
            col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            if fill_method == "mean":
                fill_value = df[col].mean()
                df.loc[col_mask, col] = fill_value
            elif fill_method == "median":
                fill_value = df[col].median()
                df.loc[col_mask, col] = fill_value
            elif fill_method == "mode":
                fill_value = df[col].mode()[0]
                df.loc[col_mask, col] = fill_value
            elif fill_method in ["ffill", "bfill"]:
                # For forward/backward fill, we need to mark outliers as NaN first
                df.loc[col_mask, col] = np.nan
                df[col] = df[col].fillna(method=fill_method)
            else:
                # Default to median if method not recognized
                fill_value = df[col].median()
                df.loc[col_mask, col] = fill_value
    
    # Save the cleaned dataset to session state
    st.session_state.dataset = df
    
    # Save the cleaned data to a file
    try:
        os.makedirs("data/processed", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/processed/cleaned_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
    except Exception as e:
        filename = "Could not save file"
    
    # Generate report
    report = "## Outlier Removal Report\n\n"
    report += f"**Method:** IQR (Interquartile Range) with multiplier {iqr_multiplier}\n\n"
    action_taken = 'Dropped outliers' if drop_outliers else f'Replaced outliers using {fill_method}'
    report += f"**Action:** {action_taken}\n\n"
    # Add note about excluded columns
    excluded_in_dataset = [col for col in excluded_columns if col in df.columns]
    if excluded_in_dataset:
        report += f"**Note:** Excluded columns from outlier detection: {', '.join(excluded_in_dataset)}\n\n"
    report += f"**Original dataset shape:** {original_shape[0]} rows × {original_shape[1]} columns\n"
    report += f"**New dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns\n"
    report += f"**Rows removed:** {original_shape[0] - df.shape[0]} ({((original_shape[0] - df.shape[0]) / original_shape[0]) * 100:.2f}%)\n\n"
    report += f"**Data saved to:** {filename}\n\n"
    
    # Add outlier information per column
    report += "**Outliers per column:**\n\n"
    for col, info in outliers_info.items():
        report += f"- **{col}**: {info['count']} outliers ({info['percentage']:.2f}%)\n"
        report += f"  - Range: [{info['range'][0]}, {info['range'][1]}]\n"
        report += f"  - Outlier boundaries: [{info['boundaries'][0]}, {info['boundaries'][1]}]\n\n"
    
    return report
