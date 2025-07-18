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
from typing import List, Optional, Union, Tuple


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
        # Get the dataset
        df = st.session_state.dataset.copy()
        original_shape = df.shape
        
        # Get all numeric columns - use a broader approach to detect numeric columns
        numeric_cols = []
        
        # First try pandas' built-in detection
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # If that fails, try to convert columns to numeric and see which ones succeed
        if not numeric_cols:
            st.warning("No numeric columns detected automatically. Attempting to convert columns to numeric...")
            for col in df.columns:
                try:
                    # Try to convert to numeric, coercing errors to NaN
                    pd.to_numeric(df[col], errors='coerce')
                    # If we get here without error and there are non-NaN values, it's numeric
                    if not pd.to_numeric(df[col], errors='coerce').isna().all():
                        numeric_cols.append(col)
                except:
                    continue
        
        if not numeric_cols:
            return "Error: The dataset has no numeric columns to remove outliers from. Please ensure your dataset contains numeric data."
        
        # Convert input to list of columns
        if isinstance(columns, str):
            # Handle 'all' case - use all numeric columns
            if columns.lower() == "all":
                columns = numeric_cols
            else:
                # Split by comma and clean each column name
                columns = [col.strip() for col in columns.split(',')]
        
        # Validate columns
        valid_columns = []
        invalid_columns = []
        
        # Print available columns for debugging
        st.info(f"Available columns in dataset: {', '.join(df.columns.tolist())}")
        st.info(f"Detected numeric columns: {', '.join(numeric_cols)}")
        
        for col in columns:
            if col in numeric_cols:
                valid_columns.append(col)
            elif col in df.columns:
                # Column exists but is not numeric - try to convert it
                try:
                    # Try to convert to numeric, coercing errors to NaN
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    # If there are valid numeric values, use this column
                    if not numeric_series.isna().all():
                        # Replace the column with the numeric version
                        df[col] = numeric_series
                        valid_columns.append(col)
                        st.info(f"Successfully converted column '{col}' to numeric type.")
                    else:
                        invalid_columns.append(f"{col} (not convertible to numeric)")
                except Exception as e:
                    invalid_columns.append(f"{col} (conversion error: {str(e)})")
            else:
                invalid_columns.append(f"{col} (not found)")
        
        if invalid_columns:
            st.warning(f"Columns not processed: {', '.join(invalid_columns)}")
        
        if not valid_columns:
            return "Error: No valid numeric columns specified for outlier removal. Please check column names and ensure they contain numeric data."
        
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
