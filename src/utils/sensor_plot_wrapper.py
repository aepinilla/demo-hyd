"""
Wrapper functions for sensor data visualization.

This module provides wrapper functions that make it easier for the LangChain agent
to use the time series visualization tools with the sensor data.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Union

from src.utils.time_series import plot_time_series, compare_time_periods

def plot_sensor_time_series(df_or_data_type: Union[pd.DataFrame, str, Dict[str, Any]],
                           value_column: Optional[str] = None,
                           time_column: str = 'timestamp',
                           group_by: Optional[str] = None,
                           title: str = 'Time Series Plot',
                           resample_freq: Optional[str] = None) -> None:
    """
    Wrapper for plot_time_series that handles different input formats.
    
    Args:
        df_or_data_type: Either a DataFrame, a string with a data type, or a dict with parameters
        value_column: Column containing the values to plot
        time_column: Column containing the timestamps
        group_by: Optional column to group by
        title: Plot title
        resample_freq: Optional frequency to resample data
    """
    # Handle the case where df is a dictionary
    if isinstance(df_or_data_type, dict):
        # Extract parameters from the dictionary
        params = df_or_data_type
        
        # If 'df' is in the dictionary, use it as the DataFrame
        if 'df' in params:
            df_param = params.get('df')
            
            # Check if df_param is a string (e.g., "latest_sensor_data")
            if isinstance(df_param, str) and 'latest_sensor_data' in df_param:
                # Use the most recently fetched data
                if 'latest_data' in st.session_state:
                    df = st.session_state.latest_data
                else:
                    st.error("No sensor data has been fetched yet. Please fetch data first.")
                    return
            else:
                # Assume df_param is the actual DataFrame
                df = df_param
        else:
            # No DataFrame provided, check if we have latest data in session state
            if 'latest_data' in st.session_state:
                df = st.session_state.latest_data
            else:
                st.error("No sensor data has been fetched yet. Please fetch data first.")
                return
        
        # Extract other parameters if provided
        value_column = params.get('value_column', value_column)
        time_column = params.get('time_column', time_column)
        group_by = params.get('group_by', group_by)
        title = params.get('title', title)
        resample_freq = params.get('resample_freq', resample_freq)
    
    # Handle the case where df is a string (e.g., a sensor type)
    elif isinstance(df_or_data_type, str):
        # Check if we have latest data in session state
        if 'latest_data' in st.session_state:
            df = st.session_state.latest_data
            
            # If the string is a sensor type, filter the data
            if df_or_data_type in ['SDS011', 'DHT22', 'BME280']:
                df = df[df['sensor_type'] == df_or_data_type]
                
                # If no value_column is provided or it's just 'value', try to filter by value_type
                if value_column is None or value_column == 'value':
                    # For SDS011, we need to pick a specific value_type (P1 or P2)
                    if df_or_data_type == 'SDS011':
                        # Get a subset of data for P1 (PM10)
                        p1_df = df[df['value_type'] == 'P1']
                        if not p1_df.empty:
                            df = p1_df
                            value_column = 'value'  # Use the 'value' column for the actual readings
                            print(f"Using P1 (PM10) values for SDS011 sensor")
                        else:
                            # Try P2 if P1 is not available
                            p2_df = df[df['value_type'] == 'P2']
                            if not p2_df.empty:
                                df = p2_df
                                value_column = 'value'
                                print(f"Using P2 (PM2.5) values for SDS011 sensor")
                    elif df_or_data_type in ['DHT22', 'BME280']:
                        # Get a subset of data for temperature
                        temp_df = df[df['value_type'] == 'temperature']
                        if not temp_df.empty:
                            df = temp_df
                            value_column = 'value'  # Use the 'value' column for the actual readings
                            print(f"Using temperature values for {df_or_data_type} sensor")
                        else:
                            # Try humidity if temperature is not available
                            humidity_df = df[df['value_type'] == 'humidity']
                            if not humidity_df.empty:
                                df = humidity_df
                                value_column = 'value'
                                print(f"Using humidity values for {df_or_data_type} sensor")
        else:
            st.error("No sensor data has been fetched yet. Please fetch data first.")
            return
    
    # Handle the case where df is already a DataFrame
    else:
        df = df_or_data_type
    
    # Ensure we have a value column
    if value_column is None:
        st.error("No value column specified for plotting.")
        return
    
    # Call the actual plotting function
    plot_time_series(df, value_column, time_column, group_by, title, resample_freq)

def compare_sensor_time_periods(df_or_data_type: Union[pd.DataFrame, str, Dict[str, Any]],
                               value_column: Optional[str] = None,
                               time_column: str = 'timestamp',
                               period1: str = 'morning',
                               period2: str = 'evening',
                               title: str = 'Time Period Comparison') -> None:
    """
    Wrapper for compare_time_periods that handles different input formats.
    
    Args:
        df_or_data_type: Either a DataFrame, a string with a data type, or a dict with parameters
        value_column: Column containing the values to compare
        time_column: Column containing the timestamps
        period1: Name of the first period
        period2: Name of the second period
        title: Plot title
    """
    # Handle the case where df is a dictionary
    if isinstance(df_or_data_type, dict):
        # Extract parameters from the dictionary
        params = df_or_data_type
        
        # If 'df' is in the dictionary, use it as the DataFrame
        if 'df' in params:
            df_param = params.get('df')
            
            # Check if df_param is a string (e.g., "latest_sensor_data")
            if isinstance(df_param, str) and 'latest_sensor_data' in df_param:
                # Use the most recently fetched data
                if 'latest_data' in st.session_state:
                    df = st.session_state.latest_data
                else:
                    st.error("No sensor data has been fetched yet. Please fetch data first.")
                    return
            else:
                # Assume df_param is the actual DataFrame
                df = df_param
        else:
            # No DataFrame provided, check if we have latest data in session state
            if 'latest_data' in st.session_state:
                df = st.session_state.latest_data
            else:
                st.error("No sensor data has been fetched yet. Please fetch data first.")
                return
        
        # Extract other parameters if provided
        value_column = params.get('value_column', value_column)
        time_column = params.get('time_column', time_column)
        period1 = params.get('period1', period1)
        period2 = params.get('period2', period2)
        title = params.get('title', title)
    
    # Handle the case where df is a string (e.g., a sensor type)
    elif isinstance(df_or_data_type, str):
        # Check if we have latest data in session state
        if 'latest_data' in st.session_state:
            df = st.session_state.latest_data
            
            # If the string is a sensor type, filter the data
            if df_or_data_type in ['SDS011', 'DHT22', 'BME280']:
                df = df[df['sensor_type'] == df_or_data_type]
                
                # If no value_column is provided or it's just 'value', try to filter by value_type
                if value_column is None or value_column == 'value':
                    # For SDS011, we need to pick a specific value_type (P1 or P2)
                    if df_or_data_type == 'SDS011':
                        # Get a subset of data for P1 (PM10)
                        p1_df = df[df['value_type'] == 'P1']
                        if not p1_df.empty:
                            df = p1_df
                            value_column = 'value'  # Use the 'value' column for the actual readings
                            print(f"Using P1 (PM10) values for SDS011 sensor")
                        else:
                            # Try P2 if P1 is not available
                            p2_df = df[df['value_type'] == 'P2']
                            if not p2_df.empty:
                                df = p2_df
                                value_column = 'value'
                                print(f"Using P2 (PM2.5) values for SDS011 sensor")
                    elif df_or_data_type in ['DHT22', 'BME280']:
                        # Get a subset of data for temperature
                        temp_df = df[df['value_type'] == 'temperature']
                        if not temp_df.empty:
                            df = temp_df
                            value_column = 'value'  # Use the 'value' column for the actual readings
                            print(f"Using temperature values for {df_or_data_type} sensor")
                        else:
                            # Try humidity if temperature is not available
                            humidity_df = df[df['value_type'] == 'humidity']
                            if not humidity_df.empty:
                                df = humidity_df
                                value_column = 'value'
                                print(f"Using humidity values for {df_or_data_type} sensor")
        else:
            st.error("No sensor data has been fetched yet. Please fetch data first.")
            return
    
    # Handle the case where df is already a DataFrame
    else:
        df = df_or_data_type
    
    # Ensure we have a value column
    if value_column is None:
        st.error("No value column specified for comparison.")
        return
    
    # Call the actual comparison function
    compare_time_periods(df, value_column, time_column, period1, period2, title)
