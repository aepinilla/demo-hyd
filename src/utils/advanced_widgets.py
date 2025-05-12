"""
Advanced widgets for interactive data visualization in Streamlit.
"""

import streamlit as st
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

def create_time_range_selector(
    default_time_period: str = "24h",
    key_prefix: str = "time_range"
) -> Dict[str, Any]:
    """
    Create an advanced time range selector widget with multiple options.
    
    Args:
        default_time_period: Default time period selection ('24h', '7d', '30d', 'custom')
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
        
    Returns:
        Dict containing the selected time range parameters
    """
    time_period_options = {
        "Last 24 Hours": "24h",
        "Last 7 Days": "7d", 
        "Last 30 Days": "30d",
        "Custom Range": "custom"
    }
    
    # Create columns for time period selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_period = st.selectbox(
            "Time Period", 
            options=list(time_period_options.keys()),
            key=f"{key_prefix}_period"
        )
    
    # Initialize result dictionary
    result = {
        "time_period": time_period_options[selected_period],
        "start_time": None,
        "end_time": None,
        "custom_range": False
    }
    
    # If custom range is selected, show date pickers
    if selected_period == "Custom Range":
        with col2:
            date_cols = st.columns(2)
            with date_cols[0]:
                start_date = st.date_input(
                    "Start Date", 
                    datetime.datetime.now() - datetime.timedelta(days=7),
                    key=f"{key_prefix}_start"
                )
            with date_cols[1]:
                end_date = st.date_input(
                    "End Date", 
                    datetime.datetime.now(),
                    key=f"{key_prefix}_end"
                )
            
            result["start_time"] = datetime.datetime.combine(start_date, datetime.time.min)
            result["end_time"] = datetime.datetime.combine(end_date, datetime.time.max)
            result["custom_range"] = True
            
    return result

def create_sensor_filter_widget(
    all_sensors: List[str],
    key_prefix: str = "sensor_filter"
) -> List[str]:
    """
    Create a widget for filtering sensors with multiple selection options.
    
    Args:
        all_sensors: List of all available sensor IDs or types
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
        
    Returns:
        List of selected sensors
    """
    # Create a multi-select widget for sensors
    filter_method = st.radio(
        "Filter Method",
        options=["Select All", "Select Specific Sensors"],
        horizontal=True,
        key=f"{key_prefix}_method"
    )
    
    if filter_method == "Select All":
        return all_sensors
    else:
        return st.multiselect(
            "Select Sensors",
            options=all_sensors,
            default=all_sensors[:5] if len(all_sensors) > 5 else all_sensors,
            key=f"{key_prefix}_selection"
        )

def create_visualization_controls(
    df: pd.DataFrame,
    numeric_columns: List[str],
    key_prefix: str = "viz_controls"
) -> Dict[str, Any]:
    """
    Create widgets for controlling visualization parameters.
    
    Args:
        df: DataFrame containing the data
        numeric_columns: List of numeric columns that can be visualized
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
        
    Returns:
        Dict containing the selected visualization parameters
    """
    st.subheader("Visualization Controls")
    
    # Create tabs for different control categories
    tab1, tab2, tab3 = st.tabs(["Basic", "Advanced", "Appearance"])
    
    result = {}
    
    # Basic controls
    with tab1:
        result["plot_type"] = st.selectbox(
            "Plot Type",
            options=["Line Plot", "Area Chart", "Bar Chart", "Scatter Plot"],
            key=f"{key_prefix}_plot_type"
        )
        
        result["y_column"] = st.selectbox(
            "Value Column", 
            options=numeric_columns,
            key=f"{key_prefix}_y_column"
        )
        
        result["group_by"] = st.selectbox(
            "Group By",
            options=["None"] + [col for col in df.columns if col not in numeric_columns and col != "timestamp"],
            key=f"{key_prefix}_group_by"
        )
        if result["group_by"] == "None":
            result["group_by"] = None
    
    # Advanced controls
    with tab2:
        result["resample_freq"] = st.selectbox(
            "Resample Frequency",
            options=[
                ("None", None),
                ("1 Minute", "1min"),
                ("5 Minutes", "5min"),
                ("15 Minutes", "15min"),
                ("30 Minutes", "30min"),
                ("1 Hour", "1H"),
                ("3 Hours", "3H"),
                ("12 Hours", "12H"),
                ("1 Day", "1D")
            ],
            format_func=lambda x: x[0],
            key=f"{key_prefix}_resample"
        )
        result["resample_freq"] = result["resample_freq"][1]
        
        result["rolling_window"] = st.slider(
            "Rolling Average Window",
            min_value=1,
            max_value=30,
            value=1,
            key=f"{key_prefix}_rolling"
        )
        
        result["show_confidence"] = st.checkbox(
            "Show Confidence Interval",
            value=False,
            key=f"{key_prefix}_confidence"
        )
    
    # Appearance controls
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            result["color_palette"] = st.selectbox(
                "Color Palette",
                options=["viridis", "plasma", "inferno", "magma", "cividis", "muted", "deep", "pastel"],
                key=f"{key_prefix}_palette"
            )
        
        with col2:
            result["plot_height"] = st.slider(
                "Plot Height",
                min_value=300,
                max_value=800,
                value=400,
                step=50,
                key=f"{key_prefix}_height"
            )
        
        result["show_grid"] = st.checkbox(
            "Show Grid",
            value=True,
            key=f"{key_prefix}_grid"
        )
        
        result["title"] = st.text_input(
            "Custom Title",
            value="",
            key=f"{key_prefix}_title"
        )
    
    return result
