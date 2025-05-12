"""
Enhanced tools for sensor.community data visualization.

This module integrates advanced visualization features with LangChain for
AI-powered querying and interaction.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool
from langchain.schema import Document

from src.utils.enhanced_time_series import (
    prepare_time_series_data,
    display_enhanced_time_series,
    create_pm_comparison_plot
)
from src.utils.sensor_api import fetch_latest_data

class EnhancedTimeSeriesInput(BaseModel):
    """Input schema for enhanced time series visualization."""
    df_or_data_type: Union[str, Dict[str, Any]] = Field(
        ..., 
        description="Either a DataFrame identifier, a sensor type, or a dictionary with parameters for data fetching"
    )
    value_column: str = Field(
        ..., 
        description="Column containing the values to plot"
    )
    time_column: str = Field(
        "timestamp", 
        description="Column containing the timestamps"
    )
    group_by: Optional[str] = Field(
        None, 
        description="Optional column to group by"
    )
    title: str = Field(
        "Time Series Plot", 
        description="Plot title"
    )
    plot_type: str = Field(
        "Line Plot", 
        description="Type of plot ('Line Plot', 'Area Chart', 'Bar Chart', 'Scatter Plot')"
    )
    color_palette: str = Field(
        "viridis", 
        description="Seaborn color palette"
    )
    resample_freq: Optional[str] = Field(
        None, 
        description="Frequency for resampling time series data (e.g., '1H' for hourly)"
    )
    rolling_window: int = Field(
        1, 
        description="Window size for rolling average calculation"
    )
    show_confidence: bool = Field(
        False, 
        description="Whether to show confidence intervals"
    )
    time_period: str = Field(
        "24h", 
        description="Time period to analyze ('24h', '7d', '30d')"
    )

class PMComparisonInput(BaseModel):
    """Input schema for PM10 vs PM2.5 comparison."""
    df_or_data_type: Union[str, Dict[str, Any]] = Field(
        ..., 
        description="Either a DataFrame identifier, a sensor type, or a dictionary with parameters for data fetching"
    )
    time_period: str = Field(
        "24h", 
        description="Time period to analyze ('24h', '7d', '30d')"
    )
    resample_freq: str = Field(
        "1H", 
        description="Frequency for resampling time series data (e.g., '1H' for hourly, '30min' for half-hourly)"
    )

def create_enhanced_time_series_plot(
    df_or_data_type: Union[pd.DataFrame, str, Dict[str, Any]],
    value_column: str,
    time_column: str = 'timestamp',
    group_by: Optional[str] = None,
    title: str = 'Time Series Plot',
    plot_type: str = 'Line Plot',
    color_palette: str = 'viridis',
    resample_freq: Optional[str] = None,
    rolling_window: int = 1,
    show_confidence: bool = False,
    time_period: str = '24h'
) -> None:
    """
    Create an enhanced time series visualization using advanced plotting features.
    
    Args:
        df_or_data_type: Either a DataFrame, a string with a data type, or a dict with parameters
        value_column: Column containing the values to plot
        time_column: Column containing the timestamps
        group_by: Optional column to group by
        title: Plot title
        plot_type: Type of plot ('Line Plot', 'Area Chart', 'Bar Chart', 'Scatter Plot')
        color_palette: Seaborn color palette
        resample_freq: Frequency for resampling time series data
        rolling_window: Window size for rolling average calculation
        show_confidence: Whether to show confidence intervals
        time_period: Time period to analyze ('24h', '7d', '30d')
    """
    # Handle the case where df is a dictionary
    # Check if input might be a JSON string and parse it
    if isinstance(df_or_data_type, str) and df_or_data_type.startswith('{') and df_or_data_type.endswith('}'): 
        try:
            import json
            params = json.loads(df_or_data_type)
            # Now handle as dictionary
            df_or_data_type = params
        except:
            # If parsing fails, continue handling as a regular string
            pass
    
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
                    # Fetch new data
                    df = fetch_latest_data()
            else:
                # Assume df_param is the actual DataFrame
                df = df_param
        else:
            # No DataFrame provided, fetch latest data 
            data_type = params.get('data_type', 'SDS011')
            df = fetch_latest_data(data_type=data_type)
    
    # Handle the case where df is a string (e.g., a sensor type)
    elif isinstance(df_or_data_type, str):
        # Fetch data for the specified sensor type
        df = fetch_latest_data(data_type=df_or_data_type)
    
    # Handle the case where df is already a DataFrame
    else:
        df = df_or_data_type
    
    # Prepare the data for visualization
    prepared_df = prepare_time_series_data(
        df,
        value_column=value_column,
        time_column=time_column,
        group_by=group_by,
        resample_freq=resample_freq,
        rolling_window=rolling_window
    )
    
    # Display the enhanced time series visualization
    display_enhanced_time_series(
        prepared_df,
        value_column=value_column,
        time_column=time_column,
        group_by=group_by,
        title=title,
        plot_type=plot_type,
        color_palette=color_palette,
        show_confidence=show_confidence
    )
    
    return "Enhanced time series visualization created successfully."

def create_enhanced_pm_comparison(
    df_or_data_type: Union[pd.DataFrame, str, Dict[str, Any]],
    time_period: str = '24h',
    resample_freq: str = '1H'
) -> None:
    """
    Create an enhanced visualization comparing PM10 and PM2.5 levels.
    
    Args:
        df_or_data_type: Either a DataFrame, a string with a data type, or a dict with parameters
        time_period: Time period to analyze ('24h', '7d', '30d')
        resample_freq: Frequency for resampling time series data
    """
    # Print debug information to help with trouble-shooting
    print(f"Input parameters: df_or_data_type={df_or_data_type}, time_period={time_period}, resample_freq={resample_freq}")
    
    # Check if input might be a JSON string and parse it
    if isinstance(df_or_data_type, str) and '{' in df_or_data_type and '}' in df_or_data_type:
        try:
            import json
            # Try to extract a JSON object using a simple heuristic
            json_start = df_or_data_type.find('{')
            json_end = df_or_data_type.rfind('}') + 1
            json_str = df_or_data_type[json_start:json_end]
            params = json.loads(json_str)
            # Now handle as dictionary
            df_or_data_type = params
            # Also extract other parameters if they exist in the parsed JSON
            if 'time_period' in params:
                time_period = params.get('time_period', time_period)
            if 'resample_freq' in params:
                resample_freq = params.get('resample_freq', resample_freq)
        except Exception as e:
            # If parsing fails, continue handling as a regular string
            print(f"Failed to parse JSON string: {e}")
            pass
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
                if 'latest_data' in pd.options.mode.use_inf_as_na:
                    df = pd.options.mode.use_inf_as_na.latest_data
                else:
                    # Fetch new data (SDS011 sensors for PM data)
                    df = fetch_latest_data(data_type="SDS011")
            else:
                # Assume df_param is the actual DataFrame
                df = df_param
        else:
            # No DataFrame provided, fetch latest data (SDS011 sensors for PM data)
            df = fetch_latest_data(data_type="SDS011")
    
    # Handle the case where df is a string (e.g., a sensor type)
    elif isinstance(df_or_data_type, str):
        # If the string is "SDS011" or similar, use it directly
        if df_or_data_type in ["SDS011", "sds011"]:
            df = fetch_latest_data(data_type=df_or_data_type)
        else:
            # Otherwise, assume it's a general request for PM data
            df = fetch_latest_data(data_type="SDS011")
    
    # Handle the case where df is already a DataFrame
    else:
        df = df_or_data_type
    
    # Create the PM comparison visualization
    create_pm_comparison_plot(
        df,
        time_period=time_period,
        resample_freq=resample_freq
    )
    
    return "PM10 vs PM2.5 comparison visualization created successfully."

def get_enhanced_visualization_tools() -> List[StructuredTool]:
    """
    Get the list of enhanced visualization tools for sensor.community data.
    
    Returns:
        List[StructuredTool]: A list of enhanced visualization tool objects
    """
    return [
        StructuredTool.from_function(
            func=create_enhanced_time_series_plot,
            name="create_enhanced_time_series",
            description="Create an enhanced time series visualization with advanced features like confidence intervals, annotations, and statistical summaries.",
            args_schema=EnhancedTimeSeriesInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=create_enhanced_pm_comparison,
            name="create_pm_comparison",
            description="Create a comprehensive comparison of PM10 and PM2.5 pollution levels, including correlation analysis, ratio calculation, and threshold visualization.",
            args_schema=PMComparisonInput,
            return_direct=False
        )
    ]
