"""
Comprehensive tools module for the visualization chatbot application.

This consolidated module includes all tools for data visualization, sensor data
operations, and enhanced visualization capabilities.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import streamlit as st
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.schema import Document

# Import all visualization utilities
from src.utils.visualization import (
    load_dataset,
    plot_histogram,
    plot_scatter,
    plot_line,
    plot_heatmap,
    get_column_types
)

# Import sensor API utilities
from src.utils.sensor_api import (
    fetch_latest_data,
    fetch_sensor_data,
    fetch_average_data,
    get_sensor_types,
    get_value_types
)

# Import all time series and data analysis utilities
from src.utils.time_series import plot_time_series, compare_time_periods
from src.utils.pollution_analysis import analyze_pollution_data, compare_pm10_pm25
from src.utils.sensor_plot_wrapper import plot_sensor_time_series, compare_sensor_time_periods
from src.utils.enhanced_time_series import (
    prepare_time_series_data,
    display_enhanced_time_series,
    create_pm_comparison_plot
)

# Import dataframe utilities
from src.utils.dataframe_utils import ensure_string_columns

#
# INPUT SCHEMAS
#

# Basic visualization input schemas
class LoadDataInput(BaseModel):
    """Input schema for loading a dataset."""
    file_path: str = Field(..., description="Path to the CSV file to load")

class PlotHistogramInput(BaseModel):
    """Input schema for histogram plotting."""
    column: str = Field(..., description="Column to plot histogram for")
    bins: int = Field(10, description="Number of bins for the histogram")
    title: str = Field("Histogram", description="Plot title")

class PlotScatterInput(BaseModel):
    """Input schema for scatter plot."""
    x_column: str = Field(..., description="Column for x-axis")
    y_column: str = Field(..., description="Column for y-axis")
    hue_column: Optional[str] = Field(None, description="Column for color grouping (optional)")
    title: str = Field("Scatter Plot", description="Plot title")

class PlotLineInput(BaseModel):
    """Input schema for line plot."""
    x_column: str = Field(..., description="Column for x-axis")
    y_column: str = Field(..., description="Column for y-axis")
    title: str = Field("Line Plot", description="Plot title")

class PlotHeatmapInput(BaseModel):
    """Input schema for heatmap."""
    columns: List[str] = Field(..., description="List of columns to include in correlation heatmap")
    title: str = Field("Correlation Heatmap", description="Plot title")

# Simplified input schemas for more flexible handling
class PlotScatterSimpleInput(BaseModel):
    """Input schema for scatter plot with simplified interface."""
    input_str: str = Field(..., description="Column for x-axis or JSON with x_column and y_column")
    
class PlotHeatmapSimpleInput(BaseModel):
    """Input schema for heatmap with simplified interface."""
    input_str: str = Field(..., description="Comma-separated list of columns or 'all' for all numeric columns")

class GetColumnTypesInput(BaseModel):
    """Input schema for getting column types."""
    columns: Optional[str] = Field(None, description="Optional comma-separated list of columns to check. If not provided, checks all columns.")

# Sensor data input schemas
class FetchLatestDataInput(BaseModel):
    """Input schema for fetching latest sensor data."""
    data_type: Optional[str] = Field(None, description="Sensor type to filter by (e.g., 'SDS011' for dust sensors or 'BME280' for temperature/humidity)")
    country: Optional[str] = Field(None, description="Country code to filter by (e.g., 'DE' for Germany, 'FR' for France)")
    area: Optional[str] = Field(None, description="Area filter in format 'lat,lon,distance' (e.g., '52.5200,13.4050,10')")
    box: Optional[str] = Field(None, description="Box filter in format 'lat1,lon1,lat2,lon2' (e.g., '52.1,13.0,53.5,13.5')")
    
    class Config:
        extra = "allow"  # Allow extra fields to be passed

class FetchSensorDataInput(BaseModel):
    """Input schema for fetching data for a specific sensor."""
    sensor_id: str = Field(..., description="The API ID of the sensor")

class FetchAverageDataInput(BaseModel):
    """Input schema for fetching average data."""
    timespan: str = Field("5m", description="Time window for which to get data ('5m', '1h', or '24h')")
    data_type: Optional[str] = Field(None, description="Filter for dust or temperature sensors ('dust' or 'temp')")

# Time series and visualization input schemas
class TimeSeriesPlotInput(BaseModel):
    """Input schema for time series plotting."""
    value_column: str = Field(..., description="Column containing the values to plot (e.g., 'value' or a specific value_type like 'P1', 'P2', 'temperature')")
    time_column: str = Field("timestamp", description="Column containing the timestamps")
    group_by: Optional[str] = Field(None, description="Optional column to group by (e.g., 'sensor_id' or 'sensor_type')")
    title: str = Field("Time Series Plot", description="Plot title")
    resample_freq: Optional[str] = Field(None, description="Optional frequency to resample data (e.g., '1H' for hourly)")

class TimePeriodsComparisonInput(BaseModel):
    """Input schema for comparing time periods."""
    value_column: str = Field(..., description="Column containing the values to compare")
    time_column: str = Field("timestamp", description="Column containing the timestamps")
    period1: str = Field("morning", description="Name of the first period")
    period2: str = Field("evening", description="Name of the second period")
    title: str = Field("Time Period Comparison", description="Plot title")

class PollutionAnalysisInput(BaseModel):
    """Input schema for pollution data analysis."""
    pollutant_type: str = Field("P1", description="Type of pollutant to analyze ('P1' for PM10 or 'P2' for PM2.5)")
    calculate_aqi: bool = Field(True, description="Whether to calculate Air Quality Index")
    show_statistics: bool = Field(True, description="Whether to show descriptive statistics")
    show_visualization: bool = Field(True, description="Whether to create visualizations")

class PM10PM25ComparisonInput(BaseModel):
    """Input schema for comparing PM10 and PM2.5 levels."""
    time_period: str = Field("24h", description="Time period to analyze ('24h', '7d', '30d')")
    resample_freq: str = Field("1H", description="Frequency for resampling time series data (e.g., '1H' for hourly, '30min' for half-hourly)")

# Enhanced visualization input schemas
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

#
# TOOL WRAPPERS AND HELPER FUNCTIONS
#

def scatter_plot_wrapper(input_str):
    """
    Wrapper for plot_scatter that handles both structured and single-parameter inputs.
    
    This wrapper helps the agent use the scatter plot tool more easily by accepting
    either a JSON string or separate parameters.
    """
    # If input is a string that might contain JSON
    if isinstance(input_str, str) and ('{' in input_str and '}' in input_str):
        # The visualization function will handle the JSON parsing
        return plot_scatter(input_str, "", None)
    else:
        # Otherwise, assume it's the x_column and call with empty y_column
        # The visualization function will handle getting the first two numeric columns
        return plot_scatter(input_str, "", None)
        
def heatmap_wrapper(input_str):
    """
    Wrapper for plot_heatmap that handles flexible inputs.
    
    This wrapper helps the agent use the heatmap tool more easily by accepting
    a string input that can be 'all' or a comma-separated list of columns.
    
    Args:
        input_str (str): Either 'all' for all numeric columns, or a comma-separated
                         list of column names (e.g. 'col1,col2,col3')
    """
    # Remove any surrounding quotes that might cause issues
    input_str = input_str.strip('"\'')
    return plot_heatmap(input_str)

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
        # If it's a recognized sensor type, filter for that
        if df_or_data_type in ['SDS011', 'DHT22', 'BME280', 'dust', 'temp']:
            df = fetch_latest_data(data_type=df_or_data_type)
        else:
            # Otherwise, just fetch the latest data
            df = fetch_latest_data()
    
    # Handle the case where df is already a DataFrame
    else:
        df = df_or_data_type
    
    # Prepare the data
    prepared_data = prepare_time_series_data(
        df, 
        value_column=value_column,
        time_column=time_column,
        group_by=group_by,
        resample_freq=resample_freq,
        rolling_window=rolling_window,
        time_period=time_period
    )
    
    # Ensure string columns for safety
    prepared_data = ensure_string_columns(prepared_data)
    
    # Create the visualization
    display_enhanced_time_series(
        prepared_data,
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
                if 'latest_data' in st.session_state:
                    df = st.session_state.latest_data
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

#
# TOOL COLLECTION FUNCTIONS
#

def get_visualization_tools() -> List[StructuredTool]:
    """
    Get the list of standard data visualization tools.
    
    Returns:
        List[StructuredTool]: A list of data visualization tool objects
    """
    return [
        StructuredTool.from_function(
            func=load_dataset,
            name="load_dataset",
            description="Load a dataset from a CSV file for visualization. Must be called before other visualization tools.",
            args_schema=LoadDataInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=get_column_types,
            name="get_column_types",
            description="Get detailed information about column data types and basic statistics. Use this before plotting to ensure you're working with the right data types.",
            args_schema=GetColumnTypesInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_histogram,
            name="plot_histogram",
            description="Create a histogram to visualize the distribution of a numerical column.",
            args_schema=PlotHistogramInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=scatter_plot_wrapper,
            name="plot_scatter",
            description="Create a scatter plot to show the relationship between two numerical columns. You can specify both columns like: {\"x_column\": \"column1\", \"y_column\": \"column2\"}",
            args_schema=PlotScatterSimpleInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_line,
            name="plot_line",
            description="Create a line plot to show trends in data over a continuous variable.",
            args_schema=PlotLineInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=heatmap_wrapper,
            name="plot_heatmap",
            description="Create a correlation heatmap to visualize relationships between numeric variables. First use get_column_types to identify numeric columns, then use 'all' for all numeric columns, or provide a comma-separated list.",
            args_schema=PlotHeatmapSimpleInput,
            return_direct=False
        )
    ]

def get_sensor_tools() -> List[StructuredTool]:
    """
    Get the list of sensor data tools.
    
    Returns:
        List[StructuredTool]: A list of sensor data tool objects
    """
    return [
        StructuredTool.from_function(
            func=fetch_latest_data,
            name="fetch_latest_sensor_data",
            description="Fetch the latest data from sensor.community API. Returns data from the past 5 minutes for various sensors based on optional filters.",
            # Don't use args_schema to avoid parameter handling issues
            return_direct=False
        ),
        StructuredTool.from_function(
            func=fetch_sensor_data,
            name="fetch_specific_sensor_data",
            description="Fetch data for a specific sensor by its API ID. Use this when you want data from a particular sensor.",
            args_schema=FetchSensorDataInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=fetch_average_data,
            name="fetch_average_sensor_data",
            description="Fetch average data for all sensors within a specific timespan (5 minutes, 1 hour, or 24 hours).",
            args_schema=FetchAverageDataInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=get_sensor_types,
            name="get_available_sensor_types",
            description="Get a list of available sensor types in the sensor.community network.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=get_value_types,
            name="get_sensor_value_types",
            description="Get a mapping of value types to their descriptions, explaining what each measurement represents.",
            return_direct=False
        ),
        # Time series visualization tools
        StructuredTool.from_function(
            func=plot_sensor_time_series,
            name="plot_sensor_time_series",
            description="Create a time series line plot for sensor data. Perfect for visualizing how measurements change over time.",
            args_schema=TimeSeriesPlotInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=compare_sensor_time_periods,
            name="compare_sensor_time_periods",
            description="Compare sensor readings between two time periods (e.g., morning vs evening) with statistical analysis.",
            args_schema=TimePeriodsComparisonInput,
            return_direct=False
        ),
        # Pollution analysis tools
        StructuredTool.from_function(
            func=analyze_pollution_data,
            name="analyze_air_quality",
            description="Analyze air quality data with Air Quality Index (AQI) calculation, statistics, and visualizations. Useful for understanding pollution levels and health implications.",
            args_schema=PollutionAnalysisInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=compare_pm10_pm25,
            name="compare_pm10_pm25_levels",
            description="Create a specialized visualization comparing PM10 and PM2.5 levels over time, including correlation analysis and statistics. Useful for understanding the relationship between different particulate matter sizes.",
            args_schema=PM10PM25ComparisonInput,
            return_direct=False
        )
    ]

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

def get_all_tools() -> List[StructuredTool]:
    """
    Get a complete list of all available tools.
    
    This function returns all tools from all categories in a single list.
    
    Returns:
        List[StructuredTool]: A complete list of all tool objects
    """
    return get_visualization_tools() + get_sensor_tools() + get_enhanced_visualization_tools()
