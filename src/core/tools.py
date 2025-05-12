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
# from src.utils.time_series import plot_time_series, compare_time_periods
from src.utils.pollution_analysis import analyze_pollution_data, compare_pm10_pm25

# Basic visualization input schemas
class LoadDataInput(BaseModel):
    """Input schema for loading a dataset."""
    file_path: str = Field(..., description="Path to the CSV file to load")

class PlotHistogramInput(BaseModel):
    """Input schema for histogram."""
    column: Optional[str] = Field(None, description="Column to plot. If not provided, will auto-select first numeric column")
    bins: int = Field(10, description="Number of bins")
    title: str = Field("Histogram", description="Plot title")

class PlotScatterInput(BaseModel):
    """Input schema for scatter plot."""
    x_column: Optional[str] = Field(None, description="Column for x-axis. If not provided, will auto-select first numeric column")
    y_column: Optional[str] = Field(None, description="Column for y-axis. If not provided, will auto-select second numeric column")
    hue_column: Optional[str] = Field(None, description="Column for color grouping (optional)")
    title: str = Field("Scatter Plot", description="Plot title")

class PlotLineInput(BaseModel):
    """Input schema for line plot."""
    x_column: Optional[str] = Field(None, description="Column for x-axis. If not provided, will auto-select first numeric column")
    y_column: Optional[str] = Field(None, description="Column for y-axis. If not provided, will auto-select second numeric column")
    title: str = Field("Line Plot", description="Plot title")

class PlotHeatmapInput(BaseModel):
    """Input schema for heatmap."""
    columns: Union[List[str], str] = Field("all", description="List of columns or 'all' for all numeric columns")
    title: str = Field("Correlation Heatmap", description="Plot title")

# We're simplifying by removing the simplified input schemas

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
# SIMPLIFIED HELPER FUNCTIONS 
#

# Removed complex wrappers for scatter plots and heatmaps to simplify the codebase

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
            func=plot_scatter,
            name="plot_scatter",
            description="Create a scatter plot to show the relationship between two numerical columns.",
            args_schema=PlotScatterInput,
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
            func=plot_heatmap,
            name="plot_heatmap",
            description="Create a correlation heatmap to visualize relationships between numeric variables.",
            args_schema=PlotHeatmapInput,
            return_direct=False
        )
    ]

def plot_sensor_data(x_column: str = "timestamp", y_column: str = "value", title: str = "Sensor Data Time Series", hue_column: str = "value_type") -> str:
    """
    Create a time series plot of sensor data. This is a specialized function for visualizing
    sensor data with sensible defaults for sensor data visualization.
    
    Args:
        x_column (str): Column for x-axis, defaults to 'timestamp'
        y_column (str): Column for y-axis, defaults to 'value'
        title (str): Plot title
        hue_column (str): Column to use for grouping/coloring the lines, defaults to 'value_type'
        
    Returns:
        str: Result message with plot statistics
    """
    import streamlit as st
    import pandas as pd
    from src.utils.visualization import plot_line
    
    # Check if we have sensor data in session state
    if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        return "Error: No sensor data available. Please fetch sensor data first using fetch_latest_sensor_data."
    
    # Get the data and make a copy to avoid modifying the original
    df = st.session_state.latest_data.copy()
    
    # Verify that we have a timestamp column that can be used for time series visualization
    if 'timestamp' not in df.columns:
        return "Error: The sensor data does not contain a timestamp column required for time series visualization."
    
    # Check if the timestamp column is already a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        st.warning("Converting timestamp column to datetime format...")
        try:
            # Try to convert the timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Check if conversion was successful (no NaT values)
            if df['timestamp'].isna().all():
                return "Error: Could not convert timestamp column to datetime format. Please check the data."
            
            # Drop rows with NaT values if any
            if df['timestamp'].isna().any():
                original_len = len(df)
                df = df.dropna(subset=['timestamp'])
                st.warning(f"Dropped {original_len - len(df)} rows with invalid timestamps.")
                
        except Exception as e:
            return f"Error converting timestamp column: {str(e)}"
    
    # Store the processed data in session state
    st.session_state.dataset = df
    
    # Filter for SDS011 sensor data if requested in the title
    if "SDS011" in title and "sensor_type" in df.columns:
        df = df[df.sensor_type == "SDS011"]
        st.session_state.dataset = df
        st.info(f"Filtered data for SDS011 sensors: {len(df)} readings")
    
    # Filter for specific pollutant types if needed
    if "PM10" in title or "P1" in title:
        df = df[df.value_type == "P1"]
        st.session_state.dataset = df
        st.info(f"Filtered data for PM10 (P1) readings: {len(df)} readings")
    elif "PM2.5" in title or "P2" in title:
        df = df[df.value_type == "P2"]
        st.session_state.dataset = df
        st.info(f"Filtered data for PM2.5 (P2) readings: {len(df)} readings")
    
    # Check if we have enough data points for a meaningful visualization
    if len(df) < 2:
        return "Error: Not enough data points for a time series visualization after filtering."
    
    # Call the plot_line function with the specified parameters
    return plot_line(x_column, y_column, title, hue_column)

def prepare_sensor_data_for_visualization() -> str:
    """
    Prepare the latest fetched sensor data for visualization with the plotting tools.
    This function makes the sensor data available to the visualization tools without
    requiring the user to save and load a CSV file.
    
    Returns:
        str: Information about the prepared dataset
    """
    import streamlit as st
    
    # Check if we have sensor data in session state
    if 'latest_data' in st.session_state and not st.session_state.latest_data.empty:
        # Copy the data to the dataset session state variable used by visualization tools
        st.session_state.dataset = st.session_state.latest_data
        
        # Get information about the dataset
        df = st.session_state.latest_data
        return f"""
## Sensor Data Prepared for Visualization

**Source:** sensor.community API
**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Columns:** {', '.join(df.columns.tolist())}

**Column Types:**
```
{df.dtypes.to_string()}
```

**Sample Data:**
```
{df.head(5).to_string()}
```

You can now use visualization tools like plot_scatter, plot_histogram, etc. directly on this data.
        """
    else:
        return "Error: No sensor data available. Please fetch sensor data first using fetch_latest_sensor_data."

def get_sensor_tools() -> List[StructuredTool]:
    """
    Get the list of sensor data tools focused on reliable SDS011 dust sensor data.
    
    Returns:
        List[StructuredTool]: A list of sensor data tool objects
    """
    return [
        StructuredTool.from_function(
            func=fetch_latest_data,
            name="fetch_latest_sensor_data",
            description="Fetch the latest data from sensor.community API, focusing on SDS011 dust sensors measuring PM10 (P1) and PM2.5 (P2) levels.",
            # Don't use args_schema to avoid parameter handling issues
            return_direct=False
        ),
        StructuredTool.from_function(
            func=prepare_sensor_data_for_visualization,
            name="prepare_sensor_data",
            description="Prepare the latest fetched sensor data for visualization with the plotting tools. Use this after fetch_latest_sensor_data and before using any visualization tools.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_data,
            name="plot_sensor_data",
            description="Create a time series plot of sensor data with sensible defaults. This is a specialized function for visualizing sensor data that automatically handles datetime formatting and grouping by value_type.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=fetch_sensor_data,
            name="fetch_specific_sensor_data",
            description="Fetch data for a specific sensor by its API ID.",
            args_schema=FetchSensorDataInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=get_sensor_types,
            name="get_available_sensor_types",
            description="Get a list of available sensor types in the sensor.community network, with SDS011 being the most reliable for dust measurements.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=get_value_types,
            name="get_sensor_value_types",
            description="Get a mapping of value types to their descriptions, particularly P1 (PM10) and P2 (PM2.5) values from dust sensors.",
            return_direct=False
        ),
        # Simple time series visualization for sensor data
        StructuredTool.from_function(
            func=plot_line,
            name="plot_sensor_data",
            description="Create a simple line plot for sensor data values over time.",
            args_schema=PlotLineInput,
            return_direct=False
        ),
        # Pollution analysis tools
        StructuredTool.from_function(
            func=analyze_pollution_data,
            name="analyze_air_quality",
            description="Analyze air quality data with Air Quality Index (AQI) calculation, statistics, and visualizations.",
            args_schema=PollutionAnalysisInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=compare_pm10_pm25,
            name="compare_pm10_pm25_levels",
            description="Compare PM10 and PM2.5 levels over time, including correlation analysis.",
            args_schema=PM10PM25ComparisonInput,
            return_direct=False
        )
    ]

def get_enhanced_visualization_tools() -> List[StructuredTool]:
    """
    Get the list of simplified visualization tools for sensor.community data.
    
    Returns:
        List[StructuredTool]: A list of visualization tool objects focused on PM data
    """
    return [
        StructuredTool.from_function(
            func=compare_pm10_pm25,
            name="create_pm_comparison",
            description="Create a simplified comparison of PM10 (P1) and PM2.5 (P2) pollution levels from SDS011 sensors.",
            args_schema=PM10PM25ComparisonInput,
            return_direct=False
        )
    ]

def get_all_tools() -> List[StructuredTool]:
    """
    Get all available tools with a simplified approach focusing on core functionality.
    
    Returns:
        List[StructuredTool]: A list of all available tool objects
    """
    # Combine all tool lists but remove duplicates that might exist in multiple categories
    all_tools = get_visualization_tools() + get_sensor_tools()
    
    # We'll only add enhanced tools that aren't already in the other categories
    # For this simplified version, we're not including enhanced visualization tools
    # to avoid duplication with compare_pm10_pm25_levels
    
    return all_tools
