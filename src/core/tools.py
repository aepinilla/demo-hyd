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

def plot_sensor_data(x_column: str = "timestamp", y_column: str = "value", title: str = "Sensor Data Time Series", hue_column: str = "value_type", time_range: str = "recent", variables_to_plot: str = None) -> str:
    """
    Create a time series plot of sensor data. This is a specialized function for visualizing
    sensor data with sensible defaults for sensor data visualization.
    
    Args:
        x_column (str): Column for x-axis, defaults to 'timestamp'
        y_column (str): Column for y-axis, defaults to 'value'
        title (str): Plot title
        hue_column (str): Column to use for grouping/coloring the lines, defaults to 'value_type'
        time_range (str): Time range for data, options: 'recent', 'daily', 'weekly', 'monthly', 'yearly'
                          Note: The sensor.community API only provides recent data (last 5 minutes).
                          Historical data requires using the archive at https://archive.sensor.community/
        variables_to_plot (str): Comma-separated list of variables to plot (e.g., "PM10,PM2.5,temperature").
                                 Will be mapped to corresponding value_types in the dataset.
                                 If not provided, will ask user to specify variables (max 5).
        
    Returns:
        str: Result message with plot statistics
    """
    import streamlit as st
    import pandas as pd
    from src.utils.visualization import plot_line
    from datetime import datetime, timedelta
    
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
    
    # Check the time range requested and provide appropriate information
    if time_range != "recent":
        # Get the actual time range of the data
        min_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        time_diff = df['timestamp'].max() - df['timestamp'].min()
        
        # Inform the user about the data limitations
        st.warning(f"⚠️ **Data Limitation Notice**\n\n"
                  f"You requested '{time_range}' data, but the sensor.community API only provides data from the last 5 minutes.\n"
                  f"Current data range: {min_date} to {max_date} (approximately {time_diff.total_seconds()/60:.1f} minutes)\n\n"
                  f"For historical data, you would need to use the archive at https://archive.sensor.community/")
    
    # Store the processed data in session state
    st.session_state.dataset = df
    
    # Filter for SDS011 sensor data if requested in the title
    if "SDS011" in title and "sensor_type" in df.columns:
        df = df[df.sensor_type == "SDS011"]
        st.session_state.dataset = df
        st.info(f"Filtered data for SDS011 sensors: {len(df)} readings")
    
    # Get available value types in the dataset before filtering
    available_value_types = df['value_type'].unique().tolist()
    st.info(f"Available variables in the dataset: {', '.join(available_value_types)}")
    
    # Define common variable mappings
    variable_mappings = {
        "pm10": "P1",
        "pm2.5": "P2",
        "pm2_5": "P2",
        "temperature": "temperature",
        "humidity": "humidity",
        "pressure": "pressure",
        "p1": "P1",
        "p2": "P2"
    }
    
    # If variables_to_plot is not provided, either ask user to specify or limit to P1 and P2
    if variables_to_plot is None:
        # Always limit the number of variables when none are specified
        if len(available_value_types) > 5:
            st.warning("⚠️ Too many variables available. Please specify which ones you want to visualize or I'll show just PM10 (P1) and PM2.5 (P2).")
            
            # Default to just showing P1 and P2 if available
            default_vars = [var for var in available_value_types if var in ["P1", "P2"]]
            
            if default_vars:
                df = df[df['value_type'].isin(default_vars)]
                st.session_state.dataset = df
                st.info(f"Showing only the most common particulate matter measurements: {', '.join(default_vars)}")
            else:
                # If P1 and P2 aren't available, just take the first 2 value types
                limited_vars = available_value_types[:2]
                df = df[df['value_type'].isin(limited_vars)]
                st.session_state.dataset = df
                st.info(f"Limited to first two variables: {', '.join(limited_vars)}")
        else:
            # If there are 5 or fewer variables, we can show all of them
            st.info(f"Showing all available variables: {', '.join(available_value_types)}")
            
        # Update the session state with the filtered data
        st.session_state.dataset = df
    else:
        # Parse the variables to plot
        requested_vars = [v.strip().lower() for v in variables_to_plot.split(',')]
        
        # Map user-friendly names to actual value_types
        mapped_vars = []
        for var in requested_vars:
            if var in variable_mappings:
                mapped_vars.append(variable_mappings[var])
            else:
                # Try to find a close match
                for available_var in available_value_types:
                    if var in available_var.lower() or available_var.lower() in var:
                        mapped_vars.append(available_var)
                        break
        
        # Limit to 5 variables
        mapped_vars = mapped_vars[:5]
        
        if not mapped_vars:
            st.warning(f"None of the requested variables ({variables_to_plot}) could be mapped to available value types.")
            st.info(f"Available variables: {', '.join(available_value_types)}")
            return f"Error: Could not map requested variables to available value types. Available: {', '.join(available_value_types)}"
        
        # Filter the dataframe to only include the requested variables
        df = df[df['value_type'].isin(mapped_vars)]
        st.session_state.dataset = df
        st.info(f"Filtered data for variables: {', '.join(mapped_vars)} ({len(df)} readings)")
    
    # Limit the number of variables if there are too many
    if len(df['value_type'].unique()) > 5:
        top_value_types = df['value_type'].value_counts().head(5).index.tolist()
        df = df[df['value_type'].isin(top_value_types)]
        st.warning(f"Limiting visualization to the top 5 most frequent variables: {', '.join(top_value_types)}")
        st.session_state.dataset = df
    
    # Check if we have enough data points for a meaningful visualization
    if len(df) < 2:
        return "Error: Not enough data points for a time series visualization after filtering."
    
    # Update the title to reflect the actual time range and variables
    if "last year" in title.lower() or "yearly" in title.lower() or time_range in ["yearly", "monthly", "weekly", "daily"]:
        min_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        title = f"{title.split('(')[0].strip()} (Available data: {min_date} to {max_date})"
    
    # Update title to show which variables are being plotted
    plotted_vars = df['value_type'].unique().tolist()
    if len(plotted_vars) <= 5:
        title = f"{title} - Variables: {', '.join(plotted_vars)}"
    
    # Create a new figure for our plot to ensure we're only showing the filtered data
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the line plot with only our filtered data
    sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    # Improve datetime axis formatting
    if pd.api.types.is_datetime64_any_dtype(df[x_column]):
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Determine appropriate date format based on the time range
        time_range = df[x_column].max() - df[x_column].min()
        if time_range.total_seconds() < 3600:  # Less than an hour
            date_format = '%H:%M:%S'
        elif time_range.total_seconds() < 86400:  # Less than a day
            date_format = '%H:%M'
        elif time_range.days < 7:  # Less than a week
            date_format = '%m-%d %H:%M'
        elif time_range.days < 365:  # Less than a year
            date_format = '%b %d'
        else:  # More than a year
            date_format = '%Y-%m-%d'
            
        # Format the date ticks nicely
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
        
        # Add grid lines for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set appropriate number of ticks based on data size
        if len(df) > 20:
            # For larger datasets, limit the number of ticks to avoid overcrowding
            ax.xaxis.set_major_locator(MaxNLocator(10))
        
        # Adjust layout to make room for rotated labels
        fig.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Add some statistics
    y_stats = df[y_column].describe()
    
    # For datetime columns, show date range
    if pd.api.types.is_datetime64_any_dtype(df[x_column]):
        date_range = f"From {df[x_column].min()} to {df[x_column].max()}"
        x_stats_text = f"**Date Range:** {date_range}"
    else:
        x_stats = df[x_column].describe()
        x_stats_text = f"**{x_column}:**\n- Mean: {x_stats['mean']:.2f}\n- Std Dev: {x_stats['std']:.2f}"
    
    # Return statistics about the plot
    return f"""
## Line Plot Statistics

{x_stats_text}

**{y_column}:**
- Mean: {y_stats['mean']:.2f}
- Std Dev: {y_stats['std']:.2f}

**Range:**
- {x_column}: {df[x_column].min()} to {df[x_column].max()}
- {y_column}: {df[y_column].min()} to {df[y_column].max()}
    """

def prepare_sensor_data_for_visualization() -> str:
    """
    Prepare the latest fetched sensor data for visualization with the plotting tools.
    This function makes the sensor data available to the visualization tools without
    requiring the user to save and load a CSV file.
    
    Returns:
        str: Information about the prepared dataset
    """
    import streamlit as st
    import pandas as pd
    
    # Check if we have sensor data in session state
    if 'latest_data' in st.session_state and not st.session_state.latest_data.empty:
        # Get the data and make a copy to avoid modifying the original
        df = st.session_state.latest_data.copy()
        
        # Verify that we have a timestamp column that can be used for time series visualization
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            st.warning("Converting timestamp column to datetime format...")
            try:
                # Try to convert the timestamp column to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Check if conversion was successful (no NaT values)
                if df['timestamp'].isna().all():
                    st.warning("Could not convert timestamp column to datetime format. Some visualizations may not work properly.")
                
                # Drop rows with NaT values if any
                if df['timestamp'].isna().any():
                    original_len = len(df)
                    df = df.dropna(subset=['timestamp'])
                    st.warning(f"Dropped {original_len - len(df)} rows with invalid timestamps.")
                    
            except Exception as e:
                st.warning(f"Error converting timestamp column: {str(e)}")
        
        # Copy the data to the dataset session state variable used by visualization tools
        st.session_state.dataset = df
        
        # Get information about the dataset
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


def plot_sensor_histogram(variables_to_plot: str = "P1,P2", bins: int = 30, title: str = "Distribution of Sensor Values") -> str:
    """
    Create a histogram of sensor data values. This is specialized for visualizing the distribution
    of particulate matter levels from sensors.
    
    Args:
        variables_to_plot (str): Comma-separated list of variables to plot (e.g., "P1,P2")
        bins (int): Number of bins for the histogram
        title (str): Plot title
        
    Returns:
        str: Result message with histogram statistics
    """
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Check if we have sensor data in session state and fetch it if not available
    if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        st.info("Automatically fetching sensor data...")
        # Fetch the latest data
        fetch_result = fetch_latest_data()
        # Prepare the data for visualization
        prepare_result = prepare_sensor_data_for_visualization()
        
        # Check again if data is available
        if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
            return "Error: Unable to automatically fetch sensor data. Please try again."
    
    # Get the data and make a copy to avoid modifying the original
    df = st.session_state.latest_data.copy()
    
    # Define common variable mappings
    variable_mappings = {
        "pm10": "P1",
        "pm2.5": "P2",
        "pm2_5": "P2",
        "p1": "P1",
        "p2": "P2"
    }
    
    # Parse the variables to plot
    requested_vars = [v.strip().lower() for v in variables_to_plot.split(',')]
    
    # Map user-friendly names to actual value_types
    mapped_vars = []
    for var in requested_vars:
        if var in variable_mappings:
            mapped_vars.append(variable_mappings[var])
        else:
            # Try to find a close match
            for available_var in df['value_type'].unique():
                if var in available_var.lower() or available_var.lower() in var:
                    mapped_vars.append(available_var)
                    break
    
    # Limit to 5 variables
    mapped_vars = mapped_vars[:5]
    
    if not mapped_vars:
        available_vars = df['value_type'].unique().tolist()
        st.warning(f"None of the requested variables ({variables_to_plot}) could be mapped to available value types.")
        st.info(f"Available variables: {', '.join(available_vars)}")
        return f"Error: Could not map requested variables to available value types. Available: {', '.join(available_vars)}"
    
    # Filter the dataframe to only include the requested variables
    filtered_df = df[df['value_type'].isin(mapped_vars)]
    
    if filtered_df.empty:
        return "Error: No data available for the requested variables after filtering."
    
    # Create a figure with subplots for each variable
    fig, axes = plt.subplots(len(mapped_vars), 1, figsize=(10, 4 * len(mapped_vars)), sharex=False)
    
    # If only one variable, axes is not an array
    if len(mapped_vars) == 1:
        axes = [axes]
    
    # Plot histograms for each variable
    for i, var in enumerate(mapped_vars):
        var_df = filtered_df[filtered_df['value_type'] == var]
        
        # Skip if no data for this variable
        if var_df.empty:
            continue
        
        # Plot histogram
        sns.histplot(var_df['value'], bins=bins, kde=True, ax=axes[i])
        
        # Set title and labels
        axes[i].set_title(f"Distribution of {var}")
        axes[i].set_xlabel(f"{var} Value")
        axes[i].set_ylabel("Frequency")
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Display the plot
    st.pyplot(fig)
    
    # Generate statistics for each variable
    stats = []
    for var in mapped_vars:
        var_df = filtered_df[filtered_df['value_type'] == var]
        if not var_df.empty:
            var_stats = var_df['value'].describe()
            stats.append(f"### {var} Statistics")
            stats.append(f"- **Count:** {var_stats['count']:.0f}")
            stats.append(f"- **Mean:** {var_stats['mean']:.2f}")
            stats.append(f"- **Std Dev:** {var_stats['std']:.2f}")
            stats.append(f"- **Min:** {var_stats['min']:.2f}")
            stats.append(f"- **25%:** {var_stats['25%']:.2f}")
            stats.append(f"- **Median:** {var_stats['50%']:.2f}")
            stats.append(f"- **75%:** {var_stats['75%']:.2f}")
            stats.append(f"- **Max:** {var_stats['max']:.2f}\n")
    
    return f"## Histogram Analysis of Sensor Data\n\n{''.join(stats)}"


def plot_sensor_scatter(x_variable: str = "P1", y_variable: str = "P2", title: str = "Relationship Between Sensor Variables") -> str:
    """
    Create a scatter plot comparing two sensor variables. This is specialized for visualizing
    relationships between particulate matter levels from sensors.
    
    Args:
        x_variable (str): Variable for x-axis (e.g., "P1")
        y_variable (str): Variable for y-axis (e.g., "P2")
        title (str): Plot title
        
    Returns:
        str: Result message with correlation statistics
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr
    
    # Check if we have sensor data in session state and fetch it if not available
    if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        st.info("Automatically fetching sensor data...")
        # Fetch the latest data
        fetch_latest_data()
        # Prepare the data for visualization
        prepare_sensor_data_for_visualization()
        
        # Check again if data is available
        if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
            return "Error: Unable to automatically fetch sensor data. Please try again."
    
    # Get the data and make a copy to avoid modifying the original
    df = st.session_state.latest_data.copy()
    
    # Define common variable mappings
    variable_mappings = {
        "pm10": "P1",
        "pm2.5": "P2",
        "pm2_5": "P2",
        "p1": "P1",
        "p2": "P2"
    }
    
    # Map user-friendly names to actual value_types
    def map_variable(var):
        var = var.strip().lower()
        if var in variable_mappings:
            return variable_mappings[var]
        
        # Try to find a close match
        for available_var in df['value_type'].unique():
            if var in available_var.lower() or available_var.lower() in var:
                return available_var
        
        return None
    
    x_var = map_variable(x_variable)
    y_var = map_variable(y_variable)
    
    if not x_var or not y_var:
        available_vars = df['value_type'].unique().tolist()
        st.warning(f"Could not map requested variables to available value types.")
        st.info(f"Available variables: {', '.join(available_vars)}")
        return f"Error: Could not map requested variables to available value types. Available: {', '.join(available_vars)}"
    
    # Filter data for each variable
    x_data = df[df['value_type'] == x_var]['value'].reset_index(drop=True)
    y_data = df[df['value_type'] == y_var]['value'].reset_index(drop=True)
    
    # If the datasets have different lengths, take the shorter one
    min_length = min(len(x_data), len(y_data))
    x_data = x_data[:min_length]
    y_data = y_data[:min_length]
    
    if min_length < 2:
        return f"Error: Not enough data points for variables {x_var} and {y_var} to create a scatter plot."
    
    # Create a dataframe for the scatter plot
    scatter_df = pd.DataFrame({
        x_var: x_data,
        y_var: y_data
    })
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with regression line
    sns.regplot(x=x_var, y=y_var, data=scatter_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(f"{x_var} Value")
    ax.set_ylabel(f"{y_var} Value")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate correlation
    corr, p_value = pearsonr(x_data, y_data)
    
    # Add correlation annotation
    correlation_text = f"Correlation: {corr:.4f} (p-value: {p_value:.4f})"
    ax.annotate(correlation_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Display the plot
    st.pyplot(fig)
    
    # Generate correlation interpretation
    corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
    corr_direction = "positive" if corr > 0 else "negative"
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    
    return f"""## Scatter Plot Analysis: {x_var} vs {y_var}

### Correlation Statistics:
- **Pearson Correlation:** {corr:.4f}
- **P-value:** {p_value:.4f}
- **Sample Size:** {min_length} data points

### Interpretation:
- The scatter plot shows a **{corr_strength} {corr_direction}** correlation between {x_var} and {y_var}.
- This correlation is **{significance}** (p-value {'<' if p_value < 0.05 else '>'} 0.05).
- {'This suggests that as ' + x_var + ' increases, ' + y_var + ' tends to ' + ('increase' if corr > 0 else 'decrease') + ' as well.' if abs(corr) > 0.3 else 'There does not appear to be a strong linear relationship between these variables.'}
"""


def plot_sensor_heatmap(variables_to_plot: str = "all", title: str = "Correlation Heatmap of Sensor Variables") -> str:
    """
    Create a correlation heatmap for sensor variables. This is specialized for visualizing
    relationships between particulate matter levels and other sensor readings.
    
    Args:
        variables_to_plot (str): Comma-separated list of variables to include in the heatmap (e.g., "P1,P2,temperature")
        title (str): Plot title
        
    Returns:
        str: Result message with correlation insights
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Check if we have sensor data in session state and fetch it if not available
    if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        st.info("Automatically fetching sensor data...")
        # Fetch the latest data
        fetch_latest_data()
        # Prepare the data for visualization
        prepare_sensor_data_for_visualization()
        
        # Check again if data is available
        if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
            return "Error: Unable to automatically fetch sensor data. Please try again."
        return "Error: No sensor data available. Please fetch sensor data first using fetch_latest_sensor_data."
    
    # Get the data and make a copy to avoid modifying the original
    df = st.session_state.latest_data.copy()
    
    # Define common variable mappings
    variable_mappings = {
        "pm10": "P1",
        "pm2.5": "P2",
        "pm2_5": "P2",
        "temperature": "temperature",
        "humidity": "humidity",
        "pressure": "pressure",
        "p1": "P1",
        "p2": "P2"
    }
    
    # Parse the variables to plot
    if variables_to_plot.lower() == "all":
        # Use all available variables
        mapped_vars = df["value_type"].unique().tolist()
        st.info(f"Using all available variables: {', '.join(mapped_vars)}")
    else:
        requested_vars = [v.strip().lower() for v in variables_to_plot.split(",")]
        
        # Map user-friendly names to actual value_types
        mapped_vars = []
        for var in requested_vars:
            if var in variable_mappings:
                mapped_vars.append(variable_mappings[var])
            else:
                # Try to find a close match
                for available_var in df["value_type"].unique():
                    if var in available_var.lower() or available_var.lower() in var:
                        mapped_vars.append(available_var)
                        break
                    break
    
    # Ensure we have at least 2 variables
    if len(mapped_vars) < 2:
        available_vars = df['value_type'].unique().tolist()
        if len(mapped_vars) == 0:
            st.warning(f"None of the requested variables ({variables_to_plot}) could be mapped to available value types.")
        else:
            st.warning(f"Need at least 2 variables for a correlation heatmap. Only found: {mapped_vars[0]}")
        
        st.info(f"Available variables: {', '.join(available_vars)}")
        
        # If we have at least 2 available variables, use them instead
        if len(available_vars) >= 2:
            mapped_vars = available_vars[:min(5, len(available_vars))]
            st.info(f"Using these variables instead: {', '.join(mapped_vars)}")
        else:
            return f"Error: Not enough variables available for a correlation heatmap."
    
    # Limit to 5 variables for readability
    mapped_vars = mapped_vars[:5]
    
    # Create a pivot table with one column per variable
    pivot_data = {}
    
    # For each variable, extract its values
    for var in mapped_vars:
        var_data = df[df['value_type'] == var]['value'].reset_index(drop=True)
        if not var_data.empty:
            pivot_data[var] = var_data
    
    # Convert to dataframe
    if not pivot_data:
        return "Error: No data available for the requested variables after filtering."
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Drop rows with NaN values
    pivot_df = pivot_df.dropna()
    
    if len(pivot_df) < 2:
        return "Error: Not enough data points after filtering for NaN values."
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # Set title
    ax.set_title(title, fontsize=16)
    
    # Display the plot
    st.pyplot(fig)
    
    # Find strongest correlations
    # Get the upper triangle of the correlation matrix (excluding diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Convert to 1D series and sort by absolute value
    sorted_corrs = upper_tri.unstack().dropna().sort_values(key=abs, ascending=False)
    
    # Get top 3 correlations
    top_corrs = sorted_corrs[:3]
    
    # Generate insights
    insights = ["## Correlation Insights\n"]
    
    if not top_corrs.empty:
        insights.append("**Strongest correlations:**")
        for (var1, var2), corr in top_corrs.items():
            corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            corr_direction = "positive" if corr > 0 else "negative"
            insights.append(f"- {var1} and {var2}: {corr:.4f} ({corr_strength} {corr_direction})")
    
    return "\n".join(insights)

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
            description="Create a time series plot of sensor data with sensible defaults. This is a specialized function for visualizing sensor data that automatically handles datetime formatting and grouping by value_type. Specify variables_to_plot to limit which variables are shown (max 5). Note: The sensor.community API only provides recent data (last 5 minutes).",
            return_direct=False
        ),
        # Specialized visualization tools for sensor data
        StructuredTool.from_function(
            func=plot_sensor_histogram,
            name="plot_sensor_histogram",
            description="Create histograms for sensor data variables like PM10 (P1) and PM2.5 (P2) to visualize their distributions.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_scatter,
            name="plot_sensor_scatter",
            description="Create a scatter plot comparing two sensor variables (e.g., PM10 vs PM2.5) to visualize their relationship.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_heatmap,
            name="plot_sensor_heatmap",
            description="Create a correlation heatmap for sensor variables to visualize relationships between different measurements.",
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
