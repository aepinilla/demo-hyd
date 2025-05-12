"""
Sensor data tools for the chatbot application.

This module defines sensor.community API-focused tools that can be used by the LangChain agent.
"""

from typing import List, Optional
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

from src.utils.sensor_api import (
    fetch_latest_data, 
    fetch_sensor_data, 
    fetch_average_data,
    get_sensor_types,
    get_value_types
)
from src.utils.pollution_analysis import analyze_pollution_data, compare_pm10_pm25
from src.utils.sensor_plot_wrapper import plot_sensor_time_series, compare_sensor_time_periods

# Define input schemas for sensor tools
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
    

def get_sensor_tools() -> List[StructuredTool]:
    """
    Get the list of sensor data tools available to the agent.
    
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
