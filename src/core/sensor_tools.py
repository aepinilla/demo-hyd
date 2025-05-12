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
from src.utils.time_series import plot_time_series, compare_time_periods

# Define input schemas for sensor tools
class FetchLatestDataInput(BaseModel):
    """Input schema for fetching latest sensor data."""
    data_type: Optional[str] = Field(None, description="Comma-separated list of sensor types (e.g., 'SDS011,BME280')")
    country: Optional[str] = Field(None, description="Comma-separated list of country codes (e.g., 'DE,BE')")
    area: Optional[str] = Field(None, description="Area filter in format 'lat,lon,distance' (e.g., '52.5200,13.4050,10')")
    box: Optional[str] = Field(None, description="Box filter in format 'lat1,lon1,lat2,lon2' (e.g., '52.1,13.0,53.5,13.5')")

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
            args_schema=FetchLatestDataInput,
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
            func=plot_time_series,
            name="plot_sensor_time_series",
            description="Create a time series line plot for sensor data. Perfect for visualizing how measurements change over time.",
            args_schema=TimeSeriesPlotInput,
            return_direct=False
        ),
        StructuredTool.from_function(
            func=compare_time_periods,
            name="compare_sensor_time_periods",
            description="Compare sensor readings between two time periods (e.g., morning vs evening) with statistical analysis.",
            args_schema=TimePeriodsComparisonInput,
            return_direct=False
        )
    ]
