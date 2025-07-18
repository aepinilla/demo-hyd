from typing import List
from langchain.tools import StructuredTool

from .get_sensor_variable_stats import get_sensor_variable_stats
from .prepare_sensor_data_for_visualization import prepare_sensor_data_for_visualization
from .plot_sensor_histogram import plot_sensor_histogram
from .plot_sensor_scatter import plot_sensor_scatter
from .plot_sensor_heatmap import plot_sensor_heatmap
from .sync_sensor_data import sync_sensor_data
from src.utils.sensor_api import fetch_latest_data, get_sensor_types
from src.utils.remove_outliers import remove_outliers_iqr
from src.utils.visualization import plot_sensor_map

def get_sensor_tools() -> List[StructuredTool]:
    """
    Get the list of sensor data visualization tools.
    
    Returns:
        List[StructuredTool]: A list of sensor data visualization tool objects
    """
    return [        
        StructuredTool.from_function(
            func=get_sensor_variable_stats,
            name="sensor_get_variable_stats",
            description="[SENSOR DATA] Get information about available variables in the sensor API and their descriptive statistics.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=fetch_latest_data,
            name="fetch_latest_sensor_data",
            description="Fetch the latest data from sensor.community API, focusing on the data_type or sensor type indicated by the user.",
            # Don't use args_schema to avoid parameter handling issues
            return_direct=False
        ),
        StructuredTool.from_function(
            func=prepare_sensor_data_for_visualization,
            name="prepare_sensor_data",
            description="Prepare the latest fetched sensor data for visualization with the plotting tools. Use this after fetch_latest_sensor_data and before using any visualization tools.",
            return_direct=False
        ),
        # Specialized visualization tools for sensor data
        StructuredTool.from_function(
            func=plot_sensor_histogram,
            name="plot_sensor_histogram",
            description="Create histograms for sensor data variables to visualize their distributions.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_scatter,
            name="plot_sensor_scatter",
            description="Create a scatter plot comparing all variables mentioned by the user. If the user does not mention any variables, use all sensor variables to visualize their relationship.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_heatmap,
            name="plot_sensor_heatmap",
            description="Create a correlation heatmap for sensor variables to visualize relationships between different measurements. Do not include latitude and longitude in the heatmap.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=get_sensor_types,
            name="get_available_sensor_types",
            description="Get a list of available sensor types in the sensor.community network.",
            return_direct=False
        ),
        # Data synchronization tool to ensure proper data flow between components
        StructuredTool.from_function(
            func=sync_sensor_data,
            name="sync_sensor_data",
            description="Synchronize sensor data between different session state variables. Use this after fetch_latest_sensor_data and before using remove_outliers_iqr.",
            return_direct=False
        ),
        # Outlier removal tool for sensor data
        StructuredTool.from_function(
            func=remove_outliers_iqr,
            name="remove_outliers_from_sensor_data",
            description="Remove outliers from sensor data using the IQR method. Must use sync_sensor_data before calling this function.",
            return_direct=False
        ),
        # Plotly visualization tool for interactive maps
        StructuredTool.from_function(
            func=plot_sensor_map,
            name="plot_sensor_map",
            description="Create an interactive map visualization of sensor locations with color and size based on pollution values (P1, P2). Use this to show geographic distribution of sensor readings.",
            return_direct=False
        ),
    ]

