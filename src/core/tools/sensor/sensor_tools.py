from typing import List
from langchain.tools import StructuredTool

from .get_sensor_variable_stats import get_sensor_variable_stats
from .prepare_sensor_data_for_visualization import prepare_sensor_data_for_visualization
from .plot_sensor_histogram import plot_sensor_histogram
from .plot_sensor_scatter import plot_sensor_scatter
from .plot_sensor_heatmap import plot_sensor_heatmap
from src.utils.sensor_api import fetch_latest_data, get_sensor_types

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
    ]



# from datetime import datetime, timedelta

# from langchain.tools import StructuredTool
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# import numpy as np
# import pandas as pd
# from scipy.stats import pearsonr
# import seaborn as sns
# import streamlit as st

# from src.utils.sensor_api import fetch_latest_data, get_sensor_types
# from src.utils.visualization import (
#     check_sensor_data_quality,
#     create_sensor_pivot_table,
#     prepare_sensor_data,
# )

# from src.utils.visualization import create_sensor_pivot_table, prepare_sensor_data