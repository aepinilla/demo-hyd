"""
Simplified tools module for the visualization chatbot application.

This module focuses on core visualization tools: heatmaps, histograms, and scatterplots
for both CSV and sensor data.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool

# Import visualization utilities
from src.utils.visualization import (
    load_dataset,
    plot_histogram,
    plot_scatter,
    plot_heatmap,
    get_column_types,
    # plot_line
)

# Core visualization input schemas
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

class PlotHeatmapInput(BaseModel):
    """Input schema for heatmap."""
    columns: Union[List[str], str] = Field("all", description="List of columns or 'all' for all numeric columns")
    title: str = Field("Correlation Heatmap", description="Plot title")

class GetColumnTypesInput(BaseModel):
    """Input schema for getting column types."""
    columns: Optional[str] = Field(None, description="Optional comma-separated list of columns to check. If not provided, checks all columns.")

# Sensor data input schemas
class FetchLatestDataInput(BaseModel):
    """Input schema for fetching latest sensor data."""
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

def get_standard_tools() -> List[StructuredTool]:
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
        # StructuredTool.from_function(
        #     func=plot_line,
        #     name="plot_line",
        #     description="Create a line plot to show trends in data over a continuous variable.",
        #     args_schema=PlotLineInput,
        #     return_direct=False
        # ),
        StructuredTool.from_function(
            func=plot_heatmap,
            name="plot_heatmap",
            description="Create a correlation heatmap to visualize relationships between numeric variables.",
            args_schema=PlotHeatmapInput,
            return_direct=False
        )
    ]