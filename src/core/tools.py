"""
Data visualization tools for the chatbot application.

This module defines visualization-focused tools that can be used by the LangChain agent.
"""

from typing import List, Optional
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

from src.utils.visualization import load_dataset, plot_histogram, plot_scatter, plot_line, plot_heatmap

# Define input schemas for visualization tools
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

def get_tools() -> List[StructuredTool]:
    """
    Get the list of data visualization tools available to the agent.
    
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
            description="Create a correlation heatmap to visualize relationships between multiple variables.",
            args_schema=PlotHeatmapInput,
            return_direct=False
        )
    ]
