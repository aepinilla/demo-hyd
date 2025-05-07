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

# Define simplified input schemas for tools that need more flexible handling
class PlotScatterSimpleInput(BaseModel):
    """Input schema for scatter plot with simplified interface."""
    input_str: str = Field(..., description="Column for x-axis or JSON with x_column and y_column")
    
class PlotHeatmapSimpleInput(BaseModel):
    """Input schema for heatmap with simplified interface."""
    input_str: str = Field(..., description="Comma-separated list of columns or 'all' for all numeric columns")

# Helper functions for tool wrappers

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
        # Use a wrapper for scatter plot to handle JSON input better
        StructuredTool.from_function(
            func=scatter_plot_wrapper,  # Use the wrapper instead
            name="plot_scatter",
            description="Create a scatter plot to show the relationship between two numerical columns. You can specify both columns like: {\"x_column\": \"column1\", \"y_column\": \"column2\"}",
            # Use the proper Pydantic model
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
            description="Create a correlation heatmap to visualize relationships between numeric variables. Use 'all' for all numeric columns, or provide a comma-separated list like 'col1,col2,col3'. Do not use quotes around column names.",
            args_schema=PlotHeatmapSimpleInput,
            return_direct=False
        )
    ]
