from src.core.tools.standard.standard_tools import get_standard_tools
from src.core.tools.sensor.sensor_tools import get_sensor_tools
from langchain.tools import StructuredTool
from typing import List

def get_all_tools() -> List[StructuredTool]:
    """
    Get all available tools with a simplified approach focusing on core functionality.
    
    Returns:
        List[StructuredTool]: A list of all available tool objects
    """
    # Combine all tool lists but remove duplicates that might exist in multiple categories
    all_tools = get_standard_tools() + get_sensor_tools()
    
    # We'll only add enhanced tools that aren't already in the other categories
    # For this simplified version, we're not including enhanced visualization tools
    # to avoid duplication with compare_pm10_pm25_levels
    
    return all_tools
