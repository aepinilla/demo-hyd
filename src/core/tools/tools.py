from src.core.tools.standard.standard_tools import get_standard_tools as get_std_tools
from src.core.tools.sensor.sensor_tools import get_sensor_tools as get_sns_tools
from langchain.tools import StructuredTool
from typing import List, Literal

# Define tool contexts for better organization
ToolContext = Literal["standard", "sensor", "all"]

def get_standard_tools() -> List[StructuredTool]:
    """
    Get the list of standard data visualization tools for CSV data.
    
    Returns:
        List[StructuredTool]: A list of standard visualization tool objects
    """
    # Rename standard tools to make them more specific
    tools = get_std_tools()
    
    # Update tool names and descriptions to clarify they're for CSV data
    for tool in tools:
        if not tool.name.startswith("csv_"):
            tool.name = f"csv_{tool.name}"
            tool.description = f"[CSV DATA] {tool.description}"
    
    return tools

def get_sensor_tools() -> List[StructuredTool]:
    """
    Get the list of sensor data visualization tools.
    
    Returns:
        List[StructuredTool]: A list of sensor visualization tool objects
    """
    # Rename sensor tools to make them more specific
    tools = get_sns_tools()
    
    # Update tool names and descriptions to clarify they're for sensor data
    for tool in tools:
        if not tool.name.startswith("sensor_") and "sensor" not in tool.name:
            tool.name = f"sensor_{tool.name}"
            tool.description = f"[SENSOR DATA] {tool.description}"
    
    return tools

def get_all_tools() -> List[StructuredTool]:
    """
    Get all available tools with clear context separation.
    
    Returns:
        List[StructuredTool]: A list of all available tool objects with proper naming
    """
    # Get tools with proper naming to avoid conflicts
    standard_tools = get_standard_tools()
    sensor_tools = get_sensor_tools()
    
    # Combine all tools with proper context
    all_tools = standard_tools + sensor_tools
    
    return all_tools

def get_tools_by_context(context: ToolContext = "all") -> List[StructuredTool]:
    """
    Get tools filtered by context.
    
    Args:
        context: The context to filter tools by ("standard", "sensor", or "all")
        
    Returns:
        List[StructuredTool]: A list of tools filtered by context
    """
    if context == "standard":
        return get_standard_tools()
    elif context == "sensor":
        return get_sensor_tools()
    else:  # "all"
        return get_all_tools()
